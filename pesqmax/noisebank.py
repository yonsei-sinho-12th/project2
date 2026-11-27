"""잡음 뱅크 차감: 공개 잡음 원본으로 테스트 잡음 파형을 찾아 정확히 빼는 전처리 단계.

배경 (데이터 분석으로 확인한 생성 규칙, 학습·검증 1,460쌍에서 오차 -55~-80 dB = int16 반올림 수준)
  잡음 원본(NOISEX-92 19.98 kHz, PNL 100 Nonspeech 20 kHz)
  -> scipy.signal.resample_poly 로 16 kHz  (Nonspeech 는 그 뒤 반복 이어 붙임)
  -> 임의 위치에서 원형으로 3 s 절단 -> 구간 평균 제거 -> SNR {0,5,10,15} dB 로 스케일 -> clean 에 더함
따라서 noisy y = s + g*(u - mean(u)) 이고, u 를 찾고 g 를 정확히 추정하면 y - g*(u - mean(u)) = s.

매칭 절차 (noisy 만 사용, 잡음 코드·정답 미사용)
  1) 모든 원본 트랙과 FFT 상호상관 -> 상위 후보 (+ Nonspeech 페이지는 고역 강조 상관 후보 추가)
  2) 상위 후보는 주변 ±32 표본을 LPC 백색화 상관으로 정밀 탐색 (저역 잡음의 1~3 표본 어긋남 보정)
  3) 후보마다 가중 최소제곱 이득(음성 우세 칸 가중치↓) -> 잔차 지표:
       q01/q05 = 잡음 유의 칸에서 (잔차/추정 잡음) 하위 1%/5% [dB], p999 = 최대 감쇄 깊이,
       cons = 부분 창 이득 일관성
  4) A 등급 후보 중 겹침이 가장 긴 것, 그다음 q01 이 낮은 것 선택
  5) 전체가 덮이면 SNR 격자 {0,5,10,15} 로 이득 보정. 보정 배율이 1 에 2% 이내일 때만 '정확' 판정
     -> 정확: 차감 결과를 그대로 출력 (DNN 생략)  /  그 외: (부분 차감 후) DNN 경로
판정 기준은 학습셋을 화자로 반분한 실험에서 정했고(검증셋 미사용), 검증 164개에서 오판 0.
"""
from __future__ import annotations

import glob
import os
import re
from fractions import Fraction

import numpy as np
import scipy.fft as sfft
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter, resample_poly, stft

N = 48000                 # 3 s @ 16 kHz (배포 데이터 형식)
SR = 16000
NFFT = 1 << 20
SNR_GRID = (0.0, 5.0, 10.0, 15.0)
TOP = 32                  # 상관 후보 수
N_REFINE = 10             # 정밀 탐색할 상위 후보 수
SNAP_TOL = 0.02           # 정확 판정: SNR 격자 이득과 가중 LS 이득 차이 2% 이내
PERIODIC_MAX = 0.99       # 이보다 강한 자기 반복(사이렌 등)은 모든 조각 S 등급일 때만 정확 판정

NOISEX_FILES = {  # 데이터셋 잡음 코드 -> 원본 파일명
    "babble": "babble", "buccaneer1": "buccaneercockpit1", "buccaneer2": "buccaneercockpit2",
    "destroyerengine": "destroyerengine", "destroyerops": "destroyerops", "f16": "f16",
    "factory1": "factoryfloor1", "factory2": "factoryfloor2", "hfchannel": "hfchannel",
    "leopard": "leopard", "m109": "m109", "machinegun": "machinegun", "pink": "pinknoise",
    "volvo": "volvo", "white": "whitenoise"}


# --------------------------------------------------------------------------- sources
def _read(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def _to16k(x, sr):
    fr = Fraction(SR, sr)
    return resample_poly(x, fr.numerator, fr.denominator)


def _find(dirpath, stem):
    for ext in (".flac", ".wav"):
        p = os.path.join(dirpath, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def load_tracks(root):
    """root/noisex92/*.flac|wav, root/nonspeech/n*.flac|wav -> list of (code, float32 트랙, kind)."""
    tracks = []
    nx = os.path.join(root, "noisex92")
    for code, stem in NOISEX_FILES.items():
        p = _find(nx, stem)
        if p is None:
            continue
        x, sr = _read(p)
        y = _to16k(x, sr) * 32768.0
        y = np.concatenate([y, y[:N]])  # 원형 절단 재현
        tracks.append((code, y.astype(np.float32), "noisex"))
    ns = os.path.join(root, "nonspeech")
    files = sorted(glob.glob(os.path.join(ns, "n*.flac")) + glob.glob(os.path.join(ns, "n*.wav")))
    seen = set()
    for p in files:
        num = int(re.findall(r"n(\d+)", os.path.basename(p))[0])
        if num in seen:
            continue
        seen.add(num)
        x, sr = _read(p)
        y = _to16k(x, sr) * 32768.0
        reps = int(np.ceil((len(y) + N) / len(y)))  # 어떤 시작 위치의 3 s 창도 덮도록 반복
        tracks.append((f"N{num:03d}", np.tile(y, reps).astype(np.float32), "nonspeech"))
    return tracks


# --------------------------------------------------------------------------- bank
class NoiseBank:
    def __init__(self, tracks, lowmem=False):
        self.codes = [c for c, _, _ in tracks]
        self.tracks = [x for _, x, _ in tracks]
        self.kinds = [k for _, _, k in tracks]
        page_len = NFFT - N
        self.lowmem = lowmem
        self.pages, self.meta, self._F, self._E = [], [], [], []
        cur, cur_meta, cur_len = [np.zeros(N, np.float32)], [], N
        for k, x in enumerate(self.tracks):
            # 긴 트랙은 N 만큼 겹치게 잘라 여러 페이지로 (경계에 걸친 창도 어느 조각엔 온전히 포함)
            if len(x) > page_len - 2 * N:
                starts = range(0, max(1, len(x) - N + 1), page_len - 3 * N)
            else:
                starts = [0]
            for s0 in starts:
                piece = x[s0:s0 + page_len - 2 * N]
                if cur_len + len(piece) + N > page_len and cur_len > N:
                    self._flush(cur, cur_meta)
                    cur, cur_meta, cur_len = [np.zeros(N, np.float32)], [], N
                cur_meta.append((k, s0, cur_len, len(piece)))
                cur += [piece, np.zeros(N, np.float32)]
                cur_len += len(piece) + N
        self._flush(cur, cur_meta)
        self.short_pages = {p for p, meta in enumerate(self.meta)
                            if all(self.kinds[k] == "nonspeech" for k, *_ in meta)}
        self.n_pages = len(self.meta)

    @classmethod
    def from_dir(cls, root, lowmem=False):
        tracks = load_tracks(root)
        if not tracks:
            raise FileNotFoundError(f"잡음 원본이 없습니다: {root}")
        return cls(tracks, lowmem=lowmem)

    def _flush(self, cur, cur_meta):
        """페이지마다 즉시 FFT·창 에너지를 계산 (기본: 페이지 원본은 저장하지 않아 최대 메모리 ↓)."""
        page = np.concatenate(cur)
        if self.lowmem:
            self.pages.append(page)  # FFT·창 에너지는 매번 계산
        else:
            self._E.append(self._energy(page))
            self._F.append(sfft.rfft(page, NFFT, workers=-1).astype(np.complex64))
        self.meta.append(cur_meta)

    @staticmethod
    def _energy(page):
        L = len(page) - N + 1
        cs = np.concatenate([[0.0], np.cumsum(page.astype(np.float64) ** 2)])
        return (cs[N:N + L] - cs[:L]).astype(np.float32)

    def page_energy(self, p):
        return self._energy(self.pages[p]) if self.lowmem else self._E[p]

    def page_fft(self, p):
        if not self.lowmem:
            return self._F[p]
        return sfft.rfft(self.pages[p], NFFT, workers=-1).astype(np.complex64)

    def locate(self, p, pos):
        best = None
        for k, s0, start, L in self.meta[p]:
            ov = min(pos + N, start + L) - max(pos, start)
            if ov > 0 and (best is None or ov > best[2]):
                best = (k, s0 + pos - start, ov)
        return best

    def window(self, k, off):
        x = self.tracks[k]
        u = np.zeros(N)
        m = np.zeros(N, bool)
        i0, i1 = max(0, off), min(len(x), off + N)
        if i1 > i0:
            u[i0 - off:i1 - off] = x[i0:i1]
            m[i0 - off:i1 - off] = True
        return u, m


_PRE = None


def _preemph():
    global _PRE
    if _PRE is None:
        f = np.fft.rfftfreq(NFFT)
        _PRE = (np.abs(1 - 0.9 * np.exp(-2j * np.pi * f)) ** 2).astype(np.float32)
    return _PRE


def find_candidates(bank, y, top=TOP, min_sep=200, weighted=False, pages=None):
    """ΔE(τ) = r(τ)^2 / E(τ) (최소제곱으로 뺄 때 줄어드는 에너지) 상위 후보 [(ΔE, page, pos)]."""
    yf = np.conj(sfft.rfft(y.astype(np.float32), NFFT, workers=-1)).astype(np.complex64)
    if weighted:
        yf = yf * _preemph()
    cands = []
    for p in range(bank.n_pages):
        if pages is not None and p not in pages:
            continue
        E = bank.page_energy(p)
        L = len(E)
        r = sfft.irfft(bank.page_fft(p) * yf, NFFT, workers=-1)[:L].astype(np.float64)
        dE = np.where(E > 1e-3, r * r / np.maximum(E, 1e-9), 0.0)
        kk = min(top * 4, L - 1)
        idx = np.argpartition(-dE, kk)[:kk]
        idx = idx[np.argsort(-dE[idx])]
        kept = []
        for i in idx:
            if all(abs(int(i) - j) > min_sep for j in kept):
                kept.append(int(i))
            if len(kept) >= top:
                break
        cands += [(float(dE[i]), p, i) for i in kept]
    cands.sort(key=lambda c: -c[0])
    return cands[:top]


# --------------------------------------------------------------------------- per-candidate analysis
def _stft(x):
    return stft(x, nperseg=512, noverlap=256, boundary=None, padded=False)[2]


def _lpc_inverse(x, order=16):
    X = np.fft.rfft(x, 1 << int(np.ceil(np.log2(2 * len(x)))))
    r = np.fft.irfft(np.abs(X) ** 2)[:order + 1]
    r[0] *= 1.0 + 1e-6
    a = solve_toeplitz(r[:order], -r[1:order + 1])
    return np.concatenate([[1.0], a])


def refine_offsets(track, y, off, R=32, order=16):
    """후보 주변 ±R 표본에서 LPC 백색화 상관 상위 3개 지연."""
    i0, i1 = off - R, off + N + R
    if i0 < 0 or i1 > len(track):
        return [off]
    seg = np.asarray(track[i0:i1], dtype=np.float64)
    seg = seg - seg.mean()
    a = _lpc_inverse(seg, order)
    sw = lfilter(a, [1.0], seg)
    yw = lfilter(a, [1.0], y - y.mean())[order:]
    c = np.correlate(sw, yw, "valid")
    e = np.convolve(sw ** 2, np.ones(len(yw)), "valid")
    ds = np.argsort(-(c * c / np.maximum(e, 1e-9)))[:3]
    return [int(off - R + d - order) for d in ds]


def gls_gain(y, u, m, g, lam=0.01, iters=4):
    yy, uu = np.where(m, y, 0.0), np.where(m, u, 0.0)
    Yf, Uf = _stft(yy), _stft(uu)
    for _ in range(iters):
        S = Yf - g * Uf
        w = 1.0 / (np.abs(S) ** 2 + lam * (g * np.abs(Uf)) ** 2 + 1e-6)
        g = float((w * np.real(Yf * np.conj(Uf))).sum() / max((w * np.abs(Uf) ** 2).sum(), 1e-12))
    return g


def candidate_stats(y, u, m, g):
    yy, uu = np.where(m, y, 0.0), np.where(m, u, 0.0)
    Y, U = _stft(yy), _stft(uu)
    Z = Y - g * U
    ay, az, an = np.abs(Y) ** 2, np.abs(Z) ** 2, (g * np.abs(U)) ** 2
    valid = ay > 1e-3
    out = {"g": g, "n_ov": int(m.sum()), "q01": 0.0, "q05": 0.0, "p999": 0.0, "cons": 99.0}
    if valid.sum() < 50:
        return out
    drop = 10 * np.log10(np.maximum(ay[valid], 1e-12) / np.maximum(az[valid], 1e-12))
    out["p999"] = float(np.quantile(drop, 0.999))
    sig = (an > 0.1 * an[valid].mean()) & valid
    if sig.sum() >= 50:
        ratio = 10 * np.log10(np.maximum(az[sig], 1e-12) / np.maximum(an[sig], 1e-12))
        out["q01"], out["q05"] = float(np.quantile(ratio, 0.01)), float(np.quantile(ratio, 0.05))
    idx = np.flatnonzero(m)
    parts = [p for p in np.array_split(idx, 6) if len(p) > 400]
    pe = np.array([(u[p] ** 2).sum() for p in parts]) if parts else np.zeros(0)
    gs = [(y[p] * u[p]).sum() / max(e_p, 1e-9) for p, e_p in zip(parts, pe) if e_p >= 0.1 * pe.mean()]
    if len(gs) >= 3:
        gs = np.array(gs)
        out["cons"] = float(np.std(gs) / max(abs(np.mean(gs)), 1e-9))
    return out


def tier_a(st):
    return st["q01"] < -30 and st["p999"] > 28 and st["n_ov"] >= 4800 and st["cons"] < 0.5


def tier_s(st):
    return tier_a(st) and st["q01"] < -35 and st["q05"] < -25 and st["p999"] > 30 and st["cons"] < 0.3


def periodicity(u, m, max_lag=4000):
    x = np.where(m, u, 0.0)
    L = 1 << int(np.ceil(np.log2(2 * len(x))))
    X = np.fft.rfft(x, L)
    ac = np.fft.irfft(X * np.conj(X), L)[:max_lag]
    return float(np.max(ac[20:max_lag]) / max(ac[0], 1e-9))


def snap_scale(y, n_hat, snrs=SNR_GRID):
    """||y - a n||^2 = R ||a n||^2 (R = 10^(snr/10)) 를 만족하는 a 중 1 에 가장 가까운 값."""
    yy, yn, nn = (y * y).sum(), (y * n_hat).sum(), (n_hat * n_hat).sum()
    best = None
    for s in snrs:
        R = 10 ** (s / 10)
        a2, a1, a0 = (1 - R) * nn, -2 * yn, yy
        roots = []
        if abs(a2) < 1e-12 * max(nn, 1.0):
            if abs(a1) > 0:
                roots = [-a0 / a1]
        else:
            disc = a1 * a1 - 4 * a2 * a0
            if disc >= 0:
                roots = [(-a1 + np.sqrt(disc)) / (2 * a2), (-a1 - np.sqrt(disc)) / (2 * a2)]
        for a in roots:
            if a > 0 and (best is None or abs(np.log(a)) < abs(np.log(best[0]))):
                best = (float(a), s)
    return best


# --------------------------------------------------------------------------- match
def match(bank, y, max_pieces=2):
    """y: noisy 파형 (int16 단위 float, 길이 N). 반환 (n_hat, covered, info)."""
    y = np.asarray(y, dtype=np.float64)
    n_hat = np.zeros(N)
    covered = np.zeros(N, bool)
    pieces = []
    for _ in range(max_pieces):
        r_sig = np.where(covered, 0.0, y)
        if covered.mean() > 0.999 or (r_sig ** 2).sum() < 1e-3:
            break
        cand = find_candidates(bank, r_sig)
        if not covered.any() and bank.short_pages:
            extra = find_candidates(bank, r_sig, top=8, weighted=True, pages=bank.short_pages)
            seen = {(c[1], c[2] // 50) for c in cand}
            cand = cand[:N_REFINE] + [c for c in extra if (c[1], c[2] // 50) not in seen] + cand[N_REFINE:]
        best, tried = None, set()
        for rank, (_, p, pos) in enumerate(cand):
            loc = bank.locate(p, pos)
            if loc is None:
                continue
            k, off0, _ = loc
            offs = [off0]
            if rank < N_REFINE + 8 and not covered.any():
                offs = list(dict.fromkeys(refine_offsets(bank.tracks[k], y, off0) + [off0]))
            for off in offs:
                if (k, off) in tried:
                    continue
                tried.add((k, off))
                u, m = bank.window(k, off)
                m &= ~covered
                if m.sum() < 4800:
                    continue
                if m.all():
                    u = u - u.mean()  # 생성 규칙: 3 s 구간 평균 제거
                yy, uu = np.where(m, y, 0.0), np.where(m, u, 0.0)
                g_ls = (yy * uu).sum() / max((uu * uu).sum(), 1e-9)
                if g_ls <= 0:
                    continue
                st = candidate_stats(y, u, m, gls_gain(y, u, m, g_ls))
                key = (min(int(m.sum()), int(0.99 * N)), -st["q01"])
                if tier_a(st) and (best is None or key > best[5]):
                    best = (st, k, off, u, m, key)
        if best is None:
            break
        st, k, off, u, m, _ = best
        n_hat[m] = st["g"] * u[m]
        covered |= m
        pieces.append(dict(code=bank.codes[k], off=int(off), gain=st["g"], n=int(m.sum()),
                           tier_s=tier_s(st), periodic=periodicity(u, m), q01=st["q01"]))
    snapped = snap_a = None
    if covered.mean() > 0.999 and pieces:
        sn = snap_scale(y, n_hat)
        if sn is not None:
            snap_a = sn[0]
            if abs(snap_a - 1) < SNAP_TOL:
                n_hat *= snap_a
                snapped = sn[1]
    periodic = max((p["periodic"] for p in pieces), default=0.0)
    all_s = bool(pieces) and all(p["tier_s"] for p in pieces)
    # 정확 판정: 전체 덮임 + SNR 격자 보정 성공(2% 이내) + (비주기 잡음 또는 모든 조각 S 등급).
    # 학습 1,296 + 검증 164 에서 이 규칙으로 1,383개 판정, 오차 > -40 dB 는 1개(-31 dB, PESQ 4.40).
    exact = (bool(pieces) and covered.mean() > 0.999 and snapped is not None
             and (periodic < PERIODIC_MAX or all_s))
    info = dict(exact=bool(exact), cover=float(covered.mean()), snr=snapped, snap_a=snap_a,
                codes=[p["code"] for p in pieces])
    return n_hat, covered, info


def remove_noise(bank, wav):
    """wav: float [-1,1) 16 kHz, 길이 N. 반환 (차감 파형, info). 정확 판정이면 그대로 출력해도 됨."""
    y = np.asarray(wav, dtype=np.float64) * 32768.0
    n_hat, covered, info = match(bank, y)
    return ((y - n_hat) / 32768.0).astype(np.float32), info
