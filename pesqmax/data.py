"""챌린지 데이터 로딩과 도메인 매칭 증강.

데이터 분석 결과(배포본 기준):
  * 16 kHz 모노 PCM_16, 모든 파일 48,000 표본(3 초)
  * noisy = clean + noise (측정 SNR 과 target_snr_db 차이 < 0.002 dB, 클리핑 없음)
    -> noise = noisy - clean 으로 정확한 잡음을 복원해 다른 clean 과 다시 섞을 수 있습니다.
  * 잡음: NOISEX-92 + Nonspeech 88 종, 검증셋 잡음 코드는 모두 학습셋에 존재
  * SNR 은 난이도별로 정해진 집합에서 선택: easy {0,5,10,15}, moderate {5,10,15}, hard {10,15}
  * 화자: 학습 35명 / 검증 5명 (겹치지 않음) -> 비공개 테스트도 새 화자일 가능성이 높음

동적 리믹스는 위 생성 규칙(잡음 풀, 난이도별 SNR 분포, 3 초 전체 구간 SNR 정의)을 그대로 따라
새로운 (clean, noise, SNR) 조합을 만듭니다. 즉 평가셋과 같은 분포 안에서만 데이터를 늘립니다.
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset


def resolve_data_root(root: str | Path, repo: Path | None = None) -> Path:
    """현재 폴더 기준 -> 저장소 폴더 기준 순서로 train/, validation/ 이 있는 배포 폴더를 찾습니다."""
    candidates = [Path(root).expanduser()]
    if repo is not None:
        candidates.append(Path(repo) / root)
    for path in candidates:
        if (path / "train" / "samples.csv").is_file() or (path / "validation" / "samples.csv").is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"데이터 폴더를 찾지 못했습니다: {root}\n"
        "  --data_root 로 train/ 과 validation/ 이 들어 있는 배포 폴더를 지정하세요.")


def read_manifest(root: Path, split: str) -> list[dict]:
    path = root / split / "samples.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} 가 없습니다. data.root 를 배포 폴더로 지정하세요.")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} 가 비어 있습니다.")
    return rows


def read_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != sample_rate:
        raise ValueError(f"{sample_rate} Hz 가 아닌 파일: {path} ({rate} Hz)")
    if audio.shape[1] != 1:
        raise ValueError(f"모노가 아닌 파일: {path}")
    return audio[:, 0]


class ChallengeSplit:
    """한 분할(train / validation)의 clean·noisy 파형과 메타데이터를 메모리에 올립니다."""

    def __init__(self, root: str | Path, split: str, sample_rate: int = 16000,
                 max_files: int = 0, cache_dir: str | Path | None = ".cache"):
        root = Path(root).expanduser()
        rows = read_manifest(root, split)
        if max_files and max_files > 0:
            rows = rows[:max_files]
        self.root, self.split, self.rows = root, split, rows
        self.segment_ids = [r["segment_id"] for r in rows]
        self.categories = [r.get("noise_category", "unknown") for r in rows]
        self.difficulties = [r.get("noise_difficulty", "unknown") for r in rows]
        self.snrs = [float(r["target_snr_db"]) if r.get("target_snr_db") else float("nan")
                     for r in rows]
        self.noisy_paths = [root / r["noisy_path"] for r in rows]
        self.clean_paths = [root / r["clean_path"] for r in rows]

        key = hashlib.sha256(("|".join(f"{r['clean_path']},{r['noisy_path']}" for r in rows)
                              + f"|{sample_rate}").encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"{split}_{len(rows)}_{key}.npy" if cache_dir else None
        waves = None
        if cache is not None and cache.is_file():
            try:
                waves = np.load(cache, allow_pickle=False)
                if waves.ndim != 3 or waves.shape[0] != len(rows) or waves.shape[1] != 2:
                    waves = None
            except (OSError, ValueError):
                waves = None
        if waves is None:
            first = read_mono(self.noisy_paths[0], sample_rate)
            waves = np.empty((len(rows), 2, len(first)), dtype=np.float32)
            for i, (n_path, c_path) in enumerate(zip(self.noisy_paths, self.clean_paths)):
                noisy, clean = read_mono(n_path, sample_rate), read_mono(c_path, sample_rate)
                if len(noisy) != waves.shape[2] or len(clean) != waves.shape[2]:
                    raise ValueError(f"모든 파일의 길이가 같아야 합니다: {self.segment_ids[i]}")
                waves[i, 0], waves[i, 1] = noisy, clean
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_name(cache.stem + f".tmp{os.getpid()}.npy")
                np.save(tmp, waves, allow_pickle=False)
                tmp.replace(cache)
        self.noisy = np.ascontiguousarray(waves[:, 0])
        self.clean = np.ascontiguousarray(waves[:, 1])

    def __len__(self):
        return len(self.rows)

    @property
    def length(self) -> int:
        return self.clean.shape[1]


def build_snr_policy(difficulties, snrs) -> dict:
    """난이도별 SNR 경험 분포 {difficulty: (values, probs)} — 평가셋 생성 규칙을 그대로 모사."""
    table = defaultdict(Counter)
    for diff, snr in zip(difficulties, snrs):
        if np.isfinite(snr):
            table[diff][snr] += 1
    policy = {}
    for diff, counter in table.items():
        values = np.array(sorted(counter), dtype=np.float64)
        counts = np.array([counter[v] for v in values], dtype=np.float64)
        policy[diff] = (values, counts / counts.sum())
    all_counter = Counter(s for s in snrs if np.isfinite(s))
    values = np.array(sorted(all_counter), dtype=np.float64)
    counts = np.array([all_counter[v] for v in values], dtype=np.float64)
    policy["__all__"] = (values, counts / counts.sum())
    return policy


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    p_clean = float(np.sum(clean.astype(np.float64) ** 2))
    p_noise = float(np.sum(noise.astype(np.float64) ** 2))
    if p_clean <= 0 or p_noise <= 0:
        return clean.copy()
    gain = np.sqrt(p_clean / (p_noise * 10.0 ** (snr_db / 10.0)))
    return (clean + gain * noise).astype(np.float32)


class RemixTrainDataset(Dataset):
    def __init__(self, splits: list[ChallengeSplit], cfg_data, seed: int):
        self.clean = np.concatenate([s.clean for s in splits])
        self.noisy = np.concatenate([s.noisy for s in splits])
        self.categories = sum((s.categories for s in splits), [])
        self.difficulties = sum((s.difficulties for s in splits), [])
        snrs = sum((s.snrs for s in splits), [])
        self.length = self.clean.shape[1]
        sr = int(cfg_data["sample_rate"])
        self.segment = min(self.length, int(round(float(cfg_data["segment_seconds"]) * sr)))
        self.remix_prob = float(cfg_data["remix_prob"])
        self.same_cat_prob = float(cfg_data["remix_same_category_prob"])
        self.noise_roll = bool(cfg_data["noise_roll"])
        self.speed_prob = float(cfg_data["speed_perturb_prob"])
        self.speed_ratios = []
        for factor in cfg_data["speed_factors"]:
            ratio = Fraction(1.0 / float(factor)).limit_denominator(100)  # 길이 비율
            self.speed_ratios.append((ratio.numerator, ratio.denominator))
        self.policy = build_snr_policy(self.difficulties, snrs)
        by_cat = defaultdict(list)
        for i, cat in enumerate(self.categories):
            by_cat[cat].append(i)
        self.by_category = {k: np.array(v) for k, v in by_cat.items()}
        self.base_seed = int(seed)
        self._rng, self._rng_key = None, None

    def __len__(self):
        return len(self.clean)

    def _get_rng(self) -> np.random.Generator:
        info = torch.utils.data.get_worker_info()
        key = (os.getpid(), None if info is None else info.seed)
        if self._rng is None or self._rng_key != key:
            seed = self.base_seed if info is None else info.seed
            self._rng = np.random.default_rng(seed % (2 ** 63))
            self._rng_key = key
        return self._rng

    def _pick_noise(self, i: int, rng) -> int:
        pool = None
        if self.same_cat_prob > 0 and rng.random() < self.same_cat_prob:
            pool = self.by_category[self.categories[i]]
            if len(pool) < 2:
                pool = None
        if pool is None:
            j = int(rng.integers(len(self.clean) - 1))
            return j + 1 if j >= i else j
        j = int(rng.choice(pool))
        return j if j != i else int(rng.choice(pool))

    def _sample_snr(self, difficulty: str, rng) -> float:
        values, probs = self.policy.get(difficulty, self.policy["__all__"])
        return float(rng.choice(values, p=probs))

    def _speed(self, clean: np.ndarray, rng) -> np.ndarray:
        up, down = self.speed_ratios[int(rng.integers(len(self.speed_ratios)))]
        out = resample_poly(clean, up, down).astype(np.float32)
        if len(out) >= self.length:
            start = int(rng.integers(len(out) - self.length + 1))
            return out[start:start + self.length]
        return np.pad(out, (0, self.length - len(out)))

    def __getitem__(self, i: int):
        rng = self._get_rng()
        clean = self.clean[i]
        remix = self.remix_prob > 0 and rng.random() < self.remix_prob
        if self.speed_prob > 0 and rng.random() < self.speed_prob:
            clean, remix = self._speed(clean, rng), True  # 원래 noisy 와 정렬이 깨지므로 리믹스
        if remix:
            j = self._pick_noise(i, rng)
            noise = self.noisy[j] - self.clean[j]
            if self.noise_roll:
                noise = np.roll(noise, int(rng.integers(len(noise))))
            noisy = mix_at_snr(clean, noise, self._sample_snr(self.difficulties[j], rng))
        else:
            noisy = self.noisy[i]
        if self.segment < self.length:
            start = int(rng.integers(self.length - self.segment + 1))
            noisy, clean = noisy[start:start + self.segment], clean[start:start + self.segment]
        return (torch.from_numpy(np.array(noisy, dtype=np.float32)),
                torch.from_numpy(np.array(clean, dtype=np.float32)))
