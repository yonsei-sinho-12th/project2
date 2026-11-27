"""미분 가능한 PESQ 손실 — torch-pesq 0.1.2 (audiolabs/torch-pesq, MIT License) 이식본.

원본: https://github.com/audiolabs/torch-pesq  (Lorenz Schmidt, AudioLabs Erlangen)
원본 저작권 및 MIT 허가 고지는 LICENSES/torch-pesq-LICENSE 에 그대로 포함했습니다.

원본과 다른 점 (계산 내용은 동일하게 유지):
  * torchaudio / torchtyping / typeguard 의존성 제거 -> torch, numpy, scipy 만 필요.
    (최신 torch 에는 짝이 맞는 torchaudio 배포가 없을 수 있습니다.)
  * torchaudio.lfilter(IIR) 를 1024 탭으로 자른 임펄스 응답의 FFT 합성곱으로 대체.
    두 필터 모두 1024 탭 이후 잔여 에너지가 1e-26 미만이라 float32 에서 동일한 결과입니다.
  * 상수 텐서를 float32 buffer 로 등록(GPU 에서 float64 연산 방지, optimizer 에서 제외).
  * in-place 연산을 out-of-place 로 바꾸고, |X|^2 를 re^2+im^2 로 계산해 0 에서의 NaN 기울기를 제거.
  * 마지막 L6/L2 시간 집계만 float64 로 계산(1e-20 클램프 값의 6제곱이 float32 에서 0 이 되는 문제 방지).

입력: 16 kHz 파형 [B, N] (reference=clean, degraded=복원). 출력: 배치별 손실 [B].
원본과 같이 시간 정렬(time alignment)은 하지 않습니다(음성 향상에서는 정렬되어 있음).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import fft as sp_fft
from scipy.signal import butter, lfilter
from torch import nn

# fmt: off
NR_OF_HZ_BANDS_PER_BARK_BAND_16K = [
    1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4,
    3, 4, 5, 4, 5, 6, 6, 7, 8, 9, 9, 12, 12, 15, 16, 18, 21, 25, 20]
CENTRE_OF_BAND_BARK_16K = [
    0.078672, 0.316341, 0.636559, 0.961246, 1.290450, 1.624217, 1.962597, 2.305636, 2.653383,
    3.005889, 3.363201, 3.725371, 4.092449, 4.464486, 4.841533, 5.223642, 5.610866, 6.003256,
    6.400869, 6.803755, 7.211971, 7.625571, 8.044611, 8.469146, 8.899232, 9.334927, 9.776288,
    10.223374, 10.676242, 11.134952, 11.599563, 12.070135, 12.546731, 13.029408, 13.518232,
    14.013264, 14.514566, 15.022202, 15.536238, 16.056736, 16.583761, 17.117382, 17.657663,
    18.204674, 18.758478, 19.319147, 19.886751, 20.461355, 21.043034]
WIDTH_OF_BAND_BARK_16K = [
    0.157344, 0.317994, 0.322441, 0.326934, 0.331474, 0.336061, 0.340697, 0.345381, 0.350114,
    0.354897, 0.359729, 0.364611, 0.369544, 0.374529, 0.379565, 0.384653, 0.389794, 0.394989,
    0.400236, 0.405538, 0.410894, 0.416306, 0.421773, 0.427297, 0.432877, 0.438514, 0.444209,
    0.449962, 0.455774, 0.461645, 0.467577, 0.473569, 0.479621, 0.485736, 0.491912, 0.498151,
    0.504454, 0.510819, 0.517250, 0.523745, 0.530308, 0.536934, 0.543629, 0.550390, 0.557220,
    0.564119, 0.571085, 0.578125, 0.585232]
POW_DENS_CORRECTION_FACTOR_16K = [
    100.000000, 99.999992, 100.000000, 100.000008, 100.000008, 100.000015, 99.999992,
    99.999969, 50.000027, 100.000000, 99.999969, 100.000015, 99.999947, 100.000061, 53.047077,
    110.000046, 117.991989, 65.000000, 68.760147, 69.999931, 71.428818, 75.000038, 76.843384,
    80.968781, 88.646126, 63.864388, 68.155350, 72.547775, 75.584831, 58.379192, 80.950836,
    64.135651, 54.384785, 73.821884, 64.437073, 59.176456, 65.521278, 61.399822, 58.144047,
    57.004543, 64.126297, 54.311001, 61.114979, 55.077751, 56.849335, 55.628868, 53.137054,
    54.985844, 79.546974]
ABS_THRESH_POWER_16K = [
    51286152.000000, 2454709.500000, 70794.593750, 4897.788574, 1174.897705, 389.045166,
    104.712860, 45.708820, 17.782795, 9.772372, 4.897789, 3.090296, 1.905461, 1.258925,
    0.977237, 0.724436, 0.562341, 0.457088, 0.389045, 0.331131, 0.295121, 0.269153, 0.257040,
    0.251189, 0.251189, 0.251189, 0.251189, 0.263027, 0.288403, 0.309030, 0.338844, 0.371535,
    0.398107, 0.436516, 0.467735, 0.489779, 0.501187, 0.501187, 0.512861, 0.524807, 0.524807,
    0.524807, 0.512861, 0.478630, 0.426580, 0.371535, 0.363078, 0.416869, 0.537032]
# fmt: on
SP_16K = 6.910853e-006
SL_16K = 1.866055e-001
ZWICKER_POWER = 0.23
NBARKS = 49


def _impulse_response(b, a, taps: int) -> torch.Tensor:
    impulse = np.zeros(taps)
    impulse[0] = 1.0
    return torch.tensor(lfilter(np.asarray(b, float), np.asarray(a, float), impulse),
                        dtype=torch.float32)


class PesqLoss(nn.Module):
    def __init__(self, factor: float = 1.0, sample_rate: int = 16000, win_length: int = 512,
                 n_fft: int = 512, hop_length: int = 256, iir_taps: int = 1024):
        super().__init__()
        if sample_rate != 16000:
            raise ValueError("이 PESQ 손실은 16 kHz 입력만 지원합니다.")
        if n_fft != 512:
            raise ValueError("Bark 필터뱅크 표는 n_fft=512 (256 bins) 기준입니다.")
        self.factor = float(factor)
        self.win_length, self.n_fft, self.hop_length = win_length, n_fft, hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        b, a = butter(5, [325, 3250], fs=16000, btype="band")
        self.register_buffer("power_ir", _impulse_response(b, a, iir_taps), persistent=False)
        self.register_buffer("pre_ir", _impulse_response(
            [2.740826, -5.4816519, 2.740826], [1.0, -1.9444777, 0.94597794], iir_taps),
            persistent=False)
        self.register_buffer("edge_ramp", torch.linspace(0, 15, 16)[1:] / 16.0, persistent=False)

        fbank = torch.zeros(NBARKS, n_fft // 2)
        current = 0
        for i, width in enumerate(NR_OF_HZ_BANDS_PER_BARK_BAND_16K):
            fbank[i, current:current + width] = 1.0
            current += width
        self.register_buffer("fbank", fbank, persistent=False)
        self.register_buffer("pow_dens_correction", torch.tensor(
            POW_DENS_CORRECTION_FACTOR_16K, dtype=torch.float32) * SP_16K, persistent=False)
        width_bark = torch.tensor(WIDTH_OF_BAND_BARK_16K, dtype=torch.float32)
        self.register_buffer("width_bark", width_bark, persistent=False)
        self.total_width = float(width_bark[1:].sum())
        self.register_buffer("threshs", torch.tensor(ABS_THRESH_POWER_16K, dtype=torch.float32)
                             .view(1, 1, -1), persistent=False)
        exp = 6.0 / (torch.tensor(CENTRE_OF_BAND_BARK_16K, dtype=torch.float32) + 2.0)
        self.register_buffer("loud_exp", exp.clamp(min=1.0, max=2.0) ** 0.15 * ZWICKER_POWER,
                             persistent=False)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _fir(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        nfft = sp_fft.next_fast_len(n + h.shape[-1] - 1, real=True)
        y = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(h, nfft), nfft)
        return y[..., :n]

    def _align_level(self, x):
        filtered = self._fir(x, self.power_ir)
        power = filtered.square().sum(dim=1, keepdim=True) / (filtered.shape[1] + 5120) / 1.04684
        return x * (1e7 / power.clamp_min(1e-20)).sqrt()

    def _preemphasize(self, x):
        n = x.shape[1]
        ramp = torch.ones(n, device=x.device, dtype=x.dtype)
        ramp = torch.cat([self.edge_ramp, ramp[15:n - 15], torch.flip(self.edge_ramp, (0,))])
        return self._fir(x * ramp, self.pre_ir)

    def _power_spec(self, x):
        spec = torch.stft(x, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window, center=False, normalized=False, return_complex=True)
        power = spec.real.square() + spec.imag.square()  # [B, 257, frames]
        power = power.transpose(1, 2)
        dc_mask = torch.ones(power.shape[-1], device=x.device, dtype=power.dtype)
        dc_mask[0] = 0.0  # 원본과 같이 에너지(DC) 항 제외
        return power * dc_mask

    def _bark(self, power):
        return torch.einsum("ij,klj->kli", self.fbank, power[:, :, :-1]) * self.pow_dens_correction

    def _total_audible(self, x, factor):
        return (x * (x > self.threshs * factor)).sum(dim=2)

    def _time_avg_audible(self, x, silent):
        mask = (x > self.threshs * 100.0) & (~silent.unsqueeze(2))
        return (x * mask).mean(dim=1)

    def _loudness(self, pow_dens):
        loud = (2.0 * self.threshs) ** self.loud_exp * (
            (0.5 + 0.5 * pow_dens / self.threshs) ** self.loud_exp - 1)
        loud = torch.where(pow_dens <= self.threshs, torch.zeros_like(loud), loud)
        return loud * SL_16K

    def _weighted_norm(self, x, p):
        weighted = self.width_bark * x / self.total_width ** (1.0 / p)
        return self.total_width * torch.linalg.vector_norm(weighted[:, :, 1:], ord=p, dim=2)

    # ------------------------------------------------------------------ core
    def raw(self, ref: torch.Tensor, deg: torch.Tensor):
        with torch.autocast(device_type=ref.device.type, enabled=False):
            ref, deg = torch.atleast_2d(ref).float(), torch.atleast_2d(deg).float()
            max_val = torch.maximum(deg.abs().amax(dim=1, keepdim=True),
                                    ref.abs().amax(dim=1, keepdim=True)).clamp_min(1e-8)
            deg, ref = deg / max_val, ref / max_val
            ref, deg = self._align_level(ref), self._align_level(deg)
            ref, deg = self._preemphasize(ref), self._preemphasize(deg)
            deg = F.pad(deg, (0, deg.shape[1] % 256))
            ref = F.pad(ref, (0, ref.shape[1] % 256))

            deg, ref = self._bark(self._power_spec(deg)), self._bark(self._power_spec(ref))
            silent = self._total_audible(ref, 1e2) < 1e7
            mean_deg = self._time_avg_audible(deg, silent)
            mean_ref = self._time_avg_audible(ref, silent)
            band_ratio = ((mean_deg + 1000) / (mean_ref + 1000)).unsqueeze(1).clamp(0.01, 100.0)
            equ_ref = band_ratio * ref

            frame_ratio = (self._total_audible(equ_ref, 1.0) + 5e3) / (
                self._total_audible(deg, 1.0) + 5e3)
            frame_ratio = torch.cat(
                [frame_ratio[:, :1], frame_ratio[:, 1:] * 0.8 + frame_ratio[:, :-1] * 0.2], dim=1)
            frame_ratio = frame_ratio.clamp(3e-4, 5.0)
            equ_deg = frame_ratio.unsqueeze(2) * deg

            deg_loud, ref_loud = self._loudness(equ_deg), self._loudness(equ_ref)
            deadzone = 0.25 * torch.minimum(deg_loud, ref_loud)
            dist = deg_loud - ref_loud
            dist = dist.sign() * (dist.abs() - deadzone).clamp(min=0)

            symm = self._weighted_norm(dist, 2).clamp(min=1e-20)
            asymm_scale = ((equ_deg + 50.0) / (equ_ref + 50.0)) ** 1.2
            asymm_scale = torch.where(asymm_scale < 3.0, torch.zeros_like(asymm_scale),
                                      asymm_scale).clamp(max=12.0)
            asymm = self._weighted_norm(dist * asymm_scale, 1).clamp(min=1e-20)

            h = ((self._total_audible(equ_ref, 1.0) + 1e5) / 1e7) ** 0.04
            symm = (symm / h).clamp(max=45.0).double()
            asymm = (asymm / h).clamp(max=45.0).double()

            def aggregate(d):
                win = F.unfold(d.unsqueeze(1).unsqueeze(1), (1, 20), stride=10)
                psqm = (win ** 6).mean(dim=1) ** (1.0 / 6)
                return psqm.square().mean(dim=1).sqrt()

            return aggregate(symm).float(), aggregate(asymm).float()

    def mos(self, ref, deg):
        """PESQ-WB MOS-LQO 근사값 (P.862.2 매핑)."""
        d_symm, d_asymm = self.raw(ref, deg)
        mos = 4.5 - 0.1 * d_symm - 0.0309 * d_asymm
        return 0.999 + 4.0 / (1.0 + torch.exp(-1.3669 * mos + 3.8224))

    def forward(self, ref, deg):
        d_symm, d_asymm = self.raw(ref, deg)
        return self.factor * (0.1 * d_symm + 0.0309 * d_asymm)
