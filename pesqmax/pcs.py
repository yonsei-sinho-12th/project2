"""Perceptual Contrast Stretching (PCS) — 학습 가능한 주파수별 대비 스트레칭 레이어.

원 PCS(Chao et al., Interspeech 2022)는 int16 피크로 정규화한 신호의 STFT 크기에 대해
    L = log1p(|X|),  L' = gamma(f) * L,  |X'| = expm1(L')
를 적용합니다. gamma(f) 는 청각 대역 중요도(SII)에 따라 0.4~4.4 kHz 대역에서 1.4 로 가장 크고,
저역·고역으로 갈수록 1.0 에 가까워집니다(PCS400 표, n_fft=400 기준).

여기서는 gamma(f) 를 학습 가능한 파라미터로 두고(시그모이드로 [gamma_min, gamma_max] 범위 제한),
출력 에너지를 입력 에너지에 맞춰 다시 정규화합니다. PESQ 는 전체 레벨을 정렬하므로
에너지 재정규화는 점수에 영향을 주지 않으면서 학습을 안정시킵니다.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# (시작 bin, 끝 bin(미포함), gamma) — PCS 공개 구현의 PCS400 표 (201 bins, 40 Hz 간격)
PCS400_SEGMENTS = [
    (0, 3, 1.0),
    (3, 5, 1.070175439),
    (5, 8, 1.182456140),
    (8, 10, 1.287719298),
    (10, 110, 1.4),
    (110, 130, 1.322807018),
    (130, 160, 1.238596491),
    (160, 190, 1.161403509),
    (190, 201, 1.077192982),
]


def pcs400_table() -> np.ndarray:
    gamma = np.ones(201, dtype=np.float64)
    for start, end, value in PCS400_SEGMENTS:
        gamma[start:end] = value
    return gamma


def pcs_profile(n_fft: int, sample_rate: int = 16000) -> np.ndarray:
    """PCS400 표를 임의의 n_fft 주파수 축에 주파수 기준으로 옮깁니다."""
    table = pcs400_table()
    freqs = np.arange(n_fft // 2 + 1) * sample_rate / n_fft
    index = np.clip(np.round(freqs / (sample_rate / 400)).astype(int), 0, 200)
    return table[index]


class LearnablePCS(nn.Module):
    def __init__(self, n_freq: int, n_fft: int, sample_rate: int = 16000, init: str = "pcs400",
                 gamma_min: float = 0.8, gamma_max: float = 2.0, learnable: bool = True):
        super().__init__()
        if init == "pcs400":
            gamma0 = pcs_profile(n_fft, sample_rate)
        elif init == "identity":
            gamma0 = np.ones(n_freq)
        else:
            raise ValueError(f"알 수 없는 PCS 초기화: {init}")
        if len(gamma0) != n_freq:
            raise ValueError("PCS 주파수 bin 수가 STFT 와 다릅니다.")
        self.gamma_min, self.gamma_max = float(gamma_min), float(gamma_max)
        ratio = np.clip((gamma0 - gamma_min) / (gamma_max - gamma_min), 1e-4, 1 - 1e-4)
        theta = np.log(ratio / (1 - ratio))
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32), requires_grad=learnable)

    def gamma(self, strength: float = 1.0) -> torch.Tensor:
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(self.theta)
        if strength != 1.0:  # 추론 시 검증셋으로 조정하는 전체 강도 (1 = 학습된 값 그대로)
            gamma = 1.0 + strength * (gamma - 1.0)
        return gamma

    def forward(self, mag: torch.Tensor, scale: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """mag: 선형 크기 [B, F, T] (>=0), scale: 발화별 int16 피크 스케일 [B]."""
        gamma = self.gamma(strength).view(1, -1, 1).to(mag.dtype)
        log_mag = torch.log1p(mag * scale.view(-1, 1, 1))
        stretched = gamma * log_mag
        peak = stretched.amax(dim=(1, 2), keepdim=True).detach()
        # expm1(stretched) * exp(-peak): 작은 값도 정확하고, 제곱 에너지가 넘치지 않도록 스케일 조정
        # (stretched <= gamma*log1p(32767*200) < 40 이므로 expm1 자체는 float32 범위 안)
        out = torch.expm1(stretched) * torch.exp(-peak)
        energy_in = mag.square().sum(dim=(1, 2), keepdim=True)
        energy_out = out.square().sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        return out * torch.sqrt(energy_in / energy_out)

    def smoothness(self) -> torch.Tensor:
        gamma = self.gamma()
        return (gamma[1:] - gamma[:-1]).abs().mean()

    def extra_repr(self) -> str:
        return f"n_freq={self.theta.numel()}, gamma_range=({self.gamma_min}, {self.gamma_max})"
