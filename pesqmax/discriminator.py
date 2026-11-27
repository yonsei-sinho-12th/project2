"""MetricGAN+ 스타일 PESQ 예측 판별자.

입력: (clean 압축 크기, 비교 대상 압축 크기) -> 정규화된 PESQ-WB 예측값 (0~1).
정규화: q = (PESQ - 1) / 3.5  => q = 1.0 이 PESQ 4.5(만점 목표)에 해당합니다.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm


def normalize_pesq(scores: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(scores, dtype=np.float64) - 1.0) / 3.5, 0.0, 1.0)


class LearnableSigmoid1d(nn.Module):
    def __init__(self, features: int = 1, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class MetricDiscriminator(nn.Module):
    def __init__(self, dim: int = 16, in_channel: int = 2):
        super().__init__()

        def block(cin, cout):
            return [spectral_norm(nn.Conv2d(cin, cout, (4, 4), (2, 2), (1, 1), bias=False)),
                    nn.InstanceNorm2d(cout, affine=True), nn.PReLU(cout)]

        self.layers = nn.Sequential(
            *block(in_channel, dim), *block(dim, dim * 2), *block(dim * 2, dim * 4),
            *block(dim * 4, dim * 8),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            spectral_norm(nn.Linear(dim * 8, dim * 4)), nn.Dropout(0.3), nn.PReLU(dim * 4),
            spectral_norm(nn.Linear(dim * 4, 1)), LearnableSigmoid1d(1))

    def forward(self, clean_mag_c, est_mag_c):  # [B, F, T] 각각
        return self.layers(torch.stack((clean_mag_c, est_mag_c), dim=1)).flatten()


class ReplayBuffer:
    """MetricGAN+ 의 historical set: 과거 생성 출력과 그 실제 PESQ 를 판별자 학습에 재사용."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.items: list = []
        self._next = 0

    def __len__(self):
        return len(self.items)

    def add(self, clean_mag_c, est_mag_c, q):
        if self.capacity <= 0:
            return
        for c, e, v in zip(clean_mag_c, est_mag_c, q):
            item = (c.detach().to("cpu", torch.float16), e.detach().to("cpu", torch.float16),
                    float(v))
            if len(self.items) < self.capacity:
                self.items.append(item)
            else:
                self.items[self._next] = item
                self._next = (self._next + 1) % self.capacity

    def sample(self, k: int, device, rng: np.random.Generator):
        if k <= 0 or len(self.items) < k:
            return None
        picks = rng.choice(len(self.items), size=k, replace=False)
        shapes = {self.items[i][0].shape for i in picks}
        if len(shapes) != 1:  # 길이가 다른 항목은 한 배치로 묶지 않음
            picks = [picks[0]]
        clean = torch.stack([self.items[i][0] for i in picks]).to(device, torch.float32)
        est = torch.stack([self.items[i][1] for i in picks]).to(device, torch.float32)
        q = torch.tensor([self.items[i][2] for i in picks], device=device, dtype=torch.float32)
        return clean, est, q
