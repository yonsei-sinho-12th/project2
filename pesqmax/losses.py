"""보조 손실: 위상 anti-wrapping 손실, Multi-Resolution STFT 손실."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def anti_wrapping(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x - torch.round(x / (2 * math.pi)) * 2 * math.pi)


def phase_losses(phase_ref: torch.Tensor, phase_est: torch.Tensor):
    """[B, F, T]: 순시 위상(IP), 군지연(GD, 주파수 차분), 순시 각주파수(IAF, 시간 차분)."""
    ip = anti_wrapping(phase_ref - phase_est).mean()
    gd = anti_wrapping(torch.diff(phase_ref, dim=1) - torch.diff(phase_est, dim=1)).mean()
    iaf = anti_wrapping(torch.diff(phase_ref, dim=2) - torch.diff(phase_est, dim=2)).mean()
    return ip, gd, iaf


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1 을 여러 해상도에서 평균."""

    def __init__(self, fft_sizes=(1024, 2048, 512), hop_sizes=(120, 240, 50),
                 win_lengths=(600, 1200, 240)):
        super().__init__()
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))
        for i, (_, _, win) in enumerate(self.resolutions):
            self.register_buffer(f"window_{i}", torch.hann_window(win), persistent=False)

    def _mag(self, x, i):
        n_fft, hop, win = self.resolutions[i]
        spec = torch.stft(x, n_fft, hop_length=hop, win_length=win,
                          window=getattr(self, f"window_{i}"), return_complex=True)
        return torch.sqrt(spec.real.square() + spec.imag.square() + 1e-7)

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=est.device.type, enabled=False):
            est, ref = est.float(), ref.float()
            total = est.new_zeros(())
            for i in range(len(self.resolutions)):
                mag_e, mag_r = self._mag(est, i), self._mag(ref, i)
                sc = torch.linalg.norm(mag_r - mag_e) / torch.linalg.norm(mag_r).clamp_min(1e-7)
                log_mag = F.l1_loss(torch.log(mag_e), torch.log(mag_r))
                total = total + sc + log_mag
            return total / len(self.resolutions)
