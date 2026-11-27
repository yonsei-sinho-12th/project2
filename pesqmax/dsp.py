"""STFT / iSTFT 와 크기 압축 (MP-SENet 과 동일한 규약)."""

from __future__ import annotations

import torch


def stft_mag_pha(wav: torch.Tensor, n_fft: int, hop: int, win: int, compress: float,
                 window: torch.Tensor):
    """파형 [B, N] -> 압축 크기 [B, F, T], 위상 [B, F, T], 압축 복소 [B, F, T, 2]."""
    spec = torch.stft(wav, n_fft, hop_length=hop, win_length=win, window=window,
                      center=True, pad_mode="reflect", normalized=False, return_complex=True)
    real, imag = spec.real, spec.imag
    mag = torch.sqrt(real.square() + imag.square() + 1e-9)
    pha = torch.atan2(imag + 1e-10, real + 1e-5)
    mag_c = mag.pow(compress)
    com_c = torch.stack((mag_c * torch.cos(pha), mag_c * torch.sin(pha)), dim=-1)
    return mag_c, pha, com_c


def istft_from_linear(mag: torch.Tensor, pha: torch.Tensor, n_fft: int, hop: int, win: int,
                      window: torch.Tensor, length: int) -> torch.Tensor:
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    return torch.istft(com, n_fft, hop_length=hop, win_length=win, window=window,
                       center=True, length=length)


def istft_from_compressed(mag_c: torch.Tensor, pha: torch.Tensor, n_fft: int, hop: int,
                          win: int, compress: float, window: torch.Tensor,
                          length: int) -> torch.Tensor:
    mag = mag_c.clamp_min(0.0).pow(1.0 / compress)
    return istft_from_linear(mag, pha, n_fft, hop, win, window, length)
