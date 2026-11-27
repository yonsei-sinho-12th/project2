"""추론 공통 함수: 체크포인트 로딩, TTA(시프트 평균), 긴 입력 분할 처리, 안전장치."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import merge_with_defaults, to_plain
from .model import Enhancer
from .utils import atomic_save

CHECKPOINT_FORMAT = "pesqmax-enhancer-v1"
DEFAULT_POSTPROC = {"use_post_pcs": True, "pcs_strength": 1.0, "tta_shifts": 1}


def export_checkpoint(path, cfg, generator_state: dict, postproc: dict, meta: dict) -> None:
    atomic_save({"format": CHECKPOINT_FORMAT, "config": to_plain(cfg),
                 "generator": generator_state, "postproc": to_plain(postproc),
                 "meta": to_plain(meta)}, path)


def load_enhancer(path: str | Path, device: torch.device):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} 는 추론용 체크포인트가 아닙니다 (best.pt 를 사용하세요).")
    cfg = merge_with_defaults(ckpt["config"])
    model = Enhancer(cfg)
    model.load_state_dict(ckpt["generator"])
    model.to(device).eval()
    postproc = dict(DEFAULT_POSTPROC)
    postproc.update(ckpt.get("postproc") or {})
    return model, postproc, ckpt


def tta_offsets(hop: int, shifts: int) -> list[int]:
    shifts = max(1, int(shifts))
    return [int(round(k * hop / shifts)) for k in range(shifts)]


@torch.no_grad()
def enhance_tensor(model: Enhancer, wav: torch.Tensor, use_post_pcs: bool = True,
                   pcs_strength: float = 1.0, tta_shifts: int = 1) -> torch.Tensor:
    """wav [B, N] -> [B, N]. tta_shifts>1 이면 STFT 프레임 격자를 hop/shifts 씩 옮겨 평균."""
    length = wav.shape[-1]
    short = max(0, 2 * model.n_fft - length)  # 아주 짧은 입력은 STFT 가 가능하도록 0 패딩
    if short:
        wav = F.pad(wav, (0, short))
    out = None
    offsets = tta_offsets(model.hop, tta_shifts)
    for off in offsets:
        x = F.pad(wav, (off, 0)) if off else wav
        y = model.enhance(x, use_post_pcs=use_post_pcs, pcs_strength=pcs_strength)
        y = y[..., off:off + length]
        out = y if out is None else out + y
    return out / len(offsets)


@torch.no_grad()
def enhance_long(model: Enhancer, wav: torch.Tensor, chunk: int, overlap: int,
                 **kwargs) -> torch.Tensor:
    """메모리 보호용: chunk 보다 긴 입력은 겹침-더하기(선형 crossfade)로 나눠 처리."""
    length = wav.shape[-1]
    if chunk <= 0 or length <= chunk:
        return enhance_tensor(model, wav, **kwargs)
    overlap = min(overlap, chunk // 2)
    step = chunk - overlap
    starts = list(range(0, length - chunk + 1, step))
    if starts[-1] + chunk < length:
        starts.append(length - chunk)
    acc = torch.zeros_like(wav)
    weight = torch.zeros(length, device=wav.device, dtype=wav.dtype)
    ramp = torch.linspace(0.0, 1.0, overlap + 2, device=wav.device, dtype=wav.dtype)[1:-1]
    for start in starts:
        w = torch.ones(chunk, device=wav.device, dtype=wav.dtype)
        if start > 0:
            w[:overlap] = ramp
        if start + chunk < length:
            w[-overlap:] = torch.flip(ramp, (0,))
        y = enhance_tensor(model, wav[..., start:start + chunk], **kwargs)
        acc[..., start:start + chunk] += y * w
        weight[start:start + chunk] += w
    return acc / weight.clamp_min(1e-8)


def safe_output(noisy: np.ndarray, enhanced: np.ndarray) -> tuple[np.ndarray, str | None]:
    """NaN/Inf 또는 사실상 무음 출력이면 입력을 그대로 내보내 PESQ 계산 오류를 막습니다."""
    if not np.all(np.isfinite(enhanced)):
        return noisy.copy(), "non-finite output"
    e_in = float(np.sum(noisy.astype(np.float64) ** 2))
    e_out = float(np.sum(enhanced.astype(np.float64) ** 2))
    if e_in > 0 and e_out < 1e-6 * e_in:
        return noisy.copy(), "near-silent output"
    return enhanced, None


def fit_for_pcm(wav: np.ndarray, subtype: str) -> np.ndarray:
    """정수 PCM 으로 저장할 때 클리핑 방지(PESQ 는 전체 레벨을 정렬하므로 점수 영향 없음)."""
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if subtype.startswith("PCM") and peak > 0.999:
        wav = wav * (0.999 / peak)
    return wav.astype(np.float32)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return int(math.floor(seconds * sample_rate))
