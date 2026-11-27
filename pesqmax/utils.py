"""공용 유틸리티: 시드, EMA, 원자적 체크포인트 저장, 장치/AMP 선택."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    """auto: CUDA 가 있으면 CUDA, 없으면 CPU. (Apple MPS 는 STFT 지원이 버전마다 달라 명시할 때만 사용)"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def amp_dtype(device: torch.device, enabled: bool):
    """CUDA 에서만 AMP 사용. Ampere(sm80) 이상은 bf16, 그 외(T4 등)는 fp16 + GradScaler."""
    if not enabled or device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


class EMA:
    """생성자 가중치의 지수 이동 평균. 검증·저장은 EMA 가중치로 합니다."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for shadow, param in zip(self.shadow.parameters(), model.parameters()):
            shadow.mul_(decay).add_(param.detach(), alpha=1 - decay)
        for shadow, buf in zip(self.shadow.buffers(), model.buffers()):
            shadow.copy_(buf)

    def state_dict(self):
        return {"shadow": self.shadow.state_dict(), "num_updates": self.num_updates}

    def load_state_dict(self, state):
        self.shadow.load_state_dict(state["shadow"])
        self.num_updates = int(state["num_updates"])


def atomic_save(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    tmp.replace(path)


def state_dict_fp32_cpu(module: torch.nn.Module) -> dict:
    return {k: v.detach().to("cpu", torch.float32) if v.is_floating_point() else v.detach().cpu()
            for k, v in module.state_dict().items()}
