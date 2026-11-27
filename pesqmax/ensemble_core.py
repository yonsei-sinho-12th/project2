"""앙상블 추론: 여러 체크포인트의 복원 파형을 평균해 PESQ를 더 끌어올립니다.

각 모델은 자기 후처리(PCS 강도·TTA)를 그대로 적용한 뒤, 출력 파형을 입력 RMS 기준으로
정렬해 평균합니다. 모델들이 같은 noisy 입력에서 출발하므로 시간 정렬은 이미 맞습니다.

사용처:
  - inference.py --ckpt a.pt,b.pt,c.pt  (쉼표로 여러 개)  또는  --ckpt checkpoints/  (폴더 전체)
  - 제출 시 checkpoints/ 에 best*.pt 를 여러 개 두면 조교가 inference.py 만 돌려도 앙상블됩니다.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .inference_core import enhance_long, load_enhancer


def resolve_ckpts(spec: str) -> list[Path]:
    """'a.pt,b.pt' 또는 폴더 경로를 받아 체크포인트 경로 목록으로 바꿉니다."""
    paths: list[Path] = []
    for part in str(spec).split(","):
        p = Path(part.strip())
        if p.is_dir():
            paths += sorted(q for q in p.glob("*.pt") if q.name != "last.pt")
        elif p.is_file():
            paths.append(p)
    # 중복 제거(순서 유지)
    seen, out = set(), []
    for p in paths:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


class EnsembleEnhancer:
    def __init__(self, ckpt_spec: str, device: torch.device):
        self.paths = resolve_ckpts(ckpt_spec)
        if not self.paths:
            raise FileNotFoundError(f"앙상블할 체크포인트를 찾지 못했습니다: {ckpt_spec}")
        self.models, self.postprocs, self.metas = [], [], []
        for path in self.paths:
            model, post, ckpt = load_enhancer(path, device)
            self.models.append(model)
            self.postprocs.append(post)
            self.metas.append(ckpt.get("meta", {}))
        self.sample_rate = self.models[0].sample_rate
        if any(m.sample_rate != self.sample_rate for m in self.models):
            raise ValueError("앙상블 체크포인트의 표본률이 서로 다릅니다.")

    def __len__(self):
        return len(self.models)

    @torch.no_grad()
    def enhance(self, wav: torch.Tensor, chunk: int) -> torch.Tensor:
        """wav [B, N] -> 앙상블 복원 [B, N] (각 모델 후처리 적용 후 평균)."""
        acc = None
        for model, post in zip(self.models, self.postprocs):
            y = enhance_long(model, wav, chunk, overlap=self.sample_rate,
                             use_post_pcs=bool(post["use_post_pcs"]),
                             pcs_strength=float(post["pcs_strength"]),
                             tta_shifts=int(post["tta_shifts"]))
            acc = y if acc is None else acc + y
        return acc / len(self.models)

    def describe(self) -> str:
        items = [f"{p.name}(PESQ {m.get('val_pesq_wb')})" for p, m in zip(self.paths, self.metas)]
        return f"{len(self.models)}개 앙상블: " + ", ".join(items)
