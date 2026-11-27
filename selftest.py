#!/usr/bin/env python
"""설치 확인용 자가 진단 (데이터 불필요, CPU 1분 이내).

    python selftest.py

검사 항목: 모델 순전파/역전파와 NaN, PCS 초기값, 미분 가능 PESQ 손실, 판별자,
체크포인트 저장/로딩 왕복, 추론 함수(TTA·분할 처리)의 출력 길이, pesq 패키지.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def check(name, cond):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def main():
    import numpy as np
    import torch

    from pesqmax.config import load_config
    from pesqmax.discriminator import MetricDiscriminator
    from pesqmax.inference_core import enhance_long, export_checkpoint, load_enhancer
    from pesqmax.model import Enhancer, count_parameters
    from pesqmax.pcs import pcs_profile
    from pesqmax.pesq_loss import PesqLoss
    from pesqmax.utils import state_dict_fp32_cpu

    torch.manual_seed(0)
    print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    full = Enhancer(load_config(None))
    check(f"MP-SENet 본 설정 파라미터 수 {count_parameters(full):,} (~2.05M)",
          2_000_000 < count_parameters(full) < 2_100_000)

    cfg = load_config(REPO / "configs" / "smoke_cpu.yaml")
    model = Enhancer(cfg)
    x = torch.randn(2, 8000) * 0.1
    out = model(x * torch.sqrt(x.shape[1] / x.pow(2).sum(1, keepdim=True)))
    check("출력 길이 = 입력 길이", out["wav"].shape == x.shape and out["wav_pre"].shape == x.shape)
    loss = PesqLoss()(x, out["wav"]).mean() + out["mag_c"].mean()
    disc = MetricDiscriminator()
    q = disc(model.stft(x)[0], model.stft(out["wav"])[0])
    check("판별자 출력 범위 (0, 1)", bool(((q > 0) & (q < 1)).all()))
    (loss + q.mean()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("모든 기울기가 유한", all(torch.isfinite(g).all() for g in grads))
    check("후처리 PCS 로 기울기 전달", model.post_pcs.theta.grad.abs().sum() > 0)
    check("PCS400 초기값 (0.4~4.4 kHz 에서 1.4)",
          abs(float(model.post_pcs.gamma().detach()[50]) - 1.4) < 1e-3 and abs(pcs_profile(400)[50] - 1.4) < 1e-9)

    ref = torch.randn(1, 16000) * 0.1
    check("PESQ 손실: 같은 신호면 0", float(PesqLoss()(ref, ref.clone())) < 1e-6)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best.pt"
        export_checkpoint(path, cfg, state_dict_fp32_cpu(model), {"use_post_pcs": True,
                          "pcs_strength": 1.0, "tta_shifts": 2}, {"epoch": 0})
        loaded, post, _ = load_enhancer(path, torch.device("cpu"))
        wav = torch.randn(1, 37_123) * 0.05
        y = enhance_long(loaded, wav, chunk=16000, overlap=4000, use_post_pcs=True,
                         pcs_strength=1.0, tta_shifts=post["tta_shifts"])
        check("체크포인트 왕복 + 분할 처리 + TTA 출력 길이", y.shape == wav.shape and
              bool(torch.isfinite(y).all()))
        tiny = enhance_long(loaded, torch.randn(1, 300) * 0.05, chunk=16000, overlap=4000)
        check("아주 짧은 입력(300 표본) 처리", tiny.shape == (1, 300))

    try:
        from pesq import pesq
        t = np.arange(32000) / 16000
        voiced = (np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        score = pesq(16000, voiced, voiced, "wb")
        check(f"pesq 패키지 동작 (동일 신호 PESQ-WB {score:.2f})", score > 4.0)
    except ImportError:
        print("  [WARN] pesq 패키지 없음: 학습·채점에는 필요합니다 (pip install pesq==0.0.4).")
    print("자가 진단 통과")


if __name__ == "__main__":
    main()
