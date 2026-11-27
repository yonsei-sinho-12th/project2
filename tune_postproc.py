#!/usr/bin/env python
"""검증셋 실제 PESQ-WB 로 추론 후처리 값을 고르고 체크포인트에 기록합니다 (재학습 불필요).

탐색 대상
  * use_post_pcs : 학습된 후처리 PCS 사용 여부
  * pcs_strength : gamma' = 1 + s * (gamma - 1)  (s=0 이면 PCS 없음, 1 이면 학습값 그대로)
  * tta_shifts   : STFT 프레임 격자를 hop/k 씩 옮겨 k 번 추론한 뒤 평균

    python tune_postproc.py --ckpt runs/mpsenet_pesqmax/best.pt \
        --data_root ../deep_learning_noise_reduction_challenge --out checkpoints/best.pt

네트워크는 시프트마다 한 번만 실행하고, PCS 강도별 합성만 반복하므로 빠릅니다.
주의: 검증셋으로 고른 값이므로 보고하는 검증 점수는 약간 낙관적일 수 있습니다(탐색 변수 3개).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def floats(text):
    return [float(x) for x in text.split(",") if x.strip()]


def ints(text):
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="../deep_learning_noise_reduction_challenge")
    parser.add_argument("--out", default=None, help="결과 체크포인트 (기본: 입력 체크포인트 덮어쓰기)")
    parser.add_argument("--strengths", default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--tta", default="1,2,4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()

    from pesqmax.metrics import PesqPool
    pool = PesqPool(args.workers)  # CUDA 초기화 전에 생성
    try:
        run(args, pool)
    finally:
        pool.close()


def run(args, pool):
    import numpy as np
    import torch
    import torch.nn.functional as F

    from pesqmax.data import ChallengeSplit, resolve_data_root
    from pesqmax.inference_core import export_checkpoint, load_enhancer, tta_offsets
    from pesqmax.utils import pick_device

    device = pick_device(args.device)
    model, postproc, ckpt = load_enhancer(args.ckpt, device)
    val = ChallengeSplit(resolve_data_root(args.data_root, REPO), "validation", model.sample_rate,
                         args.max_files,
                         REPO / ".cache")
    noisy, clean = val.noisy, val.clean
    n_files, length = noisy.shape
    tta_list = sorted(set(ints(args.tta)))
    strengths = sorted(set(floats(args.strengths)))
    offsets = sorted({o for t in tta_list for o in tta_offsets(model.hop, t)})
    print(f"검증 {n_files}개 | 시프트 {offsets} | PCS 강도 {strengths} | TTA {tta_list}")

    # 1) 시프트별 네트워크 출력 캐시
    t0 = time.time()
    cache = {}
    with torch.inference_mode():
        for off in offsets:
            items = []
            for i in range(0, n_files, args.batch_size):
                x = torch.from_numpy(noisy[i:i + args.batch_size]).to(device)
                x = F.pad(x, (off, 0)) if off else x
                factor, _ = model.rms_norm_factor(x)
                mag_c, pha, scale = model.analyze(x * factor)
                items.append((mag_c.cpu(), pha.cpu(), scale.cpu(), factor.cpu()))
            cache[off] = items
    print(f"네트워크 실행 {time.time() - t0:.1f}s")

    def synth(off, use_post, strength):
        outs = []
        with torch.inference_mode():
            for mag_c, pha, scale, factor in cache[off]:
                y = model.synthesize(mag_c.to(device), pha.to(device), scale.to(device),
                                     length + off, use_post, strength)
                outs.append((y / factor.to(device))[:, off:off + length].float().cpu().numpy())
        return np.concatenate(outs)

    configs = [(False, 1.0)] + ([(True, s) for s in strengths] if model.post_pcs is not None else [])
    results = []
    for use_post, strength in configs:
        per_offset = {off: synth(off, use_post, strength) for off in offsets}
        for tta in tta_list:
            subset = tta_offsets(model.hop, tta)
            wav = sum(per_offset[o] for o in subset) / len(subset)
            score = float(np.nanmean(pool.map(list(clean), list(wav))))
            results.append({"use_post_pcs": use_post, "pcs_strength": strength,
                            "tta_shifts": tta, "val_pesq_wb": score})
            print(f"  post_pcs={str(use_post):5s} strength={strength:4.2f} tta={tta}  "
                  f"PESQ-WB {score:.4f}", flush=True)

    best = max(results, key=lambda r: r["val_pesq_wb"])
    base = next((r for r in results if r["use_post_pcs"] == bool(postproc["use_post_pcs"])
                 and r["pcs_strength"] == 1.0 and r["tta_shifts"] == 1), None)
    print(f"\n최적: {best}")
    if base is not None:
        print(f"기존 설정 대비 {best['val_pesq_wb'] - base['val_pesq_wb']:+.4f}")
    new_post = {k: best[k] for k in ("use_post_pcs", "pcs_strength", "tta_shifts")}
    meta = dict(ckpt.get("meta", {}))
    meta["val_pesq_wb_before_postproc_tuning"] = meta.get("val_pesq_wb")
    meta["val_pesq_wb"] = best["val_pesq_wb"]
    meta["postproc_tuning"] = sorted(results, key=lambda r: -r["val_pesq_wb"])[:10]
    out = Path(args.out or args.ckpt)
    export_checkpoint(out, ckpt["config"], ckpt["generator"], new_post, meta)
    print(f"저장: {out}  (postproc={new_post})")


if __name__ == "__main__":
    main()
