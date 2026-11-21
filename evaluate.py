"""검증셋 전체 평가: 파일별·잡음 종류별 PESQ-WB와 SNR을 CSV로 저장합니다.

예)  python evaluate.py --data-root <배포 폴더> --checkpoint checkpoints/best_generator.pt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_split
from infer import enhance_wave, pick_device
from metrics import PesqPool, snr_db
from model import load_generator


def evaluate(model, manifest, waves, device, pesq_pool):
    enhanced = [enhance_wave(model, waves[i, 0], device) for i in range(len(waves))]
    clean = list(waves[:, 1])
    pesq_in = pesq_pool.map(clean, list(waves[:, 0]))
    pesq_out = pesq_pool.map(clean, enhanced)
    rows = []
    for i, row in enumerate(manifest.itertuples(index=False)):
        snr_in, snr_out = snr_db(clean[i], waves[i, 0]), snr_db(clean[i], enhanced[i])
        rows.append({"segment_id": row.segment_id,
                     "noise_category": getattr(row, "noise_category", ""),
                     "target_snr_db": getattr(row, "target_snr_db", np.nan),
                     "input_pesq_wb": pesq_in[i], "model_pesq_wb": pesq_out[i],
                     "delta_pesq_wb": pesq_out[i] - pesq_in[i],
                     "input_snr_db": snr_in, "model_snr_db": snr_out, "delta_snr_db": snr_out - snr_in})
    return pd.DataFrame(rows), enhanced


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--checkpoint", default="checkpoints/best_generator.pt")
    p.add_argument("--split", default="validation")
    p.add_argument("--out-dir", default=None, help="결과 CSV 폴더 (기본: 가중치 폴더)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    pool = PesqPool()
    device = pick_device(args.device)
    model, state = load_generator(args.checkpoint, device)
    manifest, waves = load_split(args.data_root, args.split, args.cache_dir)
    detail, _ = evaluate(model, manifest, waves, device, pool)
    pool.close()

    out_dir = Path(args.out_dir or Path(args.checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / f"{args.split}_per_file.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame({
        "비교": ["잡음 입력", "모델 복원", "개선량"],
        "파일 수": len(detail),
        "평균 PESQ-WB": [detail.input_pesq_wb.mean(), detail.model_pesq_wb.mean(), detail.delta_pesq_wb.mean()],
        "평균 SNR (dB)": [detail.input_snr_db.mean(), detail.model_snr_db.mean(), detail.delta_snr_db.mean()],
    })
    summary.to_csv(out_dir / f"{args.split}_summary.csv", index=False, encoding="utf-8-sig")
    by_noise = (detail.groupby("noise_category")[["input_pesq_wb", "model_pesq_wb", "delta_pesq_wb", "delta_snr_db"]]
                .mean().sort_values("model_pesq_wb"))
    by_noise.to_csv(out_dir / f"{args.split}_by_noise.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 160)
    print(summary.round(4).to_string(index=False))
    print("\n잡음 종류별 평균 (낮은 순)\n", by_noise.round(3).to_string())


if __name__ == "__main__":
    main()
