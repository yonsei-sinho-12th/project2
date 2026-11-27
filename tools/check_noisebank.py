#!/usr/bin/env python
"""잡음 뱅크 단독 점검: 검증셋 noisy 만으로 차감 -> 정확 판정 수와 그 파일들의 PESQ-WB (DNN 불필요, CPU).

    python tools/check_noisebank.py --data_root ../deep_learning_noise_reduction_challenge [--limit 20]

참고 결과 (검증 164개): 정확 판정 157개, 해당 파일 PESQ-WB 평균 4.642 (최저 4.548), SNR 추정 157/157 일치.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(REPO.parent / "deep_learning_noise_reduction_challenge"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--bank_dir", default=str(REPO / "noise_sources"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from pesq import pesq
    from pesqmax.noisebank import NoiseBank, remove_noise

    root = Path(args.data_root)
    meta = pd.read_csv(root / args.split / "samples.csv", encoding="utf-8-sig")
    if args.limit:
        meta = meta.head(args.limit)
    t0 = time.time()
    bank = NoiseBank.from_dir(args.bank_dir)
    print(f"뱅크 준비 {time.time() - t0:.1f}s (원본 {len(bank.tracks)}개)")
    rows = []
    for k, row in enumerate(meta.itertuples(index=False), 1):
        noisy, _ = sf.read(str(root / row.noisy_path), dtype="float32")
        clean, _ = sf.read(str(root / row.clean_path), dtype="float32")
        out, info = remove_noise(bank, noisy)
        out = np.clip(np.round(out * 32768), -32768, 32767) / 32768.0
        rows.append(dict(file=Path(row.noisy_path).name, code=row.noise_code, snr=row.target_snr_db,
                         exact=info["exact"], cover=info["cover"], snr_est=info["snr"],
                         pesq_noisy=pesq(16000, clean, noisy, "wb"),
                         pesq_sub=pesq(16000, clean, out.astype(np.float32), "wb")))
        if k % 20 == 0:
            print(f"  {k}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
    r = pd.DataFrame(rows)
    ex = r[r.exact]
    print(f"정확 판정 {len(ex)}/{len(r)} | PESQ-WB 평균 {ex.pesq_sub.mean():.4f} (최저 {ex.pesq_sub.min():.3f}) | "
          f"SNR 추정 일치 {(ex.snr_est == ex.snr).sum()}/{len(ex)}")
    print("정확 판정 안 된 파일 (실제 추론에서는 DNN 이 처리):")
    print(r[~r.exact][["file", "code", "snr", "cover", "pesq_noisy"]].to_string(index=False))


if __name__ == "__main__":
    main()
