#!/usr/bin/env python
"""검증 PESQ-WB 를 목적함수로 하는 하이퍼파라미터 무작위 탐색 (경량 모델, 짧은 학습).

    python tune_hparams.py --trials 12 --epochs 20 --data_root ../deep_learning_noise_reduction_challenge

각 trial 은 train.py 를 별도 프로세스로 실행하고 runs/tune/trial_XX/summary.json 의 최고 검증 PESQ 를
읽습니다. 결과는 runs/tune/results.csv 에 쌓이며, 이미 끝난 trial 은 다시 실행하지 않습니다.
상위 설정을 configs/mpsenet_pesqmax.yaml 에 옮긴 뒤 본 학습을 하세요.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# 탐색 공간 — PESQ 에 영향이 큰 순서대로. 필요에 따라 직접 수정하세요.
SEARCH_SPACE = {
    "loss.pesq": [1.0, 2.0, 3.0, 5.0],
    "loss.metric": [0.05, 0.1, 0.3, 0.6],
    "train.lr": [3.0e-4, 5.0e-4, 1.0e-3],
    "data.remix_prob": [0.0, 0.3, 0.5, 0.8],
    "data.speed_perturb_prob": [0.0, 0.2],
    "pcs.post": [True, False],
    "pcs.pre": [True, False],
    "pcs.init": ["pcs400", "identity"],
    "pcs.target": [False, True],
    "model.dropout": [0.0, 0.1, 0.2],
}


def fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "configs" / "mpsenet_lite.yaml"))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="탐색 표본 추출 시드")
    parser.add_argument("--out", default=str(REPO / "runs" / "tune"))
    parser.add_argument("--set", nargs="*", default=[], help="모든 trial 에 공통 적용할 설정")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    results_path = out / "results.csv"
    for k in range(args.trials):
        choice = {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}
        trial_dir = out / f"trial_{k:02d}"
        summary = trial_dir / "summary.json"
        overrides = [f"{key}={fmt(v)}" for key, v in choice.items()]
        overrides += [f"train.epochs={args.epochs}", "train.early_stop_patience=0"] + args.set
        cmd = [sys.executable, str(REPO / "train.py"), "--config", args.config,
               "--out_dir", str(trial_dir), "--set", *overrides]
        if args.data_root:
            cmd += ["--data_root", args.data_root]
        if summary.is_file() and json.loads(summary.read_text(encoding="utf-8")).get("status") != "running":
            print(f"[trial {k:02d}] 이미 완료 — 건너뜀")
            continue
        print(f"[trial {k:02d}] " + " ".join(overrides))
        if args.dry_run:
            continue
        resume = ["--resume"] if (trial_dir / "last.pt").is_file() else []
        code = subprocess.run(cmd + resume).returncode
        best = None
        if summary.is_file():
            best = json.loads(summary.read_text(encoding="utf-8")).get("best_val_pesq_wb")
        row = {"trial": k, "returncode": code, "best_val_pesq_wb": best, **choice}
        new_file = not results_path.is_file()
        with open(results_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"[trial {k:02d}] best val PESQ-WB = {best}")

    if results_path.is_file():
        with open(results_path, encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["best_val_pesq_wb"] not in ("", "None")]
        rows.sort(key=lambda r: -float(r["best_val_pesq_wb"]))
        print("\n상위 trial:")
        for r in rows[:5]:
            print(f"  trial {r['trial']}: {float(r['best_val_pesq_wb']):.4f}  " +
                  " ".join(f"{k}={r[k]}" for k in SEARCH_SPACE))


if __name__ == "__main__":
    main()
