#!/usr/bin/env python
"""제출 전 리허설: 조교 평가 절차를 검증셋으로 그대로 재현합니다.

  1) 검증 noisy WAV 를 <workdir>/input/ 에 같은 파일명으로 복사 (clean 은 넣지 않음)
  2) 조교와 똑같이  python inference.py --input <workdir>/input --output <workdir>/output  실행
  3) evaluate.py 로 형식 검사 + 평균 PESQ-WB 채점

    python check_submission.py --data_root ../deep_learning_noise_reduction_challenge
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default="../deep_learning_noise_reduction_challenge")
    parser.add_argument("--ckpt", default=str(REPO / "checkpoints" / "best.pt"))
    parser.add_argument("--workdir", default=str(REPO / "runs" / "ta_rehearsal"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_files", type=int, default=0, help="0 = 검증 전체")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    from pesqmax.data import read_manifest, resolve_data_root

    root = resolve_data_root(args.data_root, REPO)
    rows = read_manifest(root, "validation")
    if args.max_files > 0:
        rows = rows[: args.max_files]
    work = Path(args.workdir)
    in_dir, out_dir = work / "input", work / "output"
    for folder in (in_dir, out_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    for row in rows:
        src = root / row["noisy_path"]
        shutil.copy2(src, in_dir / src.name)
    print(f"[1/3] 입력 준비: {len(rows)}개 -> {in_dir}")

    cmd = [sys.executable, str(REPO / "inference.py"), "--input", str(in_dir), "--output",
           str(out_dir), "--ckpt", args.ckpt, "--device", args.device]
    print("[2/3] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    manifest_subset = None
    if args.max_files > 0:  # 일부만 검사할 때는 폴더 짝 맞추기 모드로 채점
        manifest_subset = ["--clean_dir", str(root / "validation" / "clean"), "--noisy_dir", str(in_dir)]
    eval_cmd = [sys.executable, str(REPO / "evaluate.py"), "--enhanced", str(out_dir),
                "--csv", str(work / "per_file.csv")]
    eval_cmd += manifest_subset or ["--data_root", str(root), "--split", "validation"]
    print("[3/3] " + " ".join(eval_cmd))
    result = subprocess.run(eval_cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
