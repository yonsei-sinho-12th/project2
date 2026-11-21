"""제출 형식 점검 + (기준 음성이 있으면) PESQ-WB 계산.

1) 검증셋을 테스트셋처럼 꺼내기:
   python check_submission.py export --data-root <배포 폴더> --dest val_test
   → val_test/input/*.wav (noisy 원본 복사), val_test/clean/*.wav

2) 추론:  python infer.py --input val_test/input --output val_test/output

3) 점검:  python check_submission.py check --input val_test/input --output val_test/output --clean val_test/clean
   파일 누락·추가 WAV·표본률·채널·표본 수를 확인하고, PESQ-WB 산술평균을 출력합니다.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from metrics import PesqPool


def export(args):
    root, dest = Path(args.data_root), Path(args.dest)
    manifest = pd.read_csv(root / args.split / "samples.csv")
    (dest / "input").mkdir(parents=True, exist_ok=True)
    (dest / "clean").mkdir(parents=True, exist_ok=True)
    for row in manifest.itertuples(index=False):
        shutil.copyfile(root / row.noisy_path, dest / "input" / f"{row.segment_id}.wav")
        shutil.copyfile(root / row.clean_path, dest / "clean" / f"{row.segment_id}.wav")
    print(f"{len(manifest)}개 파일을 {dest}/input, {dest}/clean에 복사했습니다.")


def check(args):
    in_dir, out_dir = Path(args.input), Path(args.output)
    inputs = {f.name: f for f in in_dir.iterdir() if f.suffix.lower() == ".wav"}
    outputs = {f.name: f for f in out_dir.iterdir() if f.suffix.lower() == ".wav"}
    problems = []
    missing, extra = sorted(set(inputs) - set(outputs)), sorted(set(outputs) - set(inputs))
    if missing:
        problems.append(f"누락 {len(missing)}개: {missing[:5]}")
    if extra:
        problems.append(f"추가 WAV {len(extra)}개: {extra[:5]}")
    for name in sorted(set(inputs) & set(outputs)):
        i, o = sf.info(inputs[name]), sf.info(outputs[name])
        if o.samplerate != 16_000 or o.channels != 1 or o.frames != i.frames:
            problems.append(f"{name}: {o.samplerate} Hz, {o.channels}ch, {o.frames} 표본 (입력 {i.frames})")
    print(f"입력 {len(inputs)}개 / 출력 {len(outputs)}개")
    print("형식 점검: " + ("통과" if not problems else "문제 있음\n  " + "\n  ".join(problems[:20])))

    if args.clean:
        clean_dir = Path(args.clean)
        names = sorted(set(inputs) & set(outputs))
        refs = [sf.read(clean_dir / n, dtype="float32")[0] for n in names]
        noisy = [sf.read(inputs[n], dtype="float32")[0] for n in names]
        outs = [sf.read(outputs[n], dtype="float32")[0] for n in names]
        pool = PesqPool()
        s_in, s_out = pool.map(refs, noisy), pool.map(refs, outs)
        pool.close()
        print(f"PESQ-WB 평균: 입력 {np.nanmean(s_in):.4f} → 출력 {np.nanmean(s_out):.4f} "
              f"(계산 실패 {int(np.isnan(s_out).sum())}개)")
    if problems:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--data-root", required=True)
    e.add_argument("--split", default="validation")
    e.add_argument("--dest", default="val_test")
    c = sub.add_parser("check")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--clean", default=None)
    args = p.parse_args()
    export(args) if args.cmd == "export" else check(args)


if __name__ == "__main__":
    main()
