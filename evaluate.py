#!/usr/bin/env python
"""리더보드와 같은 방식(전체 파일 PESQ-WB 산술평균, pesq 패키지 wb 모드)으로 결과를 채점합니다.

(1) 배포 폴더의 manifest 로 짝을 맞추는 방법 (검증셋 자가 채점):
    python evaluate.py --enhanced output --data_root ../deep_learning_noise_reduction_challenge
(2) 폴더끼리 파일명으로 짝을 맞추는 방법:
    python evaluate.py --enhanced output --clean_dir CLEAN_DIR [--noisy_dir INPUT_DIR]
    (출력 파일명에 '_noisy' 가 붙어 있고 clean 에는 없으면 자동으로 맞춰 봅니다)

형식 검사: 누락/추가 WAV, 16 kHz 모노 여부, 입력과 같은 표본 수, NaN 여부.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enhanced", required=True, help="복원 결과 폴더 (output/)")
    parser.add_argument("--data_root", default=None, help="manifest 를 쓸 배포 폴더")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--clean_dir", default=None)
    parser.add_argument("--noisy_dir", default=None, help="입력 폴더 (형식 검사·noisy 기준점수)")
    parser.add_argument("--csv", default=None, help="파일별 결과 CSV 경로")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=16000)
    return parser.parse_args()


def build_pairs(args):
    """[(enhanced_path, clean_path, noisy_path|None, meta dict)]"""
    enhanced = Path(args.enhanced)
    pairs = []
    if args.data_root:
        from pesqmax.data import read_manifest, resolve_data_root
        root = resolve_data_root(args.data_root, REPO)
        for row in read_manifest(root, args.split):
            noisy = root / row["noisy_path"]
            pairs.append((enhanced / noisy.name, root / row["clean_path"], noisy,
                          {"segment_id": row["segment_id"],
                           "noise_category": row.get("noise_category", ""),
                           "target_snr_db": row.get("target_snr_db", "")}))
        return pairs
    if not args.clean_dir:
        sys.exit("--data_root 또는 --clean_dir 중 하나가 필요합니다.")
    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir) if args.noisy_dir else None
    names = sorted(p.name for p in (noisy_dir or enhanced).iterdir() if p.suffix.lower() == ".wav")
    for name in names:
        clean = clean_dir / name
        if not clean.is_file():
            stem = Path(name).stem
            for suffix in ("_noisy", "-noisy", "_mix"):
                if stem.endswith(suffix):
                    clean = clean_dir / (stem[: -len(suffix)] + Path(name).suffix)
                    break
        pairs.append((enhanced / name, clean, (noisy_dir / name) if noisy_dir else None,
                      {"segment_id": Path(name).stem}))
    return pairs


def main():
    args = parse_args()
    import numpy as np
    import soundfile as sf

    from pesqmax.metrics import PesqPool, snr_db

    pairs = build_pairs(args)
    enhanced_dir = Path(args.enhanced)
    problems = []
    expected = {p[0].name for p in pairs}
    extra = sorted(p.name for p in enhanced_dir.iterdir()
                   if p.suffix.lower() == ".wav" and p.name not in expected)
    if extra:
        problems.append(f"추가 WAV {len(extra)}개: {extra[:5]}")

    refs, degs, noisies, rows = [], [], [], []
    for enh_path, clean_path, noisy_path, meta in pairs:
        if not enh_path.is_file():
            problems.append(f"누락: {enh_path.name}")
            continue
        if not clean_path.is_file():
            problems.append(f"clean 없음: {clean_path}")
            continue
        enh, rate = sf.read(str(enh_path), dtype="float32", always_2d=True)
        clean, rate_c = sf.read(str(clean_path), dtype="float32")
        if enh.shape[1] != 1:
            problems.append(f"모노 아님: {enh_path.name} ({enh.shape[1]}ch)")
        if rate != args.sample_rate:
            problems.append(f"표본률 {rate} Hz: {enh_path.name}")
        enh = enh[:, 0]
        noisy = None
        if noisy_path is not None and noisy_path.is_file():
            noisy, _ = sf.read(str(noisy_path), dtype="float32")
            if len(noisy) != len(enh):
                problems.append(f"표본 수 불일치: {enh_path.name} (입력 {len(noisy)}, 출력 {len(enh)})")
        if not np.all(np.isfinite(enh)):
            problems.append(f"NaN/Inf: {enh_path.name}")
            enh = np.nan_to_num(enh)
        n = min(len(clean), len(enh))
        refs.append(clean[:n])
        degs.append(enh[:n])
        noisies.append(None if noisy is None else noisy[:n])
        rows.append(dict(meta, file=enh_path.name))

    if not rows:
        sys.exit("채점할 파일이 없습니다.\n" + "\n".join(problems))
    pool = PesqPool(args.workers, args.sample_rate)
    try:
        scores = pool.map(refs, degs)
        have_noisy = all(x is not None for x in noisies)
        base = pool.map(refs, noisies) if have_noisy else None
    finally:
        pool.close()

    for i, row in enumerate(rows):
        row["pesq_wb"] = scores[i]
        row["snr_db"] = snr_db(refs[i], degs[i])
        if base is not None:
            row["input_pesq_wb"] = base[i]
            row["delta_pesq_wb"] = scores[i] - base[i]
            row["input_snr_db"] = snr_db(refs[i], noisies[i])
    errors = int(np.sum(~np.isfinite(scores)))
    if errors:
        problems.append(f"PESQ 계산 오류 {errors}개")

    print("=" * 64)
    print(f"파일 {len(rows)}/{len(pairs)}개 | 평균 PESQ-WB {np.nanmean(scores):.4f}"
          + (f" | 입력(noisy) {np.nanmean(base):.4f} | 개선 {np.nanmean(scores - base):+.4f}"
             if base is not None else ""))
    print(f"평균 SNR {np.mean([r['snr_db'] for r in rows]):.2f} dB")
    for key in ("noise_category", "target_snr_db"):
        if rows and rows[0].get(key, "") != "":
            groups = defaultdict(list)
            for r in rows:
                groups[r[key]].append(r["pesq_wb"])
            print(f"-- {key}별 PESQ-WB")

            def order(g):
                try:
                    return (0, float(g), "")
                except ValueError:
                    return (1, 0.0, str(g))

            for name in sorted(groups, key=order):
                print(f"   {str(name):24s} n={len(groups[name]):3d}  {np.nanmean(groups[name]):.4f}")
    print("형식 검사: " + ("통과" if not problems else "문제 있음"))
    for message in problems[:30]:
        print("   - " + message)
    print("=" * 64)
    if args.csv:
        keys = list(rows[0].keys())
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"파일별 결과: {args.csv}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
