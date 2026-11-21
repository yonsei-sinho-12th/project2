"""제출용 추론: input/의 모든 WAV → output/에 같은 파일명으로 저장.

예)  python infer.py --input input --output output --checkpoint checkpoints/best_generator.pt

- 출력은 16 kHz 모노, 입력과 같은 표본 수입니다.
- 입력이 PCM 정수 형식이면 같은 형식으로, 그 외에는 32-bit float로 저장합니다.
- 인터넷 연결 없이 동작합니다 (가중치 파일 하나만 필요).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from model import load_generator

SAMPLE_RATE = 16_000
INTEGER_SUBTYPES = {"PCM_16", "PCM_24", "PCM_32", "PCM_U8"}


@torch.no_grad()
def enhance_wave(model, wave, device, chunk_seconds=20.0, overlap_seconds=1.0):
    """1차원 float 파형 → 같은 길이의 복원 파형 (numpy float32)."""
    wave = np.asarray(wave, dtype=np.float32)
    length = wave.size
    if length == 0:
        return wave.copy()
    min_len = SAMPLE_RATE // 4                  # 너무 짧은 입력은 0으로 채워 처리 후 잘라냅니다.
    padded = np.pad(wave, (0, max(0, min_len - length)))

    def run(segment):
        x = torch.from_numpy(np.ascontiguousarray(segment)).unsqueeze(0).to(device)
        y = model.enhance(x)[0].float().cpu().numpy()
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    chunk = int(chunk_seconds * SAMPLE_RATE)
    if padded.size <= chunk:
        out = run(padded)
    else:                                       # 긴 파일: 겹치는 구간을 선형 크로스페이드로 합칩니다.
        total = padded.size
        fade = int(overlap_seconds * SAMPLE_RATE)
        starts = list(range(0, total - chunk, chunk - fade)) + [total - chunk]
        ramp_up = np.linspace(0, 1, fade + 2)[1:-1]
        out = np.zeros(total, np.float64)
        weight = np.zeros(total, np.float64)
        for start in starts:
            w = np.ones(chunk)
            if start > 0:
                w[:fade] = ramp_up
            if start + chunk < total:
                w[-fade:] = ramp_up[::-1]
            out[start:start + chunk] += run(padded[start:start + chunk]) * w
            weight[start:start + chunk] += w
        out = (out / np.maximum(weight, 1e-8)).astype(np.float32)
    return out[:length].astype(np.float32)


def read_any(path):
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return wave, rate, sf.info(path).subtype


def write_like_input(path, wave, subtype):
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if peak > 0.999:                            # PESQ는 레벨에 무관하므로 클리핑 대신 전체 크기만 줄입니다.
        wave = wave * (0.999 / peak)
    sf.write(path, wave, SAMPLE_RATE, subtype=subtype if subtype in INTEGER_SUBTYPES else "FLOAT")


def pick_device(name):
    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv=None):
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="input")
    p.add_argument("--output", default="output")
    p.add_argument("--checkpoint", default=str(here / "checkpoints" / "best_generator.pt"))
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0, help="CPU 스레드 수 (0: 전체)")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads or os.cpu_count() or 1)
    model, state = load_generator(args.checkpoint, device)
    print(f"가중치 {args.checkpoint} | epoch {state.get('epoch')} | 검증 PESQ-WB {state.get('val_pesq_wb', float('nan')):.4f} "
          f"| 장치 {device}")

    in_dir, out_dir = Path(args.input), Path(args.output)
    files = sorted(f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() == ".wav")
    if not files:
        sys.exit(f"{in_dir}에 WAV 파일이 없습니다.")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    for i, path in enumerate(files, 1):
        wave, rate, subtype = read_any(path)
        n_in = wave.size
        if rate != SAMPLE_RATE:
            print(f"  경고: {path.name}의 표본률 {rate} Hz → 16 kHz로 변환합니다.")
            g = np.gcd(rate, SAMPLE_RATE)
            wave = resample_poly(wave, SAMPLE_RATE // g, rate // g).astype(np.float32)
        enhanced = enhance_wave(model, wave, device)
        if rate == SAMPLE_RATE:
            assert enhanced.size == n_in
        write_like_input(out_dir / path.name, enhanced, subtype)
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)} 완료 ({time.time() - started:.1f}s)", flush=True)

    written = sorted(f.name for f in out_dir.iterdir() if f.suffix.lower() == ".wav")
    missing = sorted(set(f.name for f in files) - set(written))
    print(f"완료: {len(files)}개 처리 → {out_dir} (누락 {len(missing)}개)")
    if missing:
        sys.exit(f"누락된 출력: {missing[:10]}")


if __name__ == "__main__":
    main()
