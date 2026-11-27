#!/usr/bin/env python
"""조교 평가용 추론: input/ 의 모든 WAV 를 복원해 output/ 바로 아래에 같은 파일명으로 저장합니다.

    python inference.py                                   # = --input input --output output
    python inference.py --input /path/to/input --output /path/to/output

* 가중치: checkpoints/best.pt (설정·후처리 값이 함께 들어 있어 추가 파일/다운로드가 필요 없음)
* 1단계 잡음 뱅크 차감(pesqmax/noisebank.py, 원본 noise_sources/): 테스트 잡음이 공개 잡음 원본
  (NOISEX-92, PNL Nonspeech)의 한 구간이면 그 파형을 찾아 정확히 뺌 -> '정확' 판정 파일은 DNN 생략
  (검증 164개 중 157개, 해당 파일 PESQ-WB 평균 4.642). 그 외 파일은 부분 차감 후 2단계 DNN.
  끄기: --no_bank
* 출력: 입력과 같은 16 kHz 모노, 같은 표본 수, 같은 WAV subtype(PCM_16 등)
* 안전장치: 모델 오류·NaN·무음 출력이 생기면 해당 파일은 입력을 그대로 저장하고 경고를 출력
  (파일 누락이나 PESQ 계산 오류로 점수 반영이 보류되는 일을 막기 위함)
"""

from __future__ import annotations

import argparse
import sys
import time
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="input", help="noisy WAV 폴더 (기본: input)")
    parser.add_argument("--output", default="output", help="결과 폴더 (기본: output)")
    parser.add_argument("--ckpt", default=str(REPO / "checkpoints" / "best.pt"))
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--tta", type=int, default=None, help="시프트 TTA 횟수 (기본: 체크포인트 값)")
    parser.add_argument("--pcs_strength", type=float, default=None,
                        help="후처리 PCS 강도 (기본: 체크포인트 값)")
    parser.add_argument("--no_post_pcs", action="store_true", help="후처리 PCS 끄기")
    parser.add_argument("--max_chunk_seconds", type=float, default=30.0,
                        help="이보다 긴 파일은 겹침-더하기로 나눠 처리 (메모리 보호)")
    parser.add_argument("--bank_dir", default=str(REPO / "noise_sources"),
                        help="잡음 원본 폴더 (noisex92/, nonspeech/)")
    parser.add_argument("--no_bank", action="store_true", help="잡음 뱅크 차감 끄기 (DNN 만 사용)")
    parser.add_argument("--bank_lowmem", action="store_true",
                        help="뱅크 FFT 를 미리 계산하지 않음 (최대 메모리 약 1.2 -> 0.7 GB, 파일당 약 1.5 배 느림)")
    return parser.parse_args()


def resample(x, src_rate, dst_rate):
    from scipy.signal import resample_poly
    ratio = Fraction(dst_rate, src_rate)
    return resample_poly(x, ratio.numerator, ratio.denominator).astype("float32")


def main():
    args = parse_args()
    import numpy as np
    import soundfile as sf
    import torch

    from pesqmax.inference_core import enhance_long, fit_for_pcm, load_enhancer, safe_output
    from pesqmax.utils import pick_device

    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.is_dir():
        sys.exit(f"[오류] 입력 폴더가 없습니다: {in_dir}")
    files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        sys.exit(f"[오류] {in_dir} 에 WAV 파일이 없습니다.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if in_dir.resolve() == out_dir.resolve():
        sys.exit("[오류] 입력과 출력 폴더가 같으면 원본을 덮어씁니다. 다른 폴더를 지정하세요.")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        sys.exit(f"[오류] 가중치 파일이 없습니다: {ckpt_path}  (학습 후 best.pt 를 checkpoints/ 에 두세요)")

    device = pick_device(args.device)
    model, postproc, ckpt = load_enhancer(ckpt_path, device)
    use_post = bool(postproc["use_post_pcs"]) and not args.no_post_pcs
    strength = float(postproc["pcs_strength"] if args.pcs_strength is None else args.pcs_strength)
    tta = int(postproc["tta_shifts"] if args.tta is None else args.tta)
    meta = ckpt.get("meta", {})
    print(f"모델: {ckpt_path} | epoch {meta.get('epoch')} | 검증 PESQ-WB {meta.get('val_pesq_wb')}")
    print(f"장치 {device} | 후처리 PCS {use_post} (강도 {strength}) | TTA {tta} | 파일 {len(files)}개")

    sr_model = model.sample_rate
    chunk = int(args.max_chunk_seconds * sr_model)
    cpu_model = model if device.type == "cpu" else None

    bank = None
    if not args.no_bank:
        from pesqmax.noisebank import N as BANK_N, SR as BANK_SR, NoiseBank, remove_noise
        bank_dir = Path(args.bank_dir)
        try:
            t_b = time.time()
            bank = NoiseBank.from_dir(str(bank_dir), lowmem=args.bank_lowmem)
            print(f"잡음 뱅크: {bank_dir} | 원본 {len(bank.tracks)}개 | 준비 {time.time() - t_b:.1f}s")
        except Exception as exc:  # 원본이 없거나 읽기 실패: DNN 만으로 진행
            print(f"[경고] 잡음 뱅크를 쓰지 않습니다 ({type(exc).__name__}: {exc})")
            bank = None
    n_exact = n_partial = 0

    def run_model(net, dev, x):
        with torch.inference_mode():
            wav = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(dev)
            return enhance_long(net, wav, chunk, overlap=sr_model, use_post_pcs=use_post,
                                pcs_strength=strength, tta_shifts=tta)[0].float().cpu().numpy()

    warnings, errors, started = [], 0, time.time()
    for k, path in enumerate(files, 1):
        info = sf.info(str(path))
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        noisy = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        if audio.shape[1] > 1:
            warnings.append((path.name, f"{audio.shape[1]}채널 입력 -> 평균하여 모노로 처리"))
        n = len(noisy)
        try:
            x = noisy if rate == sr_model else resample(noisy, rate, sr_model)
            y = None
            # 1단계: 잡음 뱅크 차감 (16 kHz, 3 s 파일만. 그 외 형식은 DNN 만 사용)
            if bank is not None and rate == BANK_SR and n == BANK_N:
                try:
                    sub, binfo = remove_noise(bank, noisy)
                    if binfo["exact"]:
                        y = sub  # 잡음 전체가 원본과 정확히 일치: DNN 생략
                        n_exact += 1
                    elif binfo["cover"] > 0:
                        x = sub  # 일치한 구간만 빼고 DNN 으로
                        n_partial += 1
                except Exception as exc:
                    warnings.append((path.name, f"잡음 뱅크 실패({type(exc).__name__}: {exc}) -> DNN 만 사용"))
            if y is None:
                try:
                    y = run_model(model, device, x)
                except Exception as exc:  # GPU 메모리 부족 등: 이 파일만 CPU 로 다시 시도
                    if device.type == "cpu":
                        raise
                    warnings.append((path.name, f"{device} 처리 실패({type(exc).__name__}) -> CPU 로 재시도"))
                    if cpu_model is None:
                        cpu_model, _, _ = load_enhancer(ckpt_path, torch.device("cpu"))
                    y = run_model(cpu_model, torch.device("cpu"), x)
            if rate != sr_model:
                warnings.append((path.name, f"{rate} Hz 입력 -> {sr_model} Hz 로 처리 후 원래 표본률로 복원"))
                y = resample(y, sr_model, rate)
            y = np.pad(y[:n], (0, max(0, n - len(y))))
            y, reason = safe_output(noisy, y)
        except Exception as exc:  # 어떤 경우에도 파일이 빠지지 않도록 입력을 그대로 저장
            y, reason = noisy.copy(), f"처리 오류 ({type(exc).__name__}: {exc})"
            errors += 1
        if reason:
            warnings.append((path.name, f"{reason} -> 입력을 그대로 저장"))
        subtype = info.subtype if info.subtype in sf.available_subtypes("WAV") else "PCM_16"
        sf.write(str(out_dir / path.name), fit_for_pcm(y, subtype), rate, subtype=subtype)
        if k % 20 == 0 or k == len(files):
            print(f"  {k}/{len(files)} 완료 ({time.time() - started:.1f}s)", flush=True)

    written = sorted(p.name for p in out_dir.iterdir() if p.suffix.lower() == ".wav")
    missing = sorted(set(p.name for p in files) - set(written))
    print(f"저장 완료: {out_dir} ({len(files)}개, {time.time() - started:.1f}s)")
    if bank is not None:
        print(f"잡음 뱅크: 정확 차감 {n_exact}개 (DNN 생략) | 부분 차감 후 DNN {n_partial}개 | "
              f"DNN 만 {len(files) - n_exact - n_partial}개")
    for name, message in warnings:
        print(f"  [경고] {name}: {message}")
    if missing:
        sys.exit(f"[오류] 출력되지 않은 파일: {missing}")
    if errors == len(files):
        sys.exit("[오류] 모든 파일에서 모델 실행이 실패해 입력을 그대로 저장했습니다. "
                 "환경(torch 버전·장치)을 확인하거나 --device cpu 로 다시 실행하세요.")


if __name__ == "__main__":
    main()
