#!/usr/bin/env python
"""noise_sources/ 를 공개 원본에서 다시 만들기 (저장소에 이미 있으면 실행할 필요 없음).

출처
  NOISEX-92 (19.98 kHz 16-bit, 15종): https://github.com/panandicoding/Build-SE-Dataset  (data/NoiseX92/)
  PNL 100 Nonspeech Sounds (20 kHz, n1..n100): https://github.com/speechdnn/Noises  (Nonspeech/)
    원 배포처: http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html (G. Hu)

    git clone --depth 1 https://github.com/speechdnn/Noises /tmp/Noises
    git clone --depth 1 --filter=blob:none --sparse https://github.com/panandicoding/Build-SE-Dataset /tmp/BSE
    (cd /tmp/BSE && git sparse-checkout set data/NoiseX92)
    python tools/build_noise_sources.py --noisex /tmp/BSE/data/NoiseX92 --nonspeech /tmp/Noises/Nonspeech

주의: 다른 판본(예: 8 kHz 8-bit 의 leopard/m109/machinegun)은 주최 측 잡음과 맞지 않습니다.
MANIFEST.json 의 PCM 체크섬으로 같은 판본인지 확인합니다.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisex", required=True)
    ap.add_argument("--nonspeech", required=True)
    ap.add_argument("--out", default=str(REPO / "noise_sources"))
    args = ap.parse_args()
    out = Path(args.out)
    manifest = json.loads((out / "MANIFEST.json").read_text()) if (out / "MANIFEST.json").is_file() else {}
    jobs = [("noisex92", f) for f in sorted(glob.glob(os.path.join(args.noisex, "*.wav")))]
    jobs += [("nonspeech", f) for f in sorted(glob.glob(os.path.join(args.nonspeech, "n*.wav")))]
    mismatch = []
    for sub, f in jobs:
        x, sr = sf.read(f, dtype="int16")
        key = f"{sub}/{Path(f).stem}.flac"
        digest = hashlib.sha256(x.tobytes()).hexdigest()
        if key in manifest and manifest[key]["sha256_pcm16"] != digest:
            mismatch.append(key)
            continue
        (out / sub).mkdir(parents=True, exist_ok=True)
        sf.write(str(out / key), x, sr, format="FLAC", subtype="PCM_16")
    print(f"{len(jobs) - len(mismatch)}개 저장 -> {out}")
    if mismatch:
        sys.exit(f"[오류] 체크섬이 다른 판본 (사용하지 않음): {mismatch}")


if __name__ == "__main__":
    main()
