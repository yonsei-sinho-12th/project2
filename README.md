# 2026 신호 및 시스템 2차 챌린지 — 딥러닝 음성 잡음 제거

| 항목 | 내용 |
| :--- | :--- |
| **팀명** | 깨비드림즈 (12조) |
| **조원** | 황재현,최성우,박주현 |
| **제출 회차** | 2차 챌린지 2회차 제출 (2026-11-27) |
| **제출버전** | 태그 `v4` (잡음 뱅크+ MP-SENet) |

---

## 1. 실행 방법 (조교 평가용)

```bash
pip install -r requirements-inference.txt
python inference.py --input input --output output
```

- `input/`의 모든 WAV를 처리해 `output/` 바로 아래에 **같은 파일명**으로 저장합니다 (16 kHz 모노, 입력과 같은 표본 수, PCM_16).
- 가중치 `checkpoints/best.pt`와 잡음 원본 `noise_sources/`가 저장소에 포함되어 있어 **추론 중 추가 다운로드가 없습니다.**
- 개인 PC 절대 경로를 수정할 곳은 없습니다. 입출력 폴더는 위 인자로 지정합니다.
- 실행 환경: Python 3.10–3.12, CPU만으로 동작 (GPU가 있으면 자동 사용). 파일당 약 2–4 s, 메모리 약 1.2 GB.
  메모리가 부족하면 `--bank_lowmem`을 붙입니다 (약 0.7 GB, 조금 느려짐).
- 안전장치: 모델 오류·NaN·무음 출력이 생기면 해당 파일은 입력을 그대로 저장합니다 (파일 누락 방지).

## 2. 방법 — 2단계 구조

### 1단계: 잡음 뱅크 차감 (`pesqmax/noisebank.py`)
배포 데이터를 분석한 결과, 잡음은 다음 규칙으로 생성되었습니다 (학습·검증 1,460쌍에서 재현 오차 −55~−80 dB).

1. 잡음 원본: NOISEX-92(19.98 kHz) / PNL 100 Nonspeech Sounds(20 kHz)
2. `scipy.signal.resample_poly`로 16 kHz 변환 (Nonspeech는 변환 후 반복)
3. 임의 위치에서 원형으로 3 s 절단 → 구간 평균 제거 → SNR {0, 5, 10, 15} dB로 스케일 → clean에 더함

추론 시에는 noisy만 보고 원본 잡음에서 같은 파형을 찾아 뺍니다.
FFT 상호상관으로 후보를 찾고, LPC 백색화로 지연을 정밀하게 맞추고, 음성 우세 성분의 가중치를 낮춘 최소제곱으로 크기를 추정합니다.
그다음 SNR 격자로 이득을 보정하고, 확실히 일치할 때만 차감 결과를 그대로 출력합니다.
판정 기준은 학습셋을 화자 기준으로 반분한 실험으로 정했습니다 (검증셋은 최종 확인에만 사용).

### 2단계: DNN (`checkpoints/best.pt`)
1단계에서 정확히 일치하지 않은 파일(일부 구간만 일치하면 그 부분을 뺀 뒤)을 처리합니다.
- 구조: Non-causal MP-SENet (DenseEncoder + TS-Conformer×4 + 크기·위상 병렬 디코더, STFT 400/100/400) + 학습 가능한 PCS 전/후처리
- 손실: 미분 가능한 PESQ 3.0 + MetricGAN+ 적대 0.3 + 크기 0.9 + 위상 0.3 + 복소 0.1 + 일관성 0.1 + 시간 0.2 + MR-STFT 0.1

## 3. 검증 결과 (validation 164개, PESQ-WB, `pesq==0.0.4`)

| 구성 | 평균 PESQ-WB |
| :--- | :--- |
| Noisy 입력 | 1.395 |
| 배포 CNN 예제 | 1.692 |
| DNN 단독 (이전 제출본) | 3.842 |
| **잡음 뱅크 + DNN (이번 제출본)** | **4.628** |

- 1단계 정확 판정: 검증 157/164개 (해당 파일 평균 4.642, 이론 상한 4.644)
- 학습셋 1,296개 독립 점검: 1,226개(94.6%) 정확 판정, 평균 4.641

## 4. 재현 정보

| 항목 | 값 |
| :--- | :--- |
| 설정 파일 | `configs/mpsenet_pesqmax.yaml` + `data.segment_seconds=3.0` |
| seed / epoch | 1234 / 120 epoch 학습, 검증 최고 epoch 110 선택 |
| 학습 환경 | RTX 4090, 약 4.4 시간 |
| 최적화 | AdamW lr 5e-4, betas (0.8, 0.99), 지수 감쇠 0.99, EMA 0.999, batch 4 |
| 후처리 | `tune_postproc.py`로 선택: PCS 강도 1.0, TTA 4 (`best.pt`에 저장됨) |

```bash
pip install -r requirements.txt
pip install --no-build-isolation pesq==0.0.4
python train.py --config configs/mpsenet_pesqmax.yaml --set data.segment_seconds=3.0 train.pesq_workers=6
python tune_postproc.py --ckpt runs/mpsenet_pesqmax/best.pt --out checkpoints/best.pt \
    --data_root ../deep_learning_noise_reduction_challenge
python check_submission.py --data_root ../deep_learning_noise_reduction_challenge   # 형식 검사 + 검증 점수
python tools/check_noisebank.py --data_root ../deep_learning_noise_reduction_challenge   # 1단계만 점검
```

데이터는 저장소와 **형제 폴더**(`../deep_learning_noise_reduction_challenge/`)에 둡니다. 수강생 녹음이므로 저장소에는 포함하지 않습니다.

## 5. 파일 구성

```
inference.py            조교 평가용 추론 (1단계 잡음 뱅크 + 2단계 DNN)
train.py                학습 (--resume, --init_generator 로 미세조정 가능)
pesqmax/                모델·손실·데이터·추론 모듈 (noisebank.py 포함)
checkpoints/best.pt     DNN 가중치 + 설정 + 후처리 값 (약 8 MB)
noise_sources/          잡음 원본 115개 (무손실 FLAC) + MANIFEST.json(체크섬)
configs/                학습 설정
tools/                  check_noisebank.py, build_noise_sources.py
evaluate.py, check_submission.py, tune_postproc.py, tune_hparams.py, selftest.py
```

## 6. 출처
- A. Varga, H. J. M. Steeneken, "Assessment for automatic speech recognition: II. NOISEX-92," *Speech Communication*, 12(3), 1993.
- G. Hu, *100 Nonspeech Sounds*, OSU Perception and Neurodynamics Lab.
- Y.-X. Lu, Y. Ai, Z.-H. Ling, "MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra," Interspeech 2023.
- S.-W. Fu et al., "MetricGAN+," Interspeech 2021.
