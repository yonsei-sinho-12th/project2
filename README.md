# 2026 신호 및 시스템 2차 챌린지 - 딥러닝 음성 잡음 제거

- 팀명: 깨비드릴조(12조)
- 제출 회차: (회차 입력)
- 제출 버전: 태그 `v3`

잡음 음성의 **크기(magnitude)와 위상(phase) 스펙트럼을 병렬로 직접 추정**하는 MP-SENet 구조(약 205만 파라미터)에
**PESQ 판별기(MetricGAN)**, **동적 혼합 증강**, **EMA 가중치**를 더한 모델입니다.
추론은 **CPU만으로 동작**하며(Windows·macOS·Linux), 추가 다운로드가 필요 없습니다.

## Validation 결과 (164개, PESQ-WB)

| 방법 | PESQ-WB |
|---|---|
| Noisy 입력 | 1.395 |
| 노트북 예제 CNN | 1.692 |
| BiGRU 복소 마스크 (이전 제출, `v2`) | 2.343 |
| **MP-SENet (이번 제출, `v3`)** | **(fp32 재평가 점수 입력)** |

- 학습 중 검증 점수(bf16): 2.8272 (epoch 59, EMA 가중치)
- fp32 재평가: `python evaluate.py --data-root <데이터 폴더> --checkpoint checkpoints/best_generator.pt --device cpu`
- 파일별·잡음 종류별 결과와 학습 기록: `results/` 폴더

---

## 1. 추론 실행 방법 (제출 평가용)

`input/`의 모든 `.wav`를 처리해 `output/` 바로 아래에 **같은 파일명, 16 kHz 모노, 입력과 같은 표본 수**로 저장합니다.

```bash
python infer.py --input input --output output
```

- 가중치: `checkpoints/best_generator.pt` (`infer.py` 기준 상대 경로라 **경로 수정 불필요**)
- 장치는 자동 선택됩니다 (NVIDIA GPU가 있으면 GPU, 없으면 CPU). CPU를 강제하려면 `--device cpu`
- 3초 파일 1개당 CPU에서 약 1–5초. 인터넷 연결 불필요
- 입력이 PCM_16이면 PCM_16으로 저장합니다. 최대값이 1을 넘으면 클리핑 대신 전체 크기만 줄입니다 (PESQ는 레벨 정렬을 하므로 점수 영향 없음)
- 스테레오나 16 kHz가 아닌 입력도 모노·16 kHz로 변환해 처리합니다

### 1.1. 추론용 환경 만들기

추론에는 `torch`, `numpy`, `scipy`, `soundfile`만 필요합니다 (`pesq`는 학습·평가에만 필요).

**Windows (명령 프롬프트 또는 PowerShell)**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch numpy scipy soundfile
python infer.py --input input --output output
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy soundfile
python infer.py --input input --output output
```

**Anaconda를 쓰는 경우 (Windows 권장)**

```bat
conda create -n denoise -c conda-forge python=3.12 pesq=0.0.4
conda activate denoise
pip install torch numpy scipy soundfile pandas matplotlib tqdm
python infer.py --input input --output output
```

**Google Colab**

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/<저장소 폴더>
!python infer.py --input input --output output
```

- Python 3.10 이상, PyTorch 2.1 이상에서 동작합니다 (개발: Python 3.14, PyTorch 2.15)
- Windows PowerShell에서 `activate`가 막히면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 한 번 실행하세요

### 1.2. 전체 패키지 (학습·평가까지)

```bash
pip install -r requirements.txt
```

`pesq==0.0.4`는 C 확장이라 Windows에서 컴파일 오류가 날 수 있습니다. 이때는 위의 Anaconda 방법(`-c conda-forge pesq=0.0.4`)을 쓰세요.
Linux에서 `pip install pesq`가 실패하면 `pip install cython numpy` 후 `pip install --no-build-isolation pesq==0.0.4`를 실행하세요.

---

## 2. 모델

1. Noisy 파형 → 입력 RMS로 정규화 → STFT (Hann 400, hop 100, n_fft 400, 16 kHz)
2. 압축 크기 |Y|^0.3과 위상 ∠Y를 2채널 [시간 × 주파수] 입력으로 구성
3. **Dense 인코더**: 시간축 dilation 1·2·4·8의 dense 연결 2D 합성곱 → 주파수축 절반으로 축소 (201 → 100)
4. **TS-Conformer × 4**: 시간축 Conformer(주파수 빈마다 시간 방향 어텐션) → 주파수축 Conformer(프레임마다 주파수 방향 어텐션)
5. **크기 디코더**: 학습 가능한 시그모이드(상한 2)로 마스크를 만들어 noisy 크기에 곱함
6. **위상 디코더**: 두 출력(실수·허수 성분)의 atan2로 위상을 **직접** 추정
7. 복원 크기·위상 → iSTFT → 원래 크기로 되돌림 (입력과 같은 표본 수)

| 항목 | BiGRU (v2) | MP-SENet (v3) |
|---|---|---|
| 입력 | 크기만 (log\|Y\|) | 압축 크기 + 위상 |
| 문맥 | 시간축 GRU | 시간축 + 주파수축 어텐션 |
| 위상 | 복소 마스크로 간접 보정 | 위상 디코더로 직접 추정 |
| 손실 | 크기·복소 MSE | 크기·위상·복소·파형·STFT 일관성 + PESQ 판별기 |

---

## 3. 학습 (`train.py`)

```bash
python train.py --data-root <데이터 폴더> --out-dir runs/full --epochs 60 --val-precision amp
```

- `<데이터 폴더>`: `train/`, `validation/`이 있는 배포 폴더 (**개인 PC 경로는 이 인자만 바꾸면 됩니다**)
- 학습에는 GPU가 필요합니다. CPU로는 epoch당 1시간 이상 걸립니다
  - **NVIDIA GPU / Colab**: 일반 `pip install torch`로 바로 동작
  - **AMD GPU (Linux)**: ROCm용 PyTorch를 설치하고 아래 환경 변수를 설정
    ```bash
    pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1152]==2.15.0+rocm10.2.0"   # gfx는 GPU에 맞게
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1   # 빠른 어텐션
    export MIOPEN_LOG_LEVEL=3                          # MIOpen 경고 숨김
    ```
- 같은 `--out-dir`로 다시 실행하면 `last.pt`에서 **자동으로 이어서 학습**합니다 (`--restart`를 주면 처음부터)
- 매 epoch 검증 164개 전체의 PESQ-WB로 최고 가중치를 `runs/full/best_generator.pt`에 저장 → `checkpoints/`로 복사해 제출

### 3.1. 학습 설정

| 항목 | 값 |
|---|---|
| 학습 데이터 | Train 1,296쌍 (Validation은 가중치 선택에만 사용) |
| seed | 42 |
| epoch / 배치 | 60 / 4, 학습 구간 무작위 2초 (검증·추론은 전체 길이) |
| 최적화 | AdamW (β=0.8, 0.99, weight decay 0.01), lr 5e-4 → 1e-5 코사인, 2 epoch 워밍업, grad clip 10 |
| 손실 가중치 | PESQ 판별기 0.05, 크기 MSE 0.9, 위상 0.3, 복소 0.1, 파형 L1 0.2, STFT 일관성 0.1 |
| 위상 손실 | 순시 위상 + 군지연(주파수 차분) + 순시 각주파수(시간 차분), 2π 감김 보정 |
| 데이터 증강 | 60% 확률로 다른 Train 파일의 잡음(Noisy − Clean)을 학습셋 SNR 분포 ±3 dB로 다시 섞음, 잡음 구간 무작위·극성 반전 0.5·시간 반전 0.2·잡음 2개 중첩 0.1 |
| EMA | decay 0.999 |
| 가중치 선택 | 매 epoch 검증 PESQ-WB 최고점 (epoch 1–4: 일반·EMA 모두 fp32 / epoch 5–60: EMA만 bf16) |

### 3.2. 실제 학습 기록

| 항목 | 값 |
|---|---|
| 학습 환경 | Ubuntu 26.04, AMD Ryzen AI 7 PRO 350 내장 Radeon 860M (ROCm 10.2, PyTorch 2.15.0+rocm10.2.0, bf16 혼합 정밀도) |
| 학습 시간 | epoch당 약 30분 (학습 28분 + 검증), 총 약 30시간 (epoch 5부터 이어서 학습) |
| 선택한 가중치 | epoch 59, EMA, 학습 중 검증 PESQ-WB 2.8272 |
| 검증 PESQ 추이 | epoch 1: 1.33 → 4: 2.00 → 56: 2.826 → 59: 2.827 |

`results/training_history.csv`에 epoch별 손실·검증 점수·학습률이, `results/training_summary.json`에 종료 조건과 설정이 있습니다.
장비가 다르면 학습 결과가 조금 다를 수 있지만, 제출 평가는 저장된 가중치로 하므로 추론 결과는 같습니다.

---

## 4. 검증과 제출 전 점검

```bash
# 검증 164개 전체 평가 (파일별·잡음 종류별 CSV 저장)
python evaluate.py --data-root <데이터 폴더> --checkpoint checkpoints/best_generator.pt --device cpu

# 검증셋을 테스트셋처럼 처리해 제출 형식 점검
python check_submission.py export --data-root <데이터 폴더> --dest val_test
python infer.py --input val_test/input --output val_test/output
python check_submission.py check --input val_test/input --output val_test/output --clean val_test/clean
```

마지막 명령은 파일 누락·추가 WAV·표본률·채널·표본 수를 확인하고 PESQ-WB 평균을 출력합니다.

---

## 5. 파일 구성

```
infer.py                       ★ 제출용 추론 (input/ → output/)
checkpoints/best_generator.pt  ★ 학습된 가중치 (모델 설정 포함, 약 8 MB)
model.py                       MP-SENet 생성기, PESQ 판별기, STFT·손실 함수
data.py                        samples.csv·파형 캐시 읽기, 잡음 분리, 동적 혼합
metrics.py                     PESQ-WB(병렬)·SNR
train.py                       학습 (자동 이어 학습, EMA, 조기 종료)
evaluate.py                    검증 전체 PESQ/SNR, 잡음 종류별 결과 CSV
check_submission.py            출력 형식 점검
challenge2_mpsenet.ipynb       학습·평가·청취 비교 노트북 (Colab·Jupyter)
results/                       학습 기록과 검증 결과 CSV
requirements.txt
```

개인 식별 정보가 포함될 수 있는 원본 녹음(데이터 폴더, `input/`, `output/`, `val_test/`)은 저장소에 포함하지 않습니다 (`.gitignore`).

## 참고 문헌

- Y.-X. Lu, Y. Ai, Z.-H. Ling, "MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra," Interspeech 2023.
- S.-W. Fu et al., "MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement," Interspeech 2021.
