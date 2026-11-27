"""설정 로딩: 기본값 <- YAML 파일 <- 명령행 --set key=value 순서로 덮어씁니다."""

from __future__ import annotations

import copy
from pathlib import Path

DEFAULTS: dict = {
    "seed": 1234,
    "data": {
        # train/ 과 validation/ 이 들어 있는 배포 폴더 (상대 경로는 이 저장소 루트 기준)
        "root": "../deep_learning_noise_reduction_challenge",
        "sample_rate": 16000,
        "segment_seconds": 3.0,          # 3초 = 평가 파일 전체 길이 (평가 조건과 동일)
        "remix_prob": 0.5,               # 도메인 매칭 동적 리믹스 확률
        "remix_same_category_prob": 0.0,  # 리믹스 시 같은 잡음 범주에서 고를 확률
        "noise_roll": True,              # 잡음 원형 이동(시간 위치 다양화)
        "speed_perturb_prob": 0.0,       # 화자 다양화용 속도 변조 확률 (튜닝 대상)
        "speed_factors": [0.95, 1.05],
        "use_validation_for_training": False,  # 최종 train+val 재학습 (규정 확인 필수)
        "num_workers": 2,
        "max_train_files": 0,            # 0 = 전체 (스모크 테스트용 제한)
    },
    "stft": {"n_fft": 400, "hop": 100, "win": 400, "compress": 0.3},
    "model": {
        "dense_channel": 64,
        "num_tsblocks": 4,
        "n_heads": 4,
        "ffn_mult": 4,
        "conv_kernel": 31,
        "conv_expansion": 2,
        "dropout": 0.1,
        "dense_depth": 4,
        "mask_beta": 2.0,
    },
    "pcs": {
        "pre": True,          # 입력 특징용 학습 가능 PCS (전처리)
        "post": True,         # 출력 스펙트럼용 학습 가능 PCS (후처리, PESQ 손실로만 학습)
        "init": "pcs400",     # pcs400 | identity
        "gamma_min": 0.8,
        "gamma_max": 2.0,
        "target": False,      # 고정 PCS400 을 학습 목표(clean)에 적용 (원 논문 방식)
        "tv_weight": 0.01,    # 주파수축 gamma 평활화 정규화
        "ref_peak": 32767.0,  # 원 PCS 구현의 int16 피크 정규화 스케일 재현
    },
    "loss": {
        "mag": 0.9,
        "phase": 0.3,
        "complex": 0.1,
        "consistency": 0.1,
        "time": 0.2,
        "mrstft": 0.1,
        "pesq": 3.0,              # 미분 가능한 PESQ 손실 (높은 가중치)
        "metric": 0.3,            # MetricGAN+ 적대 손실: D 출력을 1.0(=PESQ 4.5)으로
        "pesq_start_epoch": 1,    # 이 epoch 부터 PESQ 손실 사용
        "pesq_warmup_epochs": 3,  # 시작 후 PESQ 손실 가중치를 선형으로 키우는 구간
        "metric_start_epoch": 1,  # 생성자 적대 손실 시작 epoch (판별자는 1 epoch 부터 학습)
    },
    "disc": {
        "dim": 16,
        "lr": 5.0e-4,
        "use_noisy": True,    # MetricGAN+: noisy 입력의 실제 PESQ 도 학습
        "replay_size": 256,   # MetricGAN+: 과거 출력 재사용 버퍼
        "replay_k": 2,
    },
    "train": {
        "epochs": 120,
        "batch_size": 4,
        "lr": 5.0e-4,
        "betas": [0.8, 0.99],
        "weight_decay": 0.01,
        "lr_gamma": 0.99,         # epoch 마다 ExponentialLR
        "clip_grad": 10.0,
        "amp": True,              # CUDA 에서만 사용 (bf16 지원 시 bf16, 아니면 fp16)
        "ema_decay": 0.999,
        "val_every": 1,
        "val_batch_size": 8,
        "val_max_files": 0,       # 0 = 검증 전체 (스모크 테스트용 제한)
        "pesq_workers": 4,
        "log_every": 50,
        "out_dir": "runs/pesqmax",
        "max_steps_per_epoch": 0,  # 0 = 제한 없음
        "early_stop_patience": 0,  # 0 = 사용 안 함
        "diagnose_grad_norms": False,
        "save_every": 0,           # >0 이면 해당 epoch 간격으로 EMA 스냅샷 저장
    },
}


class AttrDict(dict):
    """cfg.model.dense_channel 처럼 점 표기로 접근할 수 있는 dict."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __deepcopy__(self, memo):
        return AttrDict({k: copy.deepcopy(v, memo) for k, v in self.items()})


def to_attr(obj):
    if isinstance(obj, dict):
        return AttrDict({k: to_attr(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attr(v) for v in obj]
    return obj


def to_plain(obj):
    """체크포인트(weights_only=True)에 안전하게 저장할 수 있는 순수 파이썬 타입으로 변환."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if obj is None or type(obj) in (bool, int, float, str):
        return obj
    if hasattr(obj, "item") and callable(obj.item):  # numpy / torch 스칼라
        return obj.item()
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    return str(obj)  # Path, TorchVersion 등 문자열 하위 타입


def deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_by_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"알 수 없는 설정 키: {dotted}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"알 수 없는 설정 키: {dotted}")
    node[keys[-1]] = value


def load_config(path: str | Path | None = None, overrides=()) -> AttrDict:
    import yaml  # 학습에서만 필요 (추론은 체크포인트에 저장된 설정을 사용)

    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            user = yaml.safe_load(handle) or {}
        unknown = set(user) - set(cfg)
        if unknown:
            raise KeyError(f"알 수 없는 최상위 설정: {sorted(unknown)}")
        deep_update(cfg, user)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"--set 형식은 key=value 입니다: {item}")
        key, raw = item.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return to_attr(cfg)


def merge_with_defaults(saved: dict) -> AttrDict:
    """오래된 체크포인트에 없는 키를 기본값으로 채웁니다."""
    cfg = copy.deepcopy(DEFAULTS)
    deep_update(cfg, saved)
    return to_attr(cfg)
