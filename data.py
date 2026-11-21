"""데이터 불러오기·캐시·증강.

배포 노트북과 같은 samples.csv / .waveform_cache 형식을 그대로 씁니다.
노트북에서 이미 캐시를 만들었다면 WAV를 다시 읽지 않습니다.

학습 데이터가 1,296쌍(약 65분)으로 작기 때문에, noisy − clean으로 잡음을 분리한 뒤
매번 다른 화자 음성과 다른 잡음·SNR로 새 혼합음을 만드는 동적 혼합(dynamic mixing)을
적용합니다. 검증 데이터는 원래 쌍 그대로 사용합니다.
"""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000


def read_wav(path):
    samples, rate = sf.read(path, dtype="float32")
    if rate != SAMPLE_RATE or samples.ndim != 1:
        raise ValueError(f"16 kHz 모노 WAV가 필요합니다: {path}")
    return samples


def load_split(root, split, cache_dir=None, rebuild=False):
    """(manifest DataFrame, waveforms [N, 2(noisy/clean), S] float32)를 돌려줍니다."""
    root = Path(root)
    cache_dir = Path(cache_dir) if cache_dir else root / ".waveform_cache"
    manifest_bytes = (root / split / "samples.csv").read_bytes()
    manifest = pd.read_csv(BytesIO(manifest_bytes))
    if manifest.empty or manifest.segment_id.duplicated().any():
        raise ValueError(f"{split} manifest의 segment_id를 확인하세요.")
    lengths = manifest["num_samples"]
    sample_count = int(lengths.iloc[0])
    if not lengths.eq(sample_count).all():
        raise ValueError(f"{split}의 모든 파형 길이가 같아야 합니다.")

    key = sha256(manifest_bytes).hexdigest()[:20]
    cache_path = cache_dir / f"{split}_v1_{SAMPLE_RATE}_{key}.npy"   # 배포 노트북과 같은 이름
    shape = (len(manifest), 2, sample_count)
    if cache_path.is_file() and not rebuild:
        try:
            waves = np.load(cache_path, allow_pickle=False)
            if waves.shape == shape and waves.dtype == np.float32:
                print(f"{split}: 캐시 사용 ({len(manifest):,}쌍) {cache_path}")
                return manifest, waves
        except (OSError, ValueError, EOFError):
            pass

    print(f"{split}: WAV 읽는 중 ({len(manifest):,}쌍)…")
    waves = np.empty(shape, dtype=np.float32)
    for i, row in enumerate(manifest.itertuples(index=False)):
        noisy, clean = read_wav(root / row.noisy_path), read_wav(root / row.clean_path)
        if noisy.size != sample_count or clean.size != sample_count:
            raise ValueError(f"파형 길이가 일치하지 않습니다: {row.segment_id}")
        waves[i, 0], waves[i, 1] = noisy, clean
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_path.parent) as tmp:
        tmp_path = Path(tmp) / cache_path.name
        np.save(tmp_path, waves, allow_pickle=False)
        tmp_path.replace(cache_path)
    print(f"{split}: 캐시 저장 {cache_path}")
    return manifest, waves


def extract_noise(waves, verbose=True):
    """noisy = g·(clean + noise) 관계를 가정하고 쌍마다 g와 noise를 추정합니다.

    대부분의 합성 데이터는 g = 1 (단순 덧셈)입니다. 혼합 후 크기 정규화가 있었다면
    clean 방향으로의 사영으로 g를 추정해 보정합니다.
    """
    noisy = waves[:, 0].astype(np.float64)
    clean = waves[:, 1].astype(np.float64)
    clean_energy = np.maximum((clean ** 2).sum(1), 1e-12)
    gain = (noisy * clean).sum(1) / clean_energy
    gain = np.where(np.isfinite(gain) & (gain > 0.05), gain, 1.0)
    if np.median(np.abs(gain - 1.0)) < 0.02:
        gain = np.ones_like(gain)
    noise = noisy / gain[:, None] - clean
    # 이 SNR은 동적 혼합에서 쓰는 정의(3초 전체 평균 전력비)와 같습니다.
    snr = 10 * np.log10(clean_energy / np.maximum((noise ** 2).sum(1), 1e-12))
    if verbose:
        print(f"잡음 분리: 혼합 게인 중앙값 {np.median(gain):.4f} | "
              f"추정 SNR 범위(5~95%) {np.percentile(snr, 5):.1f} ~ {np.percentile(snr, 95):.1f} dB "
              f"(중앙값 {np.median(snr):.1f} dB)")
    return noise.astype(np.float32), snr


class TrainDataset(Dataset):
    """원래 쌍 + 동적 혼합 학습 데이터.

    dm_prob 확률로 (clean_i, noise_j, 새 SNR) 혼합을 새로 만들고, 나머지는 원래 쌍을 씁니다.
    SNR은 학습셋의 실제 SNR 분포에서 뽑아 ±snr_jitter dB 흔듭니다.
    segment 길이로 무작위 구간을 자르며, 잡음 구간은 음성 구간과 독립적으로 고릅니다.
    """

    def __init__(self, waves, segment, dm_prob=0.6, snr_jitter=3.0, snr_values=None,
                 noise_flip_prob=0.5, noise_reverse_prob=0.2, extra_noise_prob=0.1, seed=0):
        self.noisy = waves[:, 0]
        self.clean = waves[:, 1]
        self.noise, est_snr = extract_noise(waves)
        self.length = waves.shape[-1]
        self.segment = min(segment, self.length)
        self.dm_prob = dm_prob
        self.snr_jitter = snr_jitter
        self.snr_values = np.asarray(snr_values if snr_values is not None else est_snr, dtype=np.float64)
        self.noise_flip_prob = noise_flip_prob
        self.noise_reverse_prob = noise_reverse_prob
        self.extra_noise_prob = extra_noise_prob
        self.clean_power = np.maximum((self.clean.astype(np.float64) ** 2).mean(1), 1e-10)
        self.noise_power = np.maximum((self.noise.astype(np.float64) ** 2).mean(1), 1e-10)
        self.seed = seed
        self.epoch = 0
        self._rng = None

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._rng = None

    def __len__(self):
        return len(self.clean)

    def _crop(self, rng):
        return int(rng.integers(0, self.length - self.segment + 1))

    def _noise(self, rng, j):
        start = self._crop(rng)
        noise = self.noise[j, start:start + self.segment].astype(np.float32)
        if rng.random() < self.noise_reverse_prob:
            noise = noise[::-1]
        if rng.random() < self.noise_flip_prob:
            noise = -noise
        return noise, self.noise_power[j]

    def __getitem__(self, i):
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            wid = worker.id if worker else 0
            self._rng = np.random.default_rng([self.seed, self.epoch, wid])
        rng = self._rng
        start = self._crop(rng)
        clean = self.clean[i, start:start + self.segment]
        if rng.random() >= self.dm_prob:
            noisy = self.noisy[i, start:start + self.segment]
            return torch.from_numpy(noisy.copy()), torch.from_numpy(clean.copy())

        j = int(rng.integers(len(self.clean)))
        noise, noise_power = self._noise(rng, j)
        if rng.random() < self.extra_noise_prob:          # 가끔 잡음 두 개를 겹칩니다
            k = int(rng.integers(len(self.clean)))
            extra, extra_power = self._noise(rng, k)
            ratio = 10 ** (rng.uniform(-10, 0) / 20)
            noise = noise / np.sqrt(noise_power) + ratio * extra / np.sqrt(extra_power)
            noise_power = 1.0 + ratio ** 2
        snr = float(rng.choice(self.snr_values)) + rng.uniform(-self.snr_jitter, self.snr_jitter)
        # 원래 데이터처럼 3초 전체 구간의 평균 전력으로 SNR을 맞춥니다.
        scale = np.sqrt(self.clean_power[i] / (noise_power * 10 ** (snr / 10)))
        noisy = clean + scale * noise
        return torch.from_numpy(noisy.astype(np.float32)), torch.from_numpy(clean.copy())


class PairDataset(Dataset):
    """검증용: 원래 noisy/clean 쌍 전체 길이."""

    def __init__(self, waves):
        self.waves = torch.from_numpy(waves)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, i):
        noisy, clean = self.waves[i]
        return noisy, clean
