"""실제 PESQ-WB (pesq 패키지, ITU-T P.862.2) 병렬 계산과 SNR.

평가 점수와 같은 구현(pesq==0.0.4, mode="wb", 16 kHz)을 사용합니다.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np


def pesq_wb_single(args) -> float:
    ref, deg, sample_rate = args
    from pesq import pesq  # 작업 프로세스에서 지연 import

    try:
        score = float(pesq(sample_rate, np.asarray(ref, dtype=np.float32),
                           np.asarray(deg, dtype=np.float32), "wb"))
    except Exception:  # 무음 구간 등으로 PESQ 계산이 실패한 경우
        return float("nan")
    return score if np.isfinite(score) else float("nan")


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    err = clean - np.asarray(estimate, dtype=np.float64)
    signal, noise = np.sum(clean ** 2), np.sum(err ** 2)
    if signal == 0:
        return float("nan")
    return float("inf") if noise == 0 else float(10 * np.log10(signal / noise))


class _Ready:
    def __init__(self, value):
        self.value = value

    def get(self, timeout=None):
        return self.value

    def ready(self):
        return True


class PesqPool:
    """PESQ 를 CPU 프로세스 풀에서 병렬 계산합니다.

    Linux(Colab 포함)에서는 fork 를 사용하므로 CUDA 초기화 이전에 만드는 것이 안전합니다.
    workers=0 이면 메인 프로세스에서 순차 계산합니다.
    """

    def __init__(self, workers: int, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.workers = max(0, int(workers))
        self.pool = None
        if self.workers > 0:
            method = "fork" if "fork" in mp.get_all_start_methods() and os.name != "nt" else "spawn"
            self.pool = mp.get_context(method).Pool(self.workers)

    def _jobs(self, refs, degs):
        return [(r, d, self.sample_rate) for r, d in zip(refs, degs)]

    def map_async(self, refs, degs):
        jobs = self._jobs(refs, degs)
        if self.pool is None:
            return _Ready([pesq_wb_single(j) for j in jobs])
        return self.pool.map_async(pesq_wb_single, jobs, chunksize=1)

    def map(self, refs, degs) -> np.ndarray:
        return np.asarray(self.map_async(refs, degs).get(), dtype=np.float64)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
