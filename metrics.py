"""PESQ-WB / SNR 계산 (여러 프로세스로 병렬 계산)."""
import multiprocessing as mp
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SAMPLE_RATE = 16_000


def pesq_safe(reference, degraded):
    """PESQ-WB. 계산할 수 없는 경우(무음 등) NaN을 돌려줍니다."""
    from pesq import pesq
    try:
        score = float(pesq(SAMPLE_RATE, np.asarray(reference, np.float32),
                           np.asarray(degraded, np.float32), mode="wb"))
        return score if np.isfinite(score) else float("nan")
    except Exception:
        return float("nan")


def _pesq_pair(pair):
    return pesq_safe(*pair)


def _die_with_parent():
    """Linux: 부모(학습) 프로세스가 비정상 종료되면 PESQ 워커도 함께 종료되게 합니다."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
        except Exception:
            pass


def _warmup(_):
    import pesq  # noqa: F401
    return 0


class PesqPool:
    """PESQ 계산 전용 프로세스 풀. CUDA를 쓰기 전에 만들어 두세요."""

    def __init__(self, workers=None):
        workers = workers or max(1, min(8, (os.cpu_count() or 2)))
        self.workers = workers
        # spawn: CUDA를 초기화한 프로세스를 fork하지 않도록 새 인터프리터로 시작합니다.
        self.pool = (ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                         initializer=_die_with_parent)
                     if workers > 1 else None)
        if self.pool is not None:   # 워커를 미리 띄워 둡니다.
            list(self.pool.map(_warmup, range(workers)))

    def submit(self, references, degraded):
        pairs = list(zip(references, degraded))
        if self.pool is None:
            return [pesq_safe(r, d) for r, d in pairs]
        return [self.pool.submit(_pesq_pair, p) for p in pairs]

    @staticmethod
    def gather(handles):
        return np.array([h.result() if hasattr(h, "result") else h for h in handles], dtype=np.float64)

    def map(self, references, degraded):
        return self.gather(self.submit(references, degraded))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


def snr_db(clean, estimate):
    clean = np.asarray(clean, np.float64)
    error = clean - np.asarray(estimate, np.float64)
    signal_energy, error_energy = np.sum(clean ** 2), np.sum(error ** 2)
    if signal_energy == 0:
        return float("nan")
    return float("inf") if error_energy == 0 else float(10 * np.log10(signal_energy / error_energy))
