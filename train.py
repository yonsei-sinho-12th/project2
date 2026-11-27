#!/usr/bin/env python
"""PESQ-WB 극대화 학습: MP-SENet + 학습 가능 PCS + 미분 가능 PESQ 손실 + MetricGAN+ 판별자.

예)  python train.py --config configs/mpsenet_pesqmax.yaml
     python train.py --config configs/mpsenet_pesqmax.yaml --resume          # 중단된 학습 이어서
     python train.py --config configs/mpsenet_lite.yaml --set train.epochs=30 loss.pesq=4.0

학습 한 step 의 흐름
  1) noisy/clean 을 noisy 의 RMS 로 함께 정규화 (PESQ 는 레벨 불변)
  2) G(noisy) -> PCS 이전 출력(wav_pre, 복원 손실용) / PCS 이후 최종 출력(wav, PESQ 지표용)
  3) 이번 배치의 실제 PESQ-WB(pesq 패키지)를 CPU 프로세스 풀에 비동기로 요청
  4) 직전 배치의 실제 PESQ 가 도착하면 판별자 D 갱신 (MetricGAN+):
       D(clean,clean)->1,  D(clean,enhanced)->Q(enhanced),  D(clean,noisy)->Q(noisy),  + replay
  5) 생성자 손실 = 복원 손실(크기·위상·복소·일관성·시간·MR-STFT)
                 + w_pesq * 미분 가능 PESQ 손실 + w_metric * (D(clean, enhanced) - 1)^2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "configs" / "mpsenet_pesqmax.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="설정 덮어쓰기 (예: train.epochs=50 loss.pesq=4.0)")
    parser.add_argument("--data_root", default=None, help="train/, validation/ 가 있는 배포 폴더")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--resume", action="store_true", help="out_dir/last.pt 에서 이어서 학습")
    parser.add_argument("--init_generator", default=None, metavar="CKPT",
                        help="미세조정 시작점: 생성자 가중치만 불러오고 optimizer/scheduler/epoch 는 새로 "
                             "시작 (제출용 best.pt 또는 last.pt). --resume 으로 last.pt 가 복원되면 무시")
    parser.add_argument("--init_discriminator", default=None, metavar="LAST_PT",
                        help="(선택) last.pt 의 판별자 D 를 불러옴. 없으면 D 는 새로 초기화")
    parser.add_argument("--no_baseline_val", action="store_true",
                        help="--init_generator 사용 시 시작 전 기준 검증을 건너뜀 "
                             "(기본: 시작 가중치 점수를 넘을 때만 best.pt 저장)")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    from pesqmax.config import load_config
    from pesqmax.metrics import PesqPool

    cfg = load_config(args.config, args.set)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.out_dir:
        cfg.train.out_dir = args.out_dir
    # PESQ 작업 프로세스는 데이터 적재·CUDA 초기화 이전에 fork 합니다 (가볍고 안전).
    pool = PesqPool(cfg.train.pesq_workers, cfg.data.sample_rate)
    try:
        Trainer(cfg, args, pool).fit()
    finally:
        pool.close()


class Trainer:
    def __init__(self, cfg, args, pool):
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        from pesqmax.config import to_plain
        from pesqmax.data import ChallengeSplit, RemixTrainDataset, resolve_data_root
        from pesqmax.discriminator import MetricDiscriminator, ReplayBuffer
        from pesqmax.losses import MultiResolutionSTFTLoss
        from pesqmax.model import Enhancer, count_parameters
        from pesqmax.pcs import LearnablePCS
        from pesqmax.pesq_loss import PesqLoss
        from pesqmax.utils import EMA, amp_dtype, pick_device, seed_everything

        self.np, self.torch = np, torch
        self.cfg, self.args, self.pool = cfg, args, pool
        self.out_dir = Path(cfg.train.out_dir)
        if not self.out_dir.is_absolute():
            self.out_dir = Path.cwd() / self.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(int(cfg.seed))

        self.device = pick_device(args.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        self.amp = amp_dtype(self.device, bool(cfg.train.amp))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp == torch.float16)

        # ---------------- data
        root = resolve_data_root(cfg.data.root, REPO)
        sr = int(cfg.data.sample_rate)
        cache = REPO / ".cache"
        t0 = time.time()
        self.train_split = ChallengeSplit(root, "train", sr, int(cfg.data.max_train_files), cache)
        self.val_split = ChallengeSplit(root, "validation", sr, int(cfg.train.val_max_files), cache)
        splits = [self.train_split]
        if cfg.data.use_validation_for_training:
            splits.append(self.val_split)
            print("[주의] data.use_validation_for_training=true: 검증셋도 학습에 사용합니다. "
                  "검증 점수는 더 이상 일반화 성능 추정치가 아니며, 마지막 epoch 를 저장합니다.")
        self.train_set = RemixTrainDataset(splits, cfg.data, int(cfg.seed))
        workers = 0 if os.name == "nt" else int(cfg.data.num_workers)
        self.loader = DataLoader(
            self.train_set, batch_size=int(cfg.train.batch_size), shuffle=True, drop_last=True,
            num_workers=workers, pin_memory=self.device.type == "cuda",
            generator=torch.Generator().manual_seed(int(cfg.seed)))
        print(f"데이터: train {len(self.train_set)}쌍 | validation {len(self.val_split)}쌍 | "
              f"세그먼트 {self.train_set.segment / sr:.2f}s | 로딩 {time.time() - t0:.1f}s")

        # ---------------- models / optimizers
        self.G = Enhancer(cfg).to(self.device)
        self.D = MetricDiscriminator(int(cfg.disc.dim)).to(self.device)
        # warm-start: --resume 으로 복원할 last.pt 가 있으면 그쪽이 우선
        resuming = bool(args.resume) and (self.out_dir / "last.pt").is_file()
        self.init_loaded = False
        init_sd = None
        if args.init_generator and not resuming:
            init_sd = self._load_init_generator(Path(args.init_generator))
        if args.init_discriminator and not resuming:
            self._load_init_discriminator(Path(args.init_discriminator))
        self.ema = EMA(self.G, float(cfg.train.ema_decay))
        if init_sd is not None:  # EMA 구현과 무관하게 shadow 도 시작 가중치로 맞춤
            self.ema.shadow.load_state_dict(init_sd, strict=True)
            self.init_loaded = True
        betas = tuple(float(b) for b in cfg.train.betas)
        self.opt_g = torch.optim.AdamW(self.G.parameters(), lr=float(cfg.train.lr), betas=betas,
                                       weight_decay=float(cfg.train.weight_decay))
        self.opt_d = torch.optim.AdamW(self.D.parameters(), lr=float(cfg.disc.lr), betas=betas,
                                       weight_decay=float(cfg.train.weight_decay))
        gamma = float(cfg.train.lr_gamma)
        self.sched_g = torch.optim.lr_scheduler.ExponentialLR(self.opt_g, gamma=gamma)
        self.sched_d = torch.optim.lr_scheduler.ExponentialLR(self.opt_d, gamma=gamma)
        self.pesq_loss = PesqLoss(1.0, sr).to(self.device)
        self.mrstft = MultiResolutionSTFTLoss().to(self.device)
        self.target_pcs = None
        if cfg.pcs.target:
            self.target_pcs = LearnablePCS(self.G.n_fft // 2 + 1, self.G.n_fft, sr, "pcs400",
                                           cfg.pcs.gamma_min, cfg.pcs.gamma_max,
                                           learnable=False).to(self.device)
        self.replay = ReplayBuffer(int(cfg.disc.replay_size))
        self.rng = np.random.default_rng(int(cfg.seed))

        self.start_epoch, self.global_step = 1, 0
        self.best, self.best_epoch, self.stale = -math.inf, 0, 0
        self.history: list[dict] = []
        self.n_params = count_parameters(self.G)
        if args.resume:
            self._resume()
        with open(self.out_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(to_plain(cfg), handle, ensure_ascii=False, indent=2)
        print(f"장치 {self.device} | AMP {self.amp} | G {self.n_params:,} params | "
              f"D {count_parameters(self.D):,} params | {len(self.loader)} steps/epoch | "
              f"출력 {self.out_dir}")

    # ------------------------------------------------------------------ helpers
    # 가중치 모양을 결정하는 설정. 다르면 미세조정이 불가능하므로 시작 전에 중단합니다.
    _ARCH_KEYS = {
        "stft": ("n_fft", "hop", "win", "compress"),
        "model": ("dense_channel", "num_tsblocks", "n_heads", "ffn_mult", "conv_kernel",
                  "conv_expansion", "dense_depth", "mask_beta"),
        "pcs": ("pre", "post", "target", "gamma_min", "gamma_max", "ref_peak"),
    }

    def _check_arch(self, ckpt_cfg, path):
        from pesqmax.config import to_plain
        cur = to_plain(self.cfg)
        diffs = []
        for sec, keys in self._ARCH_KEYS.items():
            for key in keys:
                a, b = cur.get(sec, {}).get(key), (ckpt_cfg or {}).get(sec, {}).get(key)
                if b is not None and a != b:
                    diffs.append(f"{sec}.{key}: 현재 {a!r} / 체크포인트 {b!r}")
        sr_a = cur.get("data", {}).get("sample_rate")
        sr_b = (ckpt_cfg or {}).get("data", {}).get("sample_rate")
        if sr_b is not None and sr_a != sr_b:
            diffs.append(f"data.sample_rate: 현재 {sr_a!r} / 체크포인트 {sr_b!r}")
        if diffs:
            sys.exit(f"[init] {path} 와 현재 설정의 모델 구조가 다릅니다. --config/--set 을 맞추세요:\n  "
                     + "\n  ".join(diffs))

    def _load_init_generator(self, path):
        """제출용 체크포인트('generator', EMA 가중치) 또는 last.pt('G')에서 생성자만 불러옴."""
        if not path.is_file():
            sys.exit(f"[init] 파일이 없습니다: {path}")
        state = self.torch.load(path, map_location="cpu", weights_only=True)
        if "generator" in state:
            sd, src = state["generator"], "generator(EMA)"
        elif "G" in state:
            sd, src = state["G"], "G(last.pt, 비-EMA)"
        else:
            sys.exit(f"[init] {path} 에 생성자 가중치('generator' 또는 'G')가 없습니다.")
        self._check_arch(state.get("config"), path)
        self.G.load_state_dict(sd, strict=True)
        meta = state.get("meta", {})
        print(f"[init] 생성자 가중치 {src} <- {path} (epoch {meta.get('epoch', state.get('epoch'))}, "
              f"기록된 val PESQ-WB {meta.get('val_pesq_wb')}). optimizer/scheduler/epoch 는 새로 시작합니다.")
        L = self.cfg.loss
        if int(L.pesq_start_epoch) > 1 or int(L.pesq_warmup_epochs) > 0:
            print(f"[init][주의] loss.pesq_start_epoch={L.pesq_start_epoch}, "
                  f"pesq_warmup_epochs={L.pesq_warmup_epochs}: 초반 epoch 는 PESQ 손실이 약하거나 없습니다. "
                  "미세조정이면 --set loss.pesq_start_epoch=1 loss.pesq_warmup_epochs=0 을 권장합니다.")
        if not self.args.init_discriminator and float(L.metric) > 0 and int(L.metric_start_epoch) <= 1:
            print("[init][주의] 판별자 D 가 새로 초기화된 상태에서 epoch 1 부터 적대 손실이 생성자에 걸립니다. "
                  "--set loss.metric_start_epoch=3 (D 를 2 epoch 먼저 학습) 또는 --init_discriminator 를 권장합니다.")
        return sd

    def _load_init_discriminator(self, path):
        if not path.is_file():
            sys.exit(f"[init] 파일이 없습니다: {path}")
        state = self.torch.load(path, map_location="cpu", weights_only=True)
        if "D" not in state:
            sys.exit(f"[init] {path} 에 판별자 가중치('D')가 없습니다. 제출용 best.pt 에는 D 가 없으니 "
                     "같은 run 의 last.pt 를 지정하세요.")
        self.D.load_state_dict(state["D"], strict=True)
        print(f"[init] 판별자 D <- {path} (epoch {state.get('epoch')})")

    def _resume(self):
        path = self.out_dir / "last.pt"
        if not path.is_file():
            print(f"[resume] {path} 가 없어 처음부터 학습합니다.")
            return
        state = self.torch.load(path, map_location=self.device, weights_only=True)
        self.G.load_state_dict(state["G"])
        self.D.load_state_dict(state["D"])
        self.ema.load_state_dict(state["ema"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.sched_g.load_state_dict(state["sched_g"])
        self.sched_d.load_state_dict(state["sched_d"])
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.start_epoch = int(state["epoch"]) + 1
        self.global_step = int(state["global_step"])
        self.best, self.best_epoch = float(state["best"]), int(state["best_epoch"])
        self.stale = int(state.get("stale", 0))
        self.history = list(state.get("history", []))
        print(f"[resume] epoch {state['epoch']} 까지 학습된 상태에서 이어갑니다 "
              f"(best {self.best:.4f} @ {self.best_epoch}).")

    def _save_last(self, epoch):
        from pesqmax.config import to_plain
        from pesqmax.utils import atomic_save
        atomic_save({
            "G": self.G.state_dict(), "D": self.D.state_dict(), "ema": self.ema.state_dict(),
            "opt_g": self.opt_g.state_dict(), "opt_d": self.opt_d.state_dict(),
            "sched_g": self.sched_g.state_dict(), "sched_d": self.sched_d.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else {},
            "epoch": epoch, "global_step": self.global_step, "best": self.best,
            "best_epoch": self.best_epoch, "stale": self.stale,
            "history": to_plain(self.history), "config": to_plain(self.cfg)}, self.out_dir / "last.pt")

    def _export(self, path, epoch, val):
        from pesqmax.inference_core import export_checkpoint
        from pesqmax.utils import state_dict_fp32_cpu
        postproc = {"use_post_pcs": bool(val.get("use_post_pcs", True)), "pcs_strength": 1.0,
                    "tta_shifts": 1}
        meta = {"epoch": epoch, "global_step": self.global_step,
                "val_pesq_wb": val.get("val_pesq"), "val_pesq_pre": val.get("val_pesq_pre"),
                "val_pesq_post": val.get("val_pesq_post"), "val_files": len(self.val_split),
                "trained_on_validation": bool(self.cfg.data.use_validation_for_training),
                "seed": int(self.cfg.seed), "torch": str(self.torch.__version__),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        export_checkpoint(path, self.cfg, state_dict_fp32_cpu(self.ema.shadow), postproc, meta)

    def loss_weights(self, epoch):
        """(PESQ 손실 가중치, 생성자 적대 손실 가중치, 판별자 학습 여부)."""
        L = self.cfg.loss
        start, warm = int(L.pesq_start_epoch), int(L.pesq_warmup_epochs)
        if epoch < start:
            w_pesq = 0.0
        else:
            w_pesq = float(L.pesq) * (min(1.0, (epoch - start + 1) / warm) if warm > 0 else 1.0)
        w_metric = float(L.metric) if epoch >= int(L.metric_start_epoch) else 0.0
        return w_pesq, w_metric, float(L.metric) > 0

    def _pcs_target(self, clean):
        mag_c, pha, _ = self.G.stft(clean)
        mag = mag_c.pow(1.0 / self.G.compress)
        stretched = self.target_pcs(mag, self.G.pcs_scale(clean))
        return self.G.istft_lin(stretched, pha, clean.shape[-1])

    # ------------------------------------------------------------------ D step
    def update_discriminator(self, pending):
        np, torch = self.np, self.torch
        import torch.nn.functional as F
        from pesqmax.discriminator import normalize_pesq

        ref_mag, enh_mag, noisy_mag, job = pending
        scores = np.asarray(job.get(), dtype=np.float64)
        batch = ref_mag.shape[0]
        q, valid = normalize_pesq(scores), np.isfinite(scores)
        dev = self.device
        self.opt_d.zero_grad(set_to_none=True)
        loss = F.mse_loss(self.D(ref_mag, ref_mag), torch.ones(batch, device=dev))
        ok = valid[:batch]
        if ok.any():
            idx = torch.from_numpy(np.flatnonzero(ok)).to(dev)
            target = torch.tensor(q[:batch][ok], device=dev, dtype=torch.float32)
            loss = loss + F.mse_loss(self.D(ref_mag[idx], enh_mag[idx]), target)
        if noisy_mag is not None:
            ok_n = valid[batch:]
            if ok_n.any():
                idx_n = torch.from_numpy(np.flatnonzero(ok_n)).to(dev)
                target_n = torch.tensor(q[batch:][ok_n], device=dev, dtype=torch.float32)
                loss = loss + F.mse_loss(self.D(ref_mag[idx_n], noisy_mag[idx_n]), target_n)
        sample = self.replay.sample(int(self.cfg.disc.replay_k), dev, self.rng)
        if sample is not None:
            r_ref, r_est, r_q = sample
            loss = loss + F.mse_loss(self.D(r_ref, r_est), r_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), float(self.cfg.train.clip_grad))
        self.opt_d.step()
        if ok.any():
            self.replay.add(ref_mag[idx], enh_mag[idx], q[:batch][ok])
        return float(loss.detach()), (float(np.nanmean(scores[:batch])) if ok.any() else math.nan)

    # ------------------------------------------------------------------ G losses
    def generator_terms(self, out, targets, clean, ref_mag, enh_mag, pre_com_hat, w_pesq, w_metric):
        torch = self.torch
        import torch.nn.functional as F
        from pesqmax.losses import phase_losses

        L = self.cfg.loss
        t_mag, t_pha, t_com, clean_tgt = targets
        terms, weights = {}, {}
        terms["mag"], weights["mag"] = F.mse_loss(out["mag_c"], t_mag), float(L.mag)
        ip, gd, iaf = phase_losses(t_pha, out["pha"])
        terms["phase"], weights["phase"] = ip + gd + iaf, float(L.phase)
        terms["complex"], weights["complex"] = F.mse_loss(out["com_c"], t_com) * 2, float(L.complex)
        terms["consistency"] = F.mse_loss(out["com_c"], pre_com_hat) * 2
        weights["consistency"] = float(L.consistency)
        terms["time"], weights["time"] = F.l1_loss(out["wav_pre"], clean_tgt), float(L.time)
        if float(L.mrstft) > 0:
            terms["mrstft"], weights["mrstft"] = self.mrstft(out["wav_pre"], clean_tgt), float(L.mrstft)
        if w_pesq > 0:
            terms["pesq"], weights["pesq"] = self.pesq_loss(clean, out["wav"]).mean(), w_pesq
        if w_metric > 0:
            ones = torch.ones(enh_mag.shape[0], device=enh_mag.device)
            terms["metric"], weights["metric"] = F.mse_loss(self.D(ref_mag, enh_mag), ones), w_metric
        tv = self.G.pcs_smoothness()
        if tv is not None and float(self.cfg.pcs.tv_weight) > 0:
            terms["pcs_tv"], weights["pcs_tv"] = tv, float(self.cfg.pcs.tv_weight)
        return terms, weights

    def _diagnose(self, terms, weights):
        torch = self.torch
        params = [p for p in self.G.parameters() if p.requires_grad]
        print("\n[진단] 손실 항별 생성자 기울기 L2 norm (가중치 적용 후):")
        for key, value in terms.items():
            grads = torch.autograd.grad(weights[key] * value, params, retain_graph=True,
                                        allow_unused=True)
            norm = math.sqrt(sum(float(g.float().pow(2).sum()) for g in grads if g is not None))
            print(f"   {key:12s} value={float(value.detach()):9.4f}  weight={weights[key]:6.3f}  grad={norm:10.4f}")
        print()

    # ------------------------------------------------------------------ epoch
    def train_epoch(self, epoch):
        np, torch = self.np, self.torch
        from tqdm.auto import tqdm

        cfg, G = self.cfg, self.G
        G.train()
        self.D.train()
        w_pesq, w_metric, train_d = self.loss_weights(epoch)
        sums, counts = defaultdict(float), defaultdict(int)
        max_steps = int(cfg.train.max_steps_per_epoch) or len(self.loader)
        total_steps = min(len(self.loader), max_steps)
        pending, skipped = None, 0
        bar = tqdm(self.loader, total=total_steps, desc=f"epoch {epoch}", dynamic_ncols=True,
                   mininterval=2.0, file=sys.stdout)
        for step, (noisy, clean) in enumerate(bar):
            if step >= total_steps:
                break
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)
            factor = torch.sqrt(noisy.shape[1] / noisy.pow(2).sum(1, keepdim=True).clamp_min(1e-12))
            noisy, clean = noisy * factor, clean * factor

            with torch.no_grad():
                clean_tgt = self._pcs_target(clean) if self.target_pcs is not None else clean
                t_mag, t_pha, t_com = G.stft(clean_tgt)
                ref_mag = G.stft(clean)[0] if self.target_pcs is not None else t_mag
                use_noisy = train_d and bool(cfg.disc.use_noisy)
                noisy_mag = G.stft(noisy)[0] if use_noisy else None

            with torch.autocast(device_type=self.device.type, dtype=self.amp or torch.float32,
                                enabled=self.amp is not None):
                out = G(noisy)
            enh_mag, _, enh_com = G.stft(out["wav"])
            pre_com_hat = G.stft(out["wav_pre"])[2] if G.post_pcs is not None else enh_com
            finite = bool(torch.isfinite(out["wav"]).all()) and bool(torch.isfinite(out["wav_pre"]).all())

            # 3) 이번 배치의 실제 PESQ 를 비동기로 요청 (출력이 유한할 때만)
            job = None
            if train_d and finite:
                c_np = clean.detach().cpu().numpy()
                refs, degs = list(c_np), list(out["wav"].detach().float().cpu().numpy())
                if use_noisy:
                    refs += list(c_np)
                    degs += list(noisy.detach().cpu().numpy())
                job = self.pool.map_async(refs, degs)
            # 4) 직전 배치로 판별자 갱신
            if pending is not None:
                d_loss, tr_pesq = self.update_discriminator(pending)
                sums["d_loss"] += d_loss
                counts["d_loss"] += 1
                if math.isfinite(tr_pesq):
                    sums["train_real_pesq"] += tr_pesq
                    counts["train_real_pesq"] += 1
            pending = (ref_mag.detach(), enh_mag.detach(),
                       noisy_mag.detach() if noisy_mag is not None else None, job) if job else None
            if not finite:  # 수치 오류 배치는 건너뜀 (판별자에도 넣지 않음)
                skipped += 1
                self.opt_g.zero_grad(set_to_none=True)
                continue

            # 5) 생성자 갱신 (판별자 파라미터는 고정)
            for p in self.D.parameters():
                p.requires_grad_(False)
            terms, weights = self.generator_terms(out, (t_mag, t_pha, t_com, clean_tgt), clean,
                                                  ref_mag, enh_mag, pre_com_hat, w_pesq, w_metric)
            if cfg.train.diagnose_grad_norms and step == 0 and epoch == self.start_epoch:
                self._diagnose(terms, weights)
            total = sum(weights[k] * v for k, v in terms.items())
            for p in self.D.parameters():
                p.requires_grad_(True)
            if not torch.isfinite(total):
                skipped += 1
                self.opt_g.zero_grad(set_to_none=True)
                continue
            self.opt_g.zero_grad(set_to_none=True)
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.opt_g)
            grad_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), float(cfg.train.clip_grad))
            self.scaler.step(self.opt_g)
            self.scaler.update()
            self.ema.update(G)
            self.global_step += 1

            sums["loss"] += float(total.detach())
            counts["loss"] += 1
            if math.isfinite(float(grad_norm)):  # fp16 overflow step 은 GradScaler 가 건너뜀
                sums["grad_norm"] += float(grad_norm)
                counts["grad_norm"] += 1
            for key, value in terms.items():
                sums[key] += float(value.detach())
                counts[key] += 1
            if step % max(1, int(cfg.train.log_every)) == 0:
                post = {"loss": f"{sums['loss'] / counts['loss']:.3f}"}
                if counts["pesq"]:
                    post["pesqL"] = f"{sums['pesq'] / counts['pesq']:.3f}"
                if counts["train_real_pesq"]:
                    post["trPESQ"] = f"{sums['train_real_pesq'] / counts['train_real_pesq']:.3f}"
                if counts["d_loss"]:
                    post["D"] = f"{sums['d_loss'] / counts['d_loss']:.3f}"
                bar.set_postfix(post)
        bar.close()
        if pending is not None:
            d_loss, tr_pesq = self.update_discriminator(pending)
            sums["d_loss"] += d_loss
            counts["d_loss"] += 1
            if math.isfinite(tr_pesq):
                sums["train_real_pesq"] += tr_pesq
                counts["train_real_pesq"] += 1
        result = {f"train_{k}" if not k.startswith("train_") else k: sums[k] / counts[k]
                  for k in sums if counts[k]}
        result.update({"w_pesq": w_pesq, "w_metric": w_metric, "skipped_steps": skipped})
        return result

    # ------------------------------------------------------------------ validation
    def validate(self, model):
        np, torch = self.np, self.torch
        model.eval()
        val = self.val_split
        posts, pres = [], []
        bs = int(self.cfg.train.val_batch_size)
        with torch.no_grad():
            for i in range(0, len(val), bs):
                x = torch.from_numpy(val.noisy[i:i + bs]).to(self.device)
                if model.post_pcs is not None:
                    post, pre = model.enhance_both(x)
                    posts.append(post.float().cpu().numpy())
                else:
                    pre = model.enhance(x, use_post_pcs=False)
                pres.append(pre.float().cpu().numpy())
        refs = list(val.clean)
        res = {}
        s_pre = self.pool.map(refs, list(np.concatenate(pres)))
        res["val_pesq_pre"] = float(np.nanmean(s_pre))
        errors = int(np.sum(~np.isfinite(s_pre)))
        if posts:
            s_post = self.pool.map(refs, list(np.concatenate(posts)))
            res["val_pesq_post"] = float(np.nanmean(s_post))
            errors += int(np.sum(~np.isfinite(s_post)))
        use_post = bool(posts) and res["val_pesq_post"] >= res["val_pesq_pre"]
        res["use_post_pcs"] = use_post
        res["val_pesq"] = res["val_pesq_post"] if use_post else res["val_pesq_pre"]
        res["val_pesq_errors"] = errors
        for name, module in (("pre", model.pre_pcs), ("post", model.post_pcs)):
            if module is not None:
                gamma = module.gamma().detach().cpu().numpy()
                res[f"{name}_gamma_mean"] = float(gamma.mean())
                res[f"{name}_gamma_max"] = float(gamma.max())
        return res

    # ------------------------------------------------------------------ main loop
    def _write_history(self):
        keys = []
        for row in self.history:
            keys += [k for k in row if k not in keys]
        with open(self.out_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    def _write_summary(self, status):
        summary = {"status": status, "best_val_pesq_wb": self.best, "best_epoch": self.best_epoch,
                   "epochs_completed": self.history[-1]["epoch"] if self.history else 0,
                   "global_step": self.global_step, "generator_params": self.n_params,
                   "device": str(self.device), "out_dir": str(self.cfg.train.out_dir),
                   "trained_on_validation": bool(self.cfg.data.use_validation_for_training)}
        if self.history:
            summary["last"] = self.history[-1]
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    def fit(self):
        cfg = self.cfg
        epochs = int(cfg.train.epochs)
        use_val_train = bool(cfg.data.use_validation_for_training)
        status = "complete"
        if self.init_loaded and self.start_epoch == 1 and not self.args.no_baseline_val:
            # 시작 가중치 점수를 기준선으로: 이를 넘는 epoch 만 best.pt 로 저장 (TTA 없이 학습 중 검증과 같은 조건)
            t1 = time.time()
            val = self.validate(self.ema.shadow)
            if not use_val_train:
                self.best, self.best_epoch = val["val_pesq"], 0
            self.history.append({"epoch": 0, **val, "time_val_s": round(time.time() - t1, 1)})
            self._write_history()
            print(f"[init] 기준 검증 PESQ-WB {val['val_pesq']:.4f} (post-PCS "
                  f"{val.get('val_pesq_post', float('nan')):.4f} / pre-PCS {val['val_pesq_pre']:.4f}). "
                  "이보다 높아진 epoch 만 best.pt 로 저장합니다.", flush=True)
        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()
            row = {"epoch": epoch, "lr": self.opt_g.param_groups[0]["lr"]}
            row.update(self.train_epoch(epoch))
            self.sched_g.step()
            self.sched_d.step()
            row["time_train_s"] = round(time.time() - t0, 1)
            if epoch % max(1, int(cfg.train.val_every)) == 0 or epoch == epochs:
                t1 = time.time()
                val = self.validate(self.ema.shadow)
                row.update(val)
                row["time_val_s"] = round(time.time() - t1, 1)
                improved = val["val_pesq"] > self.best
                if improved:
                    self.best, self.best_epoch, self.stale = val["val_pesq"], epoch, 0
                else:
                    self.stale += 1
                if use_val_train or improved:
                    self._export(self.out_dir / "best.pt", epoch, val)
                if int(cfg.train.save_every) > 0 and epoch % int(cfg.train.save_every) == 0:
                    self._export(self.out_dir / f"epoch_{epoch:04d}.pt", epoch, val)
                post = val.get("val_pesq_post")
                post_txt = f"post-PCS {post:.4f} | " if post is not None else ""
                print(f"[epoch {epoch}] val PESQ-WB: {post_txt}pre-PCS {val['val_pesq_pre']:.4f}"
                      f" | best {self.best:.4f} @ {self.best_epoch}"
                      f" | train {row['time_train_s']}s, val {row['time_val_s']}s", flush=True)
            self.history.append(row)
            self._write_history()
            self._save_last(epoch)
            self._write_summary("running")
            patience = int(cfg.train.early_stop_patience)
            if patience > 0 and self.stale >= patience and not use_val_train:
                print(f"조기 종료: {patience}회 연속 검증 PESQ 개선 없음.")
                status = "early_stopped"
                break
        self._write_summary(status)
        if self.init_loaded and self.best_epoch == 0:
            print(f"학습 종료. 기준선 {self.best:.4f} 을 넘은 epoch 가 없어 best.pt 를 저장하지 않았습니다 "
                  "(시작 가중치를 그대로 쓰세요).")
            return
        print(f"학습 종료. best val PESQ-WB {self.best:.4f} (epoch {self.best_epoch}) -> "
              f"{self.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
