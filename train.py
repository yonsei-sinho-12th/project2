"""MP-SENet 학습 스크립트 (이어 학습 지원).

예)  python train.py --data-root /content/drive/MyDrive/deep_learning_noise_reduction_challenge \
                    --out-dir /content/drive/MyDrive/se_runs/mpsenet_full

같은 --out-dir로 다시 실행하면 last.pt에서 자동으로 이어서 학습합니다.
(Colab 런타임이 끊겨도 Drive에 저장된 지점부터 계속됩니다.)
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import PairDataset, TrainDataset, load_split
from metrics import PesqPool
from model import (PRESETS, MetricDiscriminator, ModelConfig, MPSENet, count_parameters,
                   mag_pha_istft, mag_pha_stft, phase_losses)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="train/, validation/ 폴더가 있는 배포 폴더")
    p.add_argument("--cache-dir", default=None, help="파형 캐시 폴더 (기본: 데이터 폴더/.waveform_cache)")
    p.add_argument("--out-dir", default="checkpoints", help="가중치·학습 기록 저장 폴더")
    p.add_argument("--preset", default="full", choices=sorted(PRESETS))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--segment-seconds", type=float, default=2.0, help="학습 구간 길이 (검증은 전체 길이)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--patience", type=int, default=40, help="검증 PESQ 개선이 없으면 종료할 epoch 수")
    p.add_argument("--min-epochs", type=int, default=40)
    # 증강
    p.add_argument("--dm-prob", type=float, default=0.6, help="동적 혼합 비율 (0이면 원래 쌍만 사용)")
    p.add_argument("--snr-jitter", type=float, default=3.0)
    # 손실 가중치 (MP-SENet 확장판 기본값)
    p.add_argument("--w-metric", type=float, default=0.05)
    p.add_argument("--w-mag", type=float, default=0.9)
    p.add_argument("--w-pha", type=float, default=0.3)
    p.add_argument("--w-com", type=float, default=0.1)
    p.add_argument("--w-time", type=float, default=0.2)
    p.add_argument("--w-consistency", type=float, default=0.1)
    p.add_argument("--no-metric-disc", action="store_true", help="PESQ 판별기(MetricGAN) 끄기")
    # 실행 환경
    p.add_argument("--amp", default="auto", choices=["auto", "off", "fp16", "bf16"])
    p.add_argument("--grad-ckpt", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--workers", type=int, default=0, help="DataLoader 워커 수")
    p.add_argument("--pesq-workers", type=int, default=0, help="PESQ 프로세스 수 (0: CPU 코어 수)")
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--val-every", type=int, default=1, help="검증 간격 (epoch). 마지막 epoch은 항상 검증")
    p.add_argument("--val-precision", default="fp32", choices=["fp32", "amp"],
                   help="amp: 학습과 같은 bf16/fp16으로 빠르게 검증 (최종 점수는 evaluate.py로 fp32 확인)")
    p.add_argument("--val-raw", action="store_true", help="EMA 외에 일반 가중치도 검증 (시간 2배)")
    p.add_argument("--max-hours", type=float, default=0, help="이 시간이 지나면 저장 후 종료 (0: 제한 없음)")
    p.add_argument("--max-steps-per-epoch", type=int, default=0, help="디버그용")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restart", action="store_true", help="last.pt가 있어도 처음부터 학습")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


# ─────────────────────────── 보조 함수 ───────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_amp(choice, device):
    if device.type != "cuda" or choice == "off":
        return None
    if choice == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[choice]


class EMA:
    """생성기 가중치의 지수 이동 평균. 검증·제출에 주로 이 가중치를 씁니다."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.steps = 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        ema_params = [p for p in self.model.parameters()]
        params = [p.detach() for p in model.parameters()]
        torch._foreach_mul_(ema_params, d)
        torch._foreach_add_(ema_params, params, alpha=1 - d)
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)

    def state_dict(self):
        return {"model": self.model.state_dict(), "steps": self.steps}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.steps = state["steps"]


def lr_lambda_factory(total_steps, warmup_steps, base_lr, min_lr):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return (min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))) / base_lr
    return fn


def normalize_pair(noisy, clean):
    norm = torch.sqrt(noisy.shape[-1] / noisy.pow(2).sum(-1, keepdim=True).clamp_min(1e-10))
    return noisy * norm, clean * norm


def generator_losses(cfg, clean, clean_mag, clean_pha, clean_com, mag_g, pha_g, com_g, audio_g, com_g_hat):
    return {
        "mag": F.mse_loss(mag_g, clean_mag),
        "pha": phase_losses(clean_pha, pha_g),
        "com": F.mse_loss(com_g, clean_com) * 2,
        "time": F.l1_loss(audio_g, clean),
        "consistency": F.mse_loss(com_g, com_g_hat) * 2,
    }


def probe_memory(generator, cfg, args, device, amp_dtype):
    """첫 배치 크기로 한 번 학습해 보고, GPU 메모리가 부족하면 gradient checkpointing을 켭니다."""
    if device.type != "cuda":
        generator.gradient_checkpointing = args.grad_ckpt == "on"
        return
    if args.grad_ckpt != "auto":
        generator.gradient_checkpointing = args.grad_ckpt == "on"
        return
    segment = int(args.segment_seconds * cfg.sample_rate)
    saved_state = copy.deepcopy(generator.state_dict())   # 점검용 가짜 배치가 BatchNorm 통계를 바꾸지 않게
    try:
        _probe(generator, cfg, args, device, amp_dtype, segment)
    finally:
        generator.load_state_dict(saved_state)
        generator.zero_grad(set_to_none=True)


def _probe(generator, cfg, args, device, amp_dtype, segment):
    for use_ckpt in (False, True):
        generator.gradient_checkpointing = use_ckpt
        try:
            x = torch.randn(args.batch_size, segment, device=device) * 0.1
            mag, pha, _ = mag_pha_stft(x, cfg)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                mag_g, pha_g, com_g = generator(mag, pha)
            audio = mag_pha_istft(mag_g, pha_g, cfg, segment)
            (mag_g.mean() + pha_g.mean() + audio.abs().mean()).backward()
            generator.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            print(f"메모리 점검: gradient checkpointing {'켬' if use_ckpt else '끔'} → "
                  f"최대 {peak:.1f} GiB / {torch.cuda.get_device_properties(device).total_memory / 2**30:.1f} GiB")
            # 여유가 너무 적으면(85% 이상) checkpointing을 켭니다.
            if use_ckpt or peak < 0.85 * torch.cuda.get_device_properties(device).total_memory / 2 ** 30:
                return
        except torch.cuda.OutOfMemoryError:
            print(f"메모리 점검: checkpointing {'켬' if use_ckpt else '끔'} 상태에서 GPU 메모리 부족")
        generator.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    raise RuntimeError("GPU 메모리가 부족합니다. --batch-size 또는 --segment-seconds를 줄이세요.")


@torch.no_grad()
def validate(model, waves, cfg, device, batch_size, pesq_pool, amp_dtype):
    model.eval()
    loader = DataLoader(PairDataset(waves), batch_size=batch_size, shuffle=False)
    refs, outs, handles = [], [], []
    for noisy, clean in loader:
        noisy = noisy.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            enhanced = model.enhance(noisy)
        enhanced = enhanced.float().cpu().numpy()
        handles += pesq_pool.submit(list(clean.numpy()), list(enhanced))
    scores = PesqPool.gather(handles)
    if np.isnan(scores).any():
        print(f"  경고: 검증 {np.isnan(scores).sum()}개 파일의 PESQ를 계산하지 못했습니다.")
    return float(np.nanmean(scores))


def save_atomic(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# ─────────────────────────── 메인 ───────────────────────────
def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    # PESQ 프로세스 풀은 CUDA 초기화 전에 만듭니다.
    pesq_pool = PesqPool(args.pesq_workers or None)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = pick_amp(args.amp, device)

    train_manifest, train_waves = load_split(args.data_root, "train", args.cache_dir)
    val_manifest, val_waves = load_split(args.data_root, "validation", args.cache_dir)

    cfg = ModelConfig(**PRESETS[args.preset])
    segment = int(round(args.segment_seconds * cfg.sample_rate))
    train_set = TrainDataset(train_waves, segment, dm_prob=args.dm_prob, snr_jitter=args.snr_jitter,
                             seed=args.seed)

    generator = MPSENet(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)
    use_metric = not args.no_metric_disc
    print(f"장치 {device} | AMP {amp_dtype} | 프리셋 {args.preset} | 생성기 {count_parameters(generator):,} "
          f"파라미터 | 판별기 {'사용' if use_metric else '끔'} | PESQ 프로세스 {pesq_pool.workers}")
    print(f"학습 {len(train_set):,}쌍 (구간 {args.segment_seconds:g}s, 동적 혼합 {args.dm_prob:.0%}) | "
          f"검증 {len(val_waves):,}쌍 (전체 길이)")

    probe_memory(generator, cfg, args, device, amp_dtype)

    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.8, 0.99),
                              weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.8, 0.99),
                              weight_decay=args.weight_decay)
    steps_per_epoch = len(train_set) // args.batch_size
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    lr_fn = lr_lambda_factory(args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch),
                              args.lr, args.min_lr)
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_fn)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_fn)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    ema = EMA(generator, args.ema_decay)

    # ── 이어 학습 ──
    last_path, best_path = out_dir / "last.pt", out_dir / "best_generator.pt"
    start_epoch, best_pesq, best_epoch, stale, history = 1, -float("inf"), 0, 0, []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if last_path.is_file() and not args.restart:
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["model_config"] != cfg.to_dict():
            raise ValueError(f"{last_path}의 모델 설정이 다릅니다. 다른 --out-dir을 쓰거나 --restart를 주세요.")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        ema.load_state_dict(state["ema"])
        opt_g.load_state_dict(state["opt_g"]); opt_d.load_state_dict(state["opt_d"])
        sched_g.load_state_dict(state["sched_g"]); sched_d.load_state_dict(state["sched_d"])
        scaler.load_state_dict(state["scaler"])
        generator.gradient_checkpointing = state.get("grad_ckpt", generator.gradient_checkpointing)
        start_epoch = state["epoch"] + 1
        best_pesq, best_epoch, stale = state["best_pesq"], state["best_epoch"], state["stale"]
        history, run_id = state["history"], state["run_id"]
        print(f"이어 학습: epoch {start_epoch}부터 (현재 최고 PESQ-WB {best_pesq:.4f} @ epoch {best_epoch})")

    started = time.time()
    stop_reason = "max_epochs"
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        generator.train(); discriminator.train()
        train_set.set_epoch(epoch)
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.workers,
                            generator=torch.Generator().manual_seed(args.seed * 1000 + epoch))
        sums, count, skipped, epoch_start = {}, 0, 0, time.time()
        for step, (noisy, clean) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            noisy, clean = normalize_pair(noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True))
            with torch.no_grad():
                clean_mag, clean_pha, clean_com = mag_pha_stft(clean, cfg)
                noisy_mag, noisy_pha, _ = mag_pha_stft(noisy, cfg)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
            audio_g = mag_pha_istft(mag_g, pha_g, cfg, segment)
            mag_g_hat, _, com_g_hat = mag_pha_stft(audio_g, cfg)

            # PESQ는 CPU에서 비동기로 계산하고, 그동안 GPU는 생성기를 갱신합니다.
            if use_metric:
                handles = pesq_pool.submit(list(clean.cpu().numpy()), list(audio_g.detach().float().cpu().numpy()))

            # ── 생성기 갱신 ──
            losses = generator_losses(cfg, clean, clean_mag, clean_pha, clean_com, mag_g, pha_g, com_g,
                                      audio_g, com_g_hat)
            total = (args.w_mag * losses["mag"] + args.w_pha * losses["pha"] + args.w_com * losses["com"]
                     + args.w_time * losses["time"] + args.w_consistency * losses["consistency"])
            if use_metric:
                discriminator.requires_grad_(False)
                metric_g = discriminator(clean_mag, mag_g_hat)
                losses["metric"] = F.mse_loss(metric_g, torch.ones_like(metric_g))
                total = total + args.w_metric * losses["metric"]
                discriminator.requires_grad_(True)

            opt_g.zero_grad(set_to_none=True)
            if not torch.isfinite(total):
                skipped += 1
                if use_metric:
                    PesqPool.gather(handles)
                continue
            scaler.scale(total).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
            scaler.step(opt_g)
            scaler.update()
            sched_g.step()
            ema.update(generator)

            # ── 판별기 갱신: D(clean, clean)→1, D(clean, 복원)→실제 PESQ ──
            if use_metric:
                scores = PesqPool.gather(handles)
                valid = torch.from_numpy(~np.isnan(scores)).to(device)
                target = torch.from_numpy(np.nan_to_num((scores - 1.0) / 3.5)).float().to(device)
                d_real = discriminator(clean_mag, clean_mag)
                loss_d = F.mse_loss(d_real, torch.ones_like(d_real))
                if valid.any():
                    d_fake = discriminator(clean_mag, mag_g_hat.detach())
                    loss_d = loss_d + F.mse_loss(d_fake[valid], target[valid])
                    losses["train_pesq"] = torch.tensor(float(np.nanmean(scores)))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                losses["disc"] = loss_d.detach()
            sched_d.step()

            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
            count += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} | total {float(total.detach()):.4f} | "
                      f"mag {float(losses['mag'].detach()):.4f} | pha {float(losses['pha'].detach()):.3f}"
                      + (f" | 학습 PESQ {float(losses['train_pesq']):.3f}" if "train_pesq" in losses else ""),
                      flush=True)

        train_time = time.time() - epoch_start
        # 검증 방식은 --val-every / --val-precision / --val-raw로 조절합니다.
        nan = float("nan")
        do_val = epoch % args.val_every == 0 or epoch == args.epochs
        val_dtype = amp_dtype if args.val_precision == "amp" else None
        val_raw = val_ema = nan
        if do_val:
            val_ema = validate(ema.model, val_waves, cfg, device, args.val_batch_size, pesq_pool, val_dtype)
            if args.val_raw:
                val_raw = validate(generator, val_waves, cfg, device, args.val_batch_size, pesq_pool, val_dtype)
        if not math.isnan(val_raw) and val_raw > val_ema:
            val_pesq, which = val_raw, "raw"
        else:
            val_pesq, which = val_ema, "ema"

        improved = do_val and val_pesq > best_pesq
        if improved:
            stale = 0 if val_pesq > best_pesq + 1e-3 else stale + 1
            best_pesq, best_epoch = val_pesq, epoch
            weights = ema.model.state_dict() if which == "ema" else generator.state_dict()
            save_atomic({"generator": weights, "model_config": cfg.to_dict(), "epoch": epoch,
                         "weights": which, "val_pesq_wb": val_pesq, "train_args": vars(args),
                         "run_id": run_id}, best_path)
        else:
            stale += 1

        row = {"epoch": epoch, **{f"train_{k}": v / max(count, 1) for k, v in sums.items()},
               "val_pesq_wb_raw": val_raw, "val_pesq_wb_ema": val_ema, "best_pesq_wb": best_pesq,
               "best_epoch": best_epoch, "lr": opt_g.param_groups[0]["lr"], "skipped_steps": skipped,
               "epoch_seconds": time.time() - epoch_start}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        print(f"Epoch {epoch:3d}/{args.epochs} | 학습 {train_time / 60:.1f}분 | 검증 PESQ-WB raw {val_raw:.4f} "
              f"ema {val_ema:.4f} | 최고 {best_pesq:.4f} (epoch {best_epoch}){' ★ 저장' if improved else ''} | "
              f"lr {row['lr']:.2e}", flush=True)

        save_atomic({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                     "ema": ema.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                     "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "best_pesq": best_pesq,
                     "best_epoch": best_epoch, "stale": stale, "history": history, "run_id": run_id,
                     "model_config": cfg.to_dict(), "grad_ckpt": generator.gradient_checkpointing},
                    last_path)

        if epoch >= args.min_epochs and stale >= args.patience:
            stop_reason = "early_stopping"
            break
        if args.max_hours and (time.time() - started) / 3600 > args.max_hours:
            stop_reason = "time_limit (같은 명령으로 다시 실행하면 이어서 학습합니다)"
            break

    pesq_pool.close()
    summary = {"run_id": run_id, "stop_reason": stop_reason, "last_epoch": epoch,
               "best_epoch": best_epoch, "best_validation_pesq_wb": best_pesq,
               "model_config": cfg.to_dict(), "train_args": vars(args),
               "torch_version": torch.__version__, "device": str(device),
               "python": sys.version.split()[0]}
    (out_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(f"학습 종료 ({stop_reason}) | 최고 검증 PESQ-WB {best_pesq:.4f} @ epoch {best_epoch} → {best_path}")


if __name__ == "__main__":
    main()
