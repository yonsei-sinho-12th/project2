"""Non-causal MP-SENet (TS-Conformer, 크기·위상 병렬 복원) + 학습 가능한 PCS 전/후처리.

참고: Lu, Ai, Ling, "MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude
and Phase Spectra", Interspeech 2023. 원 공개 구현의 nn.MultiheadAttention(batch_first=False)
축 문제를 바로잡아, 시간/주파수 경로 모두 실제 시퀀스 축에서 양방향 전체 어텐션을 수행합니다.
모든 합성곱은 대칭 패딩(non-causal)이며 미래 프레임을 그대로 참조합니다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dsp import istft_from_compressed, istft_from_linear, stft_mag_pha
from .pcs import LearnablePCS


# --------------------------------------------------------------------------- Conformer
class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * mult), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim * mult, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    """양방향(마스크 없음) 멀티헤드 셀프 어텐션. 입력 [N, L, C]."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("dense_channel 은 n_heads 로 나누어떨어져야 합니다.")
        self.heads, self.dropout = heads, dropout
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        n, length, dim = x.shape
        qkv = self.qkv(self.norm(x)).view(n, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(n, length, dim)
        return self.drop(self.proj(out))


class ConvModule(nn.Module):
    def __init__(self, dim: int, expansion: int, kernel: int, dropout: float):
        super().__init__()
        inner = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, inner * 2, 1)
        self.depthwise = nn.Conv1d(inner, inner, kernel, padding=kernel // 2, groups=inner)
        self.bn = nn.BatchNorm1d(inner)
        self.pointwise_out = nn.Conv1d(inner, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [N, L, C]
        y = self.norm(x).transpose(1, 2)
        y = F.glu(self.pointwise_in(y), dim=1)
        y = F.silu(self.bn(self.depthwise(y)))
        y = self.pointwise_out(y).transpose(1, 2)
        return self.drop(y)


class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, ffn_mult, conv_kernel, conv_expansion, dropout):
        super().__init__()
        self.ff1 = FeedForward(dim, ffn_mult, dropout)
        self.attn = SelfAttention(dim, heads, dropout)
        self.conv = ConvModule(dim, conv_expansion, conv_kernel, dropout)
        self.ff2 = FeedForward(dim, ffn_mult, dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.post_norm(x)


class TSConformerBlock(nn.Module):
    """Time-Frequency Dual-Path: 시간 축 Conformer -> 주파수 축 Conformer."""

    def __init__(self, dim, heads, ffn_mult, conv_kernel, conv_expansion, dropout):
        super().__init__()
        args = (dim, heads, ffn_mult, conv_kernel, conv_expansion, dropout)
        self.time_conformer = ConformerBlock(*args)
        self.freq_conformer = ConformerBlock(*args)

    def forward(self, x):  # [B, C, T, F]
        b, c, t, f = x.shape
        y = x.permute(0, 3, 2, 1).reshape(b * f, t, c)
        y = self.time_conformer(y) + y
        y = y.view(b, f, t, c).permute(0, 2, 1, 3).reshape(b * t, f, c)
        y = self.freq_conformer(y) + y
        return y.view(b, t, f, c).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------- Dense blocks
class DenseBlock(nn.Module):
    def __init__(self, ch: int, depth: int = 4, kernel=(3, 3)):
        super().__init__()
        layers = []
        for i in range(depth):
            dil = 2 ** i
            pad = (dil * (kernel[0] - 1) // 2, (kernel[1] - 1) // 2)
            layers.append(nn.Sequential(
                nn.Conv2d(ch * (i + 1), ch, kernel, dilation=(dil, 1), padding=pad),
                nn.InstanceNorm2d(ch, affine=True), nn.PReLU(ch)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skip = x
        for layer in self.layers:
            x = layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, in_ch: int, ch: int, depth: int):
        super().__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(in_ch, ch, 1), nn.InstanceNorm2d(ch, affine=True),
                                     nn.PReLU(ch))
        self.dense = DenseBlock(ch, depth)
        self.conv_down = nn.Sequential(nn.Conv2d(ch, ch, (1, 3), (1, 2)),
                                       nn.InstanceNorm2d(ch, affine=True), nn.PReLU(ch))

    def forward(self, x):
        return self.conv_down(self.dense(self.conv_in(x)))


class LearnableSigmoid2d(nn.Module):
    def __init__(self, n_freq: int, beta: float):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(n_freq, 1))

    def forward(self, x):  # [B, F, T]
        return self.beta * torch.sigmoid(self.slope * x)


class MaskDecoder(nn.Module):
    def __init__(self, ch: int, n_freq: int, depth: int, beta: float):
        super().__init__()
        self.dense = DenseBlock(ch, depth)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, (1, 3), (1, 2)), nn.Conv2d(ch, 1, 1),
            nn.InstanceNorm2d(1, affine=True), nn.PReLU(1), nn.Conv2d(1, 1, 1))
        self.lsigmoid = LearnableSigmoid2d(n_freq, beta)

    def forward(self, x):
        x = self.conv(self.dense(x))  # [B, 1, T, F]
        x = x.squeeze(1).transpose(1, 2).float()  # [B, F, T]
        return self.lsigmoid(x)


class PhaseDecoder(nn.Module):
    def __init__(self, ch: int, depth: int):
        super().__init__()
        self.dense = DenseBlock(ch, depth)
        self.conv = nn.Sequential(nn.ConvTranspose2d(ch, ch, (1, 3), (1, 2)),
                                  nn.InstanceNorm2d(ch, affine=True), nn.PReLU(ch))
        self.conv_r = nn.Conv2d(ch, 1, 1)
        self.conv_i = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        x = self.conv(self.dense(x))
        real = self.conv_r(x).float()
        imag = self.conv_i(x).float()
        phase = torch.atan2(imag + 1e-10, real + 1e-10)  # [B, 1, T, F]
        return phase.squeeze(1).transpose(1, 2)


class MPSENet(nn.Module):
    def __init__(self, n_freq: int, dense_channel=64, num_tsblocks=4, n_heads=4, ffn_mult=4,
                 conv_kernel=31, conv_expansion=2, dropout=0.1, dense_depth=4, mask_beta=2.0):
        super().__init__()
        if n_freq % 2 != 1:
            raise ValueError("n_fft//2+1 이 홀수가 되도록 n_fft 를 4의 배수로 설정하세요.")
        self.encoder = DenseEncoder(2, dense_channel, dense_depth)
        self.blocks = nn.ModuleList([
            TSConformerBlock(dense_channel, n_heads, ffn_mult, conv_kernel, conv_expansion, dropout)
            for _ in range(num_tsblocks)])
        self.mask_decoder = MaskDecoder(dense_channel, n_freq, dense_depth, mask_beta)
        self.phase_decoder = PhaseDecoder(dense_channel, dense_depth)

    def forward(self, mag_c, pha):  # [B, F, T] -> mask [B, F, T], phase [B, F, T]
        x = torch.stack((mag_c, pha), dim=1).transpose(2, 3)  # [B, 2, T, F]
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        return self.mask_decoder(x), self.phase_decoder(x)


# --------------------------------------------------------------------------- Wrapper
class Enhancer(nn.Module):
    """파형 -> 파형. 학습 시에는 손실 계산에 필요한 중간 스펙트럼도 함께 반환합니다.

    - pre_pcs : 네트워크 입력 특징에 적용하는 학습 가능 PCS (전처리)
    - post_pcs: 복원 스펙트럼에 적용하는 학습 가능 PCS (후처리). 복원 손실은 PCS 이전 출력에,
                PESQ 손실·판별자 손실은 PCS 이후 최종 출력에 걸리므로 gamma 는 PESQ 지표만으로 학습됩니다.
    """

    def __init__(self, cfg):
        super().__init__()
        s, m, p = cfg["stft"], cfg["model"], cfg["pcs"]
        self.n_fft, self.hop, self.win = int(s["n_fft"]), int(s["hop"]), int(s["win"])
        self.compress = float(s["compress"])
        self.sample_rate = int(cfg["data"]["sample_rate"])
        self.ref_peak = float(p["ref_peak"])
        n_freq = self.n_fft // 2 + 1
        self.register_buffer("window", torch.hann_window(self.win), persistent=False)
        self.net = MPSENet(n_freq, m["dense_channel"], m["num_tsblocks"], m["n_heads"],
                           m["ffn_mult"], m["conv_kernel"], m["conv_expansion"], m["dropout"],
                           m["dense_depth"], m["mask_beta"])
        kw = dict(n_fft=self.n_fft, sample_rate=self.sample_rate,
                  gamma_min=p["gamma_min"], gamma_max=p["gamma_max"])
        self.pre_pcs = LearnablePCS(n_freq, init=p["init"], **kw) if p["pre"] else None
        # 목표에 이미 PCS 를 적용하는 경우 후처리는 항등에서 시작해 이중 적용을 피합니다.
        post_init = "identity" if p["target"] else p["init"]
        self.post_pcs = LearnablePCS(n_freq, init=post_init, **kw) if p["post"] else None

    # ---- DSP (항상 fp32, autocast 밖에서 실행)
    def stft(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            return stft_mag_pha(wav.float(), self.n_fft, self.hop, self.win, self.compress,
                                self.window)

    def istft_c(self, mag_c, pha, length):
        with torch.autocast(device_type=mag_c.device.type, enabled=False):
            return istft_from_compressed(mag_c.float(), pha.float(), self.n_fft, self.hop,
                                         self.win, self.compress, self.window, length)

    def istft_lin(self, mag, pha, length):
        with torch.autocast(device_type=mag.device.type, enabled=False):
            return istft_from_linear(mag.float(), pha.float(), self.n_fft, self.hop, self.win,
                                     self.window, length)

    def pcs_scale(self, wav):
        return self.ref_peak / wav.detach().abs().amax(dim=1).clamp_min(1e-8)

    def analyze(self, noisy):
        """RMS 정규화 파형 [B, N] -> (복원 압축 크기 [B,F,T], 복원 위상 [B,F,T], PCS 스케일 [B])."""
        dev = noisy.device.type
        mag_c, pha, _ = self.stft(noisy)
        scale = self.pcs_scale(noisy)
        feat = mag_c
        if self.pre_pcs is not None:
            with torch.autocast(device_type=dev, enabled=False):
                lin = self.pre_pcs(mag_c.pow(1.0 / self.compress), scale)
                feat = lin.clamp_min(1e-5).pow(self.compress)
        mask, est_pha = self.net(feat, pha)
        with torch.autocast(device_type=dev, enabled=False):
            return mag_c * mask.float(), est_pha.float(), scale

    def synthesize(self, est_mag_c, est_pha, scale, length: int, use_post_pcs: bool = True,
                   pcs_strength: float = 1.0):
        """복원 스펙트럼 -> 파형. use_post_pcs 이면 학습된 후처리 PCS 를 거칩니다."""
        with torch.autocast(device_type=est_mag_c.device.type, enabled=False):
            if use_post_pcs and self.post_pcs is not None:
                mag = self.post_pcs(est_mag_c.pow(1.0 / self.compress), scale, pcs_strength)
                return self.istft_lin(mag, est_pha, length)
            return self.istft_c(est_mag_c, est_pha, length)

    def forward(self, noisy, pcs_strength: float = 1.0, need_pre_wav: bool = True):
        """noisy: RMS 정규화된 파형 [B, N]."""
        length = noisy.shape[-1]
        est_mag_c, est_pha, scale = self.analyze(noisy)
        with torch.autocast(device_type=noisy.device.type, enabled=False):
            est_com_c = torch.stack((est_mag_c * torch.cos(est_pha),
                                     est_mag_c * torch.sin(est_pha)), dim=-1)
            out = {"mag_c": est_mag_c, "pha": est_pha, "com_c": est_com_c}
            if need_pre_wav or self.post_pcs is None:
                out["wav_pre"] = self.synthesize(est_mag_c, est_pha, scale, length, False)
            if self.post_pcs is not None:
                out["wav"] = self.synthesize(est_mag_c, est_pha, scale, length, True, pcs_strength)
            else:
                out["wav"] = out["wav_pre"]
        return out

    def pcs_smoothness(self):
        terms = [m.smoothness() for m in (self.pre_pcs, self.post_pcs) if m is not None]
        return sum(terms) if terms else None

    @staticmethod
    def rms_norm_factor(wav):
        energy = wav.pow(2).sum(dim=-1, keepdim=True)
        return torch.sqrt(wav.shape[-1] / energy.clamp_min(1e-12)), energy

    @torch.no_grad()
    def enhance(self, wav, use_post_pcs: bool = True, pcs_strength: float = 1.0):
        """임의 레벨 파형 [B, N] -> 복원 파형 [B, N] (입력 레벨로 복원)."""
        factor, energy = self.rms_norm_factor(wav)
        use_post = use_post_pcs and self.post_pcs is not None
        out = self.forward(wav * factor, pcs_strength, need_pre_wav=not use_post)
        y = (out["wav"] if use_post else out["wav_pre"]) / factor
        return torch.where(energy > 1e-12, y, wav)

    @torch.no_grad()
    def enhance_both(self, wav, pcs_strength: float = 1.0):
        """검증용: (후처리 PCS 적용 출력, PCS 이전 출력)."""
        factor, energy = self.rms_norm_factor(wav)
        out = self.forward(wav * factor, pcs_strength, need_pre_wav=True)
        post = torch.where(energy > 1e-12, out["wav"] / factor, wav)
        pre = torch.where(energy > 1e-12, out["wav_pre"] / factor, wav)
        return post, pre


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
