"""MP-SENet 계열 음성 향상 모델.

크기(magnitude)와 위상(phase)을 병렬로 직접 추정하는 구조입니다.
  - Lu, Ai, Ling, "MP-SENet: A Speech Enhancement Model with Parallel Denoising of
    Magnitude and Phase Spectra", Interspeech 2023.
  - Lu et al., "Explicit Estimation of Magnitude and Phase Spectra in Parallel for
    High-Quality Speech Enhancement", Neural Networks 2025 (확장판).

구성
  noisy 파형 → STFT → (압축 크기, 위상) 2채널 입력
  → DenseEncoder → [Time-Conformer + Freq-Conformer] × N → MaskDecoder / PhaseDecoder
  → 복원 크기 = noisy 크기 × 마스크,  복원 위상 = atan2(허수, 실수)
  → iSTFT → 복원 파형

MetricDiscriminator는 (clean 크기, 복원 크기)를 보고 정규화된 PESQ를 예측하며,
생성기는 이 판별기 점수를 1(=PESQ 4.5)로 끌어올리도록 학습합니다 (MetricGAN 방식).
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────── 설정 ───────────────────────────────
@dataclass
class ModelConfig:
    sample_rate: int = 16_000
    n_fft: int = 400            # 25 ms 창
    hop: int = 100              # 6.25 ms 이동
    compress: float = 0.3       # 크기 압축 지수 |X|^0.3
    channels: int = 64          # DenseBlock / Conformer 채널 수
    num_blocks: int = 4         # TS-Conformer 블록 수
    heads: int = 4
    conv_kernel: int = 31       # Conformer 깊이별 합성곱 커널
    dropout: float = 0.2
    mask_beta: float = 2.0      # 마스크 상한 (압축 크기 영역)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**known)


PRESETS = {
    # 원 논문과 같은 크기 (약 2.0M 파라미터). GPU 권장.
    "full": dict(channels=64, num_blocks=4),
    # 학습 시간이 부족할 때 (약 0.9M 파라미터).
    "lite": dict(channels=48, num_blocks=2),
    # 코드 동작 확인용.
    "tiny": dict(channels=16, num_blocks=1, heads=2, conv_kernel=15),
}


# ─────────────────────────────── STFT 유틸 ───────────────────────────────
def mag_pha_stft(wave, cfg: ModelConfig):
    """[B, L] 파형 → 압축 크기 [B, F, T], 위상 [B, F, T], 압축 복소 [B, F, T, 2]."""
    window = torch.hann_window(cfg.n_fft, device=wave.device, dtype=torch.float32)
    spec = torch.stft(wave.float(), cfg.n_fft, hop_length=cfg.hop, win_length=cfg.n_fft,
                      window=window, center=True, pad_mode="reflect", return_complex=True)
    spec = torch.view_as_real(spec)
    mag = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    pha = torch.atan2(spec[..., 1] + 1e-10, spec[..., 0] + 1e-5)
    mag = mag.pow(cfg.compress)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_istft(mag, pha, cfg: ModelConfig, length):
    window = torch.hann_window(cfg.n_fft, device=mag.device, dtype=torch.float32)
    mag = mag.float().clamp_min(0).pow(1.0 / cfg.compress)
    spec = torch.complex(mag * torch.cos(pha.float()), mag * torch.sin(pha.float()))
    return torch.istft(spec, cfg.n_fft, hop_length=cfg.hop, win_length=cfg.n_fft,
                       window=window, center=True, length=length)


# ─────────────────────────────── 기본 블록 ───────────────────────────────
class LearnableSigmoid(nn.Module):
    """beta * sigmoid(slope * x), 주파수 빈마다 기울기를 학습합니다."""

    def __init__(self, features, beta=2.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(features))

    def forward(self, x):            # x: [..., features]
        return self.beta * torch.sigmoid(self.slope * x)


class DenseBlock(nn.Module):
    """시간축 dilation(1,2,4,8)을 쓰는 dense 연결 2D 합성곱 블록."""

    def __init__(self, channels, depth=4, kernel=(3, 3)):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            dil = 2 ** i
            pad = ((kernel[0] - 1) * dil // 2, (kernel[1] - 1) // 2)
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels * (i + 1), channels, kernel, dilation=(dil, 1), padding=pad),
                nn.InstanceNorm2d(channels, affine=True),
                nn.PReLU(channels),
            ))

    def forward(self, x):            # x: [B, C, T, F]
        skip = x
        for layer in self.layers:
            x = layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, channels, in_channels=2):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels))
        self.dense = DenseBlock(channels)
        self.down = nn.Sequential(   # 주파수축 절반으로 (201 → 100)
            nn.Conv2d(channels, channels, (1, 3), (1, 2)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels))

    def forward(self, x):
        return self.down(self.dense(self.conv_in(x)))


class MaskDecoder(nn.Module):
    def __init__(self, channels, freq_bins, beta):
        super().__init__()
        self.dense = DenseBlock(channels)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (1, 3), (1, 2)),
            nn.Conv2d(channels, 1, 1),
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1),
            nn.Conv2d(1, 1, 1))
        self.sigmoid = LearnableSigmoid(freq_bins, beta)

    def forward(self, x):            # → [B, T, F]
        x = self.up(self.dense(x))[:, 0]
        return self.sigmoid(x.float())


class PhaseDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dense = DenseBlock(channels)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (1, 3), (1, 2)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels))
        self.real = nn.Conv2d(channels, 1, 1)
        self.imag = nn.Conv2d(channels, 1, 1)

    def forward(self, x):            # → [B, T, F]
        x = self.up(self.dense(x))
        return torch.atan2(self.imag(x)[:, 0].float(), self.real(x)[:, 0].float())


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * mult), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim * mult, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    """batch_first 멀티헤드 어텐션. PyTorch SDPA로 메모리를 아낍니다."""

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):            # x: [N, L, D]
        n, length, dim = x.shape
        q, k, v = self.qkv(self.norm(x)).view(n, length, 3, self.heads, dim // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(n, length, dim))


class ConvModule(nn.Module):
    def __init__(self, dim, kernel, expansion=2, dropout=0.0):
        super().__init__()
        inner = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, inner * 2, 1)
        self.dw = nn.Conv1d(inner, inner, kernel, padding=kernel // 2, groups=inner)
        self.bn = nn.BatchNorm1d(inner)
        self.pw2 = nn.Conv1d(inner, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):            # x: [N, L, D]
        y = self.norm(x).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        y = F.silu(self.bn(self.dw(y)))
        return self.dropout(self.pw2(y).transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, kernel, dropout):
        super().__init__()
        self.ff1 = FeedForward(dim, dropout=dropout)
        self.attn = SelfAttention(dim, heads, dropout=dropout)
        self.conv = ConvModule(dim, kernel)
        self.ff2 = FeedForward(dim, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class TSConformer(nn.Module):
    """시간축 Conformer → 주파수축 Conformer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        args = (cfg.channels, cfg.heads, cfg.conv_kernel, cfg.dropout)
        self.time = ConformerBlock(*args)
        self.freq = ConformerBlock(*args)

    def forward(self, x):            # x: [B, C, T, F]
        b, c, t, f = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b * f, t, c)
        x = self.time(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).reshape(b * t, f, c)
        x = self.freq(x) + x
        return x.view(b, t, f, c).permute(0, 3, 1, 2)


# ─────────────────────────────── 생성기 ───────────────────────────────
class MPSENet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        bins = cfg.n_fft // 2 + 1
        self.encoder = DenseEncoder(cfg.channels)
        self.blocks = nn.ModuleList(TSConformer(cfg) for _ in range(cfg.num_blocks))
        self.mask_decoder = MaskDecoder(cfg.channels, bins, cfg.mask_beta)
        self.phase_decoder = PhaseDecoder(cfg.channels)
        # True이면 학습 중 중간 활성값을 저장하지 않고 역전파 때 다시 계산합니다.
        # GPU 메모리를 크게 줄이는 대신 학습 시간이 약 30% 늘어납니다.
        self.gradient_checkpointing = False

    def _run(self, module, x):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, noisy_mag, noisy_pha):
        """입력 [B, F, T] 압축 크기·위상 → (크기, 위상, 압축 복소) [B, F, T]."""
        x = torch.stack((noisy_mag, noisy_pha), dim=1).transpose(2, 3)   # [B, 2, T, F]
        x = self._run(self.encoder, x)
        for block in self.blocks:
            x = self._run(block, x)
        mask = self._run(self.mask_decoder, x).transpose(1, 2)           # [B, F, T]
        mag = noisy_mag.float() * mask
        pha = self._run(self.phase_decoder, x).transpose(1, 2)
        com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
        return mag, pha, com

    @torch.no_grad()
    def enhance(self, noisy):
        """[B, L] 파형 → 같은 길이의 복원 파형. 입력 RMS로 정규화 후 되돌립니다."""
        length = noisy.shape[-1]
        norm = torch.sqrt(length / noisy.pow(2).sum(-1, keepdim=True).clamp_min(1e-10))
        mag, pha, _ = mag_pha_stft(noisy * norm, self.cfg)
        mag_g, pha_g, _ = self(mag, pha)
        return mag_pha_istft(mag_g, pha_g, self.cfg, length) / norm


# ─────────────────────────────── 판별기 ───────────────────────────────
class MetricDiscriminator(nn.Module):
    """(clean 크기, 비교 크기) → 정규화 PESQ 예측값 (0~1)."""

    def __init__(self, dim=16):
        super().__init__()
        sn = nn.utils.spectral_norm

        def down(cin, cout):
            return [sn(nn.Conv2d(cin, cout, 4, 2, 1, bias=False)),
                    nn.InstanceNorm2d(cout, affine=True), nn.PReLU(cout)]

        self.net = nn.Sequential(
            *down(2, dim), *down(dim, dim * 2), *down(dim * 2, dim * 4), *down(dim * 4, dim * 8),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            sn(nn.Linear(dim * 8, dim * 4)), nn.Dropout(0.3), nn.PReLU(dim * 4),
            sn(nn.Linear(dim * 4, 1)))
        self.slope = nn.Parameter(torch.ones(1))

    def forward(self, clean_mag, other_mag):
        x = torch.stack((clean_mag, other_mag), dim=1)
        return torch.sigmoid(self.slope * self.net(x)).flatten()


# ─────────────────────────────── 손실 ───────────────────────────────
def anti_wrap(x):
    return torch.abs(x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi)


def phase_losses(pha_r, pha_g):
    """순시 위상, 군지연(주파수 차분), 순시 각주파수(시간 차분) 손실."""
    ip = anti_wrap(pha_r - pha_g).mean()
    gd = anti_wrap(torch.diff(pha_r, dim=1) - torch.diff(pha_g, dim=1)).mean()
    iaf = anti_wrap(torch.diff(pha_r, dim=2) - torch.diff(pha_g, dim=2)).mean()
    return ip + gd + iaf


def build_generator(cfg: ModelConfig):
    return MPSENet(cfg)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def load_generator(checkpoint_path, device="cpu"):
    """best_generator.pt (가중치 + 모델 설정)를 읽어 평가 모드의 모델을 돌려줍니다."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(state["model_config"])
    model = MPSENet(cfg).to(device)
    model.load_state_dict(state["generator"])
    model.eval()
    return model, state
