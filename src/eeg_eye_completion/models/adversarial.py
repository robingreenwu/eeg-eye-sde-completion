from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import UNet


def _prod(shape: tuple[int, ...]) -> int:
    out = 1
    for value in shape:
        out *= int(value)
    return out


def _broadcast_modality_mask(mask_col: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mask_col.view(mask_col.shape[0], *([1] * (target.dim() - 1)))


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
    weights = _broadcast_modality_mask(missing, pred)
    if float(weights.sum().detach().cpu()) < 1.0:
        weights = torch.ones_like(weights)
    err = (pred - target).pow(2) * weights
    return err.sum() / weights.sum().clamp_min(1.0) / max(1, pred[0].numel())


def _sample_missing_weights(
    mask: torch.Tensor,
    missing_eeg_weight: float = 1.0,
    missing_eye_weight: float = 1.0,
) -> torch.Tensor:
    missing_eeg = 1.0 - mask[:, 0]
    missing_eye = 1.0 - mask[:, 1]
    weights = torch.ones_like(missing_eeg)
    weights = weights + missing_eeg * (float(missing_eeg_weight) - 1.0)
    weights = weights + missing_eye * (float(missing_eye_weight) - 1.0)
    return weights.clamp_min(0.0)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _weighted_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    temperature = max(1e-3, float(temperature))
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    per_sample = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return _weighted_mean(per_sample, weights) * temperature * temperature


class MLP(nn.Module):
    def __init__(
        self,
        dims: list[int],
        activation: type[nn.Module] = nn.LeakyReLU,
        dropout: float = 0.0,
        spectral_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            if i != len(dims) - 2:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGSpatialBandEncoder(nn.Module):
    """EEG encoder that keeps band and channel structure before pooling."""

    def __init__(
        self,
        eeg_shape: tuple[int, ...],
        latent_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
    ):
        super().__init__()
        self.eeg_shape = tuple(eeg_shape)
        self.is_trial = len(eeg_shape) == 3
        if self.is_trial:
            self.time_steps, self.num_bands, self.num_channels = self.eeg_shape
        else:
            self.time_steps, self.num_bands, self.num_channels = 1, self.eeg_shape[0], self.eeg_shape[1]

        self.band_proj = nn.Linear(self.num_channels, hidden_dim)
        self.band_attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, batch_first=True)
        self.channel_proj = nn.Linear(self.num_bands, hidden_dim)
        self.channel_mix = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.adj_logits = nn.Parameter(torch.randn(self.num_channels, self.num_channels) * 0.01)
        self.window_out = MLP([hidden_dim * 2, hidden_dim, latent_dim])
        self.temporal = (
            nn.GRU(latent_dim, latent_dim // 2, batch_first=True, bidirectional=True)
            if self.is_trial
            else None
        )
        self.norm = nn.LayerNorm(latent_dim)

    def _encode_window(self, eeg: torch.Tensor) -> torch.Tensor:
        band_tokens = self.band_proj(eeg)
        band_tokens, _ = self.band_attn(band_tokens, band_tokens, band_tokens, need_weights=False)
        band_repr = band_tokens.mean(dim=1)

        channel_tokens = self.channel_proj(eeg.transpose(1, 2))
        adjacency = torch.softmax(self.adj_logits, dim=-1)
        channel_tokens = torch.einsum("ij,bjh->bih", adjacency, channel_tokens)
        channel_repr = self.channel_mix(channel_tokens).mean(dim=1)

        return self.norm(self.window_out(torch.cat([band_repr, channel_repr], dim=-1)))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.is_trial:
            batch, steps = eeg.shape[:2]
            encoded = self._encode_window(eeg.reshape(batch * steps, self.num_bands, self.num_channels))
            encoded = encoded.reshape(batch, steps, -1)
            temporal, _ = self.temporal(encoded)
            return self.norm(temporal.mean(dim=1))
        return self._encode_window(eeg)


class EyeEncoder(nn.Module):
    def __init__(
        self,
        eye_shape: tuple[int, ...],
        latent_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
    ):
        super().__init__()
        self.eye_shape = tuple(eye_shape)
        self.is_trial = len(eye_shape) == 2
        if self.is_trial:
            self.time_steps, self.num_features = self.eye_shape
            self.input_proj = nn.Linear(self.num_features, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=1)
            self.out = MLP([hidden_dim, hidden_dim, latent_dim])
        else:
            self.num_features = self.eye_shape[0]
            self.net = MLP([self.num_features, hidden_dim, hidden_dim, latent_dim])
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, eye: torch.Tensor) -> torch.Tensor:
        if self.is_trial:
            tokens = self.input_proj(eye)
            tokens = self.temporal(tokens)
            return self.norm(self.out(tokens.mean(dim=1)))
        return self.norm(self.net(eye))


class ShapeDecoder(nn.Module):
    def __init__(self, latent_dim: int, target_shape: tuple[int, ...], hidden_dim: int = 128):
        super().__init__()
        self.target_shape = tuple(target_shape)
        self.net = MLP([latent_dim, hidden_dim, hidden_dim, _prod(self.target_shape)], activation=nn.ReLU)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).reshape(latent.shape[0], *self.target_shape)


class ModalityDiscriminator(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], hidden_dim: int = 128, spectral_norm: bool = False):
        super().__init__()
        input_dim = _prod(tuple(input_shape))
        self.net = MLP([input_dim, hidden_dim, hidden_dim, 1], dropout=0.1, spectral_norm=spectral_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


class FusionDiscriminator(nn.Module):
    def __init__(self, fusion_dim: int, hidden_dim: int = 128, spectral_norm: bool = False):
        super().__init__()
        self.net = MLP([fusion_dim, hidden_dim, hidden_dim, 1], dropout=0.1, spectral_norm=spectral_norm)

    def forward(self, fusion: torch.Tensor) -> torch.Tensor:
        return self.net(fusion).squeeze(-1)


class VariableDiscriminator(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], hidden_dim: int = 128, spectral_norm: bool = False):
        super().__init__()
        self.output_dim = _prod(tuple(input_shape))
        self.net = MLP(
            [self.output_dim, hidden_dim, hidden_dim, self.output_dim],
            dropout=0.1,
            spectral_norm=spectral_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class ConditionalLatentDiscriminator(nn.Module):
    """Judge whether a target latent is realistic under the available modality condition."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, spectral_norm: bool = False):
        super().__init__()
        self.net = MLP([latent_dim * 2 + 2, hidden_dim, hidden_dim, 1], dropout=0.1, spectral_norm=spectral_norm)

    def forward(self, condition_latent: torch.Tensor, target_latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition_latent, target_latent, mask], dim=-1)
        return self.net(x).squeeze(-1)


def _compatible_heads(dim: int, requested: int) -> int:
    for heads in range(max(1, requested), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class SlotCompleteFusionTransformer(nn.Module):
    """Fixed EEG/Eye/availability slot fusion."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        slot_dim = hidden_dim
        num_heads = _compatible_heads(slot_dim, heads)
        self.slot_proj = nn.Linear(latent_dim, slot_dim)
        self.slot_type = nn.Parameter(torch.randn(3, slot_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=slot_dim,
            nhead=num_heads,
            dim_feedforward=slot_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, layers))
        self.pool_score = nn.Linear(slot_dim, 1)
        self.out = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, eeg_slot: torch.Tensor, eye_slot: torch.Tensor, availability_slot: torch.Tensor) -> torch.Tensor:
        slots = torch.stack([eeg_slot, eye_slot, availability_slot], dim=1)
        slots = self.slot_proj(slots) + self.slot_type.unsqueeze(0)
        slots = self.encoder(slots)
        weights = torch.softmax(self.pool_score(slots), dim=1)
        pooled = (slots * weights).sum(dim=1)
        return self.out(pooled)


class VPSDEDiffusionNet(nn.Module):
    """Conditional missing-modality generator with VP-SDE or rectified-flow training."""

    def __init__(
        self,
        input_size: int,
        input_channel: int = 16,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        sampling_steps: int = 50,
        eps: float = 1e-3,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        embed_dim: int = 16,
        noise_condition: bool = False,
        attention_layers: str | tuple[str, ...] = "critical",
        generation_objective: str = "sde",
        sampling_method: str = "sde",
        ddim_eta: float = 0.0,
        self_conditioning_sample: bool = False,
        self_conditioning_weight: float = 0.5,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sampling_steps = sampling_steps
        self.eps = eps
        self.noise_condition = noise_condition
        self.generation_objective = self._normalize_generation_objective(generation_objective)
        self.sampling_method = self._normalize_sampling_method(sampling_method)
        self.ddim_eta = float(ddim_eta)
        self.self_conditioning_sample = bool(self_conditioning_sample)
        self.self_conditioning_weight = min(1.0, max(0.0, float(self_conditioning_weight)))
        self.model = UNet(
            input_channel=input_channel,
            channels=list(channels),
            embed_dim=embed_dim,
            attention_layers=attention_layers,
        )

    @staticmethod
    def _normalize_sampling_method(method: str) -> str:
        normalized = str(method).lower().replace("-", "_")
        aliases = {
            "ddim": "ddim",
            "sde": "sde",
        }
        if normalized not in aliases:
            raise ValueError(f"Unknown diffusion sampling method: {method}")
        return aliases[normalized]

    @staticmethod
    def _normalize_generation_objective(objective: str) -> str:
        normalized = str(objective).lower().replace("-", "_")
        aliases = {
            "diffusion": "sde",
            "vp_sde": "sde",
            "sde": "sde",
            "flow": "flow",
            "flow_matching": "flow",
            "rectified_flow": "flow",
            "rf": "flow",
        }
        if normalized not in aliases:
            raise ValueError(f"Unknown generation objective: {objective}")
        return aliases[normalized]

    @property
    def device(self):
        return next(self.model.parameters()).device

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_mean_coeff = -0.25 * (self.beta_max - self.beta_min) * t.pow(2) - 0.5 * self.beta_min * t
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(torch.clamp(1.0 - torch.exp(2.0 * log_mean_coeff), min=1e-12))
        return mean.view(-1, 1, 1) * x, std.view(-1, 1, 1)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        log_mean_coeff = -0.25 * (self.beta_max - self.beta_min) * t.pow(2) - 0.5 * self.beta_min * t
        return torch.exp(2.0 * log_mean_coeff).clamp(min=1e-12, max=1.0)

    def _sample_condition(self, condition: torch.Tensor | None, self_condition: torch.Tensor | None) -> torch.Tensor | None:
        if condition is None or self_condition is None or not self.self_conditioning_sample:
            return condition
        weight = self.self_conditioning_weight
        return (1.0 - weight) * condition + weight * self_condition

    def _x0_from_noise(self, x: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        _, std = self.marginal_prob(x, t)
        mean_coeff = torch.sqrt(torch.clamp(1.0 - std.pow(2), min=1e-12))
        return ((x - std * pred_noise).clamp(-20.0, 20.0) / mean_coeff).clamp(-20.0, 20.0)

    def _x1_from_velocity(self, x: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        t_view = t.view(-1, 1, 1)
        return (x + (1.0 - t_view) * velocity).clamp(-20.0, 20.0)

    def perturb(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x)
        mean, std = self.marginal_prob(x, t)
        return mean + std * noise, noise

    def training_step(self, x_start: torch.Tensor, condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.generation_objective == "flow":
            return self.flow_matching_step(x_start, condition=condition)

        batch = x_start.shape[0]
        t = torch.rand(batch, device=x_start.device) * (1.0 - self.eps) + self.eps
        x_t, noise = self.perturb(x_start, t)
        condition_t = None
        if condition is not None:
            condition_t = self.perturb(condition, t)[0] if self.noise_condition else condition
        pred_noise = self.model(x_t, t, condition=condition_t)
        loss = F.mse_loss(pred_noise, noise)
        _, std = self.marginal_prob(x_start, t)
        x0_pred = (x_t - std * pred_noise).clamp(-20.0, 20.0)
        mean_coeff = torch.sqrt(torch.clamp(1.0 - std.pow(2), min=1e-12))
        x0_pred = x0_pred / mean_coeff
        return x0_pred, loss

    def flow_matching_step(self, x_start: torch.Tensor, condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x_start.shape[0]
        t = torch.rand(batch, device=x_start.device) * (1.0 - 2.0 * self.eps) + self.eps
        source = torch.randn_like(x_start)
        t_view = t.view(-1, 1, 1)
        x_t = (1.0 - t_view) * source + t_view * x_start
        condition_t = None
        if condition is not None:
            if self.noise_condition:
                condition_source = torch.randn_like(condition)
                condition_t = (1.0 - t_view) * condition_source + t_view * condition
            else:
                condition_t = condition
        target_velocity = x_start - source
        pred_velocity = self.model(x_t, t, condition=condition_t)
        loss = F.mse_loss(pred_velocity, target_velocity)
        x1_pred = self._x1_from_velocity(x_t, t, pred_velocity)
        return x1_pred, loss

    @torch.no_grad()
    def denoise_once(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor | None = None,
        t_value: float = 0.5,
    ) -> torch.Tensor:
        if self.generation_objective == "flow":
            return self.flow_predict_x1_once(x_start, condition=condition, t_value=t_value)

        batch = x_start.shape[0]
        t = torch.full((batch,), float(t_value), device=x_start.device)
        x_t, _ = self.perturb(x_start, t, noise=torch.zeros_like(x_start))
        pred_noise = self.model(x_t, t, condition=condition)
        _, std = self.marginal_prob(x_start, t)
        mean_coeff = torch.sqrt(torch.clamp(1.0 - std.pow(2), min=1e-12))
        return ((x_t - std * pred_noise).clamp(-20.0, 20.0) / mean_coeff).clamp(-20.0, 20.0)

    @torch.no_grad()
    def sample(self, shape, condition=None, print_progress: bool = False):
        if self.generation_objective == "flow":
            return self.flow_sample(shape, condition=condition, print_progress=print_progress)
        if self.sampling_method == "ddim":
            return self.ddim_sample(shape, condition=condition, print_progress=print_progress)
        return self.sde_sample(shape, condition=condition, print_progress=print_progress)

    @torch.no_grad()
    def sde_sample(self, shape, condition=None, print_progress: bool = False):
        x = torch.randn(shape, device=self.device)
        steps = max(1, int(self.sampling_steps))
        time_grid = torch.linspace(1.0, self.eps, steps + 1, device=self.device)
        iterator = range(steps)
        if print_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="VP-SDE sampling")
        self_condition = None
        for i in iterator:
            t = time_grid[i].repeat(shape[0])
            dt = time_grid[i + 1] - time_grid[i]
            beta_t = self.beta(t).view(-1, 1, 1)
            _, std = self.marginal_prob(x, t)
            condition_t = self._sample_condition(condition, self_condition)
            pred_noise = self.model(x, t, condition=condition_t)
            self_condition = self._x0_from_noise(x, t, pred_noise).detach()
            score = -pred_noise / std.clamp_min(1e-5)
            drift = -0.5 * beta_t * x - beta_t * score
            x_mean = x + drift * dt
            if i < steps - 1:
                noise = torch.randn_like(x)
                x = x_mean + torch.sqrt(beta_t * (-dt).clamp_min(1e-12)) * noise
            else:
                x = x_mean

        t_eps = torch.full((shape[0],), self.eps, device=self.device)
        _, std = self.marginal_prob(x, t_eps)
        condition_t = self._sample_condition(condition, self_condition)
        pred_noise = self.model(x, t_eps, condition=condition_t)
        mean_coeff = torch.sqrt(torch.clamp(1.0 - std.pow(2), min=1e-12))
        return ((x - std * pred_noise) / mean_coeff).clamp(-20.0, 20.0)

    @torch.no_grad()
    def flow_predict_x1_once(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor | None = None,
        t_value: float = 0.5,
    ) -> torch.Tensor:
        batch = x_start.shape[0]
        t = torch.full((batch,), float(t_value), device=x_start.device)
        t_view = t.view(-1, 1, 1)
        source = torch.zeros_like(x_start)
        x_t = (1.0 - t_view) * source + t_view * x_start
        pred_velocity = self.model(x_t, t, condition=condition)
        return self._x1_from_velocity(x_t, t, pred_velocity)

    @torch.no_grad()
    def flow_sample(self, shape, condition=None, print_progress: bool = False):
        x = torch.randn(shape, device=self.device)
        steps = max(1, int(self.sampling_steps))
        time_grid = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        iterator = range(steps)
        if print_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Rectified-flow sampling")

        self_condition = None
        for i in iterator:
            t = time_grid[i].repeat(shape[0])
            t_next = time_grid[i + 1].repeat(shape[0])
            dt = time_grid[i + 1] - time_grid[i]

            condition_t = self._sample_condition(condition, self_condition)
            velocity = self.model(x, t, condition=condition_t)
            self_condition = self._x1_from_velocity(x, t, velocity).detach()

            if i == steps - 1:
                x = x + velocity * dt
                break

            x_euler = x + velocity * dt
            condition_next = self._sample_condition(condition, self_condition)
            velocity_next = self.model(x_euler, t_next, condition=condition_next)
            x = x + 0.5 * (velocity + velocity_next) * dt
            self_condition = self._x1_from_velocity(x_euler, t_next, velocity_next).detach()

        return x.clamp(-20.0, 20.0)

    def flow_sample_differentiable(self, shape, condition=None, steps: int | None = None):
        x = torch.randn(shape, device=self.device)
        steps = max(1, int(steps or self.sampling_steps))
        time_grid = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        self_condition = None
        for i in range(steps):
            t = time_grid[i].repeat(shape[0])
            t_next = time_grid[i + 1].repeat(shape[0])
            dt = time_grid[i + 1] - time_grid[i]
            condition_t = self._sample_condition(condition, self_condition)
            velocity = self.model(x, t, condition=condition_t)
            self_condition = self._x1_from_velocity(x, t, velocity)
            if i == steps - 1:
                x = x + velocity * dt
                break

            x_euler = x + velocity * dt
            condition_next = self._sample_condition(condition, self_condition)
            velocity_next = self.model(x_euler, t_next, condition=condition_next)
            x = x + 0.5 * (velocity + velocity_next) * dt
            self_condition = self._x1_from_velocity(x_euler, t_next, velocity_next)

        return x.clamp(-20.0, 20.0)

    @torch.no_grad()
    def ddim_sample(self, shape, condition=None, print_progress: bool = False):
        x = torch.randn(shape, device=self.device)
        steps = max(1, int(self.sampling_steps))
        time_grid = torch.linspace(1.0, self.eps, steps + 1, device=self.device)
        iterator = range(steps)
        if print_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="DDIM sampling")

        x0_pred = None
        self_condition = None
        for i in iterator:
            t = time_grid[i].repeat(shape[0])
            t_next = time_grid[i + 1].repeat(shape[0])
            alpha = self.alpha_bar(t).view(-1, 1, 1)
            alpha_next = self.alpha_bar(t_next).view(-1, 1, 1)
            sqrt_alpha = alpha.sqrt()
            sqrt_one_minus_alpha = (1.0 - alpha).clamp_min(1e-12).sqrt()

            condition_t = self._sample_condition(condition, self_condition)
            pred_noise = self.model(x, t, condition=condition_t)
            x0_pred = (x - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
            self_condition = x0_pred.clamp(-20.0, 20.0).detach()

            if i == steps - 1:
                x = x0_pred
                break

            eta = max(0.0, float(self.ddim_eta))
            sigma = eta * torch.sqrt(
                ((1.0 - alpha / alpha_next).clamp_min(0.0) * (1.0 - alpha_next) / (1.0 - alpha).clamp_min(1e-12))
                .clamp_min(0.0)
            )
            direction_scale = (1.0 - alpha_next - sigma.pow(2)).clamp_min(0.0).sqrt()
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = alpha_next.sqrt() * x0_pred + direction_scale * pred_noise + sigma * noise

        if x0_pred is None:
            return x
        return x.clamp(-20.0, 20.0)


@dataclass
class LossWeights:
    diffusion: float = 1.0
    reconstruction: float = 1.0
    autoencoding: float = 0.2
    adversarial: float = 0.2
    classification: float = 1.0
    prototype: float = 0.2
    consistency: float = 0.2
    teacher_kd: float = 0.0
    teacher_fusion: float = 0.0
    teacher_proto: float = 0.0
    supervised_contrastive: float = 0.0
    prototype_margin: float = 0.0


class AdversarialEEGEyeGenerator(nn.Module):
    def __init__(
        self,
        eeg_shape: tuple[int, ...],
        eye_shape: tuple[int, ...],
        num_classes: int,
        latent_channels: int = 16,
        latent_size: int = 32,
        hidden_dim: int = 128,
        heads: int = 4,
        timesteps: int = 100,
        sampling_steps: int = 10,
        unet_channels: tuple[int, ...] = (16, 32, 64, 128),
        sde_beta_min: float = 0.1,
        sde_beta_max: float = 20.0,
        noise_condition: bool = False,
        spectral_norm_discriminators: bool = False,
        unet_attention: str | tuple[str, ...] = "critical",
        generation_objective: str = "sde",
        diffusion_sampler: str = "sde",
        ddim_eta: float = 0.0,
        fusion_type: str = "slot_transformer",
        fusion_layers: int = 2,
        fusion_dropout: float = 0.1,
        uncertainty_samples: int = 1,
        uncertainty_temperature: float = 0.1,
        min_generated_confidence: float = 0.05,
        self_conditioning_sample: bool = False,
        self_conditioning_weight: float = 0.5,
    ):
        super().__init__()
        self.eeg_shape = tuple(eeg_shape)
        self.eye_shape = tuple(eye_shape)
        self.num_classes = num_classes
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.latent_dim = latent_channels * latent_size
        self.fusion_dim = self.latent_dim
        self.fusion_type = fusion_type
        self.uncertainty_samples = max(1, int(uncertainty_samples))
        self.uncertainty_temperature = max(0.0, float(uncertainty_temperature))
        self.min_generated_confidence = min(1.0, max(0.0, float(min_generated_confidence)))

        self.eeg_encoder = EEGSpatialBandEncoder(
            self.eeg_shape,
            self.latent_dim,
            hidden_dim,
            heads,
        )
        self.eye_encoder = EyeEncoder(self.eye_shape, self.latent_dim, hidden_dim, heads)
        self.eeg_decoder = ShapeDecoder(self.latent_dim, self.eeg_shape, hidden_dim)
        self.eye_decoder = ShapeDecoder(self.latent_dim, self.eye_shape, hidden_dim)

        self.mask_encoder = MLP([2, hidden_dim, self.latent_dim])
        self.eeg_prior_classifier = MLP([self.latent_dim, hidden_dim, num_classes])
        self.eye_prior_classifier = MLP([self.latent_dim, hidden_dim, num_classes])
        self.class_embeddings = nn.Embedding(num_classes, self.latent_dim)

        cond_dim = self.latent_dim * 3
        self.eeg_condition = MLP([cond_dim, hidden_dim * 2, self.latent_dim])
        self.eye_condition = MLP([cond_dim, hidden_dim * 2, self.latent_dim])
        self.eeg_diffusion = VPSDEDiffusionNet(
            input_size=latent_size,
            input_channel=latent_channels,
            sampling_steps=sampling_steps,
            beta_min=sde_beta_min,
            beta_max=sde_beta_max,
            channels=unet_channels,
            embed_dim=latent_channels,
            noise_condition=noise_condition,
            attention_layers=unet_attention,
            generation_objective=generation_objective,
            sampling_method=diffusion_sampler,
            ddim_eta=ddim_eta,
            self_conditioning_sample=self_conditioning_sample,
            self_conditioning_weight=self_conditioning_weight,
        )
        self.eye_diffusion = VPSDEDiffusionNet(
            input_size=latent_size,
            input_channel=latent_channels,
            sampling_steps=sampling_steps,
            beta_min=sde_beta_min,
            beta_max=sde_beta_max,
            channels=unet_channels,
            embed_dim=latent_channels,
            noise_condition=noise_condition,
            attention_layers=unet_attention,
            generation_objective=generation_objective,
            sampling_method=diffusion_sampler,
            ddim_eta=ddim_eta,
            self_conditioning_sample=self_conditioning_sample,
            self_conditioning_weight=self_conditioning_weight,
        )

        if fusion_type == "mlp":
            self.fusion = MLP([self.latent_dim * 2 + self.latent_dim, hidden_dim * 2, self.fusion_dim])
        elif fusion_type == "slot_transformer":
            self.fusion = SlotCompleteFusionTransformer(
                latent_dim=self.latent_dim,
                hidden_dim=hidden_dim,
                heads=heads,
                layers=fusion_layers,
                dropout=fusion_dropout,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        self.classifier = MLP([self.fusion_dim, hidden_dim, num_classes])
        self.emotion_prototypes = nn.Parameter(torch.randn(num_classes, self.fusion_dim) * 0.02)

        self.eeg_discriminator = ModalityDiscriminator(self.eeg_shape, hidden_dim, spectral_norm_discriminators)
        self.eye_discriminator = ModalityDiscriminator(self.eye_shape, hidden_dim, spectral_norm_discriminators)
        self.fusion_discriminator = FusionDiscriminator(self.fusion_dim, hidden_dim, spectral_norm_discriminators)
        self.eeg_variable_discriminator = VariableDiscriminator(self.eeg_shape, hidden_dim, spectral_norm_discriminators)
        self.eye_variable_discriminator = VariableDiscriminator(self.eye_shape, hidden_dim, spectral_norm_discriminators)
        self.eeg_latent_discriminator = ConditionalLatentDiscriminator(
            self.latent_dim,
            hidden_dim,
            spectral_norm_discriminators,
        )
        self.eye_latent_discriminator = ConditionalLatentDiscriminator(
            self.latent_dim,
            hidden_dim,
            spectral_norm_discriminators,
        )

    def generator_parameters(self):
        discriminator_ids = {id(p) for p in self.discriminator_parameters()}
        return (p for p in self.parameters() if id(p) not in discriminator_ids)

    def discriminator_parameters(self):
        modules = [
            self.eeg_discriminator,
            self.eye_discriminator,
            self.fusion_discriminator,
            self.eeg_variable_discriminator,
            self.eye_variable_discriminator,
            self.eeg_latent_discriminator,
            self.eye_latent_discriminator,
        ]
        for module in modules:
            yield from module.parameters()

    def _to_tensor_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.reshape(latent.shape[0], self.latent_channels, self.latent_size)

    def _to_vector_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.reshape(latent.shape[0], -1)

    def _emotion_prior(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return probs @ self.class_embeddings.weight

    def _condition_for_eeg(self, eye_vec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_vec = self.mask_encoder(mask)
        prior = self._emotion_prior(self.eye_prior_classifier(eye_vec))
        return self._to_tensor_latent(self.eeg_condition(torch.cat([eye_vec, mask_vec, prior], dim=-1)))

    def _condition_for_eye(self, eeg_vec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_vec = self.mask_encoder(mask)
        prior = self._emotion_prior(self.eeg_prior_classifier(eeg_vec))
        return self._to_tensor_latent(self.eye_condition(torch.cat([eeg_vec, mask_vec, prior], dim=-1)))

    def _fuse_slots(self, eeg_slot: torch.Tensor, eye_slot: torch.Tensor, availability_slot: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "mlp":
            return self.fusion(torch.cat([eeg_slot, eye_slot, availability_slot], dim=-1))
        return self.fusion(eeg_slot, eye_slot, availability_slot)

    def _availability_slot(self, mask: torch.Tensor, real: bool = False) -> torch.Tensor:
        if self.fusion_type == "mlp" and real:
            return torch.ones(mask.shape[0], self.latent_dim, device=mask.device, dtype=mask.dtype)
        return self.mask_encoder(mask)

    def full_modality_outputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        eeg = batch["eeg"]
        eye = batch["eye"]
        mask = batch["mask"].float()
        eeg_vec = self.eeg_encoder(eeg)
        eye_vec = self.eye_encoder(eye)
        real_availability = self._availability_slot(torch.ones_like(mask), real=True)
        fusion_real = self._fuse_slots(eeg_vec, eye_vec, real_availability)
        return {
            "eeg_vec": eeg_vec,
            "eye_vec": eye_vec,
            "fusion_real": fusion_real,
            "logits_real": self.classifier(fusion_real),
            "prototype_logits_real": self.prototype_logits(fusion_real),
        }

    def _denoise_with_loss(
        self,
        diffusion: VPSDEDiffusionNet,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return diffusion.training_step(target, condition=condition)

    def _denoise_once(
        self,
        diffusion: VPSDEDiffusionNet,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return diffusion.denoise_once(target, condition=condition)

    def sample_missing_latent(
        self,
        modality: str,
        condition: torch.Tensor,
        print_progress: bool = False,
        return_uncertainty: bool = False,
        differentiable: bool = False,
        sample_steps: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diffusion = self.eeg_diffusion if modality == "eeg" else self.eye_diffusion
        shape = (condition.shape[0], self.latent_channels, self.latent_size)
        if differentiable and diffusion.generation_objective == "flow":
            latent = diffusion.flow_sample_differentiable(shape, condition=condition, steps=sample_steps)
            if not return_uncertainty:
                return latent
            uncertainty = torch.zeros(condition.shape[0], 1, device=condition.device, dtype=condition.dtype)
            confidence = torch.ones_like(uncertainty)
            return latent, uncertainty, confidence

        if not return_uncertainty or self.uncertainty_samples <= 1:
            with torch.no_grad():
                latent = diffusion.sample(shape, condition=condition, print_progress=print_progress)
            if not return_uncertainty:
                return latent
            uncertainty = torch.zeros(condition.shape[0], 1, device=condition.device, dtype=condition.dtype)
            confidence = torch.ones_like(uncertainty)
            return latent, uncertainty, confidence

        samples = [
            diffusion.sample(shape, condition=condition, print_progress=print_progress and idx == 0)
            for idx in range(self.uncertainty_samples)
        ]
        stacked = torch.stack(samples, dim=0)
        latent = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False).flatten(1).mean(dim=-1, keepdim=True)
        uncertainty = variance.clamp_min(1e-8).log()
        confidence = (1.0 / (1.0 + self.uncertainty_temperature * variance)).clamp(
            min=self.min_generated_confidence,
            max=1.0,
        )
        return latent, uncertainty, confidence

    def forward(
        self,
        batch: dict[str, Any],
        sample: bool = False,
        completion_mode: str | None = None,
        sample_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        eeg = batch["eeg"]
        eye = batch["eye"]
        mask = batch["mask"].float()

        eeg_vec = self.eeg_encoder(eeg)
        eye_vec = self.eye_encoder(eye)
        eeg_latent = self._to_tensor_latent(eeg_vec)
        eye_latent = self._to_tensor_latent(eye_vec)
        eeg_condition = self._condition_for_eeg(eye_vec, mask)
        eye_condition = self._condition_for_eye(eeg_vec, mask)

        mode = completion_mode or ("sample" if sample else "teacher")
        if mode in {"sample", "sample_grad"}:
            differentiable_sample = mode == "sample_grad"
            gen_eeg_latent, eeg_uncertainty, eeg_confidence = self.sample_missing_latent(
                "eeg",
                eeg_condition,
                return_uncertainty=True,
                differentiable=differentiable_sample,
                sample_steps=sample_steps,
            )
            gen_eye_latent, eye_uncertainty, eye_confidence = self.sample_missing_latent(
                "eye",
                eye_condition,
                return_uncertainty=True,
                differentiable=differentiable_sample,
                sample_steps=sample_steps,
            )
            eeg_diff_loss = eeg.new_tensor(0.0)
            eye_diff_loss = eeg.new_tensor(0.0)
        elif mode == "denoise":
            gen_eeg_latent = self._denoise_once(self.eeg_diffusion, eeg_latent, eeg_condition)
            gen_eye_latent = self._denoise_once(self.eye_diffusion, eye_latent, eye_condition)
            eeg_uncertainty = eeg.new_zeros(eeg.shape[0], 1)
            eye_uncertainty = eeg.new_zeros(eeg.shape[0], 1)
            eeg_confidence = eeg.new_ones(eeg.shape[0], 1)
            eye_confidence = eeg.new_ones(eeg.shape[0], 1)
            eeg_diff_loss = eeg.new_tensor(0.0)
            eye_diff_loss = eeg.new_tensor(0.0)
        elif mode == "teacher":
            gen_eeg_latent, eeg_diff_loss = self._denoise_with_loss(self.eeg_diffusion, eeg_latent, eeg_condition)
            gen_eye_latent, eye_diff_loss = self._denoise_with_loss(self.eye_diffusion, eye_latent, eye_condition)
            eeg_uncertainty = eeg.new_zeros(eeg.shape[0], 1)
            eye_uncertainty = eeg.new_zeros(eeg.shape[0], 1)
            eeg_confidence = eeg.new_ones(eeg.shape[0], 1)
            eye_confidence = eeg.new_ones(eeg.shape[0], 1)
        else:
            raise ValueError(f"Unknown completion mode: {mode}")

        gen_eeg_vec = self._to_vector_latent(gen_eeg_latent)
        gen_eye_vec = self._to_vector_latent(gen_eye_latent)
        eeg_condition_vec = self._to_vector_latent(eeg_condition)
        eye_condition_vec = self._to_vector_latent(eye_condition)

        calibrated_gen_eeg_vec = eeg_confidence * gen_eeg_vec + (1.0 - eeg_confidence) * eeg_condition_vec
        calibrated_gen_eye_vec = eye_confidence * gen_eye_vec + (1.0 - eye_confidence) * eye_condition_vec
        use_uncertainty = mode == "sample" and self.uncertainty_samples > 1
        if use_uncertainty:
            gen_eeg_vec = calibrated_gen_eeg_vec
            gen_eye_vec = calibrated_gen_eye_vec
        gen_eeg = self.eeg_decoder(gen_eeg_vec)
        gen_eye = self.eye_decoder(gen_eye_vec)
        ae_eeg = self.eeg_decoder(eeg_vec)
        ae_eye = self.eye_decoder(eye_vec)

        eeg_mask = mask[:, 0:1]
        eye_mask = mask[:, 1:2]
        completed_eeg_vec = eeg_mask * eeg_vec + (1.0 - eeg_mask) * gen_eeg_vec
        completed_eye_vec = eye_mask * eye_vec + (1.0 - eye_mask) * gen_eye_vec
        completed_eeg = _broadcast_modality_mask(eeg_mask, eeg) * eeg + _broadcast_modality_mask(1.0 - eeg_mask, eeg) * gen_eeg
        completed_eye = _broadcast_modality_mask(eye_mask, eye) * eye + _broadcast_modality_mask(1.0 - eye_mask, eye) * gen_eye

        if use_uncertainty:
            mask_for_fusion = torch.cat(
                [
                    eeg_mask + (1.0 - eeg_mask) * eeg_confidence,
                    eye_mask + (1.0 - eye_mask) * eye_confidence,
                ],
                dim=-1,
            )
        else:
            mask_for_fusion = mask
        mask_vec = self.mask_encoder(mask_for_fusion)
        real_availability = self._availability_slot(torch.ones_like(mask), real=True)
        fusion_real = self._fuse_slots(eeg_vec, eye_vec, real_availability)
        fusion_completed = self._fuse_slots(completed_eeg_vec, completed_eye_vec, mask_vec)

        return {
            "eeg_vec": eeg_vec,
            "eye_vec": eye_vec,
            "gen_eeg_vec": gen_eeg_vec,
            "gen_eye_vec": gen_eye_vec,
            "completed_eeg_vec": completed_eeg_vec,
            "completed_eye_vec": completed_eye_vec,
            "eeg_uncertainty": eeg_uncertainty,
            "eye_uncertainty": eye_uncertainty,
            "eeg_confidence": eeg_confidence,
            "eye_confidence": eye_confidence,
            "gen_eeg": gen_eeg,
            "gen_eye": gen_eye,
            "completed_eeg": completed_eeg,
            "completed_eye": completed_eye,
            "ae_eeg": ae_eeg,
            "ae_eye": ae_eye,
            "fusion_real": fusion_real,
            "fusion_completed": fusion_completed,
            "logits_real": self.classifier(fusion_real),
            "logits_completed": self.classifier(fusion_completed),
            "eeg_diff_loss": eeg_diff_loss,
            "eye_diff_loss": eye_diff_loss,
        }

    def prototype_logits(self, fusion: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        fusion = F.normalize(fusion, dim=-1)
        prototypes = F.normalize(self.emotion_prototypes, dim=-1)
        return fusion @ prototypes.t() / temperature

    def supervised_contrastive_loss(
        self,
        fusion: torch.Tensor,
        label: torch.Tensor,
        temperature: float = 0.1,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fusion.shape[0] <= 1:
            return fusion.new_tensor(0.0)
        temperature = max(1e-3, float(temperature))
        features = F.normalize(fusion, dim=-1)
        logits = features @ features.t() / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        batch = fusion.shape[0]
        self_mask = torch.eye(batch, device=fusion.device, dtype=torch.bool)
        positive_mask = label.view(-1, 1).eq(label.view(1, -1)) & ~self_mask
        logits_mask = (~self_mask).to(dtype=logits.dtype)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not bool(valid.any()):
            return fusion.new_tensor(0.0)
        per_anchor = -(log_prob * positive_mask.to(dtype=logits.dtype)).sum(dim=1) / positive_count.clamp_min(1)
        valid_weights = weights[valid] if weights is not None else None
        return _weighted_mean(per_anchor[valid], valid_weights)

    def prototype_margin_loss(
        self,
        fusion: torch.Tensor,
        label: torch.Tensor,
        margin: float = 0.2,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fusion = F.normalize(fusion, dim=-1)
        prototypes = F.normalize(self.emotion_prototypes, dim=-1)
        similarities = fusion @ prototypes.t()
        positive = similarities.gather(1, label.view(-1, 1)).squeeze(1)
        negative_mask = F.one_hot(label, num_classes=similarities.shape[1]).bool()
        hardest_negative = similarities.masked_fill(negative_mask, -1e4).max(dim=1).values
        per_sample = F.relu(float(margin) + hardest_negative - positive)
        return _weighted_mean(per_sample, weights)

    def discriminator_loss(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        use_modality_adv: bool = True,
        use_fusion_adv: bool = True,
        use_variable_adv: bool = True,
        use_latent_adv: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        eeg = batch["eeg"]
        eye = batch["eye"]
        mask = batch["mask"].float()
        real = torch.ones(eeg.shape[0], device=eeg.device)
        fake = torch.zeros(eeg.shape[0], device=eeg.device)

        losses = []
        if use_modality_adv:
            losses.extend(
                [
                    F.binary_cross_entropy_with_logits(self.eeg_discriminator(eeg), real),
                    F.binary_cross_entropy_with_logits(self.eeg_discriminator(outputs["gen_eeg"].detach()), fake),
                    F.binary_cross_entropy_with_logits(self.eye_discriminator(eye), real),
                    F.binary_cross_entropy_with_logits(self.eye_discriminator(outputs["gen_eye"].detach()), fake),
                ]
            )
        if use_fusion_adv:
            losses.extend(
                [
                    F.binary_cross_entropy_with_logits(self.fusion_discriminator(outputs["fusion_real"].detach()), real),
                    F.binary_cross_entropy_with_logits(self.fusion_discriminator(outputs["fusion_completed"].detach()), fake),
                ]
            )
        if use_variable_adv:
            eeg_var_real = self.eeg_variable_discriminator(eeg)
            eeg_var_fake = self.eeg_variable_discriminator(outputs["gen_eeg"].detach())
            eye_var_real = self.eye_variable_discriminator(eye)
            eye_var_fake = self.eye_variable_discriminator(outputs["gen_eye"].detach())
            losses.extend(
                [
                    F.binary_cross_entropy_with_logits(eeg_var_real, torch.ones_like(eeg_var_real)),
                    F.binary_cross_entropy_with_logits(eeg_var_fake, torch.zeros_like(eeg_var_fake)),
                    F.binary_cross_entropy_with_logits(eye_var_real, torch.ones_like(eye_var_real)),
                    F.binary_cross_entropy_with_logits(eye_var_fake, torch.zeros_like(eye_var_fake)),
                ]
            )
        if use_latent_adv:
            losses.extend(
                [
                    F.binary_cross_entropy_with_logits(
                        self.eeg_latent_discriminator(
                            outputs["eye_vec"].detach(),
                            outputs["eeg_vec"].detach(),
                            mask.detach(),
                        ),
                        real,
                    ),
                    F.binary_cross_entropy_with_logits(
                        self.eeg_latent_discriminator(
                            outputs["eye_vec"].detach(),
                            outputs["gen_eeg_vec"].detach(),
                            mask.detach(),
                        ),
                        fake,
                    ),
                    F.binary_cross_entropy_with_logits(
                        self.eye_latent_discriminator(
                            outputs["eeg_vec"].detach(),
                            outputs["eye_vec"].detach(),
                            mask.detach(),
                        ),
                        real,
                    ),
                    F.binary_cross_entropy_with_logits(
                        self.eye_latent_discriminator(
                            outputs["eeg_vec"].detach(),
                            outputs["gen_eye_vec"].detach(),
                            mask.detach(),
                        ),
                        fake,
                    ),
                ]
            )

        if not losses:
            loss = eeg.new_tensor(0.0)
        else:
            loss = torch.stack(losses).mean()
        return loss, {"disc": float(loss.detach().cpu())}

    def generator_loss(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        stage: int,
        weights: LossWeights,
        use_modality_adv: bool = True,
        use_fusion_adv: bool = True,
        use_variable_adv: bool = True,
        use_latent_adv: bool = True,
        semantic_temperature: float = 2.0,
        missing_eeg_semantic_weight: float = 1.0,
        missing_eye_semantic_weight: float = 1.0,
        teacher_outputs: dict[str, torch.Tensor] | None = None,
        supcon_temperature: float = 0.1,
        proto_margin: float = 0.2,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        eeg = batch["eeg"]
        eye = batch["eye"]
        label = batch["label"]
        mask = batch["mask"].float()
        missing_eeg = 1.0 - mask[:, 0]
        missing_eye = 1.0 - mask[:, 1]
        semantic_weights = _sample_missing_weights(
            mask,
            missing_eeg_weight=missing_eeg_semantic_weight,
            missing_eye_weight=missing_eye_semantic_weight,
        )

        diffusion = outputs["eeg_diff_loss"] + outputs["eye_diff_loss"]
        reconstruction = _masked_mse(outputs["gen_eeg"], eeg, missing_eeg) + _masked_mse(outputs["gen_eye"], eye, missing_eye)
        autoencoding = F.mse_loss(outputs["ae_eeg"], eeg) + F.mse_loss(outputs["ae_eye"], eye)

        loss = (
            weights.diffusion * diffusion
            + weights.reconstruction * reconstruction
            + weights.autoencoding * autoencoding
        )
        terms = {
            "diffusion": float(diffusion.detach().cpu()),
            "reconstruction": float(reconstruction.detach().cpu()),
            "autoencoding": float(autoencoding.detach().cpu()),
        }

        if stage >= 2:
            real = torch.ones(eeg.shape[0], device=eeg.device)
            adv_losses = []
            if use_modality_adv:
                adv_losses.extend(
                    [
                        F.binary_cross_entropy_with_logits(self.eeg_discriminator(outputs["gen_eeg"]), real),
                        F.binary_cross_entropy_with_logits(self.eye_discriminator(outputs["gen_eye"]), real),
                    ]
                )
            if use_fusion_adv:
                adv_losses.append(F.binary_cross_entropy_with_logits(self.fusion_discriminator(outputs["fusion_completed"]), real))
            if use_variable_adv:
                eeg_var_fake = self.eeg_variable_discriminator(outputs["gen_eeg"])
                eye_var_fake = self.eye_variable_discriminator(outputs["gen_eye"])
                adv_losses.extend(
                    [
                        F.binary_cross_entropy_with_logits(eeg_var_fake, torch.ones_like(eeg_var_fake)),
                        F.binary_cross_entropy_with_logits(eye_var_fake, torch.ones_like(eye_var_fake)),
                    ]
                )
            if use_latent_adv:
                adv_losses.extend(
                    [
                        F.binary_cross_entropy_with_logits(
                            self.eeg_latent_discriminator(outputs["eye_vec"].detach(), outputs["gen_eeg_vec"], mask),
                            real,
                        ),
                        F.binary_cross_entropy_with_logits(
                            self.eye_latent_discriminator(outputs["eeg_vec"].detach(), outputs["gen_eye_vec"], mask),
                            real,
                        ),
                    ]
                )
            adv = torch.stack(adv_losses).mean() if adv_losses else eeg.new_tensor(0.0)
            loss = loss + weights.adversarial * adv
            terms["adversarial"] = float(adv.detach().cpu())
            terms["weight_adversarial"] = float(weights.adversarial)

        if stage >= 3:
            teacher_ref = teacher_outputs if teacher_outputs is not None else outputs
            teacher_logits_real = teacher_ref["logits_real"].detach()
            teacher_fusion_real = teacher_ref["fusion_real"].detach()
            classification_real = F.cross_entropy(outputs["logits_real"], label)
            classification_completed = _weighted_mean(
                F.cross_entropy(outputs["logits_completed"], label, reduction="none"),
                semantic_weights,
            )
            classification = classification_real + classification_completed
            proto_real_logits = self.prototype_logits(outputs["fusion_real"])
            proto_completed_logits = self.prototype_logits(outputs["fusion_completed"])
            prototype_real = F.cross_entropy(proto_real_logits, label)
            prototype_completed = _weighted_mean(
                F.cross_entropy(proto_completed_logits, label, reduction="none"),
                semantic_weights,
            )
            teacher_proto_logits = teacher_ref.get("prototype_logits_real")
            if teacher_proto_logits is None:
                teacher_proto_logits = proto_real_logits
            teacher_proto_logits = teacher_proto_logits.detach()
            prototype = prototype_real + prototype_completed
            supervised_contrastive = self.supervised_contrastive_loss(
                outputs["fusion_completed"],
                label,
                temperature=supcon_temperature,
                weights=semantic_weights,
            )
            prototype_margin = self.prototype_margin_loss(
                outputs["fusion_completed"],
                label,
                margin=proto_margin,
                weights=semantic_weights,
            )
            consistency_per_sample = F.mse_loss(
                outputs["fusion_completed"],
                teacher_fusion_real,
                reduction="none",
            ).flatten(1).mean(dim=-1)
            consistency = _weighted_mean(consistency_per_sample, semantic_weights)
            loss = (
                loss
                + weights.classification * classification
                + weights.prototype * prototype
                + weights.consistency * consistency
                + weights.supervised_contrastive * supervised_contrastive
                + weights.prototype_margin * prototype_margin
            )
            if weights.teacher_kd > 0:
                teacher_kd = _weighted_kl_from_logits(
                    outputs["logits_completed"],
                    teacher_logits_real,
                    semantic_temperature,
                    semantic_weights,
                )
                loss = loss + weights.teacher_kd * teacher_kd
            else:
                teacher_kd = eeg.new_tensor(0.0)
            if weights.teacher_fusion > 0:
                fusion_cosine = F.cosine_similarity(
                    F.normalize(outputs["fusion_completed"], dim=-1),
                    F.normalize(teacher_fusion_real, dim=-1),
                    dim=-1,
                )
                teacher_fusion = _weighted_mean(1.0 - fusion_cosine, semantic_weights)
                loss = loss + weights.teacher_fusion * teacher_fusion
            else:
                teacher_fusion = eeg.new_tensor(0.0)
            if weights.teacher_proto > 0:
                teacher_proto = _weighted_kl_from_logits(
                    proto_completed_logits,
                    teacher_proto_logits,
                    semantic_temperature,
                    semantic_weights,
                )
                loss = loss + weights.teacher_proto * teacher_proto
            else:
                teacher_proto = eeg.new_tensor(0.0)
            terms.update(
                {
                    "classification": float(classification.detach().cpu()),
                    "prototype": float(prototype.detach().cpu()),
                    "consistency": float(consistency.detach().cpu()),
                    "weight_consistency": float(weights.consistency),
                    "teacher_kd": float(teacher_kd.detach().cpu()),
                    "weight_teacher_kd": float(weights.teacher_kd),
                    "teacher_fusion": float(teacher_fusion.detach().cpu()),
                    "weight_teacher_fusion": float(weights.teacher_fusion),
                    "teacher_proto": float(teacher_proto.detach().cpu()),
                    "weight_teacher_proto": float(weights.teacher_proto),
                    "supervised_contrastive": float(supervised_contrastive.detach().cpu()),
                    "weight_supervised_contrastive": float(weights.supervised_contrastive),
                    "prototype_margin": float(prototype_margin.detach().cpu()),
                    "weight_prototype_margin": float(weights.prototype_margin),
                    "semantic_weight_mean": float(semantic_weights.mean().detach().cpu()),
                }
            )

        terms["total"] = float(loss.detach().cpu())
        return loss, terms
