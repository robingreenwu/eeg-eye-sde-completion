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


class MLP(nn.Module):
    def __init__(self, dims: list[int], activation: type[nn.Module] = nn.LeakyReLU, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
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
    def __init__(self, input_shape: tuple[int, ...], hidden_dim: int = 128):
        super().__init__()
        input_dim = _prod(tuple(input_shape))
        self.net = MLP([input_dim, hidden_dim, hidden_dim, 1], dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


class FusionDiscriminator(nn.Module):
    def __init__(self, fusion_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP([fusion_dim, hidden_dim, hidden_dim, 1], dropout=0.1)

    def forward(self, fusion: torch.Tensor) -> torch.Tensor:
        return self.net(fusion).squeeze(-1)


class VariableDiscriminator(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], hidden_dim: int = 128):
        super().__init__()
        self.output_dim = _prod(tuple(input_shape))
        self.net = MLP([self.output_dim, hidden_dim, hidden_dim, self.output_dim], dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class ConditionalLatentDiscriminator(nn.Module):
    """Judge whether a target latent is realistic under the available modality condition."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP([latent_dim * 2 + 2, hidden_dim, hidden_dim, 1], dropout=0.1)

    def forward(self, condition_latent: torch.Tensor, target_latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition_latent, target_latent, mask], dim=-1)
        return self.net(x).squeeze(-1)


class VPSDEDiffusionNet(nn.Module):
    """Conditional VP-SDE diffusion with a 1D U-Net score/noise network."""

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
    ):
        super().__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sampling_steps = sampling_steps
        self.eps = eps
        self.model = UNet(input_channel=input_channel, channels=list(channels), embed_dim=embed_dim)

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

    def perturb(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x)
        mean, std = self.marginal_prob(x, t)
        return mean + std * noise, noise

    def training_step(self, x_start: torch.Tensor, condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x_start.shape[0]
        t = torch.rand(batch, device=x_start.device) * (1.0 - self.eps) + self.eps
        x_t, noise = self.perturb(x_start, t)
        condition_t = None
        if condition is not None:
            condition_t, _ = self.perturb(condition, t)
        pred_noise = self.model(x_t, t, condition=condition_t)
        loss = F.mse_loss(pred_noise, noise)
        _, std = self.marginal_prob(x_start, t)
        x0_pred = (x_t - std * pred_noise).clamp(-20.0, 20.0)
        mean_coeff = torch.sqrt(torch.clamp(1.0 - std.pow(2), min=1e-12))
        x0_pred = x0_pred / mean_coeff
        return x0_pred, loss

    @torch.no_grad()
    def sample(self, shape, condition=None, print_progress: bool = False):
        x = torch.randn(shape, device=self.device)
        steps = max(1, int(self.sampling_steps))
        time_grid = torch.linspace(1.0, self.eps, steps + 1, device=self.device)
        iterator = range(steps)
        if print_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="VP-SDE sampling")
        for i in iterator:
            t = time_grid[i].repeat(shape[0])
            dt = time_grid[i + 1] - time_grid[i]
            beta_t = self.beta(t).view(-1, 1, 1)
            _, std = self.marginal_prob(x, t)
            pred_noise = self.model(x, t, condition=condition)
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
        pred_noise = self.model(x, t_eps, condition=condition)
        mean_coeff = torch.sqrt(torch.clamp(1.0 - std.pow(2), min=1e-12))
        return ((x - std * pred_noise) / mean_coeff).clamp(-20.0, 20.0)


@dataclass
class LossWeights:
    diffusion: float = 1.0
    reconstruction: float = 1.0
    autoencoding: float = 0.2
    adversarial: float = 0.2
    classification: float = 1.0
    prototype: float = 0.2
    consistency: float = 0.2


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
    ):
        super().__init__()
        self.eeg_shape = tuple(eeg_shape)
        self.eye_shape = tuple(eye_shape)
        self.num_classes = num_classes
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.latent_dim = latent_channels * latent_size
        self.fusion_dim = self.latent_dim

        self.eeg_encoder = EEGSpatialBandEncoder(self.eeg_shape, self.latent_dim, hidden_dim, heads)
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
        )
        self.eye_diffusion = VPSDEDiffusionNet(
            input_size=latent_size,
            input_channel=latent_channels,
            sampling_steps=sampling_steps,
            beta_min=sde_beta_min,
            beta_max=sde_beta_max,
            channels=unet_channels,
            embed_dim=latent_channels,
        )

        self.fusion = MLP([self.latent_dim * 2 + self.latent_dim, hidden_dim * 2, self.fusion_dim])
        self.classifier = MLP([self.fusion_dim, hidden_dim, num_classes])
        self.emotion_prototypes = nn.Parameter(torch.randn(num_classes, self.fusion_dim) * 0.02)

        self.eeg_discriminator = ModalityDiscriminator(self.eeg_shape, hidden_dim)
        self.eye_discriminator = ModalityDiscriminator(self.eye_shape, hidden_dim)
        self.fusion_discriminator = FusionDiscriminator(self.fusion_dim, hidden_dim)
        self.eeg_variable_discriminator = VariableDiscriminator(self.eeg_shape, hidden_dim)
        self.eye_variable_discriminator = VariableDiscriminator(self.eye_shape, hidden_dim)
        self.eeg_latent_discriminator = ConditionalLatentDiscriminator(self.latent_dim, hidden_dim)
        self.eye_latent_discriminator = ConditionalLatentDiscriminator(self.latent_dim, hidden_dim)

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

    def _denoise_with_loss(
        self,
        diffusion: VPSDEDiffusionNet,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return diffusion.training_step(target, condition=condition)

    @torch.no_grad()
    def sample_missing_latent(
        self,
        modality: str,
        condition: torch.Tensor,
        print_progress: bool = False,
    ) -> torch.Tensor:
        diffusion = self.eeg_diffusion if modality == "eeg" else self.eye_diffusion
        latent = diffusion.sample(
            (condition.shape[0], self.latent_channels, self.latent_size),
            condition=condition,
            print_progress=print_progress,
        )
        return latent

    def forward(self, batch: dict[str, Any], sample: bool = False) -> dict[str, torch.Tensor]:
        eeg = batch["eeg"]
        eye = batch["eye"]
        mask = batch["mask"].float()

        eeg_vec = self.eeg_encoder(eeg)
        eye_vec = self.eye_encoder(eye)
        eeg_latent = self._to_tensor_latent(eeg_vec)
        eye_latent = self._to_tensor_latent(eye_vec)
        eeg_condition = self._condition_for_eeg(eye_vec, mask)
        eye_condition = self._condition_for_eye(eeg_vec, mask)

        if sample:
            gen_eeg_latent = self.sample_missing_latent("eeg", eeg_condition)
            gen_eye_latent = self.sample_missing_latent("eye", eye_condition)
            eeg_diff_loss = eeg.new_tensor(0.0)
            eye_diff_loss = eeg.new_tensor(0.0)
        else:
            gen_eeg_latent, eeg_diff_loss = self._denoise_with_loss(self.eeg_diffusion, eeg_latent, eeg_condition)
            gen_eye_latent, eye_diff_loss = self._denoise_with_loss(self.eye_diffusion, eye_latent, eye_condition)

        gen_eeg_vec = self._to_vector_latent(gen_eeg_latent)
        gen_eye_vec = self._to_vector_latent(gen_eye_latent)
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

        mask_vec = self.mask_encoder(mask)
        fusion_real = self.fusion(torch.cat([eeg_vec, eye_vec, torch.ones_like(mask_vec)], dim=-1))
        fusion_completed = self.fusion(torch.cat([completed_eeg_vec, completed_eye_vec, mask_vec], dim=-1))

        return {
            "eeg_vec": eeg_vec,
            "eye_vec": eye_vec,
            "gen_eeg_vec": gen_eeg_vec,
            "gen_eye_vec": gen_eye_vec,
            "completed_eeg_vec": completed_eeg_vec,
            "completed_eye_vec": completed_eye_vec,
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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        eeg = batch["eeg"]
        eye = batch["eye"]
        label = batch["label"]
        mask = batch["mask"].float()
        missing_eeg = 1.0 - mask[:, 0]
        missing_eye = 1.0 - mask[:, 1]

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

        if stage >= 3:
            classification = F.cross_entropy(outputs["logits_real"], label) + F.cross_entropy(
                outputs["logits_completed"], label
            )
            prototype = F.cross_entropy(self.prototype_logits(outputs["fusion_real"]), label) + F.cross_entropy(
                self.prototype_logits(outputs["fusion_completed"]), label
            )
            consistency = F.mse_loss(outputs["fusion_completed"], outputs["fusion_real"].detach())
            loss = (
                loss
                + weights.classification * classification
                + weights.prototype * prototype
                + weights.consistency * consistency
            )
            terms.update(
                {
                    "classification": float(classification.detach().cpu()),
                    "prototype": float(prototype.detach().cpu()),
                    "consistency": float(consistency.detach().cpu()),
                }
            )

        terms["total"] = float(loss.detach().cpu())
        return loss, terms
