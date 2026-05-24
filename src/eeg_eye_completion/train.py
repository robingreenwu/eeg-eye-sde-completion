from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_emotion_dataloaders, default_dataset_root
from .models import AdversarialEEGEyeGenerator, LossWeights
from .visualization import save_run_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial generative completion for EEG-Eye multimodal emotion recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_root", type=Path, default=default_dataset_root())
    parser.add_argument("--data_mode", choices=["window", "trial"], default="window")
    parser.add_argument(
        "--split_protocol",
        choices=["predefined", "stratified", "subject", "session"],
        default="predefined",
        help="Data split protocol. Trial mode maps predefined to stratified.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--test_subjects", type=str, default="", help="Comma-separated subject IDs for subject split.")
    parser.add_argument("--test_sessions", type=str, default="", help="Comma-separated session IDs for session split.")
    parser.add_argument("--missing_mode", choices=["random", "missing_eeg", "missing_eye", "none"], default="random")
    parser.add_argument("--missing_rate", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize", action="store_true", help="Force train-stat normalization for both modes.")
    parser.add_argument("--no_normalize", action="store_true", help="Disable automatic normalization.")

    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage3_epochs", type=int, default=40)
    parser.add_argument("--max_batches", type=int, default=0, help="Limit batches per epoch; 0 means full epoch.")
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100, help="Deprecated compatibility option; SDE uses continuous time.")
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument(
        "--generation_objective",
        choices=["sde", "flow"],
        default="sde",
        help="Missing-modality generator objective. flow uses rectified Flow Matching instead of VP-SDE noise prediction.",
    )
    parser.add_argument(
        "--diffusion_sampler",
        choices=["ddim", "sde"],
        default="sde",
        help="Reverse diffusion sampler used by the SDE objective when eval_mode=sample.",
    )
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM stochasticity; 0.0 is deterministic DDIM.")
    parser.add_argument("--sde_beta_min", type=float, default=0.1)
    parser.add_argument("--sde_beta_max", type=float, default=20.0)
    parser.add_argument("--unet_channels", type=str, default="[16, 32, 64, 128]")
    parser.add_argument(
        "--unet_attention",
        type=str,
        default="critical",
        help="U-Net attention preset/layers: all, sampling, critical, bottleneck, none, or comma-separated layers.",
    )
    parser.add_argument(
        "--fusion_type",
        choices=["slot_transformer", "mlp"],
        default="slot_transformer",
        help="Fusion interface. slot_transformer uses fixed EEG/Eye/availability slots; mlp is the older concat MLP.",
    )
    parser.add_argument("--fusion_layers", type=int, default=2)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument(
        "--uncertainty_samples",
        type=int,
        default=1,
        help="Number of stochastic sample completions used to estimate latent uncertainty; 1 disables uncertainty calibration.",
    )
    parser.add_argument(
        "--uncertainty_temperature",
        type=float,
        default=0.1,
        help="Maps sample variance to generated-modality confidence with 1 / (1 + temperature * variance).",
    )
    parser.add_argument(
        "--min_generated_confidence",
        type=float,
        default=0.05,
        help="Lower bound for generated-modality confidence when uncertainty calibration is enabled.",
    )
    parser.add_argument(
        "--self_conditioning_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="During target-free sampling, feed the previous x0 prediction back into the next sampling step condition.",
    )
    parser.add_argument("--self_conditioning_weight", type=float, default=0.5)

    parser.add_argument("--lambda_diffusion", type=float, default=1.0)
    parser.add_argument("--lambda_reconstruction", type=float, default=1.0)
    parser.add_argument("--lambda_autoencoding", type=float, default=0.2)
    parser.add_argument("--lambda_adv", type=float, default=0.05)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_proto", type=float, default=0.2)
    parser.add_argument("--lambda_consistency", type=float, default=0.05)
    parser.add_argument(
        "--lambda_supcon",
        type=float,
        default=0.05,
        help="Stage-3 supervised contrastive loss on completed fusion representations.",
    )
    parser.add_argument("--supcon_temperature", type=float, default=0.1)
    parser.add_argument(
        "--lambda_proto_margin",
        type=float,
        default=0.05,
        help="Stage-3 prototype margin loss for completed fusion representations.",
    )
    parser.add_argument("--proto_margin", type=float, default=0.2)
    parser.add_argument(
        "--teacher_checkpoint",
        type=Path,
        default=None,
        help="Optional frozen full-modality teacher checkpoint for semantic distillation.",
    )
    parser.add_argument(
        "--lambda_teacher_kd",
        type=float,
        default=0.05,
        help="Stage-3 KL distillation from full-modality logits to completed-modality logits.",
    )
    parser.add_argument(
        "--lambda_teacher_fusion",
        type=float,
        default=0.02,
        help="Stage-3 cosine semantic alignment from completed fusion to full-modality fusion.",
    )
    parser.add_argument(
        "--lambda_teacher_proto",
        type=float,
        default=0.02,
        help="Stage-3 prototype-distribution distillation from full-modality fusion to completed fusion.",
    )
    parser.add_argument("--semantic_distill_temperature", type=float, default=2.0)
    parser.add_argument(
        "--missing_eeg_semantic_weight",
        type=float,
        default=1.3,
        help="Extra semantic-loss weight for samples whose EEG modality is missing.",
    )
    parser.add_argument(
        "--missing_eye_semantic_weight",
        type=float,
        default=1.1,
        help="Extra semantic-loss weight for samples whose Eye modality is missing.",
    )
    parser.add_argument(
        "--lambda_sample_cls",
        type=float,
        default=0.2,
        help="Stage-3 lightweight sample-consistency classification loss weight.",
    )
    parser.add_argument(
        "--sample_consistency_interval",
        type=int,
        default=5,
        help="Run one target-free sample classification update every N stage-3 batches; 0 disables it.",
    )
    parser.add_argument(
        "--sample_consistency_start_fraction",
        type=float,
        default=0.5,
        help="Start lightweight sample-consistency updates after this fraction of stage-3 epochs.",
    )
    parser.add_argument(
        "--lambda_sample_distill",
        type=float,
        default=0.02,
        help="KL distillation from full-modality teacher logits to sample completed logits during sample-consistency updates.",
    )
    parser.add_argument("--sample_distill_temperature", type=float, default=2.0)
    parser.add_argument(
        "--lambda_sample_fusion_distill",
        type=float,
        default=0.02,
        help="Sample-path cosine alignment from sampled-completion fusion to full-modality fusion.",
    )
    parser.add_argument(
        "--lambda_sample_proto_distill",
        type=float,
        default=0.02,
        help="Sample-path prototype-distribution distillation from full-modality fusion to sampled completion.",
    )
    parser.add_argument(
        "--lambda_sample_supcon",
        type=float,
        default=0.02,
        help="Sample-path supervised contrastive loss on sampled completed fusion representations.",
    )
    parser.add_argument(
        "--lambda_sample_proto_margin",
        type=float,
        default=0.02,
        help="Sample-path prototype margin loss for sampled completed fusion representations.",
    )
    parser.add_argument(
        "--differentiable_sample_consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For Flow Matching, run sample-consistency through a few differentiable sampling steps.",
    )
    parser.add_argument(
        "--differentiable_sample_steps",
        type=int,
        default=2,
        help="Number of differentiable Flow sampling steps used for stage-3 sample-consistency updates.",
    )
    parser.add_argument("--adv_warmup_epochs", type=int, default=5)
    parser.add_argument("--consistency_warmup_epochs", type=int, default=5)

    parser.add_argument("--no_modality_adv", action="store_true")
    parser.add_argument("--no_fusion_adv", action="store_true")
    parser.add_argument("--no_variable_adv", action="store_true")
    parser.add_argument("--no_latent_adv", action="store_true")
    parser.add_argument("--no_prototype", action="store_true")
    parser.add_argument("--no_consistency", action="store_true")
    parser.add_argument(
        "--eval_mode",
        choices=["sample", "denoise", "teacher"],
        default="sample",
        help="sample is target-free completion; denoise is deterministic one-step diagnostic; teacher is old stochastic one-step.",
    )
    parser.add_argument("--eval_sampling", action="store_true", help="Deprecated alias for --eval_mode sample.")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N epochs within each stage.")
    parser.add_argument("--noise_condition", action="store_true", help="Also perturb condition latents during generative training.")
    parser.add_argument(
        "--spectral_norm_discriminators",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply spectral normalization to discriminator MLPs.",
    )
    parser.add_argument("--lr_scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--monitor_metric", type=str, default="macro_f1")
    parser.add_argument("--no_early_stop", action="store_true")

    parser.add_argument("--output_dir", type=Path, default=Path("runs/eeg_eye_adv"))
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end check.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def set_requires_grad(parameters, enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def parse_int_list(value: str) -> list[int] | None:
    if not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _arg(args: dict[str, Any], key: str, default: Any) -> Any:
    value = args.get(key, default)
    return default if value is None else value


def _unet_channels(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(ast.literal_eval(value))
    return tuple(value)


def load_frozen_teacher(checkpoint_path: Path, device: torch.device) -> AdversarialEEGEyeGenerator:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    meta = checkpoint["meta"]
    teacher = AdversarialEEGEyeGenerator(
        eeg_shape=tuple(meta["eeg_shape"]),
        eye_shape=tuple(meta["eye_shape"]),
        num_classes=int(meta["num_classes"]),
        latent_channels=int(_arg(ckpt_args, "latent_channels", 16)),
        latent_size=int(_arg(ckpt_args, "latent_size", 32)),
        hidden_dim=int(_arg(ckpt_args, "hidden_dim", 128)),
        heads=int(_arg(ckpt_args, "heads", 4)),
        timesteps=int(_arg(ckpt_args, "timesteps", 100)),
        sampling_steps=int(_arg(ckpt_args, "sampling_steps", 10)),
        unet_channels=_unet_channels(_arg(ckpt_args, "unet_channels", "[16, 32, 64, 128]")),
        unet_attention=_arg(ckpt_args, "unet_attention", "all"),
        sde_beta_min=float(_arg(ckpt_args, "sde_beta_min", 0.1)),
        sde_beta_max=float(_arg(ckpt_args, "sde_beta_max", 20.0)),
        noise_condition=bool(_arg(ckpt_args, "noise_condition", False)),
        spectral_norm_discriminators=bool(_arg(ckpt_args, "spectral_norm_discriminators", False)),
        generation_objective=_arg(ckpt_args, "generation_objective", "sde"),
        diffusion_sampler=_arg(ckpt_args, "diffusion_sampler", "sde"),
        ddim_eta=float(_arg(ckpt_args, "ddim_eta", 0.0)),
        fusion_type=_arg(ckpt_args, "fusion_type", "mlp"),
        fusion_layers=int(_arg(ckpt_args, "fusion_layers", 2)),
        fusion_dropout=float(_arg(ckpt_args, "fusion_dropout", 0.1)),
        uncertainty_samples=int(_arg(ckpt_args, "uncertainty_samples", 1)),
        uncertainty_temperature=float(_arg(ckpt_args, "uncertainty_temperature", 0.1)),
        min_generated_confidence=float(_arg(ckpt_args, "min_generated_confidence", 0.05)),
        self_conditioning_sample=bool(_arg(ckpt_args, "self_conditioning_sample", False)),
        self_conditioning_weight=float(_arg(ckpt_args, "self_conditioning_weight", 0.5)),
    ).to(device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def discriminator_updates_enabled(args: argparse.Namespace) -> bool:
    return not (args.no_modality_adv and args.no_fusion_adv and args.no_variable_adv and args.no_latent_adv)


class MeanMeter:
    def __init__(self):
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, values: dict[str, float]) -> None:
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def mean(self) -> dict[str, float]:
        return {key: value / max(1, self.counts.get(key, 0)) for key, value in self.totals.items()}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    f1s = []
    supports = []
    for cls in range(num_classes):
        tp = np.logical_and(y_true == cls, y_pred == cls).sum()
        fp = np.logical_and(y_true != cls, y_pred == cls).sum()
        fn = np.logical_and(y_true == cls, y_pred != cls).sum()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(float(f1))
        supports.append(int((y_true == cls).sum()))
    supports_arr = np.asarray(supports, dtype=np.float32)
    weighted = float(np.sum(np.asarray(f1s) * supports_arr) / max(1.0, supports_arr.sum()))
    return {"acc": accuracy, "macro_f1": float(np.mean(f1s)), "weighted_f1": weighted}


def masked_generation_metric(pred: torch.Tensor, target: torch.Tensor, missing: torch.Tensor) -> dict[str, float]:
    weights = missing.view(missing.shape[0], *([1] * (target.dim() - 1)))
    if float(weights.sum().detach().cpu()) < 1.0:
        weights = torch.ones_like(weights)
    diff = (pred - target) * weights
    mse = diff.pow(2).sum() / weights.sum().clamp_min(1.0) / max(1, target[0].numel())
    mae = diff.abs().sum() / weights.sum().clamp_min(1.0) / max(1, target[0].numel())
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    return {
        "mse": float(mse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "cosine": float(cosine.mean().detach().cpu()),
    }


def evaluate(
    model: AdversarialEEGEyeGenerator,
    loader,
    device: torch.device,
    num_classes: int,
    eval_mode: str,
) -> dict[str, float]:
    model.eval()
    labels = []
    preds = []
    eeg_metrics = MeanMeter()
    eye_metrics = MeanMeter()
    uncertainty_metrics = MeanMeter()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(batch, completion_mode=eval_mode)
            pred = outputs["logits_completed"].argmax(dim=-1)
            labels.append(batch["label"].detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            eeg_m = masked_generation_metric(outputs["gen_eeg"], batch["eeg"], 1.0 - batch["mask"][:, 0])
            eye_m = masked_generation_metric(outputs["gen_eye"], batch["eye"], 1.0 - batch["mask"][:, 1])
            eeg_metrics.update({f"eeg_{k}": v for k, v in eeg_m.items()})
            eye_metrics.update({f"eye_{k}": v for k, v in eye_m.items()})
            uncertainty_metrics.update(
                {
                    "eeg_confidence": float(outputs["eeg_confidence"].mean().detach().cpu()),
                    "eye_confidence": float(outputs["eye_confidence"].mean().detach().cpu()),
                    "eeg_uncertainty": float(outputs["eeg_uncertainty"].mean().detach().cpu()),
                    "eye_uncertainty": float(outputs["eye_uncertainty"].mean().detach().cpu()),
                }
            )

    y_true = np.concatenate(labels) if labels else np.asarray([], dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.asarray([], dtype=np.int64)
    result = classification_metrics(y_true, y_pred, num_classes)
    result.update(eeg_metrics.mean())
    result.update(eye_metrics.mean())
    result.update(uncertainty_metrics.mean())
    return result


def train_stage(
    stage: int,
    stage_epoch: int,
    stage_epochs: int,
    model: AdversarialEEGEyeGenerator,
    loader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    weights: LossWeights,
    args: argparse.Namespace,
    teacher_model: AdversarialEEGEyeGenerator | None = None,
) -> dict[str, float]:
    model.train()
    meter = MeanMeter()
    use_modality_adv = not args.no_modality_adv
    use_fusion_adv = not args.no_fusion_adv
    use_variable_adv = not args.no_variable_adv
    use_latent_adv = not args.no_latent_adv
    sample_start_fraction = min(1.0, max(0.0, float(args.sample_consistency_start_fraction)))
    sample_consistency_active = stage >= 3 and (stage_epoch / max(1, stage_epochs)) >= sample_start_fraction

    for step, batch in enumerate(loader, start=1):
        if args.max_batches and step > args.max_batches:
            break
        batch = move_to_device(batch, device)
        outputs = model(batch, sample=False)
        teacher_outputs = None
        if teacher_model is not None and stage >= 3:
            with torch.no_grad():
                teacher_outputs = teacher_model.full_modality_outputs(batch)

        if stage >= 2 and (use_modality_adv or use_fusion_adv or use_variable_adv or use_latent_adv):
            set_requires_grad(model.discriminator_parameters(), True)
            optimizer_d.zero_grad(set_to_none=True)
            disc_loss, disc_terms = model.discriminator_loss(
                batch,
                outputs,
                use_modality_adv=use_modality_adv,
                use_fusion_adv=use_fusion_adv,
                use_variable_adv=use_variable_adv,
                use_latent_adv=use_latent_adv,
            )
            disc_loss.backward()
            optimizer_d.step()
            meter.update(disc_terms)

        set_requires_grad(model.discriminator_parameters(), False)
        optimizer_g.zero_grad(set_to_none=True)
        gen_loss, gen_terms = model.generator_loss(
            batch,
            outputs,
            stage=stage,
            weights=weights,
            use_modality_adv=use_modality_adv,
            use_fusion_adv=use_fusion_adv,
            use_variable_adv=use_variable_adv,
            use_latent_adv=use_latent_adv,
            semantic_temperature=args.semantic_distill_temperature,
            missing_eeg_semantic_weight=args.missing_eeg_semantic_weight,
            missing_eye_semantic_weight=args.missing_eye_semantic_weight,
            teacher_outputs=teacher_outputs,
            supcon_temperature=args.supcon_temperature,
            proto_margin=args.proto_margin,
        )
        if (
            sample_consistency_active
            and (
                args.lambda_sample_cls > 0
                or args.lambda_sample_distill > 0
                or args.lambda_sample_fusion_distill > 0
                or args.lambda_sample_proto_distill > 0
                or args.lambda_sample_supcon > 0
                or args.lambda_sample_proto_margin > 0
            )
            and args.sample_consistency_interval > 0
            and step % args.sample_consistency_interval == 0
        ):
            sample_mode = (
                "sample_grad"
                if args.differentiable_sample_consistency and args.generation_objective == "flow"
                else "sample"
            )
            sample_outputs = model(
                batch,
                completion_mode=sample_mode,
                sample_steps=args.differentiable_sample_steps,
            )
            teacher_ref = teacher_outputs if teacher_outputs is not None else outputs
            semantic_weights = sample_semantic_weights(
                batch["mask"].float(),
                args.missing_eeg_semantic_weight,
                args.missing_eye_semantic_weight,
            )
            if args.lambda_sample_cls > 0:
                sample_classification = weighted_mean(
                    F.cross_entropy(sample_outputs["logits_completed"], batch["label"], reduction="none"),
                    semantic_weights,
                )
                gen_loss = gen_loss + args.lambda_sample_cls * sample_classification
                gen_terms.update(
                    {
                        "sample_classification": float(sample_classification.detach().cpu()),
                        "weight_sample_classification": float(args.lambda_sample_cls),
                    }
                )
            if args.lambda_sample_distill > 0:
                sample_distill = weighted_kl_from_logits(
                    sample_outputs["logits_completed"],
                    teacher_ref["logits_real"].detach(),
                    args.sample_distill_temperature,
                    semantic_weights,
                )
                gen_loss = gen_loss + args.lambda_sample_distill * sample_distill
                gen_terms.update(
                    {
                        "sample_distill": float(sample_distill.detach().cpu()),
                        "weight_sample_distill": float(args.lambda_sample_distill),
                    }
                )
            if args.lambda_sample_fusion_distill > 0:
                sample_fusion_cosine = F.cosine_similarity(
                    F.normalize(sample_outputs["fusion_completed"], dim=-1),
                    F.normalize(teacher_ref["fusion_real"].detach(), dim=-1),
                    dim=-1,
                )
                sample_fusion_distill = weighted_mean(1.0 - sample_fusion_cosine, semantic_weights)
                gen_loss = gen_loss + args.lambda_sample_fusion_distill * sample_fusion_distill
                gen_terms.update(
                    {
                        "sample_fusion_distill": float(sample_fusion_distill.detach().cpu()),
                        "weight_sample_fusion_distill": float(args.lambda_sample_fusion_distill),
                    }
                )
            if args.lambda_sample_proto_distill > 0:
                sample_proto_distill = weighted_kl_from_logits(
                    model.prototype_logits(sample_outputs["fusion_completed"]),
                    teacher_ref.get("prototype_logits_real", model.prototype_logits(outputs["fusion_real"])).detach(),
                    args.sample_distill_temperature,
                    semantic_weights,
                )
                gen_loss = gen_loss + args.lambda_sample_proto_distill * sample_proto_distill
                gen_terms.update(
                    {
                        "sample_proto_distill": float(sample_proto_distill.detach().cpu()),
                        "weight_sample_proto_distill": float(args.lambda_sample_proto_distill),
                    }
                )
            if args.lambda_sample_supcon > 0:
                sample_supcon = model.supervised_contrastive_loss(
                    sample_outputs["fusion_completed"],
                    batch["label"],
                    temperature=args.supcon_temperature,
                    weights=semantic_weights,
                )
                gen_loss = gen_loss + args.lambda_sample_supcon * sample_supcon
                gen_terms.update(
                    {
                        "sample_supervised_contrastive": float(sample_supcon.detach().cpu()),
                        "weight_sample_supervised_contrastive": float(args.lambda_sample_supcon),
                    }
                )
            if args.lambda_sample_proto_margin > 0:
                sample_proto_margin = model.prototype_margin_loss(
                    sample_outputs["fusion_completed"],
                    batch["label"],
                    margin=args.proto_margin,
                    weights=semantic_weights,
                )
                gen_loss = gen_loss + args.lambda_sample_proto_margin * sample_proto_margin
                gen_terms.update(
                    {
                        "sample_prototype_margin": float(sample_proto_margin.detach().cpu()),
                        "weight_sample_prototype_margin": float(args.lambda_sample_proto_margin),
                    }
                )
        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.generator_parameters(), 5.0)
        optimizer_g.step()
        set_requires_grad(model.discriminator_parameters(), True)
        meter.update(gen_terms)

    return meter.mean()


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))


def sample_semantic_weights(
    mask: torch.Tensor,
    missing_eeg_weight: float,
    missing_eye_weight: float,
) -> torch.Tensor:
    missing_eeg = 1.0 - mask[:, 0]
    missing_eye = 1.0 - mask[:, 1]
    weights = torch.ones_like(missing_eeg)
    weights = weights + missing_eeg * (float(missing_eeg_weight) - 1.0)
    weights = weights + missing_eye * (float(missing_eye_weight) - 1.0)
    return weights.clamp_min(0.0)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def weighted_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    weights: torch.Tensor,
) -> torch.Tensor:
    temperature = max(1e-3, float(temperature))
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    per_sample = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return weighted_mean(per_sample, weights) * temperature * temperature


def _warmup_scale(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0, epoch) / warmup_epochs)


def effective_weights(base: LossWeights, stage: int, stage_epoch: int, args: argparse.Namespace) -> LossWeights:
    adv_scale = 1.0
    if stage == 2:
        adv_scale = _warmup_scale(stage_epoch, args.adv_warmup_epochs)
    consistency_scale = _warmup_scale(stage_epoch, args.consistency_warmup_epochs) if stage >= 3 else 0.0
    return LossWeights(
        diffusion=base.diffusion,
        reconstruction=base.reconstruction,
        autoencoding=base.autoencoding,
        adversarial=base.adversarial * adv_scale,
        classification=base.classification,
        prototype=base.prototype,
        consistency=base.consistency * consistency_scale,
        teacher_kd=base.teacher_kd * consistency_scale,
        teacher_fusion=base.teacher_fusion * consistency_scale,
        teacher_proto=base.teacher_proto * consistency_scale,
        supervised_contrastive=base.supervised_contrastive * consistency_scale,
        prototype_margin=base.prototype_margin * consistency_scale,
    )


def save_checkpoint(path: Path, model: AdversarialEEGEyeGenerator, meta: dict, args: argparse.Namespace, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    args_dict = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(
        {
            "model": model.state_dict(),
            "meta": meta,
            "args": args_dict,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.stage1_epochs = 1
        args.stage2_epochs = 1
        args.stage3_epochs = 1
        args.max_batches = 2
        args.sampling_steps = min(args.sampling_steps, 2)
        train_limit = 128
        test_limit = 64
    else:
        train_limit = None
        test_limit = None

    if args.eval_sampling:
        args.eval_mode = "sample"

    if args.no_prototype:
        args.lambda_proto = 0.0
    if args.no_consistency:
        args.lambda_consistency = 0.0
    if args.no_modality_adv and args.no_fusion_adv and args.no_variable_adv and args.no_latent_adv:
        args.lambda_adv = 0.0

    normalize = None
    if args.normalize:
        normalize = True
    if args.no_normalize:
        normalize = False

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, meta = build_emotion_dataloaders(
        data_mode=args.data_mode,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        missing_mode=args.missing_mode,
        missing_rate=args.missing_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        normalize=normalize,
        train_limit=train_limit,
        test_limit=test_limit,
        split_protocol=args.split_protocol,
        test_ratio=args.test_ratio,
        test_subjects=parse_int_list(args.test_subjects),
        test_sessions=parse_int_list(args.test_sessions),
    )

    unet_channels = tuple(ast.literal_eval(args.unet_channels))
    if unet_channels[0] != args.latent_channels:
        raise ValueError(
            "The current diffusion U-Net requires unet_channels[0] to equal latent_channels. "
            f"Got latent_channels={args.latent_channels}, unet_channels={unet_channels}."
        )

    model = AdversarialEEGEyeGenerator(
        eeg_shape=tuple(meta["eeg_shape"]),
        eye_shape=tuple(meta["eye_shape"]),
        num_classes=int(meta["num_classes"]),
        latent_channels=args.latent_channels,
        latent_size=args.latent_size,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        timesteps=args.timesteps,
        sampling_steps=args.sampling_steps,
        unet_channels=unet_channels,
        unet_attention=args.unet_attention,
        sde_beta_min=args.sde_beta_min,
        sde_beta_max=args.sde_beta_max,
        noise_condition=args.noise_condition,
        spectral_norm_discriminators=args.spectral_norm_discriminators,
        generation_objective=args.generation_objective,
        diffusion_sampler=args.diffusion_sampler,
        ddim_eta=args.ddim_eta,
        fusion_type=args.fusion_type,
        fusion_layers=args.fusion_layers,
        fusion_dropout=args.fusion_dropout,
        uncertainty_samples=args.uncertainty_samples,
        uncertainty_temperature=args.uncertainty_temperature,
        min_generated_confidence=args.min_generated_confidence,
        self_conditioning_sample=args.self_conditioning_sample,
        self_conditioning_weight=args.self_conditioning_weight,
    ).to(device)

    teacher_model = None
    if args.teacher_checkpoint is not None:
        teacher_model = load_frozen_teacher(args.teacher_checkpoint, device)

    base_weights = LossWeights(
        diffusion=args.lambda_diffusion,
        reconstruction=args.lambda_reconstruction,
        autoencoding=args.lambda_autoencoding,
        adversarial=args.lambda_adv,
        classification=args.lambda_cls,
        prototype=args.lambda_proto,
        consistency=args.lambda_consistency,
        teacher_kd=args.lambda_teacher_kd,
        teacher_fusion=args.lambda_teacher_fusion,
        teacher_proto=args.lambda_teacher_proto,
        supervised_contrastive=args.lambda_supcon,
        prototype_margin=args.lambda_proto_margin,
    )
    optimizer_g = torch.optim.AdamW(model.generator_parameters(), lr=args.lr_g, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.AdamW(model.discriminator_parameters(), lr=args.lr_d, weight_decay=args.weight_decay)
    total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs
    scheduler_g = None
    scheduler_d = None
    if args.lr_scheduler == "cosine" and total_epochs > 0:
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=total_epochs)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=total_epochs)

    print(json.dumps({k: v for k, v in meta.items() if k != "stats"}, indent=2))
    best_score = -math.inf
    best_record: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    global_epoch = 0
    epochs_without_improvement = 0
    stop_training = False
    eval_metrics: dict[str, float] = {}
    stage_epochs = [(1, args.stage1_epochs), (2, args.stage2_epochs), (3, args.stage3_epochs)]
    for stage, epochs in stage_epochs:
        for epoch in range(1, epochs + 1):
            global_epoch += 1
            weights = effective_weights(base_weights, stage, epoch, args)
            train_metrics = train_stage(
                stage,
                epoch,
                epochs,
                model,
                train_loader,
                optimizer_g,
                optimizer_d,
                device,
                weights,
                args,
                teacher_model=teacher_model,
            )
            train_metrics["lr_g"] = optimizer_g.param_groups[0]["lr"]
            train_metrics["lr_d"] = optimizer_d.param_groups[0]["lr"]
            should_evaluate = args.eval_interval <= 1 or epoch == epochs or epoch % args.eval_interval == 0
            eval_metrics = (
                evaluate(
                    model,
                    test_loader,
                    device,
                    int(meta["num_classes"]),
                    eval_mode=args.eval_mode,
                )
                if should_evaluate
                else {}
            )
            record = {
                "stage": stage,
                "epoch": epoch,
                "global_epoch": global_epoch,
                "train": train_metrics,
                "eval": eval_metrics,
            }
            history.append(record)
            print(
                f"stage={stage} epoch={epoch}/{epochs} "
                f"train[{format_metrics(train_metrics)}] eval[{format_metrics(eval_metrics)}]"
            )
            if stage == 3 and eval_metrics:
                if args.monitor_metric not in eval_metrics:
                    raise KeyError(f"Monitor metric {args.monitor_metric!r} is not in eval metrics.")
                score = float(eval_metrics[args.monitor_metric])
                if score > best_score + args.early_stop_min_delta:
                    best_score = score
                    best_record = record
                    epochs_without_improvement = 0
                    if not args.no_save:
                        save_checkpoint(args.output_dir / "best_model.pth", model, meta, args, eval_metrics)
                else:
                    epochs_without_improvement += 1
                    if (
                        not args.no_early_stop
                        and args.early_stop_patience > 0
                        and epochs_without_improvement >= args.early_stop_patience
                    ):
                        print(
                            f"early_stop monitor={args.monitor_metric} best={best_score:.4f} "
                            f"patience={args.early_stop_patience}"
                        )
                        stop_training = True
            if scheduler_g is not None:
                scheduler_g.step()
            if scheduler_d is not None and stage >= 2 and discriminator_updates_enabled(args):
                scheduler_d.step()
            if stop_training:
                break
        if stop_training:
            break

    if not args.no_save:
        save_checkpoint(args.output_dir / "last_model.pth", model, meta, args, eval_metrics)
    save_run_artifacts(args.output_dir, args, meta, history, best_record)


if __name__ == "__main__":
    main()
