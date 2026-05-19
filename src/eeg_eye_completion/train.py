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
        description="Adversarial diffusion completion for EEG-Eye multimodal emotion recognition.",
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
    parser.add_argument("--sde_beta_min", type=float, default=0.1)
    parser.add_argument("--sde_beta_max", type=float, default=20.0)
    parser.add_argument("--unet_channels", type=str, default="[16, 32, 64, 128]")
    parser.add_argument(
        "--unet_attention",
        type=str,
        default="critical",
        help="U-Net attention preset/layers: all, sampling, critical, bottleneck, none, or comma-separated layers.",
    )

    parser.add_argument("--lambda_diffusion", type=float, default=1.0)
    parser.add_argument("--lambda_reconstruction", type=float, default=1.0)
    parser.add_argument("--lambda_autoencoding", type=float, default=0.2)
    parser.add_argument("--lambda_adv", type=float, default=0.05)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_proto", type=float, default=0.2)
    parser.add_argument("--lambda_consistency", type=float, default=0.05)
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
    parser.add_argument("--noise_condition", action="store_true", help="Also perturb condition latents during diffusion training.")
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


def discriminator_updates_enabled(args: argparse.Namespace) -> bool:
    return not (args.no_modality_adv and args.no_fusion_adv and args.no_variable_adv and args.no_latent_adv)


class MeanMeter:
    def __init__(self):
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, values: dict[str, float]) -> None:
        self.count += 1
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)

    def mean(self) -> dict[str, float]:
        return {key: value / max(1, self.count) for key, value in self.totals.items()}


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

    y_true = np.concatenate(labels) if labels else np.asarray([], dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.asarray([], dtype=np.int64)
    result = classification_metrics(y_true, y_pred, num_classes)
    result.update(eeg_metrics.mean())
    result.update(eye_metrics.mean())
    return result


def train_stage(
    stage: int,
    model: AdversarialEEGEyeGenerator,
    loader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    weights: LossWeights,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    meter = MeanMeter()
    use_modality_adv = not args.no_modality_adv
    use_fusion_adv = not args.no_fusion_adv
    use_variable_adv = not args.no_variable_adv
    use_latent_adv = not args.no_latent_adv

    for step, batch in enumerate(loader, start=1):
        if args.max_batches and step > args.max_batches:
            break
        batch = move_to_device(batch, device)
        outputs = model(batch, sample=False)

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
        )
        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.generator_parameters(), 5.0)
        optimizer_g.step()
        set_requires_grad(model.discriminator_parameters(), True)
        meter.update(gen_terms)

    return meter.mean()


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))


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
    ).to(device)

    base_weights = LossWeights(
        diffusion=args.lambda_diffusion,
        reconstruction=args.lambda_reconstruction,
        autoencoding=args.lambda_autoencoding,
        adversarial=args.lambda_adv,
        classification=args.lambda_cls,
        prototype=args.lambda_proto,
        consistency=args.lambda_consistency,
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
            train_metrics = train_stage(stage, model, train_loader, optimizer_g, optimizer_d, device, weights, args)
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
