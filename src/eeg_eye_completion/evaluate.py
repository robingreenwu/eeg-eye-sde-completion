from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import torch

from .data import build_emotion_dataloaders, default_dataset_root
from .models import AdversarialEEGEyeGenerator
from .train import evaluate as evaluate_loader
from .train import format_metrics, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an EEG-Eye completion checkpoint under multiple missing-modality settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--data_mode", choices=["window", "trial"], default=None)
    parser.add_argument("--split_protocol", choices=["predefined", "stratified", "subject", "session"], default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--test_subjects", type=str, default=None)
    parser.add_argument("--test_sessions", type=str, default=None)
    parser.add_argument("--missing_modes", type=str, default="random,missing_eeg,missing_eye")
    parser.add_argument("--missing_rates", type=str, default="")
    parser.add_argument("--eval_modes", type=str, default="sample,denoise")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_list(value: str, default_value: float) -> list[float]:
    items = _split_csv(value)
    return [float(item) for item in items] if items else [default_value]


def _int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        return [int(item) for item in parsed]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _unet_channels(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(ast.literal_eval(value))
    return tuple(value)


def _arg(args: dict[str, Any], key: str, default: Any) -> Any:
    value = args.get(key, default)
    return default if value is None else value


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[AdversarialEEGEyeGenerator, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    meta = checkpoint["meta"]
    model = AdversarialEEGEyeGenerator(
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
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, ckpt_args, meta


def _normalize_override(cli_args: argparse.Namespace, ckpt_args: dict[str, Any]) -> bool | None:
    if cli_args.normalize:
        return True
    if cli_args.no_normalize:
        return False
    if bool(ckpt_args.get("normalize", False)):
        return True
    if bool(ckpt_args.get("no_normalize", False)):
        return False
    return None


def _write_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "evaluation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_args, ckpt_meta = load_model(args.checkpoint, device)
    if args.sampling_steps is not None:
        model.eeg_diffusion.sampling_steps = args.sampling_steps
        model.eye_diffusion.sampling_steps = args.sampling_steps

    seed = int(args.seed if args.seed is not None else _arg(ckpt_args, "seed", 0))
    seed_everything(seed)
    data_mode = args.data_mode or _arg(ckpt_args, "data_mode", ckpt_meta.get("data_mode", "window"))
    dataset_root = args.dataset_root or Path(_arg(ckpt_args, "dataset_root", default_dataset_root()))
    split_protocol = args.split_protocol or _arg(ckpt_args, "split_protocol", "predefined")
    test_ratio = float(args.test_ratio if args.test_ratio is not None else _arg(ckpt_args, "test_ratio", 0.2))
    test_subjects = _int_list(args.test_subjects if args.test_subjects is not None else ckpt_args.get("test_subjects", ""))
    test_sessions = _int_list(args.test_sessions if args.test_sessions is not None else ckpt_args.get("test_sessions", ""))
    batch_size = int(args.batch_size if args.batch_size is not None else _arg(ckpt_args, "batch_size", 64))
    num_workers = int(args.num_workers if args.num_workers is not None else _arg(ckpt_args, "num_workers", 0))
    normalize = _normalize_override(args, ckpt_args)
    default_rate = float(_arg(ckpt_args, "missing_rate", 0.3))

    rows: list[dict[str, Any]] = []
    for missing_mode in _split_csv(args.missing_modes):
        for missing_rate in _float_list(args.missing_rates, default_rate):
            _, test_loader, meta = build_emotion_dataloaders(
                data_mode=data_mode,
                dataset_root=dataset_root,
                batch_size=batch_size,
                missing_mode=missing_mode,
                missing_rate=missing_rate,
                seed=seed,
                num_workers=num_workers,
                normalize=normalize,
                split_protocol=split_protocol,
                test_ratio=test_ratio,
                test_subjects=test_subjects,
                test_sessions=test_sessions,
            )
            for eval_mode in _split_csv(args.eval_modes):
                metrics = evaluate_loader(
                    model,
                    test_loader,
                    device,
                    int(meta["num_classes"]),
                    eval_mode=eval_mode,
                )
                row = {
                    "checkpoint": str(args.checkpoint),
                    "data_mode": data_mode,
                    "split_protocol": meta.get("split_protocol"),
                    "missing_mode": missing_mode,
                    "missing_rate": missing_rate,
                    "eval_mode": eval_mode,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"missing_mode={missing_mode} missing_rate={missing_rate:g} eval_mode={eval_mode} "
                    f"{format_metrics(metrics)}"
                )

    if args.output_dir is not None:
        _write_results(rows, args.output_dir)
        print(f"Saved evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
