from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                flat[f"{key}.{child_key}"] = child_value
        else:
            flat[key] = value
    return flat


def save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "metrics.jsonl"
    csv_path = output_dir / "metrics.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in history:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    flat_records = [_flatten_record(record) for record in history]
    fieldnames: list[str] = []
    for record in flat_records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_records)


def save_summary(
    output_dir: Path,
    args: Any,
    meta: dict[str, Any],
    history: list[dict[str, Any]],
    best_record: dict[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best": best_record,
        "final": history[-1] if history else None,
        "args": vars(args),
        "meta": {k: v for k, v in meta.items() if k != "stats"},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)


def _prepare_matplotlib(output_dir: Path):
    mpl_dir = output_dir / ".mplconfig"
    cache_dir = output_dir / ".cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _series(history: list[dict[str, Any]], key: str) -> list[float | None]:
    values: list[float | None] = []
    for record in history:
        cursor: Any = record
        for part in key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        values.append(float(cursor) if isinstance(cursor, (int, float)) else None)
    return values


def _plot_lines(plt, x: list[int], history: list[dict[str, Any]], keys: list[str], title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    plotted = False
    for key in keys:
        values = _series(history, key)
        if any(value is not None for value in values):
            ax.plot(x, values, marker="o", linewidth=1.8, markersize=3.5, label=key)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("Global epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_training_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    if not history:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib(output_dir)

    x = [int(record.get("global_epoch", index + 1)) for index, record in enumerate(history)]
    _plot_lines(
        plt,
        x,
        history,
        ["eval.acc", "eval.macro_f1", "eval.weighted_f1"],
        "Emotion Recognition Metrics",
        "Score",
        plot_dir / "classification_metrics.png",
    )
    _plot_lines(
        plt,
        x,
        history,
        [
            "train.total",
            "train.diffusion",
            "train.reconstruction",
            "train.classification",
            "train.prototype",
            "train.consistency",
        ],
        "Training Objective Curves",
        "Loss",
        plot_dir / "training_losses.png",
    )
    _plot_lines(
        plt,
        x,
        history,
        ["eval.eeg_mse", "eval.eye_mse", "eval.eeg_mae", "eval.eye_mae"],
        "Generated Modality Error",
        "Error",
        plot_dir / "generation_error.png",
    )
    _plot_lines(
        plt,
        x,
        history,
        ["eval.eeg_cosine", "eval.eye_cosine"],
        "Generated Modality Cosine Similarity",
        "Cosine similarity",
        plot_dir / "generation_cosine.png",
    )


def save_run_artifacts(
    output_dir: Path,
    args: Any,
    meta: dict[str, Any],
    history: list[dict[str, Any]],
    best_record: dict[str, Any] | None,
) -> None:
    save_history(history, output_dir)
    save_summary(output_dir, args, meta, history, best_record)
    try:
        plot_training_history(history, output_dir)
    except Exception as exc:  # pragma: no cover - visualization should not break training.
        with (output_dir / "plot_error.txt").open("w", encoding="utf-8") as f:
            f.write(f"Failed to generate plots: {exc}\n")
