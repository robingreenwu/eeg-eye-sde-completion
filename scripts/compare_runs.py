from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare EEG-Eye completion training runs.")
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Run directories containing summary.json.")
    parser.add_argument("--output_dir", type=Path, default=Path("runs/comparison"))
    return parser.parse_args()


def load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def row_from_summary(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    args = summary.get("args", {})
    record = summary.get("best") or summary.get("final") or {}
    metrics = record.get("eval", {})
    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "data_mode": args.get("data_mode"),
        "missing_mode": args.get("missing_mode"),
        "missing_rate": args.get("missing_rate"),
        "seed": args.get("seed"),
        "best_stage": record.get("stage"),
        "best_epoch": record.get("epoch"),
        "best_global_epoch": record.get("global_epoch"),
        "acc": metrics.get("acc"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "eeg_mse": metrics.get("eeg_mse"),
        "eye_mse": metrics.get("eye_mse"),
        "eeg_mae": metrics.get("eeg_mae"),
        "eye_mae": metrics.get("eye_mae"),
        "eeg_cosine": metrics.get("eeg_cosine"),
        "eye_cosine": metrics.get("eye_cosine"),
    }


def write_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


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


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        values.append(float(value) if isinstance(value, (int, float)) else 0.0)
    return values


def plot_comparison(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib(output_dir)
    names = [str(row["run"]) for row in rows]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.3), 5), dpi=180)
    width = 0.25
    for offset, key in [(-width, "acc"), (0.0, "macro_f1"), (width, "weighted_f1")]:
        ax.bar([i + offset for i in x], _values(rows, key), width=width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Emotion Recognition Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "recognition_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(rows) * 1.4), 4.5), dpi=180)
    for ax, keys, title, ylabel in [
        (axes[0], ["eeg_mse", "eye_mse"], "Generation Error", "MSE"),
        (axes[1], ["eeg_cosine", "eye_cosine"], "Generation Similarity", "Cosine"),
    ]:
        width = 0.35
        for offset, key in [(-width / 2, keys[0]), (width / 2, keys[1])]:
            ax.bar([i + offset for i in x], _values(rows, key), width=width, label=key)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "generation_comparison.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = [row_from_summary(run_dir, load_summary(run_dir)) for run_dir in args.run_dirs]
    write_table(rows, args.output_dir)
    plot_comparison(rows, args.output_dir)
    print(f"Saved comparison artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
