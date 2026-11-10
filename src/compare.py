"""Combine per-model metrics JSON files into comparison CSV and charts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
METRICS = (
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
)


def collect_metrics(results_dir: str | Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Read all model metrics files and validate the shared contract."""
    folder = Path(results_dir)
    rows = []
    for path in sorted(folder.glob("*_metrics.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read metrics file {path}: {exc}") from exc
        missing = [key for key in ("model", *METRICS) if key not in data]
        if missing:
            raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
        rows.append({"model": data["model"], **{metric: data[metric] for metric in METRICS}})
    if not rows:
        raise FileNotFoundError(f"No *_metrics.json files found in {folder}")
    return rows


def create_comparison(results_dir: str | Path = RESULTS_DIR) -> tuple[Path, Path]:
    """Write comparison.csv and comparison_metrics.png; return their paths."""
    from matplotlib import pyplot as plt

    folder = Path(results_dir)
    rows = collect_metrics(folder)
    csv_path = folder / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    # Compact grouped chart of all key scores, on a consistent 0-1 scale.
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.5), 5))
    x = list(range(len(rows)))
    width = 0.11
    for i, metric in enumerate(METRICS):
        offsets = [value + (i - (len(METRICS) - 1) / 2) * width for value in x]
        ax.bar(offsets, [float(row[metric]) for row in rows], width, label=metric.replace("_", " "))
    ax.set_xticks(x, [str(row["model"]) for row in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Human Activity Recognition model comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize="small", loc="lower right")
    fig.tight_layout()
    chart_path = folder / "comparison_metrics.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return csv_path, chart_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    csv_path, chart_path = create_comparison(args.results_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {chart_path}")


if __name__ == "__main__":
    main()
