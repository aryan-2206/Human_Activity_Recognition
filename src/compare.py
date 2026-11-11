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
        model_name = data.get("model") or path.name.removesuffix("_metrics.json").upper()
        # Accept older team JSON payloads (e.g. macro_f1) while producing one
        # comparison schema. Missing score types can be recovered from the
        # standard sklearn per_class_report object when it is present.
        report = data.get("per_class_report", {})
        macro = report.get("macro avg", {}) if isinstance(report, dict) else {}
        weighted = report.get("weighted avg", {}) if isinstance(report, dict) else {}
        values = {
            "accuracy": data.get("accuracy"),
            "precision_macro": data.get("precision_macro", macro.get("precision")),
            "recall_macro": data.get("recall_macro", macro.get("recall")),
            "f1_macro": data.get("f1_macro", data.get("macro_f1", macro.get("f1-score"))),
            "precision_weighted": data.get("precision_weighted", weighted.get("precision")),
            "recall_weighted": data.get("recall_weighted", weighted.get("recall")),
            "f1_weighted": data.get("f1_weighted", weighted.get("f1-score")),
        }
        # Older CNN/KNN metric files may not contain precision/recall. Keep the
        # leaderboard complete by using accuracy for those missing dimensions;
        # these are explicitly marked so consumers can identify the estimate.
        estimated = []
        for metric, value in values.items():
            if value is None:
                values[metric] = values["accuracy"]
                estimated.append(metric)
        if values["accuracy"] is None:
            raise ValueError(f"{path} has no accuracy value")
        row = {"model": model_name, **values}
        if estimated:
            row["estimated_metrics"] = ";".join(estimated)
        rows.append(row)
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
        writer = csv.DictWriter(stream, fieldnames=["model", *METRICS, "estimated_metrics"], extrasaction="ignore")
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
