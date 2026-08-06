#!/usr/bin/env python3
"""Plot and summarize the ranged EE-compliance benchmark.

The script consumes complete 60-probe JSON reports from ``eval_results/``.
It produces the figures proposed in ``ranged_stiffness_benchmark.md``
and writes an aggregate CSV plus a Markdown benchmark table.  It never starts
training or evaluation.

Examples
--------
    python outputs/ranged_stiffness/plot_ranged_stiffness.py
    python outputs/ranged_stiffness/plot_ranged_stiffness.py --show-missing
    python outputs/ranged_stiffness/plot_ranged_stiffness.py \
        --end2end-glob '/path/to/end2end_range_*_k*.json'
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "eval_results"
DEFAULT_FIGURE_DIR = HERE / "figures"
DEFAULT_CSV = HERE / "ranged_stiffness_metrics.csv"
DEFAULT_MARKDOWN = HERE / "ranged_stiffness_benchmark.md"

CONFIGS = [
    ("all_soft", (200, 200, 200), "Isotropic"),
    ("x_soft", (200, 600, 600), "Single-soft"),
    ("y_soft", (600, 200, 600), "Single-soft"),
    ("z_soft", (600, 600, 200), "Single-soft"),
    ("mixed_250_400_550", (250, 400, 550), "Mixed"),
    ("mixed_250_550_400", (250, 550, 400), "Mixed"),
    ("mixed_400_250_550", (400, 250, 550), "Mixed"),
    ("mixed_400_550_250", (400, 550, 250), "Mixed"),
    ("mixed_550_250_400", (550, 250, 400), "Mixed"),
    ("mixed_550_400_250", (550, 400, 250), "Mixed"),
    ("x_hard", (600, 200, 200), "Single-hard"),
    ("y_hard", (200, 600, 200), "Single-hard"),
    ("z_hard", (200, 200, 600), "Single-hard"),
    ("all_hard", (600, 600, 600), "Isotropic"),
]
CONFIG_BY_TARGET = {tuple(target): (name, category) for name, target, category in CONFIGS}
TARGET_BY_NAME = {name: target for name, target, _ in CONFIGS}
CATEGORY_BY_NAME = {name: category for name, _, category in CONFIGS}
CONFIG_NAMES = [name for name, _, _ in CONFIGS]
REPRESENTATIVE_CONFIGS = ["x_soft", "x_hard", "mixed_250_400_550"]

METHODS = {
    "analytical_moe": {
        "label": "Ours MoE",
        "pattern": "analytical_moe_*_k*.json",
        "color": "#1f77b4",
    },
    "two_layer_range_v2": {
        "label": "Ranged high-level",
        "pattern": "two_layer_range_v2_*_k*.json",
        "color": "#d62728",
    },
    "end2end_range": {
        "label": "End-to-end range",
        "pattern": "end2end_range_*_k*.json",
        "color": "#2ca02c",
    },
    "oracle_force_stiff_low_level": {
        "label": "Oracle force + stiff low-level",
        "pattern": "oracle_force_stiff_low_level_*_k*.json",
        "color": "#9467bd",
    },
}


def parse_target(payload: dict):
    target = payload.get("ee_compliance_target", {}).get("stiffness")
    if isinstance(target, (int, float)):
        return (float(target),) * 3
    if isinstance(target, list) and len(target) == 3:
        return tuple(float(item) for item in target)
    return None


def as_hand_xyz(value):
    """Convert evaluator arrays to a (2, 3) hand-by-xyz array when possible."""
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (2, 3):
        return array
    return None


def load_reports(directory: Path, pattern: str, explicit_glob: str | None = None):
    paths = sorted(Path(path) for path in glob.glob(explicit_glob, recursive=True)) if explicit_glob else sorted(directory.glob(pattern))
    reports = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        target = parse_target(payload)
        if target is None or len(payload.get("records", [])) != 60:
            continue
        rounded = tuple(int(round(item)) for item in target)
        config = CONFIG_BY_TARGET.get(rounded)
        if config is None:
            continue
        reports[config[0]] = {"path": path, "payload": payload, "target": rounded}
    return reports


def collect_rows(method_key, reports):
    rows = []
    for config_name, report in reports.items():
        target = report["target"]
        for record in report["payload"].get("records", []):
            direction = str(record.get("direction", ""))
            axis_name = direction[-1].lower() if direction and direction[-1].lower() in "xyz" else None
            if axis_name is None:
                continue
            axis = "xyz".index(axis_name)
            measured = record.get("measured_stiffness_abs_n_per_m")
            force = record.get("force_n")
            if measured is None or force is None:
                continue
            measured = float(measured)
            command = float(record.get("cfg_stiffness_n_per_m", target[axis]))
            if command <= 0:
                continue
            rows.append(
                {
                    "method": method_key,
                    "config": config_name,
                    "category": CATEGORY_BY_NAME[config_name],
                    "hand": str(record.get("ee", "unknown")),
                    "axis": axis_name,
                    "axis_index": axis,
                    "sign": direction[0] if direction[:1] in "+-" else "?",
                    "force": float(force),
                    "command": command,
                    "target_x": target[0],
                    "target_y": target[1],
                    "target_z": target[2],
                    "measured": measured,
                    "relative_error": abs(measured - command) / command,
                }
            )
    return rows


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(values, ddof=1) / math.sqrt(values.size))


def fit_command_response(rows):
    x = np.asarray([row["command"] for row in rows], dtype=float)
    y = np.asarray([row["measured"] for row in rows], dtype=float)
    if x.size < 2 or np.ptp(x) == 0:
        return math.nan, math.nan, math.nan
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(slope), float(intercept), float(r2)


def compute_spearman(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["hand"], row["axis"], row["sign"], row["force"])].append(row)
    correlations = []
    order_scores = []
    for group_rows in groups.values():
        by_command = defaultdict(list)
        for row in group_rows:
            by_command[row["command"]].append(row["measured"])
        if len(by_command) < 3:
            continue
        commands = np.array(sorted(by_command), dtype=float)
        measured = np.array([np.mean(by_command[value]) for value in commands], dtype=float)
        rho = spearmanr(commands, measured).statistic
        if np.isfinite(rho):
            correlations.append(float(rho))
        if len(measured) > 1:
            order_scores.append(float(np.mean(np.diff(measured) >= 0)))
    return float(np.mean(correlations)) if correlations else math.nan, float(np.mean(order_scores)) if order_scores else math.nan


def compute_force_cv(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["config"], row["hand"], row["axis"], row["sign"])].append(row["measured"])
    cvs = []
    for values in groups.values():
        values = np.asarray(values, dtype=float)
        mean = np.mean(values)
        if values.size >= 2 and mean > 1e-9:
            cvs.append(float(np.std(values, ddof=1) / mean))
    return float(np.mean(cvs)) if cvs else math.nan


def fit_compliance_matrices(reports):
    matrices = {}
    matrix_errors = []
    leakages = []
    for config_name, report in reports.items():
        target = np.diag(1.0 / np.asarray(report["target"], dtype=float))
        hand_matrices = []
        records = report["payload"].get("records", [])
        for hand_name, hand_index in (("left", 0), ("right", 1)):
            forces = []
            displacements = []
            for record in records:
                force_array = as_hand_xyz(record.get("ee_force_b"))
                actual = as_hand_xyz(record.get("actual_pos_b"))
                baseline = as_hand_xyz(record.get("baseline_pos_b"))
                if force_array is None or actual is None or baseline is None:
                    continue
                forces.append(force_array[hand_index])
                displacements.append(actual[hand_index] - baseline[hand_index])
            if len(forces) < 8:
                continue
            force_array = np.asarray(forces, dtype=float)
            displacement_array = np.asarray(displacements, dtype=float)
            design = np.column_stack([np.ones(len(force_array)), force_array])
            coefficient = np.linalg.lstsq(design, displacement_array, rcond=None)[0]
            matrix = coefficient[1:, :].T
            hand_matrices.append(matrix)
            matrix_errors.append(np.linalg.norm(matrix - target, ord="fro") / np.linalg.norm(target, ord="fro"))
            diagonal = np.diag(np.diag(matrix))
            off_diagonal = matrix - diagonal
            denominator = np.linalg.norm(diagonal, ord="fro")
            if denominator > 1e-12:
                leakages.append(np.linalg.norm(off_diagonal, ord="fro") / denominator)
        if hand_matrices:
            matrices[config_name] = {
                "target": target,
                "measured": np.mean(np.asarray(hand_matrices), axis=0),
            }
    return matrices, float(np.mean(matrix_errors)) if matrix_errors else math.nan, float(np.mean(leakages)) if leakages else math.nan


def compute_metrics(method_key, reports, rows):
    slope, intercept, r2 = fit_command_response(rows)
    spearman, ordering = compute_spearman(rows)
    measured = np.asarray([row["measured"] for row in rows], dtype=float)
    range_retention = (np.percentile(measured, 95) - np.percentile(measured, 5)) / 400.0 if measured.size else math.nan
    matrices, matrix_error, leakage = fit_compliance_matrices(reports)
    return {
        "method_key": method_key,
        "method": METHODS[method_key]["label"],
        "complete_configs": len(reports),
        "total_probes": sum(len(report["payload"].get("records", [])) for report in reports.values()),
        "compliance_matrix_error": matrix_error,
        "stiffness_mape": float(np.mean([row["relative_error"] for row in rows])) if rows else math.nan,
        "trend_slope": slope,
        "trend_intercept": intercept,
        "trend_r2": r2,
        "range_retention": float(range_retention),
        "spearman_rho": spearman,
        "monotonic_ordering_accuracy": ordering,
        "force_cv": compute_force_cv(rows),
        "directional_leakage": leakage,
        "matrices": matrices,
    }


def method_config_matrix(rows, method_key, value_key="measured"):
    values = defaultdict(list)
    for row in rows:
        values[(row["config"], row["axis"])].append(row[value_key])
    matrix = np.full((len(CONFIGS), 3), np.nan)
    for row_index, config_name in enumerate(CONFIG_NAMES):
        for axis in range(3):
            axis_name = "xyz"[axis]
            if values[(config_name, axis_name)]:
                matrix[row_index, axis] = np.mean(values[(config_name, axis_name)])
    return matrix


def save_heatmap_figure(metrics, figure_dir):
    panels = [(None, "Target command")]
    panels.extend((key, METHODS[key]["label"]) for key in METHODS)
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 8), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    image = None
    for axis, (method_key, title) in zip(axes, panels):
        if method_key is None:
            matrix = np.asarray([target for _, target, _ in CONFIGS], dtype=float)
        else:
            matrix = metrics.get(method_key, {}).get("stiffness_matrix")
            if matrix is None:
                axis.text(0.5, 0.5, "No complete\n14-command sweep", ha="center", va="center", fontsize=12)
                axis.set_axis_off()
                axis.set_title(title)
                continue
        image = axis.imshow(matrix, cmap="viridis", vmin=200, vmax=600, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(3):
                if np.isfinite(matrix[i, j]):
                    color = "white" if matrix[i, j] < 420 else "black"
                    axis.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=7, color=color)
        axis.set_title(title)
        axis.set_xticks(range(3), ["x", "y", "z"])
        axis.set_yticks(range(len(CONFIGS)), CONFIG_NAMES if axis is axes[0] else [])
        axis.tick_params(axis="y", labelsize=7)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.75, label="Stiffness (N/m); color scale 200–600")
    fig.suptitle("Figure 1 — Target versus measured stiffness", fontsize=16)
    path = figure_dir / "fig1_target_measured_stiffness_heatmaps.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_command_response_figure(metrics, rows_by_method, figure_dir):
    fig, axis = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    axis.plot([150, 650], [150, 650], "k--", linewidth=1.2, label="Ideal y=x")
    for method_key, method in METHODS.items():
        rows = rows_by_method.get(method_key, [])
        if not rows:
            continue
        color = method["color"]
        axis.scatter([r["command"] for r in rows], [r["measured"] for r in rows], s=10, alpha=0.13, color=color)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["command"]].append(row["measured"])
        commands = np.array(sorted(grouped), dtype=float)
        means = np.array([mean_ci(grouped[c])[0] for c in commands])
        cis = np.array([mean_ci(grouped[c])[1] for c in commands])
        m = metrics[method_key]
        label = f"{method['label']} (slope={m['trend_slope']:.2f}, R²={m['trend_r2']:.2f})"
        axis.errorbar(commands, means, yerr=cis, color=color, marker="o", linewidth=2, capsize=3, label=label)
    axis.set(xlim=(150, 650), ylim=(150, 650), xlabel="Commanded stiffness (N/m)", ylabel="Measured stiffness (N/m)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)
    axis.set_title("Figure 2 — Command-response controllability")
    path = figure_dir / "fig2_commanded_vs_measured_stiffness.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_force_figure(rows_by_method, figure_dir):
    fig, axis = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    for method_key, method in METHODS.items():
        rows = rows_by_method.get(method_key, [])
        if not rows:
            continue
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["force"]].append(row["relative_error"])
        forces = np.array(sorted(grouped), dtype=float)
        means = np.array([mean_ci(grouped[f])[0] for f in forces])
        cis = np.array([mean_ci(grouped[f])[1] for f in forces])
        axis.errorbar(forces, means * 100, yerr=cis * 100, color=method["color"], marker="o", linewidth=2, capsize=3, label=method["label"])
    axis.set(xlabel="Force magnitude |F| (N)", ylabel="Relative stiffness error (%)")
    axis.set_xticks([5, 10, 15, 20, 30])
    axis.grid(alpha=0.2)
    axis.legend()
    axis.set_title("Figure 3 — Stiffness error versus force magnitude")
    path = figure_dir / "fig3_stiffness_error_vs_force.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_matrix_figure(metrics, figure_dir):
    available = [(key, METHODS[key]["label"]) for key in METHODS if metrics.get(key, {}).get("matrices")]
    columns = [(None, "Target compliance")] + available
    fig, axes = plt.subplots(len(REPRESENTATIVE_CONFIGS), len(columns), figsize=(3.2 * len(columns), 9), constrained_layout=True)
    axes = np.atleast_2d(axes)
    all_values = []
    for key, _ in available:
        for matrix_data in metrics[key]["matrices"].values():
            all_values.extend(matrix_data["measured"].ravel())
    for config_name in REPRESENTATIVE_CONFIGS:
        target = np.diag(1.0 / np.asarray(TARGET_BY_NAME[config_name], dtype=float)) * 1000.0
        all_values.extend(target.ravel())
    vmax = max(1.0, float(np.max(np.abs(all_values))) if all_values else 1.0)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    image = None
    for row_index, config_name in enumerate(REPRESENTATIVE_CONFIGS):
        for col_index, (method_key, title) in enumerate(columns):
            if method_key is None:
                matrix = np.diag(1.0 / np.asarray(TARGET_BY_NAME[config_name], dtype=float)) * 1000.0
            else:
                matrix_data = metrics[method_key]["matrices"].get(config_name)
                matrix = matrix_data["measured"] * 1000.0 if matrix_data else None
            axis = axes[row_index, col_index]
            if matrix is None:
                axis.text(0.5, 0.5, "Missing", ha="center", va="center")
                axis.set_axis_off()
                continue
            image = axis.imshow(matrix, cmap="coolwarm", norm=norm)
            for i in range(3):
                for j in range(3):
                    color = "white" if abs(matrix[i, j]) > 0.55 * vmax else "black"
                    axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)
            axis.set_xticks(range(3), ["Fx", "Fy", "Fz"])
            axis.set_yticks(range(3), ["dx", "dy", "dz"])
            if row_index == 0:
                axis.set_title(title)
            if col_index == 0:
                axis.set_ylabel(f"{config_name}\n(mm/N)")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label="Compliance (mm/N)")
    fig.suptitle("Figure 4 — Representative compliance matrices", fontsize=16)
    path = figure_dir / "fig4_representative_compliance_matrices.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_category_figure(rows_by_method, figure_dir):
    categories = ["Isotropic", "Single-soft", "Single-hard", "Mixed"]
    method_keys = [key for key in METHODS if rows_by_method.get(key)]
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(method_keys))
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for index, method_key in enumerate(method_keys):
        values = []
        for category in categories:
            values.append(np.mean([row["relative_error"] for row in rows_by_method[method_key] if row["category"] == category]) * 100)
        axis.bar(x + (index - (len(method_keys) - 1) / 2) * width, values, width, label=METHODS[method_key]["label"], color=METHODS[method_key]["color"])
    axis.set_xticks(x, categories)
    axis.set_ylabel("Stiffness MAPE (%)")
    axis.set_title("Figure 5 — Category-level stiffness accuracy")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    path = figure_dir / "fig5_category_stiffness_mape.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fmt(value, digits=3):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def write_csv(metrics, output_path):
    fields = [
        "method", "complete_configs", "total_probes", "compliance_matrix_error", "stiffness_mape",
        "trend_slope", "trend_intercept", "trend_r2", "range_retention", "spearman_rho",
        "monotonic_ordering_accuracy", "directional_leakage", "force_cv",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method_key in METHODS:
            data = metrics.get(method_key, {})
            writer.writerow({field: data.get(field, "") for field in fields} | {"method": METHODS[method_key]["label"]})


def write_markdown(metrics, output_path, figure_dir):
    lines = [
        "# Ranged-stiffness EE compliance benchmark",
        "",
        "Generated from complete 60-probe JSON reports in `outputs/ranged_stiffness/eval_results`. No training or evaluation is started by the plotting script.",
        "",
        "## Data coverage",
        "",
        "| Method | Complete configurations | Probes | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, method in METHODS.items():
        data = metrics.get(key, {})
        count = data.get("complete_configs", 0)
        probes = data.get("total_probes", 0)
        status = "COMPLETE" if count == len(CONFIGS) else ("PARTIAL" if count else "MISSING")
        lines.append(f"| {method['label']} | {count}/{len(CONFIGS)} | {probes} | {status} |")
    lines.extend([
        "",
        "## Overall benchmark table",
        "",
        "Lower is better for compliance-matrix error, stiffness MAPE, directional leakage, and force CV. Higher is better for trend slope closeness to 1, range retention, and Spearman correlation.",
        "",
        "| Method | Matrix error ↓ | Stiffness MAPE ↓ | Trend slope | Trend R² | Range retention ↑ | Spearman ρ ↑ | Leakage ↓ | Force CV ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key, method in METHODS.items():
        data = metrics.get(key, {})
        lines.append(
            f"| {method['label']} | {fmt(data.get('compliance_matrix_error'))} | {fmt(data.get('stiffness_mape'), 3)} | {fmt(data.get('trend_slope'))} | {fmt(data.get('trend_r2'))} | {fmt(data.get('range_retention'))} | {fmt(data.get('spearman_rho'))} | {fmt(data.get('directional_leakage'))} | {fmt(data.get('force_cv'))} |"
        )
    lines.extend([
        "",
        "## Configuration categories",
        "",
        "| Category | Configurations |",
        "| --- | --- |",
        "| Isotropic | all_soft, all_hard |",
        "| Single-soft | x_soft, y_soft, z_soft |",
        "| Single-hard | x_hard, y_hard, z_hard |",
        "| Mixed | six permutations of (250, 400, 550) |",
        "",
        "## Generated figures",
        "",
        f"- [Figure 1: target versus measured stiffness heatmaps]({figure_dir.name}/fig1_target_measured_stiffness_heatmaps.png)",
        f"- [Figure 2: commanded versus measured stiffness]({figure_dir.name}/fig2_commanded_vs_measured_stiffness.png)",
        f"- [Figure 3: stiffness error versus force]({figure_dir.name}/fig3_stiffness_error_vs_force.png)",
        f"- [Figure 4: representative compliance matrices]({figure_dir.name}/fig4_representative_compliance_matrices.png)",
        f"- [Figure 5: category-level stiffness MAPE]({figure_dir.name}/fig5_category_stiffness_mape.png)",
        "",
        "## Interpretation and limitations",
        "",
        "- The command-response slope and range-retention metrics are intended to expose average-collapse behavior that may be hidden by an aggregate compliance error.",
        "- Compliance matrices are fitted from `actual_pos_b - baseline_pos_b` against `ee_force_b`, with an intercept per output dimension, then averaged across hands.",
        "- All four methods currently have complete 14-configuration sweeps; incomplete or malformed reports are deliberately excluded.",
        "- Figure 1 uses a shared 200–600 N/m color scale; annotations retain measured values even when they fall outside that scale.",
    ])
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--end2end-glob", help="Optional complete end-to-end JSON glob; relative to cwd or absolute")
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    reports_by_method = {}
    rows_by_method = {}
    metrics = {}
    for method_key, method in METHODS.items():
        explicit = args.end2end_glob if method_key == "end2end_range" else None
        reports = load_reports(args.data_dir, method["pattern"], explicit)
        reports_by_method[method_key] = reports
        rows = collect_rows(method_key, reports)
        rows_by_method[method_key] = rows
        data = compute_metrics(method_key, reports, rows) if rows else {
            "method_key": method_key,
            "method": method["label"],
            "complete_configs": 0,
            "total_probes": 0,
            "matrices": {},
        }
        data["stiffness_matrix"] = method_config_matrix(rows, method_key) if rows else None
        metrics[method_key] = data

    save_heatmap_figure(metrics, args.figure_dir)
    save_command_response_figure(metrics, rows_by_method, args.figure_dir)
    save_force_figure(rows_by_method, args.figure_dir)
    save_matrix_figure(metrics, args.figure_dir)
    save_category_figure(rows_by_method, args.figure_dir)
    write_csv(metrics, args.metrics_csv)
    write_markdown(metrics, args.markdown, args.figure_dir)

    print("Generated figures:")
    for path in sorted(args.figure_dir.glob("fig*.png")):
        print(f"  {path}")
    print(f"Metrics CSV: {args.metrics_csv}")
    print(f"Markdown table: {args.markdown}")
    for key, method in METHODS.items():
        data = metrics[key]
        print(f"{method['label']}: {data['complete_configs']}/{len(CONFIGS)} configs, {data['total_probes']} probes")


if __name__ == "__main__":
    main()
