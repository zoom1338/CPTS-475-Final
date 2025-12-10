import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_FILES = {
    "Global CTGAN": "global_xgboost.json",
    "Clustered CTGAN": "clustered_xgboost.json",
    "3-Class CTGAN": "3-class_run.json",
    "7-Class CTGAN": "7-class_run.json",
}

WEAK_ATTACKS = [
    "DDoS",
    "Worms",
    "injection",
    "password",
    "ransomware",
    "scanning",
    "xss",
]

TARGET_ATTACKS = WEAK_ATTACKS

OUTPUT_DIR = Path("figures_xgboost")
OUTPUT_DIR.mkdir(exist_ok=True)


# Helper Functions
def load_run(path):
    with open(path, "r") as f:
        return json.load(f)


def get_scenario(run_json, name_substring):
    for scen in run_json["scenarios"]:
        if name_substring in scen["name"]:
            return scen
    raise ValueError(f"Scenario containing '{name_substring}' not found")


def per_class_f1(report_dict):
    out = {}
    for k, v in report_dict.items():
        if isinstance(v, dict) and "f1-score" in v:
            out[k] = v["f1-score"]
    return out


def extract_metrics(report_dict):
    return {
        "accuracy": report_dict["accuracy"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
    }



# load all runs
def load_all_runs():
    runs = {}
    for run_name, fname in RUN_FILES.items():
        j = load_run(fname)

        # Mixed (Real + Synthetic)
        mixed = get_scenario(j, "Mixed")
        mixed_report = mixed["report"]

        # TSTR (Synthetic-only)
        tstr = get_scenario(j, "Synthetic-only")
        tstr_report = tstr["report"]

        runs[run_name] = {
            "mixed_report": mixed_report,
            "mixed_metrics": extract_metrics(mixed_report),
            "mixed_f1": per_class_f1(mixed_report),
            "tstr_report": tstr_report,
            "tstr_metrics": extract_metrics(tstr_report),
            "tstr_f1": per_class_f1(tstr_report),
        }
    return runs


runs = load_all_runs()

# Real-only baseline
real_only_json = load_run(RUN_FILES["Global CTGAN"])
real_only_scen = get_scenario(real_only_json, "Real-only")
BASELINE_F1 = per_class_f1(real_only_scen["report"])


# Plot 1: Heatmap F1 vs Real-only (Weak Attacks, Mixed)
def plot_heatmap_weak_attacks_delta(runs):
    methods_ctgan = list(RUN_FILES.keys())
    delta_matrix = np.zeros((len(methods_ctgan), len(WEAK_ATTACKS)))

    for i, method in enumerate(methods_ctgan):
        for j, attack in enumerate(WEAK_ATTACKS):
            f1_m = runs[method]["mixed_f1"][attack]
            f1_base = BASELINE_F1[attack]
            delta_matrix[i, j] = f1_m - f1_base

    plt.figure(figsize=(10, 4))
    im = plt.imshow(delta_matrix, aspect="auto", cmap="bwr", vmin=-0.2, vmax=0.2)

    plt.colorbar(im, label="ΔF1 vs Real-only")
    plt.xticks(np.arange(len(WEAK_ATTACKS)), WEAK_ATTACKS, rotation=45)
    plt.yticks(np.arange(len(methods_ctgan)), methods_ctgan)
    plt.xlabel("Attack Type")
    plt.title("Change in F1 vs Real-only (Weak Attacks, Mixed Training)")

    for i in range(delta_matrix.shape[0]):
        for j in range(delta_matrix.shape[1]):
            val = delta_matrix[i, j]
            plt.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8, color="black")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "weak_attacks_delta_heatmap.png"
    plt.savefig(out_path, dpi=200)
    plt.close()



# Plot 2: Bar Plot TSTR Macro F1 across methods
def plot_tstr_macro_f1_bar(runs):
    labels = list(RUN_FILES.keys())
    vals = [runs[m]["tstr_metrics"]["macro_f1"] for m in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.ylim(0, 1)
    plt.ylabel("Macro F1")
    plt.title("Macro F1 (Synthetic-only / TSTR)")
    plt.xticks(rotation=20)
    plt.xlabel("Method")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "tstr_macro_f1_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()



# Plot 3: Heatmap – F1 (TSTR) vs Global CTGAN for targeted attacks
def plot_tstr_delta_vs_global(runs):
    rename = {
        "Global CTGAN": "Global",
        "Clustered CTGAN": "Clustered",
        "3-Class CTGAN": "3-Class",
        "7-Class CTGAN": "7-Class",
    }

    all_classes = set()
    for m in RUN_FILES.keys():
        for cls in runs[m]["tstr_f1"].keys():
            all_classes.add(cls)

    all_classes = sorted(all_classes)

    rows: List[Dict[str, Any]] = []
    for attack in all_classes:
        row = {"Attack": attack}
        for m in RUN_FILES.keys():
            f1_map = runs[m]["tstr_f1"]
            row[rename[m]] = f1_map.get(attack, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Attack")

    df = df.loc[[a for a in TARGET_ATTACKS if a in df.index]]

    baseline = df["Global"]
    delta_plot = pd.DataFrame({
        "Clustered": df["Clustered"] - baseline,
        "3-Class": df["3-Class"] - baseline,
        "7-Class": df["7-Class"] - baseline,
    })

    fig, ax = plt.subplots(figsize=(10, 4))

    vmax = np.nanmax(np.abs(delta_plot.values))
    im = ax.imshow(delta_plot.T.values, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_yticks(np.arange(delta_plot.T.shape[0]))
    ax.set_yticklabels(delta_plot.T.index)
    ax.set_xticks(np.arange(delta_plot.T.shape[1]))
    ax.set_xticklabels(delta_plot.T.columns, rotation=45, ha="right")

    ax.set_xlabel("Attack type")
    ax.set_ylabel("Method (ΔF1 vs Global)")
    ax.set_title("Change in TSTR F1 relative to Global CTGAN")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ΔF1 = F1(method) - F1(Global)")

    for i in range(delta_plot.T.shape[0]):
        for j in range(delta_plot.T.shape[1]):
            val = delta_plot.T.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "tstr_delta_vs_global.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_heatmap_weak_attacks_delta(runs)
    plot_tstr_macro_f1_bar(runs)
    plot_tstr_delta_vs_global(runs)
