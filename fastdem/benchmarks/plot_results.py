#!/usr/bin/env python3
"""
Generate paper figures from FastDEM benchmark CSV outputs.

Dependencies:
    pip install matplotlib pandas numpy

Usage:
    # Run from the directory where the CSVs were generated:
    python3 plot_results.py

Inputs (must be in the current directory):
    convergence_rmse.csv        - Section B output
    dynamic_response.csv        - Section C output
    traversability_accuracy.csv - Section E output

Outputs (written to ./figures/):
    fig1_convergence.pdf/png       - RMSE convergence curves (3 terrains)
    fig2_dynamic_response.pdf/png  - Step response RMSE time series
    fig3_traversability_f1.pdf/png - F1 score vs scan count
    fig4_traversability_noise.pdf/png - Final F1 across noise levels (bar chart)
    fig5_boundary_rmse.pdf/png     - Step boundary RMSE vs scan count
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# =============================================================================
# Global style
# =============================================================================

# Consistent color / line style per estimator across all figures
STYLE = {
    "Kalman":     {"color": "#2166ac", "ls": "-",  "lw": 1.6, "label": "Kalman"},
    "P2Quantile": {"color": "#d6604d", "ls": "--", "lw": 1.6, "label": "P2Quantile"},
    "StatMean":   {"color": "#1a9641", "ls": "-.", "lw": 1.6, "label": "StatMean"},
    "MovingAvg":  {"color": "#7b2d8b", "ls": ":",  "lw": 1.8, "label": "MovingAvg"},
}
ORDER = ["Kalman", "P2Quantile", "StatMean", "MovingAvg"]

# IEEE two-column paper dimensions
COL1 = 3.5    # single-column figure width [in]
COL2 = 7.16   # double-column figure width [in]
ROW_H = 2.4   # typical row height [in]

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "lines.linewidth":  1.6,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: figures/{name}.pdf / .png")


def _legend_handles():
    """Build consistent legend handles for all estimators."""
    return [
        plt.Line2D([0], [0], color=STYLE[e]["color"],
                   ls=STYLE[e]["ls"], lw=STYLE[e]["lw"],
                   label=STYLE[e]["label"])
        for e in ORDER
    ]


# =============================================================================
# Figure 1 — Convergence Accuracy
# =============================================================================

def fig1_convergence(csv="convergence_rmse.csv", rep_sigma=0.03):
    """
    RMSE vs scan count for three terrain types at a representative noise level.
    One subplot per terrain; 4 estimator curves per subplot.
    """
    df = pd.read_csv(csv)

    scan_cols = [c for c in df.columns if c.startswith("scan")]
    df_long = df.melt(
        id_vars=["terrain", "sigma", "estimator"],
        value_vars=scan_cols,
        var_name="scan_col",
        value_name="rmse",
    )
    df_long["scan"] = df_long["scan_col"].str.replace("scan", "").astype(int)
    df_long["rmse"] = pd.to_numeric(df_long["rmse"], errors="coerce")

    terrains = ["Flat", "Slope", "Sinusoidal"]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, ROW_H), sharey=False)

    for ax, terrain in zip(axes, terrains):
        sub = df_long[
            (df_long["terrain"] == terrain) &
            np.isclose(df_long["sigma"], rep_sigma)
        ]
        for est in ORDER:
            s = STYLE[est]
            data = sub[sub["estimator"] == est].sort_values("scan")
            # Convert RMSE from m → cm for readability
            ax.plot(data["scan"].to_numpy(), (data["rmse"] * 100).to_numpy(),
                    color=s["color"], ls=s["ls"], lw=s["lw"])

        ax.set_title(terrain)
        ax.set_xlabel("Scan count")
        ax.set_xlim(1, int(df_long["scan"].max()))
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    axes[0].set_ylabel("RMSE [cm]")

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.10), frameon=True)
    fig.suptitle(f"Convergence Accuracy  (σ = {rep_sigma} m)", y=1.03)
    fig.tight_layout()
    _save(fig, "fig1_convergence")
    plt.close(fig)


# =============================================================================
# Figure 2 — Dynamic Step Response
# =============================================================================

def fig2_dynamic(csv="dynamic_response.csv"):
    """
    RMSE time series with two phases (h0 → h1).
    Vertical dashed line marks the step change.
    """
    df = pd.read_csv(csv)
    df = df.replace("", np.nan)

    phase1_end = int(df[df["phase"] == "h0"]["scan"].max())
    total = int(df["scan"].max())

    fig, ax = plt.subplots(figsize=(COL1, ROW_H + 0.3))

    # Shaded phase regions
    ax.axvspan(1, phase1_end + 0.5,       alpha=0.07, color="#2166ac", zorder=0)
    ax.axvspan(phase1_end + 0.5, total,   alpha=0.07, color="#d6604d", zorder=0)
    ax.axvline(phase1_end + 0.5, color="gray", ls="--", lw=1.0, zorder=1)

    rmse_cols = [c for c in df.columns if c.endswith("_rmse")]
    for col in rmse_cols:
        est = col.replace("_rmse", "")
        if est not in STYLE:
            continue
        s = STYLE[est]
        vals = (pd.to_numeric(df[col], errors="coerce") * 100).to_numpy()  # → cm
        ax.plot(df["scan"].to_numpy(), vals,
                color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])

    ymax = ax.get_ylim()[1]
    ax.text(phase1_end * 0.5, ymax * 0.92,
            "$h_0 = 0$ m", ha="center", va="top", fontsize=7.5, color="#2166ac")
    ax.text(phase1_end + (total - phase1_end) * 0.5, ymax * 0.92,
            "$h_1 = 0.3$ m", ha="center", va="top", fontsize=7.5, color="#d6604d")

    ax.set_xlabel("Scan number")
    ax.set_ylabel("RMSE [cm]")
    ax.set_xlim(1, total)
    ax.set_ylim(bottom=0)

    handles = _legend_handles()
    handles.append(plt.Line2D([0], [0], color="gray", ls="--", lw=1.0, label="Step change"))
    ax.legend(handles=handles, loc="upper right", frameon=True)

    ax.set_title("Dynamic Response to Step Height Change")
    fig.tight_layout()
    _save(fig, "fig2_dynamic_response")
    plt.close(fig)


# =============================================================================
# Figure 3 — Traversability F1 vs Scan Count
# =============================================================================

def fig3_f1_vs_scan(csv="traversability_accuracy.csv", rep_sigma=0.03):
    """
    F1 score vs scan count at a representative noise level.
    Horizontal dashed line at F1 = 0.95 marks 'navigation-reliable' threshold.
    """
    df = pd.read_csv(csv)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    sub = df[np.isclose(df["sigma"], rep_sigma)]

    fig, ax = plt.subplots(figsize=(COL1, ROW_H))

    for est in ORDER:
        s = STYLE[est]
        data = sub[sub["estimator"] == est].sort_values("scan")
        ax.plot(data["scan"].to_numpy(), data["f1"].to_numpy(),
                color=s["color"], ls=s["ls"], lw=s["lw"])

    ax.axhline(0.95, color="gray", ls=":", lw=1.2, label="F1 = 0.95")

    ax.set_xlabel("Scan count")
    ax.set_ylabel("F1 Score")
    ax.set_xlim(1, int(sub["scan"].max()))
    ax.set_ylim(0, 1.05)

    handles = _legend_handles()
    handles.append(plt.Line2D([0], [0], color="gray", ls=":", lw=1.2, label="F1 = 0.95 threshold"))
    ax.legend(handles=handles, loc="lower right", frameon=True)

    ax.set_title(f"Traversability F1 Score  (σ = {rep_sigma} m)")
    fig.tight_layout()
    _save(fig, "fig3_traversability_f1")
    plt.close(fig)


# =============================================================================
# Figure 4 — Final F1 across Noise Levels (grouped bar chart)
# =============================================================================

def fig4_f1_vs_noise(csv="traversability_accuracy.csv"):
    """
    Grouped bar chart: one group per noise level, one bar per estimator.
    Shows how F1 degrades as measurement noise increases.
    """
    df = pd.read_csv(csv)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

    # Take the last scan for each (sigma, estimator) pair
    final = (df.sort_values("scan")
               .groupby(["sigma", "estimator"], as_index=False)
               .last())

    noise_levels = sorted(final["sigma"].unique())
    x = np.arange(len(noise_levels))
    n = len(ORDER)
    width = 0.18
    offsets = (np.arange(n) - (n - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(COL1, ROW_H))

    for i, est in enumerate(ORDER):
        s = STYLE[est]
        vals = []
        for sig in noise_levels:
            row = final[(np.isclose(final["sigma"], sig)) & (final["estimator"] == est)]
            vals.append(row["f1"].values[0] if len(row) > 0 else np.nan)

        bars = ax.bar(x + offsets[i], vals, width,
                      color=s["color"], label=s["label"],
                      edgecolor="white", linewidth=0.4, zorder=2)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90)

    ax.set_xlabel("Noise σ [m]")
    ax.set_ylabel("F1 Score (after convergence)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.2f}" for s in noise_levels])
    ax.set_ylim(0, 1.18)
    ax.legend(loc="lower left", frameon=True)
    ax.set_title("Traversability F1 Score vs Noise Level")

    fig.tight_layout()
    _save(fig, "fig4_traversability_noise")
    plt.close(fig)


# =============================================================================
# Figure 5 — Step Boundary RMSE vs Scan Count
# =============================================================================

def fig5_boundary_rmse(csv="traversability_accuracy.csv", rep_sigma=0.03):
    """
    Boundary RMSE vs scan count at a representative noise level.
    Boundary region: cells within ±1 m of the step edge.
    """
    df = pd.read_csv(csv)
    df["boundary_rmse"] = pd.to_numeric(df["boundary_rmse"], errors="coerce")
    sub = df[np.isclose(df["sigma"], rep_sigma)]

    fig, ax = plt.subplots(figsize=(COL1, ROW_H))

    for est in ORDER:
        s = STYLE[est]
        data = sub[sub["estimator"] == est].sort_values("scan")
        ax.plot(data["scan"].to_numpy(), (data["boundary_rmse"] * 100).to_numpy(),
                color=s["color"], ls=s["ls"], lw=s["lw"])

    ax.set_xlabel("Scan count")
    ax.set_ylabel("Boundary RMSE [cm]")
    ax.set_xlim(1, int(sub["scan"].max()))
    ax.set_ylim(bottom=0)

    ax.legend(handles=_legend_handles(), loc="upper right", frameon=True)
    ax.set_title(f"Step Boundary RMSE  (σ = {rep_sigma} m)")

    fig.tight_layout()
    _save(fig, "fig5_boundary_rmse")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

TASKS = [
    ("convergence_rmse.csv",        fig1_convergence,    "Fig 1 — Convergence curves"),
    ("dynamic_response.csv",         fig2_dynamic,        "Fig 2 — Dynamic response"),
    ("traversability_accuracy.csv",  fig3_f1_vs_scan,     "Fig 3 — Traversability F1 vs scan"),
    ("traversability_accuracy.csv",  fig4_f1_vs_noise,    "Fig 4 — F1 vs noise level"),
    ("traversability_accuracy.csv",  fig5_boundary_rmse,  "Fig 5 — Boundary RMSE vs scan"),
]

if __name__ == "__main__":
    print("FastDEM — Generating paper figures\n")

    skipped, failed = [], []

    for csv_path, fn, desc in TASKS:
        print(f"[{desc}]")
        if not Path(csv_path).exists():
            print(f"  SKIP: '{csv_path}' not found\n")
            skipped.append(csv_path)
            continue
        try:
            fn(csv_path)
        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            failed.append(desc)
            continue
        print()

    print("─" * 50)
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(set(skipped))}")
    if failed:
        print(f"Failed  ({len(failed)}): {', '.join(failed)}")
    if not skipped and not failed:
        print(f"All 5 figures saved to ./figures/")
