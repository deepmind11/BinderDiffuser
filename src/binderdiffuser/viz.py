"""Figure generation for BinderDiffuser results.

Produces the README-facing artifacts:

    1. self_consistency_scatter:
        scRMSD vs mean pLDDT, colored by ipTM. The classic "good designs in
        the lower-left/upper-right" plot used in every binder design paper.
    2. metrics_violin:
        Distribution of each metric across the cohort, with the top-K
        marked. Helps eyeball whether the run is producing a meaningful
        tail of high-quality designs.
    3. interactive_3d_view:
        py3Dmol HTML widget combining target (gray cartoon) + ranked binders
        (rainbow), with the motif highlighted.

Figures are saved as ``.png`` (300 dpi) for embedding in the README and
``.html`` for interactive notebooks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PLOT_STYLE = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.family": "DejaVu Sans",
}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams.update(PLOT_STYLE)


def self_consistency_scatter(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Self-consistency landscape",
) -> Path:
    """Scatter scRMSD vs mean_plddt, coloured by ipTM."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    color_col = df["iptm"].fillna(df["iptm"].mean() if df["iptm"].notna().any() else 0.5)
    sc = ax.scatter(
        df["sc_rmsd"],
        df["mean_plddt"],
        c=color_col,
        cmap="viridis",
        s=42,
        edgecolors="white",
        linewidths=0.6,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("ipTM", rotation=270, labelpad=14)
    ax.set_xlabel("scRMSD (Å)")
    ax.set_ylabel("mean pLDDT")
    ax.set_title(title, loc="left", weight="bold")
    ax.axvline(2.0, color="firebrick", ls="--", lw=1, alpha=0.6, label="scRMSD = 2 Å")
    ax.axhline(70, color="steelblue", ls="--", lw=1, alpha=0.6, label="pLDDT = 70")
    ax.legend(loc="lower right", frameon=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def metrics_violin(
    df: pd.DataFrame,
    out_path: str | Path,
    metrics: tuple[str, ...] = ("sc_rmsd", "sc_tm", "mean_plddt", "iptm"),
    top_k: int | None = 10,
) -> Path:
    """Per-metric violin plots with the top-K (by composite_score) overlaid."""
    _setup_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.2), dpi=150)
    if len(metrics) == 1:
        axes = [axes]

    if "composite_score" in df.columns and top_k is not None:
        top = df.nlargest(top_k, "composite_score")
    else:
        top = df.head(0)

    for ax, metric in zip(axes, metrics, strict=False):
        if metric not in df.columns:
            ax.set_axis_off()
            continue
        data = df[metric].dropna()
        if data.empty:
            ax.set_axis_off()
            continue
        sns.violinplot(y=data, ax=ax, inner="quartile", color="lightsteelblue")
        if not top.empty and metric in top.columns:
            ax.scatter(
                np.zeros(len(top)),
                top[metric].values,
                color="darkorange",
                s=40,
                zorder=5,
                label=f"top {len(top)}",
            )
            ax.legend(loc="best", frameon=True)
        ax.set_title(metric, weight="bold")
        ax.set_ylabel("")

    fig.suptitle("Design metrics distribution", weight="bold", x=0.02, ha="left")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_pymol_script(
    target_pdb: str | Path,
    binder_pdbs: list[str | Path],
    out_path: str | Path,
    motif_residues: list[int] | None = None,
    target_chain: str = "B",
) -> Path:
    """Emit a PyMOL .pml script that loads target + ranked binders.

    The script is the canonical way to reproduce the README hero figure:
    open PyMOL and ``run hero.pml`` to render. We do not invoke PyMOL here
    because it is not always installed; the script is itself the artifact.
    """
    target_pdb = Path(target_pdb)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"load {target_pdb}, target",
        "hide everything, target",
        f"show cartoon, target and chain {target_chain}",
        "color gray80, target",
    ]
    if motif_residues:
        sel = "+".join(str(r) for r in motif_residues)
        lines.append(f"show sticks, target and chain {target_chain} and resi {sel}")
        lines.append(f"color magenta, target and chain {target_chain} and resi {sel}")

    palette = ["yellow", "salmon", "limon", "skyblue", "violet", "wheat", "teal"]
    for i, binder in enumerate(binder_pdbs):
        name = f"binder_{i}"
        col = palette[i % len(palette)]
        lines.extend([
            f"load {Path(binder)}, {name}",
            f"hide everything, {name}",
            f"show cartoon, {name}",
            f"color {col}, {name}",
        ])
    lines.extend([
        "bg_color white",
        "set ray_opaque_background, 0",
        "orient",
        "ray 1600, 1200",
        f"png {out_path.with_suffix('.png')}, dpi=300",
    ])
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
