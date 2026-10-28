"""Figure 5: tonic network input and operating-point inhibitory feedback."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = HERE.parent / "data" / "model"
OUT = HERE.parent / "figures"
COLORS = {"input": "#D25D40", "slope": "#8E6BB1",
          "loop": "#2B8A78", "stability": "#344054"}


def pathway_panel(ax):
    ax.axis("off")
    ax.text(-.045, 1.07, "A", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="bottom")
    ax.set_title("Proposed mechanism of network-driven stabilization", fontsize=15, pad=15)
    boxes = [
        (.03, "Tonic network input", r"$GJ_{\mathrm{NMDA}}CS_E$", COLORS["input"]),
        (.285, "I operating slope", r"$D_I=H'_I(I_I)$", COLORS["slope"]),
        (.54, "E–I–E feedback",
         r"$\frac{D_Ew_{IE}D_Iw_{EI}J_{\mathrm{NMDA}}}{\tau_I^{-1}+D_I}$",
         COLORS["loop"]),
        (.795, "Motor stability", r"$-\max\,\mathrm{Re}(\lambda)$", COLORS["stability"]),
    ]
    width = .19
    for x, title, formula, color in boxes:
        ax.add_patch(FancyBboxPatch((x, .25), width, .59,
            boxstyle="round,pad=.008,rounding_size=.02",
            facecolor=color, edgecolor=color, alpha=.13,
            transform=ax.transAxes))
        ax.text(x+width/2, .68, title, ha="center", va="center",
                color=color, fontweight="bold", fontsize=12,
                transform=ax.transAxes)
        ax.text(x+width/2, .43, formula, ha="center", va="center",
                color=color, fontsize=15 if title == "E–I–E feedback" else 12,
                transform=ax.transAxes)
    for x1, x2 in ((.22, .285), (.475, .54), (.73, .795)):
        ax.annotate("", xy=(x2-.007, .56), xytext=(x1+.007, .56),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=2, color="#667085"))


def row_heading(subspec, fig, letter, title, title_offset=.009):
    bounds = subspec.get_position(fig)
    fig.text(bounds.x0-.035, bounds.y1+.009, letter,
             fontsize=18, fontweight="bold", va="bottom")
    fig.text((bounds.x0+bounds.x1)/2, bounds.y1+title_offset, title,
             ha="center", va="bottom", fontsize=15)


def changes_panel(subspec, data, fig):
    row_heading(subspec, fig, "B", "Fitted 6-week changes")
    axes = subspec.subgridspec(1, 4, wspace=.55)
    specs = [
        ("tonic_network_input", "Tonic network input (nA)", COLORS["input"]),
        ("mean_D_I", r"Motor $D_I$ ($s^{-1}$)", COLORS["slope"]),
        ("inhibitory_loop_restoring", r"E–I–E feedback ($s^{-1}$)", COLORS["loop"]),
        ("motor_stability", r"Motor stability ($s^{-1}$)", COLORS["stability"]),
    ]
    rows = []
    for i, (key, name, color) in enumerate(specs):
        ax = fig.add_subplot(axes[0, i])
        values = data[key].to_numpy(float)
        box = ax.boxplot([values], positions=[0], widths=.4,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color="#263244", lw=1.5))
        box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=.28)
        jitter = np.random.default_rng(506+i).uniform(-.14, .14, len(values))
        ax.scatter(jitter, values, s=23, facecolor="white", edgecolor=color,
                   linewidth=1, zorder=3)
        p = wilcoxon(values).pvalue
        positive = int((values > 0).sum())
        ax.text(.5, .98, f"{positive}/{len(values)} ↑\np={p:.2g}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)
        ax.axhline(0, color="#98A2B3", lw=.85)
        low, high = min(0, values.min()), max(0, values.max())
        span = high-low
        ax.set_ylim(low-.08*span, high+.25*span)
        ax.set_xlim(-.45, .45)
        ax.set_xticks([])
        ax.set(xlabel=name, ylabel="Change")
        rows.append(dict(term=key, n=len(values), mean_delta=values.mean(),
                         median_delta=np.median(values), positive=positive,
                         wilcoxon_p_two_sided=p))
    return rows


def correlations_panel(subspec, data, fig):
    row_heading(subspec, fig, "C", "Fit-level associations along the proposed path",
                title_offset=.035)
    axes = subspec.subgridspec(1, 3, wspace=.45)
    specs = [
        ("tonic_network_input", "mean_D_I", "Δ tonic network input",
         r"Δ motor $D_I$", COLORS["input"]),
        ("mean_D_I", "inhibitory_loop_restoring", r"Δ motor $D_I$",
         "Δ E–I–E feedback", COLORS["slope"]),
        ("inhibitory_loop_restoring", "motor_stability",
         "Δ E–I–E feedback", "Δ motor stability margin", COLORS["loop"]),
    ]
    rows = []
    for i, (xkey, ykey, xlabel, ylabel, color) in enumerate(specs):
        ax = fig.add_subplot(axes[0, i])
        x, y = data[xkey].to_numpy(float), data[ykey].to_numpy(float)
        rho, p = spearmanr(x, y)
        ax.scatter(x, y, s=38, facecolor="white", edgecolor=color,
                   linewidth=1.2)
        ax.axhline(0, color="#98A2B3", lw=.8)
        ax.axvline(0, color="#98A2B3", lw=.8)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", labelsize=12, pad=5)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.text(.5, 1.02, f"Spearman ρ={rho:+.2f}; p={p:.2g}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
        rows.append(dict(x=xkey, y=ykey, n=len(x), spearman_rho=rho,
                         p_two_sided=p))
    return rows


def perturbation_panel(subspec, perturbation, fig):
    row_heading(subspec, fig, "D", "Small within-model input perturbation")
    axes = subspec.subgridspec(1, 3, width_ratios=[.35, 1.3, .35])
    ax = fig.add_subplot(axes[0, 1])
    specs = [
        ("delta_D_I", r"Motor $D_I$", COLORS["slope"]),
        ("delta_EIE_restoring", "E–I–E feedback", COLORS["loop"]),
        ("delta_motor_stability", "Motor stability", COLORS["stability"]),
    ]
    n = len(perturbation)
    for i, (key, name, color) in enumerate(specs):
        values = perturbation[key].to_numpy(float)
        positive = int((values > 0).sum())
        ax.barh(i, positive/n*100, height=.5, color=color, alpha=.76)
        ax.text(min(positive/n*100+2.5, 121), i,
                f"{positive}/{n} ↑", va="center", fontsize=11)
    ax.set_yticks(range(3), [name for _, name, _ in specs])
    ax.set_ylim(2.72, -.55)
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
    ax.set_xlabel("Fits with an increase")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--perturbation-csv", type=Path)
    parser.add_argument("--stem", default="figure6_stability_mechanism")
    args = parser.parse_args()
    perturbation_csv = (args.perturbation_csv or
                        args.analysis_dir / "network_input_DI_closed_loop_0p1pct.csv")
    data = pd.read_csv(args.analysis_dir / "paired_changes.csv")
    perturbation = pd.read_csv(perturbation_csv)
    if data.candidate_id.duplicated().any() or perturbation.candidate_id.duplicated().any():
        raise ValueError("Candidate IDs must be unique")
    if set(data.candidate_id) != set(perturbation.candidate_id):
        raise ValueError("Fitted associations and perturbations must use identical candidates")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                         "axes.titlesize": 13, "xtick.labelsize": 10,
                         "ytick.labelsize": 10})
    fig = plt.figure(figsize=(16.5, 14.5), facecolor="white")
    grid = fig.add_gridspec(4, 1, left=.085, right=.98, top=.965, bottom=.075,
                            height_ratios=[1.05, 2.25, 2.0, 1.55], hspace=.65)
    pathway_ax = fig.add_subplot(grid[0])
    pathway_panel(pathway_ax)
    pos = pathway_ax.get_position()
    pathway_ax.set_position([pos.x0, pos.y0-.03, pos.width, pos.height])
    changes = changes_panel(grid[1], data, fig)
    correlations = correlations_panel(grid[2], data, fig)
    perturbation_panel(grid[3], perturbation, fig)
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=4)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{args.stem}.{ext}", dpi=250)
    plt.close(fig)
    pd.DataFrame(changes).to_csv(OUT / f"{args.stem}_changes.csv", index=False)
    pd.DataFrame(correlations).to_csv(OUT / f"{args.stem}_correlations.csv", index=False)
    print(OUT / f"{args.stem}.png")


if __name__ == "__main__":
    main()
