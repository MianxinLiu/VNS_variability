"""Shared plotting primitives for the current model main figure."""
import numpy as np
import matplotlib.pyplot as plt

BASE = "#49657A"
POST = "#DD6845"
DARK = "#263244"
LIGHT = "#BBC7D1"

def label(ax, text):
    ax.text(-.10, 1.06, text, transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=DARK, va="bottom")

def sigstar(p):
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    if p < .05:
        return "*"
    return ""

def ptext(p):
    if p >= .1:
        return ""
    if p < .001:
        return f" (p={p:.1e})"
    return f" (p={p:.3f})"

def ptext_all(p):
    return f" (p={p:.1e})" if p < .001 else f" (p={p:.3f})"

def paired_ladder(ax, frame, value, title, ylabel=None, show_delta=True):
    pair = frame.pivot(index=["candidate_id", "seed"], columns="session", values=value)
    for row in pair.itertuples():
        ax.plot([0, 1], [row.baseline, row.followup6w], color=LIGHT,
                marker="o", ms=1.8, lw=.55, alpha=.32, zorder=1)
    means = pair[["baseline", "followup6w"]].mean().to_numpy(float)
    sds = pair[["baseline", "followup6w"]].std(ddof=1).to_numpy(float)
    ax.plot([0, 1], means, color=DARK, lw=2.2, zorder=3)
    ax.errorbar([0, 1], means, yerr=sds, fmt="o", color=DARK,
                ecolor=DARK, capsize=3.5, ms=5.2, zorder=4)
    if show_delta:
        ax.set_title(f"{title}\nmean Δ={float((pair.followup6w-pair.baseline).mean()):+.3f}", fontsize=10)
    else:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 1], ["Baseline", "6 weeks"])
    ax.set_xlim(-.25, 1.25)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    return pair

def all_seed_bar(ax, frame, value, title, ylabel, zoom=False):
    pair = frame.pivot(index=["candidate_id", "seed"], columns="session", values=value)
    means = pair[["baseline", "followup6w"]].mean().to_numpy(float)
    sds = pair[["baseline", "followup6w"]].std(ddof=1).to_numpy(float)
    ax.bar([0, 1], means, color=[BASE, POST], width=.58, zorder=1)
    ax.errorbar([0, 1], means, yerr=sds, fmt="none", color=DARK,
                ecolor=DARK, capsize=4, lw=1.3, zorder=3)
    # Keep the fitted solutions visible without treating 100 seeds as independent fits.
    fit_means = pair.groupby(level="candidate_id").mean()
    for _, row in fit_means.iterrows():
        ax.plot([0, 1], row[["baseline", "followup6w"]], color=LIGHT,
                marker="o", ms=3.5, lw=1.0, alpha=.75, zorder=2)
    ax.set_title(f"{title}\nmean Δ={float((pair.followup6w-pair.baseline).mean()):+.3f}", fontsize=10)
    ax.set_xticks([0, 1], ["Baseline", "6 weeks"])
    ax.set_xlim(-.42, 1.42)
    if zoom:
        low = float(np.min(means-sds))
        high = float(np.max(means+sds))
        span = max(high-low, abs(high)*.08, 1e-6)
        ax.set_ylim(low-.18*span, high+.24*span)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)

def horizontal_delta_boxes(ax, groups, labels, colors, title, xlabel, stars=None, pvalues=None,
                           annotation_x=.98, annotation_yoffset=0):
    """Horizontal boxplots of fit-level changes with all fitted solutions."""
    y = np.arange(len(groups))[::-1]
    bp = ax.boxplot(groups, vert=False, positions=y, widths=.52,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=DARK, linewidth=1.5),
                    whiskerprops=dict(color=DARK, linewidth=1.0),
                    capprops=dict(color=DARK, linewidth=1.0),
                    boxprops=dict(edgecolor=DARK, linewidth=1.0))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(.48)
    rng = np.random.default_rng(991)
    if stars is None:
        stars = [""] * len(groups)
    if pvalues is None:
        pvalues = [None] * len(groups)
    for yi, values, color, star, pvalue in zip(y, groups, colors, stars, pvalues):
        values = np.asarray(values, float)
        ax.scatter(values, yi+rng.uniform(-.10, .10, len(values)),
                   facecolor="white", edgecolor=color, s=25, linewidth=.9,
                   zorder=3)
        p_label = "" if pvalue is None else ptext(pvalue)
        ax.text(annotation_x, yi+annotation_yoffset, f"mean Δ={values.mean():+.3f}{star}{p_label}", va="center", ha="right",
                fontsize=8, transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=.6))
    ax.axvline(0, color=DARK, lw=.9, zorder=0)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)

