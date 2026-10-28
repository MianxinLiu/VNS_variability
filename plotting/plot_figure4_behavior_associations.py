#!/usr/bin/env python3
"""Reorganized Figure 4: FMA trajectory, longitudinal EEG models, cohort 2 check."""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from scipy.stats import chi2

import analysis_and_figure_data as source
from manuscript_style import polish, trajectory_space


OUT = Path("figures")
HEMIS = ("contra", "ipsi")
NAMES = {"contra": "Ipsilesional", "ipsi": "Contralesional"}
COLORS = {"contra": "#C65D3B", "ipsi": "#258873"}
COMPOSITE = "z(SE) − z(DFA)"


def z_composite_long(se, dfa):
    """Z-score each component per hemisphere over all 41 cohort-1 visits."""
    rows = [dict(row) for row in se["rows"]]
    for hemi in HEMIS:
        se_values = np.array([r[hemi] for r in se["rows"]], dtype=float)
        dfa_values = np.array([r[hemi] for r in dfa["rows"]], dtype=float)
        values = source.zscore_safe(se_values) - source.zscore_safe(dfa_values)
        for row, value in zip(rows, values):
            row[hemi] = float(value)
    return {"label": COMPOSITE, "rows": rows}


def z_composite_post(se, dfa):
    """Z-score each component per hemisphere over all 9 cohort-2 people/visits."""
    result = {"label": COMPOSITE, "keep": se["keep"]}
    for hemi in HEMIS:
        result[hemi] = (source.zscore_safe(np.asarray(se[hemi], float).ravel())
                        - source.zscore_safe(np.asarray(dfa[hemi], float).ravel())).reshape(se[hemi].shape)
    return result


def pformat(p):
    return f"{p:.1e}" if p < .001 else f"{p:.3f}"


def panel_title(fig, ax, letter, title):
    box = ax.get_position()
    fig.text(box.x0, box.y1 + .022, f"{letter}  {title}", fontsize=14,
             fontweight="bold", color="#263244", va="bottom")


def single_panel(fig, slot, effects, comparisons, letter="B", title="Single-metric longitudinal associations"):
    layout=slot.subgridspec(1,2,width_ratios=[2.1,1],wspace=.08)
    ax = fig.add_subplot(layout[0,0])
    table=fig.add_subplot(layout[0,1])
    table.axis('off')
    metrics = ("SE", "DFA", "MSE", "Composite")
    y_centers = np.arange(len(metrics))[::-1] * 3
    ax.axvline(0, color="#6B7280", lw=.9)
    for metric, center in zip(metrics, y_centers):
        for h_index, hemi in enumerate(HEMIS):
            row = effects[(effects.model == "Single metric") & (effects.metric == metric)
                          & (effects.hemisphere == hemi)].iloc[0]
            comp = comparisons[(comparisons.model == "Single metric") & (comparisons.metric == metric)
                               & (comparisons.hemisphere == hemi)].iloc[0]
            y = center + 1 - h_index
            ax.plot([row.ci_low, row.ci_high], [y, y], color=COLORS[hemi], lw=1.8)
            ax.scatter(row.beta, y, color=COLORS[hemi], s=31, zorder=3)
            table.text(.03, y, f"{row.beta:+.2f}", fontsize=12, color=COLORS[hemi], va="center")
            table.text(.34, y, pformat(row.p), fontsize=12, va="center",
                    fontweight="bold" if row.p < .05 else "normal")
            table.text(.70, y, pformat(comp.p_vs_time_only), fontsize=12, va="center",
                    fontweight="bold" if comp.p_vs_time_only < .05 else "normal")
    ax.set_yticks(y_centers+.5, [COMPOSITE if x == "Composite" else x for x in metrics])
    ax.set_ylim(-1.8, y_centers[0] + 2)
    ax.set_xlim(-.45, .50)
    ax.set_xticks([-.4, -.2, 0, .2, .4])
    table.set_ylim(ax.get_ylim())
    ax.set_xlabel("Standardized EEG coefficient (95% CI)")
    ax.grid(axis="x", color="#E5E7EB", lw=.7)
    ax.spines[["top", "right"]].set_visible(False)
    table.text(.03, y_centers[0] + 1.9, "β", fontsize=12, fontweight="bold")
    table.text(.34, y_centers[0] + 1.9, "p effect", fontsize=12, fontweight="bold")
    table.text(.70, y_centers[0] + 1.9, "p vs time", fontsize=12, fontweight="bold")
    panel_title(fig, ax, letter, title)
    return ax


def within_panel(fig, slot, effects, metric, letter):
    nested = slot.subgridspec(2, 2, width_ratios=[2.15, 1.35],
                              hspace=.65, wspace=.08)
    for part_i, (part, heading, limits, ticks) in enumerate((
            ("within", "Within participant", (-.4, .5), [-.4, -.2, 0, .2, .4]),
            ("between", "Between participants", (-2.3, 2.5), [-2, -1, 0, 1, 2]))):
        ax = fig.add_subplot(nested[part_i, 0])
        table = fig.add_subplot(nested[part_i, 1])
        ax.axvline(0, color="#6B7280", lw=.9)
        for h_index, hemi in enumerate(HEMIS):
            row = effects[(effects.model == "Within/between") & (effects.metric == metric)
                          & (effects.component == part) & (effects.hemisphere == hemi)].iloc[0]
            y = 1.6 - h_index
            ax.plot([row.ci_low, row.ci_high], [y, y], color=COLORS[hemi], lw=1.7)
            ax.scatter(row.beta, y, color=COLORS[hemi], s=28, zorder=3)
            table.text(.02, y, f"β={row.beta:+.2f}  p={pformat(row.p)}", fontsize=11,
                       va="center", fontweight="bold" if row.p < .05 else "normal")
        ax.set_yticks([1.6, .6], ["Ipsi", "Contra"])
        ax.set_ylim(0, 2.4)
        ax.set_xlim(limits)
        ax.set_xticks(ticks)
        ax.text(.0, 1.03, heading, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom")
        if part_i == 1:
            ax.set_xlabel("Standardized coefficient (95% CI)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", color="#E5E7EB", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
        table.set_xlim(0, 1)
        table.set_ylim(0, 2.4)
        table.axis("off")
        if part_i == 0:
            panel_title(fig, ax, letter, COMPOSITE if metric == "Composite" else metric)


def cohort2_change_panel(fig, slot, cohorts):
    layout=slot.subgridspec(1,2,width_ratios=[2.1,1],wspace=.08)
    ax = fig.add_subplot(layout[0,0])
    table=fig.add_subplot(layout[0,1]);table.axis('off')
    ax.axvline(0, color="#263244", lw=.9)
    active = np.asarray(cohorts["SE"]["keep"], bool)
    fma_delta = np.asarray(source.FMA_C1[:, 1] - source.FMA_C1[:, 0], float)
    rows = []
    metrics = ("SE", "DFA", "MSE", "Composite")
    directions = {"SE": "greater", "DFA": "less",
                  "MSE": "greater", "Composite": "greater"}
    for metric_index, metric in enumerate(metrics):
        for hemi_index, hemi in enumerate(HEMIS):
            data = np.asarray(cohorts[metric][hemi], float)
            pairs = np.column_stack([data[:, 1] - data[:, 0], fma_delta])[active]
            pairs = pairs[np.isfinite(pairs).all(axis=1)]
            r, p = pearsonr(pairs[:, 0], pairs[:, 1], alternative=directions[metric])
            fisher = 1.644854 / np.sqrt(len(pairs) - 3)
            low, high = np.tanh(np.arctanh(np.clip(r, -.999999, .999999)) + np.array([-fisher, fisher]))
            y = (len(metrics) - metric_index - 1) * 3 + (1 - hemi_index) * 1.2
            ax.plot([low, high], [y, y], color=COLORS[hemi], lw=2.0)
            ax.scatter(r, y, color=COLORS[hemi], s=40, zorder=3)
            table.text(.03, y, f"r={r:+.2f}    p={pformat(p)}    n={len(pairs)}", fontsize=12,
                    va="center", fontweight="bold" if p < .06 else "normal")
            rows.append({"metric": COMPOSITE if metric == "Composite" else metric,
                         "hemisphere": NAMES[hemi], "n_active": len(pairs),
                         "r": r, "ci90_low": low, "ci90_high": high,
                         "alternative": directions[metric], "p_one_sided": p})
    ax.set_yticks(np.arange(len(metrics))[::-1] * 3 + .6,
                  [COMPOSITE if x == "Composite" else x for x in metrics])
    ax.set_ylim(-1, len(metrics) * 3)
    ax.set_xlim(-1.05, 1.05)
    table.set_ylim(ax.get_ylim())
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_xlabel("Pearson r: post − baseline EEG change vs FMA-UE change")
    ax.grid(axis="x", color="#E5E7EB", lw=.7)
    ax.spines[["top", "right"]].set_visible(False)
    panel_title(fig, ax, "G", "Cohort 2 active group: EEG change and FMA-UE improvement")
    return pd.DataFrame(rows)


def pooled_single_metric_rows(long, post):
    """Pooled analogue of B; z-score FMA/EEG inside each cohort first."""
    metrics = ("SE", "DFA", "MSE", "Composite")
    active = np.asarray(post["SE"]["keep"], bool)
    rows, comparisons = [], []
    for hemi in HEMIS:
        for metric in metrics:
            records = []
            for row in long[metric]["rows"]:
                records.append({"subject": f"c1_{row['sid']}", "cohort": "Cohort 1",
                                "time": float(row["week"]), "fma": float(row["fma"]),
                                "eeg": float(row[hemi])})
            for person_i, is_active in enumerate(active):
                if not is_active:
                    continue
                for visit_i, week in enumerate((0, 6)):
                    records.append({"subject": f"c2_{person_i}", "cohort": "Cohort 2",
                                    "time": float(week),
                                    "fma": float(source.FMA_C1[person_i, visit_i]),
                                    "eeg": float(post[metric][hemi][person_i, visit_i])})
            df = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan).dropna()
            if df.groupby("cohort")["subject"].nunique().to_dict() != {"Cohort 1": 10, "Cohort 2": 7}:
                raise ValueError("Unexpected participant count in pooled association")
            df["resp"] = df.groupby("cohort")["fma"].transform(source.zscore_safe)
            df["metric_z"] = df.groupby("cohort")["eeg"].transform(source.zscore_safe)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base = source.fit_within_between_model(df, ["C(cohort)"])
                fit = source.fit_within_between_model(df, ["C(cohort)", "metric_z"])
            ci = fit.conf_int().loc["metric_z"]
            lrt = max(0.0, 2 * (float(fit.llf) - float(base.llf)))
            rows.append({"hemisphere": hemi, "model": "Single metric", "metric": metric,
                         "component": "overall", "beta": float(fit.params["metric_z"]),
                         "ci_low": float(ci.iloc[0]), "ci_high": float(ci.iloc[1]),
                         "p": float(fit.pvalues["metric_z"]), "n_participants": 17,
                         "n_observations": len(df)})
            comparisons.append({"hemisphere": hemi, "model": "Single metric", "metric": metric,
                                "p_vs_time_only": float(chi2.sf(lrt, 1)),
                                "LRT_vs_time_only": lrt, "n_participants": 17,
                                "n_observations": len(df)})
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def main():
    source.OUT_DIR = OUT
    source.DFA_FILENAME = "DFA.npy"
    source.OUTPUT_SUFFIX = ""
    source.PAIRED_TEST = "ttest"
    source.setup_style()
    long = {key: source.load_cohort2(key, filename_override="DFA.npy" if key == "DFA" else None)
            for key in ("SE", "DFA", "MSE")}
    long["Composite"] = z_composite_long(long["SE"], long["DFA"])
    post = {key: source.load_cohort1(key) for key in ("SE", "DFA", "MSE")}
    post["Composite"] = z_composite_post(post["SE"], post["DFA"])
    effects, comparisons = source.behavior_association_rows(cohorts_override=long, hemis=HEMIS)
    pooled_effects, pooled_comparisons = pooled_single_metric_rows(long, post)

    fig = plt.figure(figsize=(18, 24), facecolor="white")
    grid = fig.add_gridspec(6, 4, left=.10, right=.97, bottom=.065, top=.955,
                            height_ratios=[1.15, 1.4, 1.25, 1.25, 1.25, 1.4],
                            hspace=.65, wspace=.48)
    top_pair = grid[0, :].subgridspec(1, 2, wspace=.20)
    ax_comp = fig.add_subplot(top_pair[0, 0])
    comp_stats = source.cohort2_paired_week_stats(long["Composite"], "ipsi", [2, 4, 6], alt="less")
    source.plot_cohort2_key_trajectory(ax_comp, long["Composite"], "ipsi", COMPOSITE,
                                       "Cohort 1 contralesional composite", stats=comp_stats,
                                       mean_max_week=6, show_sigstars=True,
                                       equally_spaced_weeks=True, mean_color="#258873")
    ax_a = fig.add_subplot(top_pair[0, 1])
    stats = source.cohort2_paired_week_stats(long["SE"], "fma", [2, 4, 6], alt="less")
    source.plot_cohort2_key_trajectory(ax_a, long["SE"], "fma", "FMA-UE",
                                       "Cohort 1 FMA-UE trajectory", stats=stats,
                                       mean_max_week=6, show_sigstars=True,
                                       equally_spaced_weeks=True, mean_color="#3F6FA1")
    panel_title(fig, ax_comp, "A", "Cohort 1 EEG and FMA-UE trajectories")
    single_panel(fig, grid[1, :], effects, comparisons)
    for slot, metric, letter in zip((grid[2, :2], grid[2, 2:], grid[3, :2], grid[3, 2:]),
                                    ("Composite", "SE", "DFA", "MSE"), ("C", "D", "E", "F")):
        within_panel(fig, slot, effects, metric, letter)
    corr_rows = cohort2_change_panel(fig, grid[4, :], post)
    single_panel(fig, grid[5, :], pooled_effects, pooled_comparisons,
                 letter="H", title="Pooled Cohort 1 + Cohort 2 active: single-metric associations")
    handles = [Line2D([0], [0], color=COLORS[h], marker="o", lw=1.5, label=NAMES[h]) for h in HEMIS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015), ncol=2, frameon=False, fontsize=13)
    for ax in (ax_comp,ax_a):
        trajectory_space(ax)
        ax.set_title(ax.get_title(),pad=15)
    polish(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure4_behavior_associations.png", dpi=220)
    fig.savefig(OUT / "figure4_behavior_associations.pdf")
    plt.close(fig)
    effects.to_csv(OUT / "figure4_behavior_associations_effects.csv", index=False)
    comparisons.to_csv(OUT / "figure4_behavior_associations_model_comparison.csv", index=False)
    corr_rows.to_csv(OUT / "figure4_cohort2_change_fma_correlations_active.csv", index=False)
    pooled_effects.to_csv(OUT / "figure4_pooled_behavior_associations_effects.csv", index=False)
    pooled_comparisons.to_csv(OUT / "figure4_pooled_behavior_associations_model_comparison.csv", index=False)
    print(OUT / "figure4_behavior_associations.png")


if __name__ == "__main__":
    main()
