#!/usr/bin/env python3
"""Separate exploratory ANCOVA-style baseline EEG / 6-week FMA sensitivity."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import f as f_dist
from scipy.stats import pearsonr, t as t_dist

import analysis_and_figure_data as source
from manuscript_style import polish
from matplotlib.lines import Line2D


OUT = Path("figures")
REF = Path("data/baseline_zscore_reference.csv")
METRICS = ("SE", "DFA", "MSE", "Composite")
HEMIS = (("contra", "Ipsilesional", "#C65D3B"),
         ("ipsi", "Contralesional", "#258873"))
COHORT_COLORS = {"Cohort 1": "#4C78A8", "Cohort 2": "#D5822B"}


def cohort_partial_correlation(group):
    """EEG/FMA6 partial correlation controlling baseline FMA within one cohort."""
    x = smf.ols("eeg_z ~ fma0_z", data=group).fit().resid
    y = smf.ols("fma6_z ~ fma0_z", data=group).fit().resid
    r, _ = pearsonr(x, y)
    df = len(group) - 3  # One covariate plus the two correlated variables.
    t_value = r * np.sqrt(df / max(1e-15, 1 - r * r))
    return float(r), float(2 * t_dist.sf(abs(t_value), df))


def baseline_records():
    source.DFA_FILENAME = "DFA.npy"
    long = {m: source.load_cohort2(m, filename_override="DFA.npy" if m == "DFA" else None)
            for m in METRICS if m != "Composite"}
    post = {m: source.load_cohort1(m)
            for m in METRICS if m != "Composite"}
    reference = pd.read_csv(REF).set_index(["hemisphere", "metric"])
    by_sid = {}
    for row in long["SE"]["rows"]:
        by_sid.setdefault(row["sid"], {})[row["week"]] = row
    active = np.asarray(post["SE"]["keep"], bool)
    rows = []
    for hemi, hemi_label, _ in HEMIS:
        baseline_maps = {m: {r["sid"]: float(r[hemi]) for r in data["rows"] if r["week"] == 0}
                         for m, data in long.items()}
        for sid, visits in by_sid.items():
            if 0 not in visits or 6 not in visits:
                continue
            record = {"hemisphere": hemi_label, "cohort": "Cohort 1", "subject": f"c1_{sid}",
                      "fma0": float(visits[0]["fma"]), "fma6": float(visits[6]["fma"])}
            record.update({m: baseline_maps[m][sid] for m in baseline_maps})
            rows.append(record)
        for i, keep in enumerate(active):
            if not keep:
                continue
            record = {"hemisphere": hemi_label, "cohort": "Cohort 2", "subject": f"c2_{i}",
                      "fma0": float(source.FMA_C1[i, 0]), "fma6": float(source.FMA_C1[i, 1])}
            record.update({m: float(post[m][hemi][i, 0]) for m in post})
            rows.append(record)
    data = pd.DataFrame(rows)
    for hemi_label in data.hemisphere.unique():
        idx = data.hemisphere == hemi_label
        se_ref = reference.loc[(hemi_label, "SE")]
        dfa_ref = reference.loc[(hemi_label, "DFA")]
        data.loc[idx, "Composite"] = ((data.loc[idx, "SE"] - se_ref["mean"]) / se_ref["sd_population"]
                                      - (data.loc[idx, "DFA"] - dfa_ref["mean"]) / dfa_ref["sd_population"])
    if data.groupby(["hemisphere", "cohort"]).subject.nunique().to_dict() != {
            (h, c): n for _, h, _ in HEMIS for c, n in (("Cohort 1", 8), ("Cohort 2", 7))}:
        raise ValueError("Unexpected participant counts")
    return data


def main():
    source.setup_style()
    data = baseline_records()
    results = []
    residuals = {}
    cohort_results = []
    for _, hemi, _ in HEMIS:
        for metric in METRICS:
            df = data[data.hemisphere == hemi].copy()
            df["eeg_z"] = df.groupby("cohort")[metric].transform(source.zscore_safe)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["eeg_z", "fma0", "fma6"])
            df["fma6_z"] = source.zscore_safe(df["fma6"])
            df["fma0_z"] = source.zscore_safe(df["fma0"])
            base = smf.ols("fma6_z ~ fma0_z + C(cohort)", data=df).fit()
            fit = smf.ols("fma6_z ~ fma0_z + C(cohort) + eeg_z", data=df).fit()
            ci = fit.conf_int().loc["eeg_z"]
            f_stat = max(0, ((base.ssr - fit.ssr) / 1) / (fit.ssr / fit.df_resid))
            p_increment = float(f_dist.sf(f_stat, 1, fit.df_resid))
            x_adjusted = smf.ols("eeg_z ~ fma0_z + C(cohort)", data=df).fit().resid
            partial_r, _ = pearsonr(x_adjusted, base.resid)
            results.append({"metric": metric, "hemisphere": hemi, "n": len(df),
                            "n_cohort1": sum(df.cohort == "Cohort 1"),
                            "n_cohort2": sum(df.cohort == "Cohort 2"),
                            "beta_standardized": float(fit.params["eeg_z"]),
                            "partial_r": float(partial_r),
                            "ci95_low": float(ci.iloc[0]), "ci95_high": float(ci.iloc[1]),
                            "p_eeg": float(fit.pvalues["eeg_z"]),
                            "p_adds_to_baseline_cohort": p_increment,
                            "base_r2": float(base.rsquared), "full_r2": float(fit.rsquared),
                            "delta_r2": float(fit.rsquared - base.rsquared)})
            if hemi == "Contralesional" and metric in ("SE", "Composite"):
                residuals[metric] = (df.assign(eeg_adjusted=x_adjusted, fma_adjusted=base.resid),
                                     fit, partial_r)
                for cohort, group in df.groupby("cohort"):
                    r_cohort, p_cohort = cohort_partial_correlation(group)
                    cohort_results.append({"metric": metric, "hemisphere": hemi,
                                           "cohort": cohort, "n": len(group),
                                           "partial_r": r_cohort, "p_two_sided": p_cohort})
    result = pd.DataFrame(results)
    fig = plt.figure(figsize=(17.5, 15.0), facecolor="white")
    grid = fig.add_gridspec(3, 2, left=.09, right=.97, top=.945, bottom=.065,
                            height_ratios=[1.3, 1.0, 1.0], hspace=.60, wspace=.27)
    top=grid[0,:].subgridspec(1,2,width_ratios=[1,1.55],wspace=.06)
    ax = fig.add_subplot(top[0, 0])
    table=fig.add_subplot(top[0,1]);table.axis('off')
    ax.axvline(0, color="#6B7280", lw=.9)
    for metric_i, metric in enumerate(METRICS):
        center = (len(METRICS) - metric_i - 1) * 2.7
        for hemi_i, (_, hemi, color) in enumerate(HEMIS):
            row = result[(result.metric == metric) & (result.hemisphere == hemi)].iloc[0]
            y = center + .42 - .84 * hemi_i
            ax.plot([row.ci95_low, row.ci95_high], [y, y], color=color, lw=2)
            ax.scatter(row.beta_standardized, y, color=color, s=40, zorder=3)
            table.text(.02, y,
                    f"β={row.beta_standardized:+.2f}   "
                    f"95% CI [{row.ci95_low:+.2f}, {row.ci95_high:+.2f}]   "
                    f"partial r={row.partial_r:+.2f}   "
                    f"p={row.p_eeg:.3g}",
                    fontsize=11, va="center", color="#000000" if row.p_eeg < .1 else "#263244",
                    fontweight="bold" if row.p_eeg < .1 else "normal")
    ax.set_yticks(np.arange(len(METRICS))[::-1] * 2.7,
                  ["z(SE) − z(DFA)" if m == "Composite" else m for m in METRICS])
    ax.set_ylim(-1.5, len(METRICS) * 2.7 - 1.3)
    table.set_ylim(ax.get_ylim())
    ax.set_xlim(-.32, .25)
    ax.set_xticks([-.3, -.15, 0, .15])
    ax.set_xlabel("Standardized β (95% CI)")
    fig.text(.09,.973,"A  Baseline EEG association adjusted for baseline FMA and cohort",
             fontsize=15,fontweight='bold',color='#263244')
    handles=[Line2D([0],[0],color=c,marker='o',label=h) for _,h,c in HEMIS]
    table.legend(handles=handles,loc='upper left',bbox_to_anchor=(.0,1.12),
                 ncol=2,frameon=False,fontsize=12)
    ax.grid(axis="x", color="#E5E7EB", lw=.7)
    ax.spines[["top", "right"]].set_visible(False)

    for col, metric in enumerate(("SE", "Composite")):
        ax = fig.add_subplot(grid[1, col])
        df, fit, partial_r = residuals[metric]
        for cohort, group in df.groupby("cohort"):
            ax.scatter(group.eeg_adjusted, group.fma_adjusted, s=43,
                       color=COHORT_COLORS[cohort], alpha=.85, edgecolor="white",
                       linewidth=.5, label=cohort)
        xx = np.linspace(df.eeg_adjusted.min(), df.eeg_adjusted.max(), 100)
        ax.plot(xx, fit.params["eeg_z"] * xx, color="#263244", lw=2)
        ax.axhline(0, color="#9CA3AF", lw=.8)
        ax.axvline(0, color="#9CA3AF", lw=.8)
        ax.set_title(f"{'B' if col == 0 else 'C'}  Contralesional {metric}: adjusted association",
                     loc="left", fontweight="bold",pad=33)
        ax.text(.0, 1.015, f"Pooled: partial r={partial_r:+.2f}, p={fit.pvalues['eeg_z']:.3f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        ax.set_xlabel(f"Baseline {('z(SE)−z(DFA)' if metric == 'Composite' else metric)}\n(adjusted for baseline FMA and cohort)")
        if col == 0:
            ax.set_ylabel("Adjusted 6-week FMA-UE (pooled z)")
        ax.grid(color="#E5E7EB", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 1:
            ax.legend(frameon=False, fontsize=11)

    for col, metric in enumerate(("SE", "Composite")):
        ax = fig.add_subplot(grid[2, col])
        df = data[data.hemisphere == "Contralesional"].copy()
        df["fma_change"] = df.fma6 - df.fma0
        for i, (cohort, group) in enumerate(df.groupby("cohort")):
            x = group[metric].to_numpy(float)
            y = group.fma_change.to_numpy(float)
            r, p = pearsonr(x, y)
            color = COHORT_COLORS[cohort]
            ax.scatter(x, y, s=43, color=color, alpha=.85, edgecolor="white",
                       linewidth=.5, label=cohort, zorder=3)
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, slope * xx + intercept, color=color, lw=1.5,
                    ls="--", alpha=.65, zorder=2)
            ax.text(.02, .97 - .09 * i,
                    f"{cohort}: r={r:+.2f}, p={p:.3g}, n={len(group)}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=12,
                    color=color, bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        ax.axhline(0, color="#9CA3AF", lw=.8)
        ax.set_title(f"{'D' if col == 0 else 'E'}  Contralesional {metric}: 6-week FMA change",
                     loc="left", fontweight="bold",pad=15)
        ax.set_xlabel(f"Baseline {('z(SE)−z(DFA)' if metric == 'Composite' else metric)}")
        if col == 0:
            ax.set_ylabel("6-week FMA-UE change (points)")
        ax.grid(color="#E5E7EB", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)

    polish(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figureS3_baseline_adjusted_sensitivity.png", dpi=220)
    fig.savefig(OUT / "figureS3_baseline_adjusted_sensitivity.pdf")
    plt.close(fig)
    result.to_csv(OUT / "figureS3_baseline_adjusted_sensitivity.csv", index=False)
    pd.DataFrame(cohort_results).to_csv(OUT / "figureS3_baseline_adjusted_sensitivity_cohorts.csv", index=False)
    data.to_csv(OUT / "figureS3_baseline_adjusted_sensitivity_participants.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
