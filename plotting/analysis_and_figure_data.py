#!/usr/bin/env python3
"""Shared data loading, statistics, and rendering for final Figures 2–4."""
from pathlib import Path
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, pearsonr, sem, ttest_rel, wilcoxon
OUT_DIR = Path(os.environ.get('FIGURE_OUT_DIR', 'figures'))
DFA_FILENAME = os.environ.get('DFA_FILENAME', 'DFA.npy')
OUTPUT_SUFFIX = os.environ.get('OUTPUT_SUFFIX', '')
PAIRED_TEST = os.environ.get('PAIRED_TEST', 'ttest')
COLORS = {'baseline': '#445B6E', 'post': '#D96545', 'green': '#2E8B73', 'gold': '#C4972F', 'gray': '#8A96A3', 'light': '#B9C2CC', 'dark': '#263244', 'grid': '#D8DEE6'}
HEMI_PAIR_KEYS = ['contra', 'ipsi']
HEMI_PAIR_TITLES = ['ipsilesional', 'contralesional']
LEFT_ELC = [48, 41, 23, 63, 58, 43, 75, 65, 78]
RIGHT_ELC = [206, 205, 212, 184, 182, 193, 142, 163, 171]
MIN_COMMON_ROI_CHANNELS = 6
FMA_C1 = np.array([
    30, 38, 27, 27, 20, 23, 39, 41, 23, 24, 39, 43, 47, 50, 27, 30,
    21, 22,
]).reshape(-1, 2)
FMA_C2 = np.array([40, 48, 52, 51, 51, 53, 61, 63, 10, 15, 16, 24, 22, 26, 49, 48, 54, 33, 44, 55, 35, 43, 46, 13, 23, 31, 28, 24, 39, 48, 48, 48, 47, 47, 59, 60, 62, 44, 47, 54, 56])

def setup_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': COLORS['dark'], 'xtick.color': COLORS['dark'], 'ytick.color': COLORS['dark'], 'figure.dpi': 160, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

def paired_pvalue(pre, post, alternative):
    """Return a one-sided paired-test p value selected for the figure run."""
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    keep = np.isfinite(pre) & np.isfinite(post)
    pre, post = (pre[keep], post[keep])
    if PAIRED_TEST == 'wilcoxon':
        return float(wilcoxon(pre, post, alternative=alternative, method='auto').pvalue)
    return float(ttest_rel(pre, post, alternative=alternative).pvalue)

def significance_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''

def savefig(name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        plt.savefig(OUT_DIR / f'{name}{OUTPUT_SUFFIX}.{ext}', pad_inches=0.1)
    plt.close()

def panel_label(ax, label):
    bbox = ax.get_position()
    left_label_x = ax.figure.subplotpars.left - 0.035
    label_x = left_label_x if bbox.x0 <= ax.figure.subplotpars.left + 0.08 else bbox.x0 - 0.035
    ax.figure.text(label_x, bbox.y1 + 0.012, label, transform=ax.figure.transFigure, fontsize=13, fontweight='bold', color=COLORS['dark'], ha='left', va='bottom')

def load_metric(path, feature, filename_override=None):
    if feature == 'DFA':
        filename = DFA_FILENAME
    else:
        filename = f'{feature}.npy'
    if filename_override is not None:
        filename = filename_override
    arr = np.load(path / filename, allow_pickle=True)
    if feature == 'MSE':
        arr = np.nanmean(arr, axis=2)
    return np.asarray(arr, dtype=float)

def metric_transform(feature, arr):
    return (arr, feature)

def common_roi_mean(arr, channels, min_channels=MIN_COMMON_ROI_CHANNELS):
    """Average one fixed set of channels that is valid at every visit."""
    arr = np.asarray(arr, dtype=float)
    roi = arr[:, channels]
    common = np.isfinite(roi).all(axis=0)
    if common.sum() < min_channels:
        return np.full(arr.shape[0], np.nan)
    return roi[:, common].mean(axis=1)

def load_cohort1(feature, transform=True):
    """Load the anonymized baseline/post Cohort 2 feature arrays."""
    root = Path('data/cohort2')
    subjects = sorted([p for p in root.iterdir() if p.is_dir()])
    side = np.array(['R', 'R', 'L', 'R', 'L', 'L', 'R', 'R', 'L'])
    keep = np.ones(len(subjects), dtype=bool)
    keep[[6, 8]] = False
    all_ch = []
    for subject in subjects:
        arr = load_metric(subject, feature)[:2]
        if transform:
            arr, label = metric_transform(feature, arr)
        else:
            label = feature
        all_ch.append(arr)
    all_ch = np.stack(all_ch, axis=0)
    left = np.stack([common_roi_mean(arr, LEFT_ELC) for arr in all_ch])
    right = np.stack([common_roi_mean(arr, RIGHT_ELC) for arr in all_ch])
    ipsi = np.stack([left[i] if s == 'L' else right[i] for i, s in enumerate(side)], axis=0)
    contra = np.stack([left[i] if s == 'R' else right[i] for i, s in enumerate(side)], axis=0)
    bilateral = (ipsi + contra) / 2
    return {'label': label, 'subjects': [p.name for p in subjects], 'keep': keep, 'channels': all_ch, 'ipsi': ipsi, 'contra': contra, 'bilateral': bilateral}

def load_cohort2(feature, transform=True, filename_override=None):
    """Load the anonymized longitudinal Cohort 1 feature arrays."""
    root = Path('data/cohort1')
    visit_counts = {10: 3, 1: 5, 2: 6, 3: 3, 4: 3, 5: 3, 6: 5, 7: 5, 8: 4, 9: 4}
    subjects = [root / f'sub{sid:02d}' for sid in visit_counts]
    side_by_subject_id = {1: 'L', 2: 'L', 3: 'L', 4: 'L', 5: 'R', 6: 'L', 7: 'L', 8: 'L', 9: 'R', 10: 'R'}
    rows = []
    all_week_channels = []
    fma_i = 0
    for subject in subjects:
        sid = int(subject.name.removeprefix('sub'))
        n_visits = visit_counts[sid]
        weeks = np.load(subject / 'Weeks.npy', allow_pickle=True)[:n_visits]
        arr = load_metric(subject, feature, filename_override=filename_override)[:n_visits]
        if transform:
            arr, label = metric_transform(feature, arr)
        else:
            label = feature
        left = common_roi_mean(arr, LEFT_ELC)
        right = common_roi_mean(arr, RIGHT_ELC)
        ipsi = left if side_by_subject_id[sid] == 'L' else right
        contra = left if side_by_subject_id[sid] == 'R' else right
        bilateral = (ipsi + contra) / 2
        for i, week in enumerate(weeks):
            rows.append({'sid': sid, 'week': int(round(float(week))), 'ipsi': float(ipsi[i]), 'contra': float(contra[i]), 'bilateral': float(bilateral[i]), 'fma': float(FMA_C2[fma_i])})
            all_week_channels.append((int(round(float(week))), arr[i]))
            fma_i += 1
    return {'label': label, 'rows': rows, 'channels': all_week_channels}

def load_cohort2_composite(dfa_filename=None):
    se = load_cohort2('SE')
    dfa = load_cohort2('DFA', filename_override=dfa_filename)
    mse = load_cohort2('MSE')
    rows = []
    for r_se, r_dfa, r_mse in zip(se['rows'], dfa['rows'], mse['rows']):
        rows.append({'sid': r_se['sid'], 'week': r_se['week'], 'ipsi': r_se['ipsi'] + r_mse['ipsi'] - r_dfa['ipsi'], 'contra': r_se['contra'] + r_mse['contra'] - r_dfa['contra'], 'bilateral': r_se['bilateral'] + r_mse['bilateral'] - r_dfa['bilateral'], 'fma': r_se['fma']})
    channels = [(wk, se_ch + mse_ch - dfa_ch) for (wk, se_ch), (_, dfa_ch), (_, mse_ch) in zip(se['channels'], dfa['channels'], mse['channels'])]
    return {'label': 'raw composite', 'rows': rows, 'channels': channels}

def cohort2_matrix(cohort2, key='bilateral'):
    rows = cohort2['rows']
    weeks = np.array(sorted(set((r['week'] for r in rows))))
    sids = np.array(sorted(set((r['sid'] for r in rows))))
    mat = np.full((len(sids), len(weeks)), np.nan)
    sid_index = {sid: i for i, sid in enumerate(sids)}
    week_index = {week: i for i, week in enumerate(weeks)}
    for row in rows:
        mat[sid_index[row['sid']], week_index[row['week']]] = row[key]
    return (sids, weeks, mat)

def plot_cohort2_key_trajectory(ax, cohort2, key, ylabel, title, stats=None, x_scale=1.0, x_label='Week', mean_color=None, sparse_n=False, max_week=None, mean_max_week=None, show_sigstars=False, equally_spaced_weeks=False):
    _, weeks, mat = cohort2_matrix(cohort2, key=key)
    if max_week is not None:
        keep_weeks = weeks <= max_week
        weeks, mat = (weeks[keep_weeks], mat[:, keep_weeks])
    x_values = np.arange(len(weeks)) if equally_spaced_weeks else weeks * x_scale
    mean_color = mean_color or COLORS['green']
    for row in mat:
        ax.plot(x_values, row, color=COLORS['light'], lw=1.1, marker='o', ms=3.2, alpha=0.42)
    n = np.sum(np.isfinite(mat), axis=0)
    mean_keep = np.ones(len(weeks), dtype=bool) if mean_max_week is None else weeks <= mean_max_week
    mean = np.nanmean(mat[:, mean_keep], axis=0)
    se = sem(mat[:, mean_keep], axis=0, nan_policy='omit')
    ax.errorbar(x_values[mean_keep], mean, yerr=se, color=mean_color, lw=2.35, marker='o', ms=5.3, capsize=3.5)
    y0, y1 = ax.get_ylim()
    for i, (x_value, nn) in enumerate(zip(x_values, n)):
        if not sparse_n or i == 0 or x_value >= 2:
            ax.text(x_value, y0, f'n={int(nn)}', ha='center', va='bottom', fontsize=7.0, color=COLORS['gray'])
    if stats:
        y = y1 - 0.08 * (y1 - y0)
        for i, (week, p) in enumerate(stats):
            y_stat = y - (i * 0.11 if x_label == 'Months' else i % 2 * 0.07) * (y1 - y0)
            ax.text(x_values[np.where(weeks == week)[0][0]] if equally_spaced_weeks else week * x_scale, y_stat, f"p={p:.3g}{(significance_stars(p) if show_sigstars else '')}", ha='center', va='top', fontsize=7.6, bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=0.8))
    if x_label == 'Months':
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlim(-0.18, 4.1)
    else:
        ax.set_xticks(x_values, labels=weeks if equally_spaced_weeks else None)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', color=COLORS['grid'], alpha=0.75, lw=0.8)

def cohort2_paired_week_stats(cohort2, key, target_weeks, alt):
    _, weeks, mat = cohort2_matrix(cohort2, key=key)
    if 0 not in weeks:
        raise ValueError('Longitudinal paired stats require week 0 baseline data.')
    pre_idx = int(np.where(weeks == 0)[0][0])
    stats = []
    for week in target_weeks:
        if week not in weeks:
            continue
        week_idx = int(np.where(weeks == week)[0][0])
        vals = mat[:, [pre_idx, week_idx]]
        vals = vals[np.isfinite(vals).all(axis=1)]
        if vals.shape[0] < 3:
            continue
        p = paired_pvalue(vals[:, 0], vals[:, 1], alternative=alt)
        stats.append((week, p))
    return stats

def merged_pre_post_at_week(c1, c2, key, target_week=6):
    c1_vals = c1[key][c1['keep']]
    c1_vals = c1_vals[np.isfinite(c1_vals).all(axis=1)]
    sids, weeks, mat = cohort2_matrix(c2, key=key)
    if 0 not in weeks or target_week not in weeks:
        raise ValueError(f'Longitudinal merge requires paired week 0 and week {target_week} data.')
    pre_idx = int(np.where(weeks == 0)[0][0])
    post_idx = int(np.where(weeks == target_week)[0][0])
    c2_vals = mat[:, [pre_idx, post_idx]]
    c2_vals = c2_vals[np.isfinite(c2_vals).all(axis=1)]
    vals = np.vstack([c1_vals, c2_vals])
    groups = np.array(['C1'] * c1_vals.shape[0] + ['C2'] * c2_vals.shape[0])
    return (vals, groups)

def plot_merge_ladder(ax, c1, c2, key, ylabel, title, alt='greater', show_sigstars=False, target_week=6):
    vals, groups = merged_pre_post_at_week(c1, c2, key, target_week=target_week)
    x = np.array([0, 1])
    for row, group in zip(vals, groups):
        color = COLORS['light'] if group == 'C1' else '#A7BFA4'
        ax.plot(x, row, color=color, lw=1.25, marker='o', ms=3.7, alpha=0.75)
    mean = np.nanmean(vals, axis=0)
    se = sem(vals, axis=0, nan_policy='omit')
    ax.errorbar(x, mean, yerr=se, color=COLORS['dark'], lw=2.35, marker='o', ms=5.6, capsize=3.5)
    p = paired_pvalue(vals[:, 0], vals[:, 1], alternative=alt)
    ax.text(0.5, 0.96, f"n={vals.shape[0]}, p={p:.3g}{(significance_stars(p) if show_sigstars else '')}", ha='center', va='top', transform=ax.transAxes, fontsize=8.3)
    ax.set_xticks(x)
    ax.set_xticklabels(['baseline', f'Post / wk{target_week}'])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', color=COLORS['grid'], alpha=0.75, lw=0.8)

def zscore_safe(values):
    arr = np.asarray(values, dtype=float)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return arr * np.nan
    return (arr - np.nanmean(arr)) / sd

def plot_eeg_cohort_figures(part='full', pooled_week=6):
    """Cohort 1 trajectories, Cohort 2 pairs, and selected-week pooled pairs."""
    if part not in ('full', 'trajectory', 'replication'):
        raise ValueError(f'Unknown EEG figure part: {part}')
    metric_specs = [('SE', 'less'), ('DFA', 'greater'), ('MSE', 'less')]
    longitudinal = {metric: load_cohort2(metric, filename_override='DFA.npy' if metric == 'DFA' else None) for metric, _ in metric_specs}
    post = {metric: load_cohort1(metric) for metric, _ in metric_specs}
    fig = plt.figure(figsize=(15.2, 22.0) if part == 'full' else (16.5, 13.5))
    nrows = 7 if part == 'full' else 3 if part == 'trajectory' else 4
    gs = fig.add_gridspec(nrows, 6, left=0.065, right=0.985, top=0.973 if part == 'full' else 0.965, bottom=0.052 if part == 'full' else 0.085, height_ratios=[1.1] * 3 + [0.9] * 4 if part == 'full' else [1] * nrows, hspace=0.65, wspace=0.8)
    trajectory_rows = [[fig.add_subplot(gs[row, :3]), fig.add_subplot(gs[row, 3:])] for row in range(3)] if part != 'replication' else []

    def draw_trajectory_pair(axes, cohort, feature, panel, alt='less'):
        stats = {key: cohort2_paired_week_stats(cohort, key, [2, 4, 6], alt=alt) for key in HEMI_PAIR_KEYS}
        for ax, key, title in zip(axes, HEMI_PAIR_KEYS, HEMI_PAIR_TITLES):
            plot_cohort2_key_trajectory(ax, cohort, key, feature, f'Cohort 1 {title}', stats=stats[key], mean_max_week=6, show_sigstars=True, equally_spaced_weeks=True)
        panel_label(axes[0], panel)
    if part != 'replication':
        for index, (metric, alt) in enumerate(metric_specs):
            draw_trajectory_pair(trajectory_rows[index], longitudinal[metric], metric, 'ABC'[index], alt=alt)
    if part != 'trajectory':
        row_offset = 3 if part == 'full' else 0
        for row_index, (hemi, hemi_title) in enumerate(zip(HEMI_PAIR_KEYS, HEMI_PAIR_TITLES)):
            axes = [fig.add_subplot(gs[row_index + row_offset, col * 2:(col + 1) * 2]) for col in range(3)]
            for ax, (metric, alt) in zip(axes, metric_specs):
                cohort = post[metric]
                values = np.asarray(cohort[hemi], dtype=float)[:, :2]
                active = np.asarray(cohort['keep'], dtype=bool)
                for individual, is_active in zip(values, active):
                    if not np.isfinite(individual).all():
                        continue
                    ax.plot([0, 1], individual, color=COLORS['light'] if is_active else '#7B61A8', marker='o' if is_active else 'D', ms=3.8 if is_active else 4.5, lw=1.2 if is_active else 1.6, alpha=0.75 if is_active else 0.9)
                paired = values[active]
                paired = paired[np.isfinite(paired).all(axis=1)]
                ax.errorbar([0, 1], paired.mean(axis=0), yerr=sem(paired, axis=0), color=COLORS['green'], lw=2.3, marker='o', ms=5.5, capsize=3.5)
                p = paired_pvalue(paired[:, 0], paired[:, 1], alternative=alt)
                ax.text(0.5, 0.97, f'active n={len(paired)}, p={p:.3g}{significance_stars(p)}', transform=ax.transAxes, ha='center', va='top', fontsize=8.3, bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=1.0))
                ax.set_xticks([0, 1], ['Baseline', 'Post'])
                ax.set_title(metric)
                ax.set_ylabel(metric)
                ax.grid(axis='y', color=COLORS['grid'], alpha=0.75, lw=0.8)
                ax.spines[['top', 'right']].set_visible(False)
            fig.text(0.075, axes[0].get_position().y1 + (0.034 if part == 'replication' else 0.014), f"{('EF' if part == 'full' else 'AB')[row_index]}   Cohort 2 {hemi_title}", fontsize=13, fontweight='bold', color=COLORS['dark'])
        for row_index, (hemi, hemi_title) in enumerate(zip(HEMI_PAIR_KEYS, HEMI_PAIR_TITLES)):
            axes = [fig.add_subplot(gs[row_index + row_offset + 2, col * 2:(col + 1) * 2]) for col in range(3)]
            for ax, (metric, alt) in zip(axes, metric_specs):
                plot_merge_ladder(ax, post[metric], longitudinal[metric], hemi, metric, metric, alt=alt, show_sigstars=True, target_week=pooled_week)
                ax.spines[['top', 'right']].set_visible(False)
            fig.text(0.075, axes[0].get_position().y1 + (0.034 if part == 'replication' else 0.014), f"{('GH' if part == 'full' else 'CD')[row_index]}   Pooled: Cohort 1 week {pooled_week} + Cohort 2 post · {hemi_title}", fontsize=13, fontweight='bold', color=COLORS['dark'])
    handles = [plt.Line2D([0], [0], color=COLORS['light'], marker='o', lw=1.4, ms=4, label='Active individual'), plt.Line2D([0], [0], color='#7B61A8', marker='D', lw=1.4, ms=4, label='Cohort 2 sham individual'), plt.Line2D([0], [0], color=COLORS['green'], marker='o', lw=2.2, ms=5, label='Cohort mean +/- SEM'), plt.Line2D([0], [0], color=COLORS['dark'], marker='o', lw=2.2, ms=5, label='Pooled mean +/- SEM')]
    if part != 'trajectory':
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.014 if part == 'full' else 0.026), ncol=2, frameon=False, fontsize=11)
    from manuscript_style import polish, trajectory_space
    polish(fig)
    for axes in trajectory_rows:
        for ax in axes:
            trajectory_space(ax)
            ax.set_title(ax.get_title(), pad=16)
    if part == 'replication':
        for ax in fig.axes:
            low, high = ax.get_ylim()
            ax.set_ylim(low, high + 0.2 * (high - low))
            ax.set_title(ax.get_title(), pad=14)
            ax.set_xlim(-0.1, 1.1)
    savefig({'full': 'figure2_3_combined', 'trajectory': 'figure2_cohort1_trajectories', 'replication': 'figure3_cohort2_and_pooled'}[part])

def fit_within_between_model(df, terms):
    fixed = ' + '.join(['time'] + terms)
    formula = f'resp ~ {fixed}'
    last_error = None
    for method in ('lbfgs', 'powell', 'cg'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = smf.mixedlm(formula, data=df, groups=df['subject']).fit(reml=False, method=method, disp=False)
            if result.converged:
                return result
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'Mixed model failed for {formula}: {last_error}')

def behavior_association_rows(weeks=None, cohorts_override=None, hemis=None):
    """Fit single-metric, joint, and metric-specific within/between models."""
    cohorts = cohorts_override if cohorts_override is not None else {'SE': load_cohort2('SE'), 'DFA': load_cohort2('DFA', filename_override='DFA.npy'), 'MSE': load_cohort2('MSE'), 'Composite': load_cohort2_composite(dfa_filename='DFA.npy')}
    model_rows = []
    fit_rows = []
    selected_weeks = None if weeks is None else set(weeks)
    for hemi in ('contra', 'ipsi', 'bilateral') if hemis is None else hemis:
        records = []
        for feature, cohort in cohorts.items():
            for row in cohort['rows']:
                if selected_weeks is not None and row['week'] not in selected_weeks:
                    continue
                records.append({'subject': row['sid'], 'time_raw': float(row['week']), 'resp_raw': float(row['fma']), feature: float(row[hemi])})
        df = pd.DataFrame(records).groupby(['subject', 'time_raw', 'resp_raw'], as_index=False).first()
        df['time'] = df['time_raw']
        df['resp'] = zscore_safe(df['resp_raw'])
        z_features = {}
        for feature in cohorts:
            eeg_z = pd.Series(zscore_safe(df[feature]), index=df.index)
            z_features[feature] = eeg_z
            df[f'{feature}_z'] = eeg_z
            between = eeg_z.groupby(df['subject']).transform('mean')
            df[f'{feature}_within'] = eeg_z - between
            df[f'{feature}_between'] = between
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            base = smf.mixedlm('resp ~ time', data=df, groups=df['subject']).fit(reml=False, method='lbfgs', disp=False)
        model_specs = []
        for feature in cohorts:
            model_specs.append(('Single metric', feature, [f'{feature}_z']))
        model_specs.append(('Joint SE+DFA+MSE', 'Joint', ['SE_z', 'DFA_z', 'MSE_z']))
        for feature in ('Composite', 'SE', 'DFA', 'MSE'):
            model_specs.append(('Within/between', feature, [f'{feature}_within', f'{feature}_between']))
        for model_name, metric_name, terms in model_specs:
            result = fit_within_between_model(df, terms)
            lrt = max(0.0, 2 * (float(result.llf) - float(base.llf)))
            df_diff = len(terms)
            model_p = float(chi2.sf(lrt, df_diff))
            fit_rows.append({'hemisphere': hemi, 'model': model_name, 'metric': metric_name, 'n_participants': int(df['subject'].nunique()), 'n_observations': int(len(df)), 'LRT_vs_time_only': lrt, 'df': df_diff, 'p_vs_time_only': model_p, 'AIC': float(result.aic)})
            ci = result.conf_int()
            for term in terms:
                clean_term = term[:-2] if term.endswith('_z') else term
                model_rows.append({'hemisphere': hemi, 'model': model_name, 'metric': clean_term if model_name == 'Joint SE+DFA+MSE' else metric_name, 'term': clean_term, 'component': 'within' if term.endswith('_within') else 'between' if term.endswith('_between') else 'overall', 'beta': float(result.params[term]), 'ci_low': float(ci.loc[term, 0]), 'ci_high': float(ci.loc[term, 1]), 'p': float(result.pvalues[term])})
    return (pd.DataFrame(model_rows), pd.DataFrame(fit_rows))
