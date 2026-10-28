#!/usr/bin/env python3
"""Reproduce manuscript Figures 2–4 and supplementary Figure S3."""
from pathlib import Path

import analysis_and_figure_data as source
from plot_figure4_behavior_associations import main as plot_figure4
from plot_figureS3_baseline_adjusted_sensitivity import main as plot_figureS3


def main():
    source.OUT_DIR = Path("figures")
    source.DFA_FILENAME = "DFA.npy"
    source.PAIRED_TEST = "ttest"
    source.setup_style()
    source.plot_eeg_cohort_figures("trajectory")
    source.plot_eeg_cohort_figures("replication", pooled_week=4)
    plot_figure4()
    plot_figureS3()


if __name__ == "__main__":
    main()
