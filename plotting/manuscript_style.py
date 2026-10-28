"""Shared typography and annotation spacing for revised main figures."""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def polish(fig, label_size=13, tick_size=11, title_size=14):
    for ax in fig.axes:
        if not ax.axison:
            continue
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=tick_size, length=3.5, pad=5)
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        ax.xaxis.labelpad=8
        ax.yaxis.labelpad=8
        ax.title.set_fontsize(title_size)
        for text in ax.texts:
            if len(text.get_text()) == 1 and text.get_text().isupper():
                text.set_fontsize(17)
            else:
                text.set_fontsize(max(text.get_fontsize(), 10.5))


def trajectory_space(ax):
    """Reserve separate bands for visit p values and sample counts."""
    lo, hi=ax.get_ylim()
    span=hi-lo
    ax.set_ylim(lo-.04*span, hi+.20*span)
    for text in ax.texts:
        label=text.get_text()
        x, _=text.get_position()
        if label.startswith('p='):
            text.set_transform(ax.get_xaxis_transform())
            text.set_position((x, .97))
            text.set_va('top')
        elif label.startswith('n='):
            text.set_transform(ax.get_xaxis_transform())
            text.set_position((x, .015))
            text.set_va('bottom')
            text.set_color('#667085')
