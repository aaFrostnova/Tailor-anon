"""Request satisfaction with its 95% Wilson interval, one panel per scenario.

Reads only the committed evaluation snapshot, so the figure is reproducible from
the repository alone:

    python scripts/plot_confidence_intervals.py

The interval is drawn where it is read, next to the rate it qualifies, so the
reader compares it against the gaps between methods rather than against nothing.
Emphasis rather than eleven hues: the two TAILOR rows carry the accent, every
baseline is the same recessive gray, and identity comes from the row label.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent.parent
SNAPSHOT = PAPER / 'data/main_no_ctrlregen07_20260924.json'
SCENARIOS = [('S1', 'Signal'), ('S2', 'Geometric'), ('S3', 'Regen. / edit'),
             ('S4', 'Removal'), ('S5', 'Broad')]
Z = 1.959963984540054
# One representative per method family, each the strongest of its family in
# Table 1, so the rows form a ladder rather than a crowd.
ROWS = [('TrustMark', 'TrustMark'), ('TrustMarkGeo', 'TrustMark + Geo.'),
        ('EnumGeo', 'Enumeration + Geo.'), ('Greedy06', 'TAILOR-G'),
        ('SMT06', 'TAILOR-F')]
ACCENT, ACCENT_LIGHT, GRAY, RULE, INK = '#27847d', '#74aaa2', '#6b7280', '#c9d1da', '#000000'


def wilson(accepted, n):
    p = accepted / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return 100 * p, 100 * centre, 100 * half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=PAPER / 'figures')
    out = ap.parse_args().output
    plt.rcParams.update({'font.family': 'Nimbus Roman', 'mathtext.fontset': 'stix',
                         'text.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    summary = json.loads(SNAPSHOT.read_bytes())['summary']
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(5.5, 1.62), sharey=True,
                             gridspec_kw=dict(wspace=0.34, left=0.245, right=0.992,
                                              top=0.775, bottom=0.215))
    widest = (0.0, '', '')
    for ax, (key, label) in zip(axes, SCENARIOS):
        n = summary['SMT06']['scenarios'][key]['n']
        for i, (method, name) in enumerate(ROWS):
            cell = summary[method]['scenarios'][key]
            rate, centre, half = wilson(cell['accepted'], cell['n'])
            colour = ACCENT if method == 'SMT06' else ACCENT_LIGHT if method == 'Greedy06' else GRAY
            y = len(ROWS) - 1 - i
            ax.errorbar(centre, y, xerr=half, fmt='o', ms=3.0, lw=1.1, capsize=2.4,
                        color=colour, mec=colour, mfc=colour, zorder=3, clip_on=False)
            if half > widest[0]:
                widest = (half, name, key)
        ax.set_title(f'{key}  {label}\n$n$ = {n:,}', fontsize=6.4, pad=3.2, color=INK)
        ax.set_xlim(-7, 107)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(['0', '50', '100'])
        ax.tick_params(axis='x', labelsize=6.2, length=2, pad=1.6, colors=INK)
        ax.tick_params(axis='y', length=0)
        for x in (0, 50, 100):
            ax.axvline(x, color=RULE, lw=0.5, zorder=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(RULE)
        ax.spines['bottom'].set_linewidth(0.6)
    axes[0].set_yticks(range(len(ROWS)))
    axes[0].set_yticklabels([name for _, name in ROWS][::-1], fontsize=6.6, color=INK)
    axes[0].set_ylim(-0.6, len(ROWS) - 0.4)
    fig.supxlabel('Request satisfaction (%)', fontsize=7.0, y=0.018, color=INK)
    # one direct label: the widest interval in the figure, which bounds every other
    axes[3].annotate(f'widest interval: $\\pm${widest[0]:.2f} pp',
                     xy=(0.5, -0.30), xycoords='axes fraction', ha='center',
                     fontsize=6.0, color=GRAY, annotation_clip=False)
    out.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'confidence_intervals.{ext}', dpi=400, facecolor='white')
    print(f'wrote {out}/confidence_intervals.pdf; widest half-width '
          f'{widest[0]:.2f} pp at {widest[1]} in {widest[2]}')


if __name__ == '__main__':
    main()
