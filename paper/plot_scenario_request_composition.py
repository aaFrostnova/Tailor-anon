"""Plot scenario request totals, baseline successes, and attack inclusion counts."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea
import numpy as np

from plot_request_radars import ATTACKS

PAPER = Path(__file__).resolve().parents[1]
SCENES = ['S1', 'S2', 'S3', 'S4', 'S5']
INK = MUTED = '#000000'
SCENE_COLORS = ['#235784', '#328f89', '#bb9051', '#9378ad', '#c16e61']
BASELINE_LABELS = {'Greedy06':'TAILOR-G', 'EnumGeo':'Enumeration + Geo.', 'Fixed1':'Enumeration'}


def aggregate():
    hashes = {}
    def read(name):
        path = PAPER/'data'/name
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    report = read('main_no_ctrlregen07_20260924.json')
    original = read('request_specifications_6532.json')['rows']
    extension = read('editing_requests_1000_20260922.json')
    for path, sha in hashes.items():
        if path in report['source_sha256']:
            assert sha == report['source_sha256'][path], ('source changed', path)
    ids = set(report['retained_spec_ids'])
    rows = [r for r in original+extension if r['spec_id'] in ids]
    assert len(rows) == len({r['spec_id'] for r in rows}) == len(ids) == 7321
    assert {a for r in rows for a in r['inputs']['attacks']} == {a for a,_ in ATTACKS}
    assert all(len(r['inputs']['attacks']) == len(set(r['inputs']['attacks'])) for r in rows)
    summaries = report['summary']
    eligible = [k for k in summaries if not k.startswith('SMT') and k != 'Greedy04']
    selected = sorted(eligible, key=lambda k:(-summaries[k]['macro_percent'], k))[:3]
    assert selected == ['Greedy06', 'EnumGeo', 'Fixed1']
    bars, attacks, scenes = [], [], {}
    for scene in SCENES:
        part = [r for r in rows if r['cls'] == scene.replace('S', 'C')]
        counts = Counter(a for r in part for a in set(r['inputs']['attacks']))
        n = len(part)
        assert n == report['distribution'][scene]['n']
        scenes[scene] = dict(n_requests=n, attack_incidences=sum(counts.values()))
        bars.append(dict(scenario=scene, method='Total', label='Total requests', count=n))
        for method in selected:
            stat = summaries[method]['scenarios'][scene]
            assert stat['n'] == n and 0 <= stat['accepted'] <= n
            assert abs(stat['percent']-100*stat['accepted']/n) < 1e-10
            bars.append(dict(scenario=scene, method=method,
                             label=BASELINE_LABELS[method], count=stat['accepted']))
        for attack, label in ATTACKS:
            assert 0 <= counts[attack] <= n
            attacks.append(dict(scenario=scene, attack=attack, label=label,
                                count=counts[attack], scenario_requests=n))
    assert sum(r['n_requests'] for r in scenes.values()) == 7321
    for attack, _ in ATTACKS:
        assert sum(r['count'] for r in attacks if r['attack'] == attack) == report['attack_counts'][attack]
    for method in selected:
        assert sum(r['count'] for r in bars if r['method'] == method) == summaries[method]['accepted']
    return dict(n_requests=7321, attack_settings=20, scenarios=scenes,
        baseline_selection='Top three non-full methods by main-table unweighted scenario mean; margin 0.06',
        selected_baselines=[dict(method=k, label=BASELINE_LABELS[k],
            macro_satisfaction_percent=summaries[k]['macro_percent']) for k in selected],
        bars=bars, attack_counts=attacks, source_sha256=hashes,
        radar_metric='Number of distinct requests in each scenario containing the attack',
        radar_normalized=False, attacks_per_request_can_overlap=True,
        all_current_request_ids_accounted_for=True)


def save(fig, out, stem):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out/f'{stem}.{ext}', dpi=350, facecolor='white', bbox_inches='tight', pad_inches=.06)
    fig.savefig(out/f'{stem}_preview.png', dpi=160, facecolor='white', bbox_inches='tight', pad_inches=.06)
    plt.close(fig)


def plot_bars(data, out):
    methods = ['Total']+[r['method'] for r in data['selected_baselines']]
    colors = ['#dce3eb', '#328782', '#b58a50', '#d8bf99']
    labels = {'Total':'Total requests', **BASELINE_LABELS}
    fig = plt.figure(figsize=(8.3, 5.7))
    ax = fig.add_axes([.09, .105, .865, .75])
    ax.set_facecolor('#f8fafc')
    y = np.arange(5)
    offsets = [-.30, -.10, .10, .30]
    by = {(r['scenario'], r['method']):r['count'] for r in data['bars']}
    for method, color, offset in zip(methods, colors, offsets):
        vals = [by[s,method] for s in SCENES]
        ax.barh(y+offset, vals, height=.17, color=color,
                edgecolor='#b9c5d1' if method=='Total' else color,
                linewidth=.55, label=labels[method], zorder=3)
        for yy, count in zip(y+offset, vals):
            ax.text(count+22, yy, f'{count:,}', va='center', ha='left',
                    fontsize=9.7, color=INK, weight='bold' if method=='Total' else 'normal')
    ax.set_yticks(y, SCENES, fontsize=12.8)
    ax.set_ylim(4.58, -.58)
    ax.set_xlim(0, 2330)
    ax.set_xticks([0,500,1000,1500,2000], ['0','500','1,000','1,500','2,000'])
    ax.set_xlabel('Number of requests', fontsize=12.2, labelpad=7)
    ax.tick_params(axis='x', length=0, pad=5, labelsize=10.8, labelcolor=MUTED)
    ax.tick_params(axis='y', length=0, pad=12, labelcolor=INK)
    ax.grid(axis='x', color='#e0e6ed', lw=.6, zorder=0)
    for yy in [.5,1.5,2.5,3.5]:
        ax.axhline(yy, color='white', lw=2, zorder=1)
    for side in ('top','right','left'): ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#b2bfca')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.525, .986), ncol=2, frameon=False,
               fontsize=11, columnspacing=2.2, handlelength=1.7,
               handletextpad=.6, labelspacing=.6)
    save(fig, out, 'scenario_request_counts')


def plot_radar(data, out):
    names = dict(ATTACKS)
    angles = np.linspace(0, 2*np.pi, len(ATTACKS), endpoint=False)
    closed = np.r_[angles, angles[0]]
    by = {(r['scenario'], r['attack']):r['count'] for r in data['attack_counts']}
    # Designed for the half-width manuscript panel: enlarge text relative to
    # the canvas, rather than increasing a canvas that LaTeX shrinks again.
    fig = plt.figure(figsize=(6.6, 6.4))
    ax = fig.add_axes([.24, .26, .52, .64], projection='polar')
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 2200)
    ax.set_yticks([500,1000,1500,2000], ['500','1,000','1,500','2,000'])
    ax.tick_params(axis='y', labelsize=13.5, labelcolor=MUTED)
    ax.set_rlabel_position(35)
    ax.set_xticks(angles, [])
    label_boxes = []
    for (attack, _), angle in zip(ATTACKS, angles):
        horizontal = np.sin(angle)
        align = 'left' if horizontal > .15 else 'right' if horizontal < -.15 else 'center'
        label = TextArea(names[attack], textprops=dict(fontsize=15, color=INK, ha=align))
        offset = {'bright': 6, 'hflip': -4}.get(attack, 0)
        box = AnnotationBbox(label, (angle, 2420), xybox=(0, offset),
            xycoords='data', boxcoords='offset points', frameon=False, pad=0,
            box_alignment=(0 if align=='left' else 1 if align=='right' else .5, .5),
            annotation_clip=False)
        ax.add_artist(box)
        label_boxes.append(box)
    # Retain full single-line names, resolving any collisions through placement.
    for _ in range(30):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = [box.get_window_extent(renderer) for box in label_boxes]
        moved = False
        for i, a in enumerate(bounds):
            for j in range(i+1, len(bounds)):
                b = bounds[j]
                if min(a.x1, b.x1)-max(a.x0, b.x0) <= 1:
                    continue
                overlap = min(a.y1, b.y1)-max(a.y0, b.y0)
                if overlap < 1:
                    continue
                dy = (overlap+5)*72/fig.dpi/2
                sign = 1 if a.y0+a.y1 > b.y0+b.y1 else -1
                for k, direction in ((i, sign), (j, -sign)):
                    x, y = label_boxes[k].xybox
                    label_boxes[k].xybox = (x, y+direction*dy)
                moved = True
        if not moved:
            break
    ax.set_facecolor('#fafbfd')
    ax.yaxis.grid(True, color='#d6dfe8', lw=.7)
    ax.xaxis.grid(True, color='#e3e9ef', lw=.55)
    ax.spines['polar'].set_color('#c8d3dd')
    ax.spines['polar'].set_linewidth(.7)
    layers = []
    for scene, color in zip(SCENES, SCENE_COLORS):
        vals = np.array([by[scene,a] for a,_ in ATTACKS])
        layers.append((float(np.sum(vals*np.roll(vals,-1))), scene, color, np.r_[vals,vals[0]]))
    for _, scene, color, vals in sorted(layers, reverse=True):
        ax.fill(closed, vals, color=color, alpha=.025 if scene=='S5' else .045,
                linewidth=0, zorder=1)
    handles = []
    for scene, color, marker in zip(SCENES, SCENE_COLORS, ['o','s','D','^','p']):
        vals = [by[scene,a] for a,_ in ATTACKS]
        kw = dict(color=color, linewidth=1.9, marker=marker, markersize=4.2,
                  markerfacecolor='white', markeredgewidth=.9)
        ax.plot(closed, vals+[vals[0]], **kw, zorder=3)
        handles.append(Line2D([], [], **kw,
            label=f"{scene} (n={data['scenarios'][scene]['n_requests']:,})"))
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.50,.160),
        ncol=3, frameon=False, fontsize=14, handlelength=1.3,
        handletextpad=.4, columnspacing=1.0, labelspacing=.65)
    save(fig, out, 'attack_counts_by_scenario')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    data = aggregate()
    plt.rcParams.update({'font.family':'Nimbus Roman', 'mathtext.fontset':'stix', 'font.size':11,
        'text.color':INK, 'axes.labelcolor':INK, 'pdf.fonttype':42,
        'ps.fonttype':42, 'svg.fonttype':'none'})
    plot_bars(data, args.output_dir)
    plot_radar(data, args.output_dir)
    (args.output_dir/'scenario_request_composition.json').write_text(json.dumps(data, indent=2)+'\n')
    for key, stem in [('bars','scenario_request_counts'), ('attack_counts','attack_counts_by_scenario')]:
        with (args.output_dir/(stem+'.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[key][0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(data[key])
    print(json.dumps({'n_requests':data['n_requests'], 'selected':data['selected_baselines'],
        'scenarios':data['scenarios'], 'bar_records':len(data['bars']),
        'radar_records':len(data['attack_counts'])}, indent=2))


if __name__ == '__main__':
    main()
