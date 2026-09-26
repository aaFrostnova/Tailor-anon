#!/usr/bin/env python3
"""Export matched constraint sweeps, paired PSNR, and a frozen prediction diagnostic.

Only reads completed measurements. Writes all generated artifacts to --output.
Use --plot-data to redraw the figure from the portable JSON snapshot alone.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

PAPER = Path(__file__).resolve().parents[1]
METHODS = ('SMT04', 'Greedy04', 'Fixed1')


def psnr(row):
    value = row.get('psnr_live')
    if value is None:
        value = row['verdict']['psnr_live']
    assert math.isfinite(value)
    return value


def export(run_root):
    hashes = {}

    def read(path, expected=None):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None:
            assert digest == expected, f'Source changed: {path}'
        hashes[str(path)] = digest
        return json.loads(raw)

    reference = read(PAPER / 'data/s4_reduced_evaluation_20260922.json')
    # Bind every raw result to the sources used for the current manuscript tables.
    pinned = reference['source_sha256']

    def frozen(path):
        # Accept relocation of the archived run without dropping hash checks.
        suffix = str(path).split('unified_ba_20260914/')[-1]
        candidates = [digest for name, digest in pinned.items()
                      if name == str(path) or name.endswith('/' + suffix)]
        assert len(candidates) == 1, f'Unpinned or ambiguous source: {path}'
        return read(path, candidates[0])

    specs = frozen(PAPER / 'data/request_specifications_6532.json')['rows']
    specs += frozen(PAPER / 'data/editing_requests_1000_20260922.json')
    keep = set(reference['retained_spec_ids'])
    specs = [r for r in specs if r['spec_id'] in keep]
    assert len(specs) == len(keep) == 7414
    amended = frozen(run_root / 'live_failure_fallback_20260921/amended_results.json')['rows']
    editing = frozen(run_root / 'editing_request_extension_20260921/results.json')['rows']
    base = frozen(run_root / 'quality_paired_20260921/requests.json')['methods']
    methods = {
        'SMT04': [r for r in amended if r['cohort'] == 'main' and r['arm'] == '04'],
        'Greedy04': list(base['Greedy04'].values()),
        'Fixed1': frozen(run_root / 'fixed1_20260918/results.json')['rows'],
    }
    mapping = {'SMT04': ('Full', '04'), 'Greedy04': ('G_shared', '04'), 'Fixed1': ('Fixed1', 'na')}
    for method, variant in mapping.items():
        rows = methods[method] + [r for r in editing if (r['variant'], r['arm']) == variant]
        rows = [r for r in rows if r['spec_id'] in keep]
        assert len(rows) == len({r['spec_id'] for r in rows}) == len(keep)
        assert not any('pending' in r['status'] for r in rows)
        accepted = [r for r in rows if r['status'] == 'accepted']
        assert len(accepted) == reference['reduced'][method]['accepted']
        assert abs(mean(psnr(r) for r in accepted) - reference['reduced'][method]['psnr']) < 1e-8
        methods[method] = {r['spec_id']: r for r in rows}

    def sweep(variable, levels, expected_groups):
        groups = defaultdict(dict)
        for r in specs:
            inputs = r['inputs']
            key = (r['cls'], tuple(sorted(inputs['attacks'])),
                   tuple((k, inputs[k]) for k in ('fpr', 'min_psnr', 'max_ms') if k != variable))
            value = inputs[variable]
            assert value not in groups[key], 'Duplicated request specification'
            groups[key][value] = r['spec_id']
        selected = [(key, ids) for key, ids in sorted(groups.items()) if all(v in ids for v in levels)]
        assert len(selected) == expected_groups
        records = []
        for (scene, attacks, fixed), ids in selected:
            records.append(dict(scenario=scene.replace('C', 'S'), attacks=attacks,
                fixed_inputs=dict(fixed), spec_ids=[ids[v] for v in levels],
                statuses={m: [methods[m][ids[v]]['status'] for v in levels] for m in METHODS},
                accepted={m: [methods[m][ids[v]]['status'] == 'accepted' for v in levels] for m in METHODS}))
        counts = {m: [sum(r['accepted'][m][j] for r in records) for j in range(len(levels))] for m in METHODS}
        transitions = {m: [dict(start=a, end=b, n=n) for (a,b),n in sorted(Counter(
            (r['statuses'][m][0], r['statuses'][m][-1]) for r in records).items())] for m in METHODS}
        return dict(variable=variable, levels=levels, n_groups=len(records),
                    scenarios=dict(sorted(Counter(r['scenario'] for r in records).items())),
                    accepted=counts, percent={m: [100*n/len(records) for n in counts[m]] for m in METHODS},
                    endpoint_status_transitions=transitions, groups=records)

    fpr = sweep('fpr', [.1, .01, 1e-4, 1e-6], 218)
    quality = sweep('min_psnr', [32, 33, 34, 36], 173)

    # Both displayed comparisons use the identical three-pipeline intersection.
    # Equal sample counts alone would not ensure equal request difficulty.
    joint_ids = sorted(set.intersection(*[
        {i for i, r in methods[m].items() if r['status'] == 'accepted'} for m in METHODS]))
    assert joint_ids

    def paired(ours, other):
        ids = joint_ids
        assert all(ours[i]['status'] == other[i]['status'] == 'accepted' for i in ids)
        values = [[i, psnr(ours[i]), psnr(other[i])] for i in ids]
        deltas = [a-b for _, a, b in values]
        eps = 1e-8  # Numerical ties only; not a practical significance threshold.
        return dict(cohort='main_three_pipeline_common_success', n=len(values),
                    mean=mean(deltas), median=median(deltas),
                    min=min(deltas), max=max(deltas),
                    positive=sum(x > eps for x in deltas), negative=sum(x < -eps for x in deltas),
                    tied=sum(abs(x) <= eps for x in deltas),
                    columns=['spec_id', 'tailor_psnr', 'comparator_psnr'], values=values)

    pairs = {}
    for name, method in (('Greedy', 'Greedy04'), ('Enumeration', 'Fixed1')):
        pairs[name] = paired(methods['SMT04'], methods[method])
    assert [r[:2] for r in pairs['Greedy']['values']] == [r[:2] for r in pairs['Enumeration']['values']]

    diagnostic_reference = read(PAPER / 'data/evaluation_shared_table_20260916.json')
    source_hashes = diagnostic_reference['source_hashes']
    diagnostic = {}
    for filename in ('report.json', 'audit.json', 'plan.json'):
        digest = [h for p, h in source_hashes.items() if p.endswith('/combination_scope/' + filename)]
        assert len(digest) == 1
        diagnostic[filename] = read(run_root / 'combination_scope' / filename, digest[0])
    report, audit = diagnostic['report.json'], diagnostic['audit.json']
    assert report['complete'] and report['measurement_evidence_verified'] and audit['passed']
    assert not report['model_updated_from_audit'] and not report['global_error_bound_claimed']
    rows = [r for r in report['rows'] if r['fpr'] == 1e-4 and r['margin']['name'] == '04']
    assert len(rows) == len({r['task_id'] for r in rows}) == 126
    fields = ('task_id', 'kind', 'predicted_mean', 'observed_mean', 'predicted_coverage_pass', 'live_pass')
    points = [{k: r[k] for k in fields} for r in rows]
    summary = dict(n=len(points), mae=mean(abs(r['predicted_mean']-r['observed_mean']) for r in points),
                   predicted_pass=sum(r['predicted_coverage_pass'] for r in points),
                   observed_pass=sum(r['live_pass'] for r in points),
                   optimistic=sum(r['predicted_coverage_pass'] and not r['live_pass'] for r in points),
                   conservative=sum(not r['predicted_coverage_pass'] and r['live_pass'] for r in points))
    assert abs(summary['mae'] - .02479133333333337) < 1e-10
    assert (summary['predicted_pass'], summary['observed_pass'], summary['optimistic'], summary['conservative']) == (114,125,0,11)
    return dict(schema='evaluation_details_v2_common_pipeline_cohort', initial_rate_margin=.04, initial_mean_margin=.02,
                aggregation='Equal weight per matched group at each constraint value; not scenario macro-averages.',
                fpr=fpr, quality=quality, paired=pairs,
                paired_cohort=dict(n=len(joint_ids), methods=list(METHODS), spec_ids=joint_ids,
                    selection='Intersection of requests accepted by all three main-experiment pipelines; no ablation rows.'),
                prediction=dict(fpr=1e-4, mean_margin=.02, rate_margin=.04, summary=summary, points=points,
                    interpretation='Maximum predicted fragment mean versus measured unified decoder score; distinct readouts, not an FPR calibration.'),
                source_sha256=hashes, exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def plot(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'Nimbus Roman', 'mathtext.fontset': 'stix', 'text.color': 'black', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'legend.fontsize': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    blue, orange, green = '#1764AB', '#C26817', '#32866A'
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.65), layout='constrained')
    for ax, key, title in zip(axs[0], ('fpr', 'quality'),
                              ('(a) FPR budget · 218 groups', '(b) PSNR floor · 173 groups')):
        record = data[key]
        x = record['levels']
        for m, label, color, marker in zip(METHODS, ('TAILOR', 'Greedy', 'Enumeration'),
                                           (blue, orange, green), ('o', 's', '^')):
            ax.plot(x, record['percent'][m], color=color, marker=marker, markersize=4,
                    linewidth=1.5, label=label)
        ax.set_title(title, loc='left', fontweight='bold', pad=9)
        ax.set_ylabel('Request satisfaction (%)')
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(x)
        if key == 'fpr':
            ax.set_xscale('log')
            ax.invert_xaxis()
            ax.set_xticks(x)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xticklabels([r'$10^{-1}$', r'$10^{-2}$', r'$10^{-4}$', r'$10^{-6}$'])
            ax.set_xlabel('FPR budget (tighter →) · S1–S5')
        else:
            ax.set_xlabel('PSNR floor (dB; tighter →) · S2–S5')
        ax.grid(axis='y', alpha=.16)
        ax.legend(loc='lower left', frameon=False, ncol=1)
    ax = axs[1, 0]
    assert set(data['paired']) == {'Greedy', 'Enumeration'}
    cohort = data['paired_cohort']
    for (label, r), color in zip(data['paired'].items(), (orange, green)):
        assert [v[0] for v in r['values']] == cohort['spec_ids']
        values = np.sort([a-b for _, a, b in r['values']])
        # Include both tails and the jump at repeated values, including zero.
        ax.step(np.r_[values[0], values], np.r_[0, np.arange(1, len(values)+1)/len(values)],
                where='post', color=color, linewidth=1.5, label=label)
    ax.axvline(0, color='#777777', linewidth=.8, linestyle=':', zorder=0)
    ax.set_title(f"(c) Paired image quality · n={cohort['n']:,}", loc='left', fontweight='bold', pad=9)
    ax.set_xlabel('ΔPSNR: TAILOR − comparator (dB)')
    ax.set_ylabel('Cumulative fraction')
    ax.set_ylim(0, 1.025)
    ax.grid(alpha=.16)
    ax.legend(loc='lower right', frameon=False)
    ax = axs[1, 1]
    points = data['prediction']['points']
    limits = [min(min(r['predicted_mean'], r['observed_mean']) for r in points)-.025, 1.01]
    ax.plot(limits, limits, color='#888888', linewidth=.8, linestyle='--', zorder=0)
    for kind, label, color, marker in zip(('pair_plain_2d_grid', 'pair_frontend_spot', 'triple_spot'),
            ('Plain pairs', 'Pairs + frontend', 'Triples'), (blue, orange, green), ('o', '^', 's')):
        rows = [r for r in points if r['kind'] == kind]
        ax.scatter([r['predicted_mean'] for r in rows], [r['observed_mean'] for r in rows],
                   label=label, color=color, marker=marker, s=18, alpha=.75, edgecolors='white', linewidth=.3)
    ax.set_title('(d) Editing prediction · 126 configs.', loc='left', fontweight='bold', pad=9)
    ax.set_xlabel('Predicted recovery proxy')
    ax.set_ylabel('Measured decoder score')
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.text(.04, .96, f"MAE = {data['prediction']['summary']['mae']:.4f}",
            transform=ax.transAxes, ha='left', va='top', fontsize=8)
    ax.grid(alpha=.16)
    ax.legend(loc='lower right', frameon=False)
    fig.savefig(output / 'evaluation_details.pdf', metadata={'CreationDate': None})
    fig.savefig(output / 'evaluation_details.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-root', type=Path)
    source.add_argument('--plot-data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = export(args.run_root) if args.run_root else json.loads(args.plot_data.read_text())
    if args.run_root:
        (args.output / 'evaluation_details.json').write_text(json.dumps(data, separators=(',', ':')) + '\n')
    plot(data, args.output)
    print(json.dumps(dict(fpr={k:v for k,v in data['fpr'].items() if k != 'groups'},
                          quality={k:v for k,v in data['quality'].items() if k != 'groups'},
                          paired={k:{a:b for a,b in v.items() if a != 'values'} for k,v in data['paired'].items()},
                          prediction=data['prediction']['summary']), indent=2))


if __name__ == '__main__':
    main()
