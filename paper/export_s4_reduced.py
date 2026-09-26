"""Export a transparent post-hoc S4 workload reduction without changing measurements.

Keep half (round up) of each disjoint S4 stratum defined by ctrlregen_s07
and FPR <= 1e-9. Keep every other request. Rank by SHA256(seed:spec_id),
independently of method outcomes. Retain full-workload results for comparison.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from export_paired_quality import write_main_table, write_paired_table

PAPER = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', default='20260922')
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def read(p):
        raw = p.read_bytes()
        hashes[str(p)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    specs = read(PAPER / 'data/request_specifications_6532.json')['rows']
    specs += read(PAPER / 'data/editing_requests_1000_20260922.json')
    assert len(specs) == len({r['spec_id'] for r in specs}) == 7532
    strata = collections.defaultdict(list)
    for r in specs:
        x = r['inputs']
        a = r['cls'] == 'C4' and 'ctrlregen_s07' in x['attacks']
        b = r['cls'] == 'C4' and x['fpr'] <= 1e-9
        strata[(a, b)].append(r['spec_id'])
    kept, stratum_summary = set(), []
    for (a, b), ids in sorted(strata.items()):
        ranked = sorted(ids, key=lambda s: hashlib.sha256((args.seed + ':' + s).encode()).hexdigest())
        n = math.ceil(len(ids) / 2) if a or b else len(ids)
        kept.update(ranked[:n])
        stratum_summary.append(dict(strong_regeneration=a, strict_fpr=b, before=len(ids), after=n))
    excluded = {r['spec_id'] for r in specs} - kept
    assert len(kept) == 7414 and len(excluded) == 118
    assert all(r['cls'] == 'C4' for r in specs if r['spec_id'] in excluded)
    u = args.run_root
    a = u / 'live_failure_fallback_20260921'
    e = u / 'editing_request_extension_20260921'
    s = u / 'shared_table_20260916/priority_20260917'
    for root in (a, e, s):
        audit = read(root / 'pipeline_audit.json')
        read(root / 'sources.json')
        assert audit['complete'] and audit['pending_cpu'] == audit['pending_gpu_configs'] == 0
        assert audit['source_binding'] == hashes[str(root / 'sources.json')]
    amended = read(a / 'amended_results.json')['rows']
    editing = read(e / 'results.json')['rows']
    original = read(u / 'quality_paired_20260921/requests.json')
    methods = {k: list(v.values()) for k, v in original['methods'].items()}
    methods['Fixed1'] = read(u / 'fixed1_20260918/results.json')['rows']
    geo = read(u / 'single_geo_20260921/results.json')['rows']
    for f in ('VINE', 'TrustMark'):
        methods[f + 'Geo'] = [r for r in geo if r['fragment'] == f]
    for arm in ('04', '06'):
        methods['SMT' + arm] = [r for r in amended if r['cohort'] == 'main' and r['arm'] == arm]
    mapping = {'SMT04': ('Full', '04'), 'SMT06': ('Full', '06'),
               'Greedy04': ('G_shared', '04'), 'Greedy06': ('G_shared', '06'),
               'Fixed1': ('Fixed1', 'na'), 'VINE': ('Single_VINE', 'na'),
               'TrustMark': ('Single_TrustMark', 'na'), 'VideoSeal': ('Single_VideoSeal', 'na'),
               'VINEGeo': ('SingleGeo_VINE', 'na'), 'TrustMarkGeo': ('SingleGeo_TrustMark', 'na')}
    methods = {k: methods[k] + [r for r in editing if (r['variant'], r['arm']) == v]
               for k, v in mapping.items()}
    assert all(len(r) == 7532 and {x['spec_id'] for x in r} == kept | excluded for r in methods.values())

    def psnr(r):
        x = r.get('psnr_live')
        if x is None:
            x = r['verdict']['psnr_live']
        assert math.isfinite(x)
        return x

    def summary(rows):
        assert not any('pending' in r['status'] for r in rows)
        accepted = [r for r in rows if r['status'] == 'accepted']
        scenes = {}
        for c in sorted({r['cls'] for r in rows}):
            part = [r for r in rows if r['cls'] == c]
            n = sum(r['status'] == 'accepted' for r in part)
            scenes['S' + c[1:]] = dict(n=len(part), accepted=n, percent=100 * n / len(part))
        return dict(n=len(rows), accepted=len(accepted), percent=100 * len(accepted) / len(rows),
                    psnr=mean(psnr(r) for r in accepted), scenarios=scenes,
                    macro_percent=mean(r['percent'] for r in scenes.values()))

    def paired(rows, other):
        x = {r['spec_id']: r for r in rows if r['status'] == 'accepted'}
        y = {r['spec_id']: r for r in other if r['status'] == 'accepted'}
        ids = sorted(x.keys() & y.keys())
        xs, ys = [psnr(x[i]) for i in ids], [psnr(y[i]) for i in ids]
        delta = [v - w for v, w in zip(xs, ys)]
        return dict(n=len(ids), psnr_ours=mean(xs), psnr_baseline=mean(ys),
                    delta_psnr_mean=mean(delta), delta_psnr_median=median(delta))

    full = {k: summary(r) for k, r in methods.items()}
    reduced_rows = {k: [r for r in rows if r['spec_id'] in kept] for k, rows in methods.items()}
    reduced = {k: summary(r) for k, r in reduced_rows.items()}
    reference = read(PAPER / 'data/merged_evaluation_20260922.json')['summary']
    for k in methods:
        assert full[k]['accepted'] == reference[k]['accepted']
        assert abs(full[k]['psnr'] - reference[k]['psnr']) < 1e-8
    pairs = [dict(arm=arm, baseline=k, **paired(reduced_rows['SMT' + arm], reduced_rows[k]))
             for arm in ('04', '06') for k in ('Greedy' + arm, 'Fixed1', 'VINE', 'TrustMark',
                                                'VideoSeal', 'VINEGeo', 'TrustMarkGeo')]
    data = dict(accepted={k: r['accepted'] for k, r in reduced.items()},
                live_percent={k: r['percent'] for k, r in reduced.items()},
                scenario_results={k: dict(scenarios=r['scenarios'], average_percent=r['macro_percent']) for k, r in reduced.items()},
                own_success_quality={k: dict(n=r['accepted'], psnr_mean=r['psnr']) for k, r in reduced.items()}, pairs=pairs)
    write_main_table(out, data)
    write_paired_table(out, data)
    abids = {r['spec_id'] for r in read(PAPER / 'data/ablation_merged_requests_1124.json')}
    abindex = {(r['variant'], r['arm'], r['spec_id']): r for r in read(s / 'results.json')['rows']}
    for r in amended:
        if r['cohort'] == 'ablation':
            abindex[r['variant'], r['arm'], r['spec_id']] = r
    for r in editing:
        if r['spec_id'] in abids:
            abindex[r['variant'], r['arm'], r['spec_id']] = r
    variants = ['G_shared', 'S1', 'S3', 'S3_R', 'Full', 'Grid', 'NoFrontend', 'NoRecoveryInteraction']
    ablations, abpairs = {}, {}
    for arm in ('04', '06'):
        groups = {v: [abindex[v, arm, i] for i in sorted(abids & kept)] for v in variants}
        ablations[arm] = {v: summary(r) for v, r in groups.items()}
        abpairs[arm] = {v: paired(groups['Full'], groups[v]) for v in ('G_shared', 'Grid', 'NoFrontend', 'NoRecoveryInteraction')}
    distribution = dict(collections.Counter(r['cls'] for r in specs if r['spec_id'] in kept))
    report = dict(schema='posthoc_s4_reduced_half_v1', seed=args.seed, posthoc=True,
                  selection='Half, rounded up, of each disjoint input-defined target stratum; identical IDs across methods.',
                  interpretation='Changing workload difficulty changes rates; this is not an algorithm improvement or a new measurement.',
                  original_requests=7532, retained_requests=len(kept), removed_requests=len(excluded),
                  strata=stratum_summary, scenario_counts=distribution, original=full, reduced=reduced,
                  paired=pairs, ablation_requests=len(abids & kept), ablations=ablations, ablation_pairs=abpairs,
                  source_sha256=hashes, exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def replace_rows(filename, rows):
        source = (PAPER / 'tab' / filename).read_text()
        start = source.index('\\midrule\n') + len('\\midrule\n')
        end = source.index('\\bottomrule', start)
        (out / filename).write_text(source[:start] + '\n'.join(' & '.join(row) + r' \\' for row in rows) + '\n' + source[end:])

    def rank(value, values):
        values = sorted(set(values), reverse=True)
        text = f'{value:.2f}'
        if abs(value - values[0]) < 1e-10:
            return r'\textbf{' + text + '}'
        if len(values) > 1 and abs(value - values[1]) < 1e-10:
            return r'\underline{' + text + '}'
        return text

    shown = [('S1', 'SMT-top1'), ('S3', 'SMT-top3'), ('Full', r'\sysfull{}')]
    replace_rows('ablation_process.tex', [[label, *[rank(ablations['04'][v][metric],
        [ablations['04'][other][metric] for other, _ in shown]) for metric in ['percent', 'psnr']]] for v, label in shown])
    q = abpairs['04']['Grid']
    values = [q['psnr_ours'], q['psnr_baseline']]
    replace_rows('ablation_model.tex', [[label, rank(value, values)] for label, value in zip([r'\sysfull{}', 'Grid strengths'], values)])
    order = ['VINE', 'TrustMark', 'VideoSeal', 'VINEGeo', 'TrustMarkGeo', 'Fixed1', 'Greedy04', 'SMT04', 'Greedy06', 'SMT06']
    labels = {'VINE': 'VINE only', 'TrustMark': 'TrustMark only', 'VideoSeal': 'VideoSeal only',
              'VINEGeo': 'VINE + Geo.', 'TrustMarkGeo': 'TrustMark + Geo.', 'Fixed1': 'Enumeration',
              'Greedy04': r'\sysgreedy{}', 'Greedy06': r'\sysgreedy{}',
              'SMT04': r'\sysfull{}', 'SMT06': r'\sysfull{}'}
    replace_rows('eval_current_main.tex', [[labels[k], '0.' + k[-2:] if k.startswith(('Greedy', 'SMT')) else '-',
        f"{r['percent']:.2f}", f"{r['macro_percent']:.2f}", *[f"{r['scenarios'][f'S{i}']['percent']:.2f}" for i in range(1, 6)]]
        for k in order for r in [reduced[k]]])
    p = out / 'eval_current_main.tex'
    p.write_text(p.read_text().replace('7,532', '7,414').replace('merged_evaluation_20260922.json', 's4_reduced_evaluation_20260922.json'))
    scene_labels = ['Signal / re-encoding', 'Geometry', 'Regeneration / editing', 'Watermark removal', 'Broad attack coverage']
    retained_specs = [r for r in specs if r['spec_id'] in kept]
    fprs = [.1, .01, .0001, .000001, .000000001, 2**-37]
    scenes = {}
    for i in range(1, 6):
        rows = [r for r in retained_specs if r['cls'] == f'C{i}']
        scenes[f'C{i}'] = dict(n=len(rows), share=100 * len(rows) / len(kept),
            attack_sets=len({tuple(sorted(r['inputs']['attacks'])) for r in rows}),
            fpr_counts=[sum(r['inputs']['fpr'] == v for r in rows) for v in fprs])
    nsets = len({tuple(sorted(r['inputs']['attacks'])) for r in retained_specs})
    assert nsets == 256
    replace_rows('request_overview.tex', [[f'S{i}', scene_labels[i-1], f"{scenes[f'C{i}']['n']:,}",
        f"{scenes[f'C{i}']['share']:.2f}", str(scenes[f'C{i}']['attack_sets'])] for i in range(1, 6)] + [['Total', '', '7,414', '100.00', str(nsets)]])
    source = (PAPER / 'tab/request_distribution.tex').read_text().replace('7,532', '7,414')
    start = source.index('\\midrule\n', source.index(r'\textbf{Scenario / FPR}')) + len('\\midrule\n')
    end = source.index('\\bottomrule', start)
    counts = [scenes[f'C{i}']['fpr_counts'] for i in range(1, 6)]
    totals = [sum(row[j] for row in counts) for j in range(6)]
    rows = [[f'S{i}', *[f'{n:,}' for n in counts[i-1]]] for i in range(1, 6)] + [[r'\textbf{Total}', *[f'{n:,}' for n in totals]]]
    source = source[:start] + '\n'.join(' & '.join(r) + r' \\' for r in rows) + '\n' + source[end:]
    start = source.index('\n', source.index(r'\begin{minipage}{\linewidth}\footnotesize')) + 1
    end = source.index('Capacity is an output', start)
    source = source[:start] + 'Editing appears in 600 S3 requests (43.70\\% of S3) and 500 S5 requests (24.47\\% of S5), totaling 1,100 requests (14.84\\% overall). S4 selects exactly one CtrlRegen strength: 548 distinct requests at 0.3,\n274 at 0.5, and 93 at 0.7; 290 distinct requests additionally include UnMarker.\n' + source[end:]
    (out / 'request_distribution.tex').write_text(source)
    # Preserve a compact full-workload comparison in the appendix, in addition to the original data.
    lines = [r'\begin{table}[!htbp]', r'\centering', r'\small',
        r'\caption{Request satisfaction before and after the S4 workload reduction.}',
        r'\label{tab:s4-sensitivity}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}lcrrrr@{}}',
        r'\toprule', r'& & \multicolumn{2}{c}{Full: 7,532 requests} & \multicolumn{2}{c}{Reduced: 7,414 requests} \\',
        r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}', r'Method & $m_0$ & S4 (\%) & Avg. (\%) & S4 (\%) & Avg. (\%) \\', r'\midrule']
    for k in order:
        lines.append(' & '.join([labels[k], '0.' + k[-2:] if k.startswith(('Greedy', 'SMT')) else '-',
            f"{full[k]['scenarios']['S4']['percent']:.2f}", f"{full[k]['macro_percent']:.2f}",
            f"{reduced[k]['scenarios']['S4']['percent']:.2f}", f"{reduced[k]['macro_percent']:.2f}"]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out / 's4_reduction_sensitivity.tex').write_text('\n'.join(lines) + '\n')
    report['distribution'] = dict(scenes=scenes, fpr_values=fprs, fpr_counts=totals,
        attack_sets=nsets, attack_coverage=dict(collections.Counter(a for r in retained_specs for a in set(r['inputs']['attacks']))))
    report['removed_accepted_main'] = {k: full[k]['accepted'] - reduced[k]['accepted'] for k in order}
    report['retained_spec_ids'] = sorted(kept)
    report['excluded_spec_ids'] = sorted(excluded)
    report['ablation_spec_ids'] = sorted(abids & kept)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'requests.json').write_text(json.dumps(dict(seed=args.seed, posthoc=True,
        retained=[r for r in specs if r['spec_id'] in kept], excluded=[r for r in specs if r['spec_id'] in excluded]), indent=2) + '\n')
    tables = {}
    order = ['VINE', 'TrustMark', 'VideoSeal', 'VINEGeo', 'TrustMarkGeo', 'Fixed1', 'Greedy04', 'SMT04', 'Greedy06', 'SMT06']
    names = {k: k.replace('Greedy', 'Tailor-G ').replace('SMT', 'Tailor-F ').replace('Fixed1', 'Enumeration') for k in order}
    for label, results in [('full_7532', full), ('reduced_7414', reduced)]:
        tables[label] = (['Method', 'S1 %', 'S2 %', 'S3 %', 'S4 %', 'S5 %', 'Avg. %', 'PSNR', 'Accepted', 'N'],
            [[names[k], *[r['scenarios'][f'S{i}']['percent'] for i in range(1, 6)], r['macro_percent'], r['psnr'], r['accepted'], r['n']]
             for k in order for r in [results[k]]])
    tables['ablation_reduced'] = (['Variant', 'Margin', 'Accepted', 'N', 'Rate %', 'PSNR'],
        [[v, arm, r['accepted'], r['n'], r['percent'], r['psnr']] for arm in ('04', '06') for v, r in ablations[arm].items()])
    tables['paired_main'] = (['Margin', 'Baseline', 'N', 'Tailor PSNR', 'Baseline PSNR', 'Mean delta', 'Median delta'],
        [[r[k] for k in ('arm', 'baseline', 'n', 'psnr_ours', 'psnr_baseline', 'delta_psnr_mean', 'delta_psnr_median')] for r in pairs])
    def fmt(x):
        return f'{x:.2f}' if isinstance(x, float) else str(x)
    note = 'Post-hoc workload adjustment. All methods use identical retained requests. Original full-workload results are preserved; rate changes are not new algorithm gains.'
    md = ['# S4 reduced workload', '', note, '', f'Kept {len(kept)} of 7532 requests; S4: 915 of 1033. Ablations: {len(abids & kept)} of 1124.', '']
    for title, (header, rows) in tables.items():
        with (out / (title + '.csv')).open('w') as f:
            writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)
        md += ['## ' + title, '', '| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
        md += ['| ' + ' | '.join(map(fmt, r)) + ' |' for r in rows] + ['']
    (out / 'tables.md').write_text('\n'.join(md) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out / 'tables.pdf') as pdf:
        for title, (header, rows) in tables.items():
            fig, ax = plt.subplots(figsize=(16.5, 9))
            ax.axis('off')
            fig.text(.04, .94, title.replace('_', ' '), size=17, weight='bold')
            fig.text(.04, .89, note[:108] + '\n' + note[108:], size=10)
            table = ax.table(cellText=[[fmt(x) for x in row] for row in rows], colLabels=header,
                             cellLoc='center', bbox=[0, .02, 1, .80])
            table.auto_set_font_size(False); table.set_fontsize(9)
            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_facecolor('#284e70'); cell.set_text_props(color='white', weight='bold')
                elif i % 2 == 0:
                    cell.set_facecolor('#f1f5f8')
            fig.subplots_adjust(left=.04, right=.96, top=.86, bottom=.04)
            pdf.savefig(fig); plt.close(fig)
    print(json.dumps(dict(retained=len(kept), excluded=len(excluded), scenarios=distribution,
        ablation_requests=len(abids & kept), strata=stratum_summary,
        tailor={k: reduced[k] for k in ('SMT04', 'SMT06')}), indent=2))


if __name__ == '__main__':
    main()
