"""Summarize frozen live decisions and draw paired PSNR ECDFs.

Output is staged outside the paper. LPIPS is never averaged over a partial
request subset. Main-table quality summarizes each method's own accepted
requests; paired comparisons use baseline/SMT joint live success. Archived
LPIPS evidence is retained in the audit JSON, but paper tables report only PSNR.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.quality_root, args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert digest(root / 'requests.json') == manifest['request_snapshot_sha256']
    for path, expected in manifest['source_hashes'].items():
        assert digest(path) == expected, ('frozen input changed', path)
    expected_configs = {e['cfg_id']: e for e in manifest['configurations']}
    methods = json.loads((root / 'requests.json').read_text())['methods']
    assert all(len(rows) == 6532 for rows in methods.values())
    cells, hashes = {}, {}
    for p in sorted((root / 'cells').glob('*.json')):
        d = json.loads(p.read_text())
        assert d['cfg_id'] == p.stem and d['cfg'] == expected_configs[p.stem]['cfg']
        assert d['complete'] and d['images_sha256'] == manifest['images']['images_sha256']
        assert len(d['lpips']) == len(d['psnr_db']) == 100
        assert abs(statistics.mean(d['lpips']) - d['lpips_mean']) < 1e-10
        assert abs(statistics.mean(d['psnr_db']) - d['psnr_mean']) < 1e-10
        for field in ['expected_psnr_min', 'expected_psnr_max']:
            expected = expected_configs[p.stem][field]
            if expected is not None:
                assert abs(d['psnr_mean']-expected) <= manifest['psnr_reproduction_tolerance_db']
        cells[d['cfg_id']] = d
        hashes[str(p)] = digest(p)
    identities = {json.dumps(d['metric_identity'], sort_keys=True) for d in cells.values()}
    assert len(identities) <= 1, 'mixed metric versions or hardware'
    required = {r['cfg_id'] for rows in methods.values() for r in rows.values() if r['status'] == 'accepted'}
    pairs, observations = [], []
    for arm in ['04', '06']:
        ours = methods['SMT' + arm]
        for baseline in ['VINE', 'TrustMark', 'VideoSeal', 'Greedy' + arm]:
            theirs = methods[baseline]
            joint = sorted(sid for sid, r in ours.items()
                           if r['status'] == theirs[sid]['status'] == 'accepted')
            assert joint
            psnr_o = [ours[s]['psnr_live'] for s in joint]
            psnr_b = [theirs[s]['psnr_live'] for s in joint]
            delta = [o - b for o, b in zip(psnr_o, psnr_b)]
            available = all(ours[s]['cfg_id'] in cells and theirs[s]['cfg_id'] in cells for s in joint)
            lp_o = [cells[ours[s]['cfg_id']]['lpips_mean'] for s in joint] if available else []
            lp_b = [cells[theirs[s]['cfg_id']]['lpips_mean'] for s in joint] if available else []
            for s, o, b, dd in zip(joint, psnr_o, psnr_b, delta):
                observations.append(dict(arm=arm, baseline=baseline, spec_id=s, scenario='S'+ours[s]['cls'][1:],
                    ours_cfg=ours[s]['cfg_id'], baseline_cfg=theirs[s]['cfg_id'],
                    ours_psnr=o, baseline_psnr=b, delta_psnr=dd))
            pair = dict(arm=arm, baseline=baseline, n=len(joint), psnr_ours=statistics.mean(psnr_o),
                psnr_baseline=statistics.mean(psnr_b), delta_psnr_mean=statistics.mean(delta),
                delta_psnr_median=statistics.median(delta), delta_psnr_min=min(delta), delta_psnr_max=max(delta),
                higher_fraction=sum(d > 1e-4 for d in delta)/len(delta),
                lower_fraction=sum(d < -1e-4 for d in delta)/len(delta),
                tie_fraction=sum(abs(d) <= 1e-4 for d in delta)/len(delta),
                lpips_complete=available, lpips_ours=statistics.mean(lp_o) if available else None,
                lpips_baseline=statistics.mean(lp_b) if available else None)
            if available:
                pair['lpips_delta_mean'] = statistics.mean(o-b for o,b in zip(lp_o,lp_b))
            pairs.append(pair)
    counts = {m:sum(r['status']=='accepted' for r in rows.values()) for m,rows in methods.items()}
    scenario_results = {}
    for method, rows in methods.items():
        scenarios = {}
        for i in range(1, 6):
            subset = [r for r in rows.values() if r['cls'] == f'C{i}']
            assert subset
            accepted = sum(r['status'] == 'accepted' for r in subset)
            scenarios[f'S{i}'] = dict(n=len(subset), accepted=accepted,
                                       percent=100*accepted/len(subset))
        assert sum(r['n'] for r in scenarios.values()) == 6532
        assert sum(r['accepted'] for r in scenarios.values()) == counts[method]
        scenario_results[method] = dict(scenarios=scenarios,
            average_percent=statistics.mean(r['percent'] for r in scenarios.values()))
    own_success = {}
    for method, rows in methods.items():
        accepted = [r for r in rows.values() if r['status'] == 'accepted']
        available = all(r['cfg_id'] in cells for r in accepted)
        own_success[method] = dict(n=len(accepted),
            psnr_mean=statistics.mean(r['psnr_live'] for r in accepted),
            lpips_complete=available,
            lpips_mean=statistics.mean(cells[r['cfg_id']]['lpips_mean'] for r in accepted)
                       if available else None)
    summary = dict(schema='paired_live_fidelity_v1', request_denominator=6532,
        accepted=counts, live_percent={m:100*n/6532 for m,n in counts.items()},
        method_display_names={m:('Tailor-F' if m.startswith('SMT') else
                                 'Tailor-G' if m.startswith('Greedy') else m) for m in methods},
        quality_conditioning='Separate intersection of SMT and each baseline, at the matched margin; identical 100 live images',
        own_success_quality=own_success, scenario_results=scenario_results,
        reported_metrics=['request_satisfaction', 'PSNR'],
        archived_metrics_not_reported=['LPIPS'],
        main_table_satisfaction_aggregation="Per-scenario live rates; Avg. is the unweighted mean of S1-S5",
        main_table_quality_conditioning='Each method and arm averages only its own accepted requests; descriptive, not a paired comparison',
        pairs=pairs, lpips_complete=required.issubset(cells), required_configurations=len(required),
        measured_configurations=len(cells), missing_configurations=sorted(required-cells.keys()),
        psnr_tie_tolerance_db=1e-4, metric=manifest['metric'],
        validation=dict(source_hashes_checked=True,exact_configurations_checked=True,
            original_live_decisions_unchanged=True,image_role='selection',images_per_configuration=100,
            incomplete_pair_means_withheld=True),
        metric_identity=json.loads(next(iter(identities))) if identities else None,
        inputs={str(root/'manifest.json'):digest(root/'manifest.json'),str(root/'requests.json'):digest(root/'requests.json')},
        exporter_sha256=digest(__file__), evidence_sha256=hashes)
    (out/'paired_quality.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (out/'paired_psnr.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(observations[0]));writer.writeheader();writer.writerows(observations)
    draw(out, observations, pairs)
    write_main_table(out, summary)
    write_paired_table(out, summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['evidence_sha256','missing_configurations']},indent=2))


def draw(out, observations, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(2,2,figsize=(6.8,4.6),sharey=True)
    colors={'04':'#2768A8','06':'#D77A22'}
    for ax,base in zip(axes.flat,['VINE','TrustMark','VideoSeal','Greedy']):
        values=[]
        for arm in ['04','06']:
            label=base+arm if base=='Greedy' else base
            ds=np.sort([r['delta_psnr'] for r in observations if r['arm']==arm and r['baseline']==label])
            assert len(ds)
            values.extend(ds)
            xx=np.r_[ds[0],ds];yy=np.r_[0,np.arange(1,len(ds)+1)/len(ds)]
            ax.step(xx,yy,where='post',color=colors[arm],lw=1.65,
                label=r'$m_0=0.'+arm+r'$; $n='+f'{len(ds):,}'+r'$')
        span=max(values)-min(values);pad=max(.18,span*.05)
        ax.set_xlim(min(min(values)-pad,-pad),max(values)+pad)
        ax.set_ylim(0,1.025);ax.set_yticks([0,.25,.5,.75,1])
        ax.axvline(0,color='#747D87',lw=.9,ls='--',zorder=0)
        ax.grid(axis='y',color='#E0E5E9',lw=.7);ax.set_axisbelow(True)
        ax.set_title('Greedy variant' if base=='Greedy' else base,loc='left',fontweight='bold')
        ax.legend(loc='lower right',frameon=False)
        ax.set_xlabel(r'$\Delta$PSNR (Tailor $-$ baseline), dB')
    axes[0,0].set_ylabel('Fraction of paired requests')
    axes[1,0].set_ylabel('Fraction of paired requests')
    fig.tight_layout(pad=1.0,w_pad=1.6,h_pad=1.4)
    fig.savefig(out/'paired_delta_psnr.pdf',bbox_inches='tight',metadata={'Title':'Paired PSNR differences on common live-success requests'})
    fig.savefig(out/'paired_delta_psnr.png',bbox_inches='tight',dpi=200)
    plt.close(fig)


def write_paired_table(out, d):
    lines=[r'\begin{table}[!t]',r'\centering',r'\small',
      r'\caption{Request satisfaction and paired image fidelity comparison across methods.}',
      r'\label{tab:paired-quality}',r'\setlength{\tabcolsep}{6pt}',r'\begin{tabular}{@{}lrr rr@{}}',r'\toprule',
      r'& & & \multicolumn{2}{c}{PSNR (dB) $\uparrow$} \\',
      r'Baseline & Live (\%) $\uparrow$ & $n_{\cap}$ & Baseline & \sysfull{} \\',r'\midrule']
    for arm in ['04','06']:
        lines += [r'\multicolumn{5}{l}{\sysfull{}: $m_0=0.'+arm+r'$, live satisfaction '+f"{d['live_percent']['SMT'+arm]:.2f}"+r'\%} \\']
        for p in [p for p in d['pairs'] if p['arm']==arm]:
            name=p['baseline'];label=r'\sysgreedy{}' if name.startswith('Greedy') else {'Fixed1':'Enumeration','VINEGeo':'VINE + Geo.','TrustMarkGeo':'TrustMark + Geo.'}.get(name,name)
            ps_b, ps_o = f"{p['psnr_baseline']:.2f}", f"{p['psnr_ours']:.2f}"
            if p['psnr_ours'] >= p['psnr_baseline'] - 1e-10:
                ps_o = r'\textbf{' + ps_o + '}'
            else:
                ps_o = r'\underline{' + ps_o + '}'
            if p['psnr_baseline'] >= p['psnr_ours'] - 1e-10:
                ps_b = r'\textbf{' + ps_b + '}'
            else:
                ps_b = r'\underline{' + ps_b + '}'
            lines.append(' & '.join([label,f"{d['live_percent'][name]:.2f}",f"{p['n']:,}",
                ps_b,ps_o])+r' \\')
        if arm=='04':lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'paired_quality.tex').write_text('\n'.join(lines)+'\n')


def write_main_table(out, d):
    lines = [
        r'\definecolor{tblgray}{rgb}{0.5,0.5,0.5}',
        r'\begin{table}[t]', r'\centering',
        r'\caption{Request satisfaction and image fidelity comparison across scenarios.}',
        r'\label{tab:main}', r'\small', r'\setlength{\tabcolsep}{4pt}',
        r'\renewcommand{\arraystretch}{1.08}',
        r'\begin{tabular}{@{}l c cccccc @{\hspace{10pt}} c@{}}',
        r'\toprule', r'\toprule',
        r'& & \multicolumn{6}{c}{\textit{Request satisfaction (\%) $\uparrow$}} & \textit{Image fidelity} \\',
        r'\cmidrule(lr){3-8} \cmidrule(lr){9-9}',
        r'Method & $m_0$ & S1 & S2 & S3 & S4 & S5 & \textbf{Avg.} & PSNR $\uparrow$ \\',
        r'\midrule',
    ]
    order = [('VINE', None), ('TrustMark', None), ('VideoSeal', None),
             ('VINEGeo', None), ('TrustMarkGeo', None), ('Fixed1', None),
             ('Greedy', '04'), ('SMT', '04'), ('Greedy', '06'), ('SMT', '06')]
    extension_labels = {'VINEGeo': 'VINE + Geo.', 'TrustMarkGeo': 'TrustMark + Geo.',
                        'Fixed1': 'Enumeration'}
    def metric_values(method):
        results = d['scenario_results'][method]
        quality = d['own_success_quality'][method]
        return ([results['scenarios'][f'S{i}']['percent'] for i in range(1, 6)]
                + [results['average_percent'], quality['psnr_mean']])
    all_values = [metric_values(base if arm is None else base + arm) for base, arm in order
                  if (base if arm is None else base + arm) in d['scenario_results']]
    top_ranks = []
    for column in range(7):
        available = sorted({values[column] for values in all_values if values[column] is not None}, reverse=True)
        distinct = []
        for value in available:
            if not distinct or abs(value-distinct[-1]) > 1e-10:
                distinct.append(value)
        top_ranks.append(distinct[:2])
    for base, arm in order:
        control = base in {'VINE', 'TrustMark', 'VideoSeal'} and arm is None
        method = base if arm is None else base + arm
        if base in {'VINEGeo', 'Greedy'}:
            lines.append(r'\midrule')
        if method not in d['scenario_results']:
            assert base in extension_labels, ('missing original method', method)
            lines.append(' & '.join([extension_labels[base], '-'] + [''] * 5
                                    + [r'\cellcolor{orange!8}', '']) + r' \\')
            continue
        def style(value, rank=None):
            if rank == 0:
                value = r'\textbf{' + value + '}'
            elif rank == 1:
                value = r'\underline{' + value + '}'
            if control:
                value = r'\textcolor{tblgray}{' + value + '}'
            return value
        label = r'\sysfull{}' if base == 'SMT' else (r'\sysgreedy{}' if base == 'Greedy'
                                                    else extension_labels.get(base, base))
        row = [style(label), style('-' if arm is None else '0.' + arm)]
        assert d['own_success_quality'][method]['n'] == d['accepted'][method]
        for column, value in enumerate(metric_values(method)):
            formatted = f'{value:.2f}' if value is not None else r'\FILL'
            rank = next((i for i, candidate in enumerate(top_ranks[column])
                         if value is not None and abs(value-candidate) < 1e-10), None)
            cell = style(formatted, rank)
            if column == 5:
                cell = r'\cellcolor{orange!8}' + cell
            row.append(cell)
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out/'main_results.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
