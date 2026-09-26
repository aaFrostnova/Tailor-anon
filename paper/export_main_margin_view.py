"""Render the main comparison and appendix margin study from frozen measurements."""
import argparse
import hashlib
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
BASELINES = ['StegaStamp', 'MaskWM', 'VINE', 'TrustMark', 'VideoSeal',
             'VINEGeo', 'TrustMarkGeo', 'Fixed1', 'EnumGeo']
LABELS = dict(zip(BASELINES, ['StegaStamp', 'MaskWM', 'VINE', 'TrustMark',
    'VideoSeal', 'VINE + Geo.', 'TrustMark + Geo.', 'Enumeration', 'Enumeration + Geo.']))


def label(key):
    return (r'\sysfull{}' if key.startswith('SMT') else
            r'\sysgreedy{}' if key.startswith('Greedy') else LABELS[key])


def render(summary, keys, margin_column, caption, table_label, paired=None):
    metrics = {k: [*[summary[k]['scenarios'][f'S{i}']['percent'] for i in range(1, 6)],
                   summary[k]['macro_percent'], summary[k]['psnr']] for k in keys}
    ranks = [sorted({round(metrics[k][i], 9) for k in keys if metrics[k][i] is not None},
                    reverse=True)[:2] for i in range(7)]
    def cell(value, i):
        if value is None:
            return '-'
        text = f'{value:.2f}'
        for rank, best in enumerate(ranks[i]):
            if abs(value-best) < 1e-8:
                return (r'\textbf{' if rank == 0 else r'\underline{') + text + '}'
        return text
    first = 3 if margin_column else 2
    last = first+5
    quality_cols = 2 if paired is not None else 1
    quality_heading = (r'\multicolumn{2}{c}{\textit{Image fidelity (dB)}}'
                       if paired is not None else r'\textit{Image fidelity}')
    lines = [r'\begin{table}[t]', r'\centering', r'\caption{'+caption+'}',
        r'\label{'+table_label+'}', r'\papertablestyle',
        r'\begin{tabular}{@{}l '+('c ' if margin_column else '')+r'cccccc @{\hspace{10pt}} '+('c'*quality_cols)+r'@{}}',
        r'\toprule', r'\toprule',
        ('& & ' if margin_column else '& ')+r'\multicolumn{6}{c}{\textit{Request satisfaction (\%) $\uparrow$}} & '+quality_heading+r' \\',
        rf'\cmidrule(lr){{{first}-{last}}} \cmidrule(lr){{{last+1}-{last+quality_cols}}}',
        'Method & '+(r'$m_0$ & ' if margin_column else '')+r'S1 & S2 & S3 & S4 & S5 & \textbf{Avg.} & PSNR $\uparrow$'+(r' & PSNR-P' if paired is not None else '')+r' \\',
        r'\midrule']
    for key in keys:
        if (margin_column and key == 'Greedy06') or (not margin_column and key in ('VINEGeo', 'Greedy06', 'Greedy04')):
            lines.append(r'\midrule')
        values = [cell(v, i) for i, v in enumerate(metrics[key])]
        values[5] = r'\cellcolor{orange!8}' + values[5]
        if paired is not None:
            p = paired.get(key)
            # Each pair shares one intersection; different rows are not ranked.
            values.append(f"{p['psnr_baseline']:.2f} / {p['psnr_ours']:.2f}" if p and p['n'] else '-')
        lines.append(' & '.join([label(key)] + (['0.'+key[-2:]] if margin_column else []) + values) + r' \\')
    lines += [r'\bottomrule', r'\bottomrule', r'\end{tabular}']
    if paired is not None:
        lines += [r'\papertablenote{PSNR-P: row method / \sysfull{}, evaluated on jointly accepted requests.}']
    return '\n'.join(lines+[r'\end{table}'])+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=PAPER/'data/main_no_ctrlregen07_20260924.json')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--main-margin', choices=['04', '06'], default='06')
    args = p.parse_args()
    data = json.loads(args.data.read_text())
    assert data['n_requests'] == 7321
    summary = data['summary']
    main_keys = BASELINES + ['Greedy'+args.main_margin, 'SMT'+args.main_margin]
    margin_keys = ['Greedy04', 'SMT04', 'Greedy06', 'SMT06']
    full_key = 'SMT'+args.main_margin
    paired_rows = [r for r in data['paired'] if r['arm'] == args.main_margin]
    paired = {r['baseline']: r for r in paired_rows}
    assert len(paired) == len(paired_rows) == 10
    assert set(paired) == set(main_keys)-{full_key}
    for k, r in paired.items():
        assert 0 <= r['n'] <= min(summary[k]['accepted'], summary[full_key]['accepted'])
        if r['n']:
            assert abs(r['psnr_ours']-r['psnr_baseline']-r['delta_psnr_mean']) < 1e-10
        else:
            assert all(r[x] is None for x in ('psnr_ours', 'psnr_baseline', 'delta_psnr_mean'))
    for k in set(main_keys+margin_keys):
        s = summary[k]
        assert s['n'] == sum(r['n'] for r in s['scenarios'].values()) == 7321
        assert s['accepted'] == sum(r['accepted'] for r in s['scenarios'].values())
        assert abs(s['macro_percent'] - sum(r['percent'] for r in s['scenarios'].values())/5) < 1e-10
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'main_results.tex').write_text(render(summary, main_keys, False,
        'Request satisfaction and image fidelity across deployment scenarios.', 'tab:main', paired=paired))
    (args.output_dir/'screening_margin.tex').write_text(render(summary, margin_keys, True,
        'Effect of the initial screening margin on request satisfaction and image fidelity.',
        'tab:screening-margin'))
    (args.output_dir/'margin_view_audit.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'n_requests': 7321, 'main_margin': float('0.'+args.main_margin),
        'main_methods': main_keys, 'appendix_margin_methods': margin_keys,
        'measurements_changed': False,
        'ranking': 'best and second distinct value within each table column',
        'paired_deltas': paired,
        'paired_delta_direction': 'full minus the row baseline on jointly accepted requests',
        'paired_deltas_ranked': False,
        'paired_column': 'PSNR-P: baseline / full absolute means, dB',
    }, indent=2)+'\n')
    print('Rendered main and appendix tables from frozen current-cohort summaries.')


if __name__ == '__main__':
    main()
