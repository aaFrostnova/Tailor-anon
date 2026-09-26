"""Stage all five active fidelity tables; refuse incomplete SSIM measurements.

The frozen PSNR and request outcomes are read from the paper's existing reports.
New SSIM means must cover exactly the same accepted-request conditioning sets.
This script writes only to --output-dir; copying reviewed tables is a separate step.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
MAIN_ORDER = ['StegaStamp', 'MaskWM', 'VINE', 'TrustMark', 'VideoSeal',
              'VINEGeo', 'TrustMarkGeo', 'Fixed1', 'EnumGeo', 'Greedy06', 'SMT06']
ABLATION_ORDER = ['SingleOptimized', 'UnitStrength', 'S1', 'S3', 'Full']
LABELS = dict(VINEGeo='VINE + Geo.', TrustMarkGeo='TrustMark + Geo.',
              Fixed1='Enumeration', EnumGeo='Enumeration + Geo.',
              Greedy04=r'\sysgreedy{}', Greedy06=r'\sysgreedy{}',
              SMT04=r'\sysfull{}', SMT06=r'\sysfull{}', Full=r'\sysfull{}',
              SingleOptimized='Single-fragment', UnitStrength='Unit-strength',
              S1='SMT-top1', S3='SMT-top3', Grid='Grid strengths')
ENDROW = r' \\'


FROZEN_IMAGES_SHA256 = '95393c291b39219bbb7c70163e1f697a26b920fa3114e68d29b437430630600b'


def rank(value, values, digits=2):
    if value is None:
        return '-'
    number = round(value, digits)
    levels = sorted({round(v, digits) for v in values if v is not None}, reverse=True)
    text = f'{value:.{digits}f}'
    if number == levels[0]:
        return r'\textbf{' + text + '}'
    if len(levels) > 1 and number == levels[1]:
        return r'\underline{' + text + '}'
    return text


def row(values):
    return ' & '.join(values) + ENDROW


def start(caption, label, columns, standalone=True, fit=False):
    lines = [r'\begin{table}[!t]'] if standalone else []
    lines += [r'\centering', r'\caption{' + caption + '}',
              r'\label{' + label + '}', r'\papertablestyle']
    if fit:
        lines += [r'\resizebox{\linewidth}{!}{%']
    return lines + [r'\begin{tabular}{' + columns + '}', r'\toprule', r'\toprule']


def end(standalone=True, fit=False, note=None):
    lines = [r'\bottomrule', r'\bottomrule', r'\end{tabular}' + ('}' if fit else '')]
    if note:
        lines += [r'\papertablenote{' + note + '}']
    if standalone:
        lines += [r'\end{table}']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-ssim', type=Path, required=True)
    parser.add_argument('--ablation-ssim', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source_hashes = {}

    def read(path):
        data = path.read_bytes()
        source_hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    frozen = read(PAPER/'data/main_no_ctrlregen07_20260924.json')
    ablation = read(PAPER/'data/ablation_paired_psnr_20260924.json')
    grid = read(PAPER/'data/s4_reduced_evaluation_20260922.json')['ablation_pairs']['04']['Grid']
    ss = read(args.main_ssim)
    ab = read(args.ablation_ssim)
    assert frozen['n_requests'] == ss['n_requests'] == 7321
    assert ablation['n_requests'] == ab['n_requests'] == 1111
    own = {r['method']: r for r in ss['own']}
    paired = {(r['arm'], r['baseline']): r for r in ss['paired']}
    abown = ab['own']
    abpair = ab['paired']
    assert len(own) == len(ss['own'])
    assert len(paired) == len(ss['paired'])
    assert ss['images_per_configuration'] == ab['images_per_configuration'] == 100
    assert ss['images_sha256'] == ab['images_sha256'] == FROZEN_IMAGES_SHA256
    retained_hash = hashlib.sha256(json.dumps(sorted(frozen['retained_spec_ids'])).encode()).hexdigest()
    assert ss['retained_spec_ids_sha256'] == retained_hash

    def require_full(record, n, fields):
        available = record.get('available', record.get('available_n'))
        assert record['n'] == available == n, ('missing SSIM', record)
        assert n == 0 or record['complete'], ('incomplete SSIM', record)
        expected_ids = record.get('accepted_spec_ids_sha256', record.get('joint_spec_ids_sha256'))
        assert isinstance(expected_ids, str) and len(expected_ids) == 64
        if 'available_spec_ids_sha256' in record:
            assert record['available_spec_ids_sha256'] == expected_ids
        for field in fields:
            value = record[field]
            assert (value is None if n == 0 else
                    isinstance(value, (int, float)) and math.isfinite(value) and -1 <= value <= 1)

    for method, frozen_row in frozen['summary'].items():
        require_full(own[method], frozen_row['accepted'], ['ssim'])
    for pair in frozen['paired']:
        require_full(paired[pair['arm'], pair['baseline']], pair['n'],
                     ['baseline_ssim', 'full_ssim'])
    for variant in ABLATION_ORDER:
        require_full(abown[variant], ablation['metrics'][variant]['accepted'], ['ssim'])
    for variant, pair in ablation['paired'].items():
        require_full(abpair[variant], pair['n'], ['variant_ssim', 'full_ssim'])
        assert abpair[variant]['joint_spec_ids_sha256'] == pair['joint_ids_sha256']
    require_full(abpair['Grid'], grid['n'], ['variant_ssim', 'full_ssim'])
    assert abpair['Grid']['joint_spec_ids_sha256'] == '3d26672169f3697a4bc1ec019ebff01fe2489e8eab70df8243e99da431b5413d'
    assert ss['metric'] == ab['metric'], 'different SSIM implementations'
    for report in [ss, ab]:
        for group in ['source_sha256', 'cell_sha256']:
            for path, expected in report.get(group, {}).items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, ('source changed', path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    def write(name, lines):
        text = '\n'.join(lines) + '\n'
        (args.output_dir/name).write_text(text)
        generated[name] = hashlib.sha256(text.encode()).hexdigest()

    summary = frozen['summary']
    quality = {k: own[k]['ssim'] for k in MAIN_ORDER}
    metrics = {k: [*[summary[k]['scenarios'][f'S{i}']['percent'] for i in range(1, 6)],
                    summary[k]['macro_percent'], summary[k]['psnr']] for k in MAIN_ORDER}
    lines = start('Request satisfaction and image fidelity across deployment scenarios.',
                  'tab:main', r'@{}l cccccc @{\hspace{6pt}} ccc@{}', fit=True)
    lines += [r'& \multicolumn{6}{c}{\textit{Request satisfaction (\%) $\uparrow$}} & \multicolumn{3}{c}{\textit{Image fidelity}} \\',
              r'\cmidrule(lr){2-7} \cmidrule(lr){8-10}',
              r'Method & S1 & S2 & S3 & S4 & S5 & \textbf{Avg.} & PSNR $\uparrow$ & SSIM $\uparrow$ & PSNR-P \\',
              r'\midrule']
    for k in MAIN_ORDER:
        if k in ['VINEGeo', 'Greedy06']:
            lines += [r'\midrule']
        values = [rank(v, [metrics[o][j] for o in MAIN_ORDER]) for j, v in enumerate(metrics[k])]
        values[5] = r'\cellcolor{orange!8}' + values[5]
        pair = next((p for p in frozen['paired'] if p['arm'] == '06' and p['baseline'] == k), None)
        psnr_p = (f"{pair['psnr_baseline']:.2f} / {pair['psnr_ours']:.2f}" if pair and pair['n'] else '-')
        lines += [row([LABELS.get(k, k), *values, rank(quality[k], quality.values(), 4), psnr_p])]
    lines += end(fit=True, note=r'PSNR-P: row method / \sysfull{}, evaluated on jointly accepted requests.')
    write('main_results.tex', lines)

    lines = start('Ablation of candidate search, composition, and strength adaptation.',
                  'tab:ablation-process', '@{}l c ccc@{}', standalone=False, fit=True)
    lines += [r'& \textit{Satisfaction} & \multicolumn{3}{c}{\textit{Image fidelity}} \\',
              r'\cmidrule(lr){2-2} \cmidrule(lr){3-5}',
              r'Method & Rate (\%) $\uparrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & PSNR-P \\', r'\midrule']
    for k in ABLATION_ORDER:
        m = ablation['metrics'][k]
        pair = ablation['paired'].get(k)
        p = f"{pair['variant_psnr']:.2f} / {pair['full_psnr']:.2f}" if pair else '-'
        lines += [row([LABELS[k], rank(m['percent'], [v['percent'] for v in ablation['metrics'].values()]),
                       rank(m['psnr'], [v['psnr'] for v in ablation['metrics'].values()]),
                       rank(abown[k]['ssim'], [abown[v]['ssim'] for v in ABLATION_ORDER], 4), p])]
    lines += end(standalone=False, fit=True,
                 note=r'PSNR-P: row variant / \sysfull{}, evaluated on jointly accepted requests.')
    write('ablation_process.tex', lines)

    lines = start('Paired image fidelity on jointly accepted requests.',
                  'tab:paired-quality', '@{}l c ccc cc@{}', fit=True)
    lines += [r'& & \multicolumn{3}{c}{\textit{PSNR (dB) $\uparrow$}} & \multicolumn{2}{c}{\textit{SSIM $\uparrow$}} \\',
              r'\cmidrule(lr){3-5} \cmidrule(lr){6-7}',
              r'Baseline & $n_{\cap}$ & Baseline & \sysfull{} & $\Delta$ & Baseline & \sysfull{} \\', r'\midrule']
    for arm in ['04', '06']:
        if arm == '06':
            lines += [r'\midrule']
        lines += [r'\multicolumn{7}{l}{\sysfull{}: $m_0=0.' + arm + r'$} \\']
        for p in [p for p in frozen['paired'] if p['arm'] == arm]:
            k = p['baseline']
            b, f = p['psnr_baseline'], p['psnr_ours']
            sp = paired[arm, k]
            sb, sf = sp['baseline_ssim'], sp['full_ssim']
            lines += [row([LABELS.get(k, k), f"{p['n']:,}", rank(b, [b, f]), rank(f, [b, f]),
                           f"{p['delta_psnr_mean']:+.2f}" if p['n'] else '-',
                           rank(sb, [sb, sf], 4), rank(sf, [sb, sf], 4)])]
    lines += end(fit=True)
    write('paired_quality.tex', lines)

    keys = ['Greedy04', 'SMT04', 'Greedy06', 'SMT06']
    lines = start('Effect of the initial screening margin on request satisfaction and image fidelity.',
                  'tab:screening-margin', r'@{}l c cccccc @{\hspace{6pt}} cc@{}',
                  standalone=False, fit=True)
    lines += [r'& & \multicolumn{6}{c}{\textit{Request satisfaction (\%) $\uparrow$}} & \multicolumn{2}{c}{\textit{Image fidelity}} \\',
              r'\cmidrule(lr){3-8} \cmidrule(lr){9-10}',
              r'Method & $m_0$ & S1 & S2 & S3 & S4 & S5 & \textbf{Avg.} & PSNR $\uparrow$ & SSIM $\uparrow$ \\',
              r'\midrule']
    margin_metrics = {k: [*[summary[k]['scenarios'][f'S{i}']['percent'] for i in range(1, 6)],
                          summary[k]['macro_percent'], summary[k]['psnr'], own[k]['ssim']] for k in keys}
    for k in keys:
        if k == 'Greedy06':
            lines += [r'\midrule']
        values = [rank(v, [margin_metrics[o][j] for o in keys], 4 if j == 7 else 2)
                  for j, v in enumerate(margin_metrics[k])]
        values[5] = r'\cellcolor{orange!8}' + values[5]
        lines += [row([LABELS[k], '0.' + k[-2:], *values])]
    lines += end(standalone=False, fit=True)
    write('screening_margin.tex', lines)

    lines = start('Image fidelity with continuous and grid strengths.',
                  'tab:ablation-model', '@{}l cc@{}', standalone=False, fit=True)
    lines += [r'& \multicolumn{2}{c}{\textit{Image fidelity}} \\', r'\cmidrule(lr){2-3}',
              r'Method & PSNR $\uparrow$ & SSIM $\uparrow$ \\', r'\midrule']
    gpsnr = [grid['psnr_ours'], grid['psnr_baseline']]
    gssim = [abpair['Grid']['full_ssim'], abpair['Grid']['variant_ssim']]
    for i, k in enumerate(['Full', 'Grid']):
        lines += [row([LABELS[k], rank(gpsnr[i], gpsnr), rank(gssim[i], gssim, 4)])]
    lines += end(standalone=False, fit=True)
    write('ablation_model.tex', lines)

    report = dict(schema='ssim_table_export_v1', main_requests=7321, ablation_requests=1111,
                  all_required_measurements_complete=True, frozen_request_decisions_unchanged=True,
                  frozen_psnr_values_unchanged=True, source_sha256=source_hashes,
                  output_sha256=generated, exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output_dir/'table_export_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
