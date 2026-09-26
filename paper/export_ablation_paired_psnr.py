"""Add matched absolute PSNR values to the current 1,111-request ablation table."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

PAPER = Path(__file__).resolve().parents[1]
ORDER = ['SingleOptimized', 'UnitStrength', 'S1', 'S3', 'Full']
LABELS = dict(zip(ORDER, ['Single-fragment', 'Unit-strength', 'SMT-top1', 'SMT-top3', r'\sysfull{}']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    hashes = {}
    def read(path, expected=None):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert expected is None or digest == expected, ('source changed', str(path))
        hashes[str(path)] = digest
        return json.loads(raw)
    ref = read(PAPER/'data/s4_reduced_evaluation_20260922.json')
    extra = read(PAPER/'data/final_additions_20260923.json')
    ids = set(ref['ablation_spec_ids'])
    assert len(ids) == 1111
    def frozen(rel, record=ref):
        path = args.run_root/rel
        return read(path, record['source_sha256'][str(path)])
    base = frozen('shared_table_20260916/priority_20260917/results.json')['rows']
    amended = frozen('live_failure_fallback_20260921/amended_results.json')['rows']
    editing = frozen('editing_request_extension_20260921/results.json')['rows']
    component = frozen('component_ablation_20260923/results.json', extra)['rows']
    index = {(r['variant'], r['arm'], r['spec_id']): r for r in base}
    assert len(index) == len(base)
    for r in amended:
        if r['cohort'] == 'ablation': index[r['variant'], r['arm'], r['spec_id']] = r
    for r in editing + component:
        if r['spec_id'] in ids: index[r['variant'], r['arm'], r['spec_id']] = r
    def quality(r):
        return r['psnr_live'] if r.get('psnr_live') is not None else r['verdict']['psnr_live']
    def config(r):
        return json.dumps({k:r['candidate'][k] for k in ('order', 'fe', 'fe_on', 's')}, sort_keys=True)
    results, successful = {}, {}
    for variant in ORDER:
        rows = [index[variant, '04', i] for i in sorted(ids)]
        assert all(r['status'] in {'accepted', 'model_unsat', 'live_exhausted'} for r in rows)
        ok = {r['spec_id']:r for r in rows if r['status'] == 'accepted'}
        assert all(r['verdict']['pass'] and not r['verdict']['pending'] for r in ok.values())
        successful[variant] = ok
        target = (extra['component_ablations'][variant] if variant in extra['component_ablations']
                  else ref['ablations']['04'][variant])
        own = mean(quality(r) for r in ok.values())
        assert len(ok) == target['accepted'] and abs(own-target['psnr']) < 1e-10
        results[variant] = dict(accepted=len(ok), n=1111, percent=100*len(ok)/1111, psnr=own)
    pairs, observations = {}, []
    full = successful['Full']
    for variant in ORDER[:-1]:
        baseline = successful[variant]
        joint = sorted(full.keys() & baseline.keys())
        assert all(full[i]['cls'] == baseline[i]['cls'] for i in joint)
        ours = [quality(full[i]) for i in joint]
        other = [quality(baseline[i]) for i in joint]
        deltas = [a-b for a,b in zip(ours, other)]
        pairs[variant] = dict(n=len(joint), full_psnr=mean(ours), variant_psnr=mean(other),
            delta_psnr=mean(deltas), median_delta_psnr=median(deltas),
            min_delta_psnr=min(deltas), max_delta_psnr=max(deltas),
            same_config_count=sum(config(full[i]) == config(baseline[i]) for i in joint),
            same_psnr_count=sum(abs(d) < 1e-10 for d in deltas),
            joint_ids_sha256=hashlib.sha256(json.dumps(joint).encode()).hexdigest())
        observations += [dict(spec_id=i, variant=variant, full_psnr=a, variant_psnr=b, delta_psnr=d)
                         for i,a,b,d in zip(joint, ours, other, deltas)]
    report = dict(schema='ablation_paired_psnr_v1', n_requests=1111, initial_margin=.04,
        metrics=results, paired=pairs, source_sha256=hashes,
        measurements_changed=False, delta_direction='Full minus variant on joint accepted requests',
        paired_column='PSNR-P: variant / full absolute means, dB',
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'ablation_paired_psnr.json').write_text(json.dumps(report, indent=2)+'\n')
    with (args.output_dir/'ablation_paired_psnr.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(observations[0]))
        w.writeheader(); w.writerows(observations)
    def rank(value, metric):
        vs = sorted({round(r[metric],9) for r in results.values()}, reverse=True)
        txt = f'{value:.2f}'
        return ((r'\textbf{' if abs(value-vs[0]) < 1e-8 else r'\underline{')+txt+'}'
                if any(abs(value-v) < 1e-8 for v in vs[:2]) else txt)
    lines = [r'\centering',
        r'\caption{Ablation of candidate search, composition, and strength adaptation.}',
        r'\label{tab:ablation-process}', r'\papertablestyle',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{@{}l c cc@{}}',
        r'\toprule', r'\toprule',
        r'& \textit{Request satisfaction} & \multicolumn{2}{c}{\textit{Image fidelity (dB)}} \\',
        r'\cmidrule(lr){2-2} \cmidrule(lr){3-4}',
        r'Method & Rate (\%) $\uparrow$ & PSNR $\uparrow$ & PSNR-P \\', r'\midrule']
    for variant in ORDER:
        r = results[variant]
        paired_value = ('-' if variant == 'Full' else
            f"{pairs[variant]['variant_psnr']:.2f} / {pairs[variant]['full_psnr']:.2f}")
        lines.append(' & '.join([LABELS[variant], rank(r['percent'], 'percent'), rank(r['psnr'], 'psnr'), paired_value])+r' \\')
    lines += [r'\bottomrule', r'\bottomrule', r'\end{tabular}}',
        r'\papertablenote{PSNR-P: row variant / \sysfull{}, evaluated on jointly accepted requests.}']
    (args.output_dir/'ablation_process.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    main()
