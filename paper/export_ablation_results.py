"""Export audited, paired live ablations without changing experiment outputs."""
import collections
import hashlib
import json
from pathlib import Path
from statistics import mean, median

PAPER = Path(__file__).resolve().parents[1]
RUN = Path('/data/tailor/home/outputs/rigor_editing_only_20260911/unified_ba_20260914/shared_table_20260916/priority_20260917')
VARIANTS = ['G_shared', 'S1', 'S3', 'S3_R', 'Full', 'Grid', 'NoRecoveryInteraction', 'NoFrontend']


def main():
    hashes = {}
    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    audit = read(RUN / 'pipeline_audit.json')
    result = read(RUN / 'results.json')
    sources = RUN / 'sources.json'
    assert hashlib.sha256(sources.read_bytes()).hexdigest() == audit['source_binding']
    assert audit['complete'] and audit['pending_cpu'] == audit['pending_gpu_configs'] == 0
    assert result['status'] == {k: v for k, v in audit.items() if k != 'source_binding'}
    cohort = read(PAPER / 'data/ablation_unique_reporting.json')['rows']
    assert len(cohort) == len({r['spec_id'] for r in cohort}) == 924
    rows = result['rows']
    index = {(r['variant'], r['arm'], r['spec_id']): r for r in rows}
    assert len(index) == len(rows) == audit['target_evaluations']
    observed = collections.defaultdict(collections.Counter)
    for row in rows:
        observed[row['variant'] + '_' + row['arm']][row['status']] += 1
    assert dict(observed) == audit['counts']
    cohorts = {'union': cohort}
    for name, n in [('representative', 388), ('targeted', 553)]:
        cohorts[name] = [r for r in cohort if name in r['cohorts']]
        assert len(cohorts[name]) == n
    assert sum(len(r['cohorts']) == 2 for r in cohort) == 17
    metrics = {}
    for variant in VARIANTS:
        metrics[variant] = {}
        for name, part in cohorts.items():
            metrics[variant][name] = {}
            for arm in ['04', '06']:
                chosen = [index[variant, arm, r['spec_id']] for r in part]
                assert all(z['i'] == r['representative_i'] and z['cls'] == r['cls']
                           for z, r in zip(chosen, part))
                assert all(z['status'] in {'accepted', 'live_exhausted', 'model_unsat', 'greedy_stalled'}
                           for z in chosen)
                accepted = sum(z['status'] == 'accepted' for z in chosen)
                metrics[variant][name][arm] = dict(n=len(part), accepted=accepted,
                                                  percent=100 * accepted / len(part))
    paired_quality={}
    for variant in ['Grid','NoRecoveryInteraction','NoFrontend']:
        paired_quality[variant]={}
        for name,part in cohorts.items():
            paired_quality[variant][name]={}
            for arm in ['04','06']:
                pairs=[(index['Full',arm,r['spec_id']],index[variant,arm,r['spec_id']]) for r in part]
                pairs=[(a,b) for a,b in pairs if a['status']==b['status']=='accepted']
                full=[a['verdict']['psnr_live'] for a,b in pairs]
                other=[b['verdict']['psnr_live'] for a,b in pairs]
                delta=[a-b for a,b in zip(full,other)]
                paired_quality[variant][name][arm]=dict(n=len(pairs),full_psnr_db=mean(full),
                    variant_psnr_db=mean(other),mean_delta_psnr_db=mean(delta),median_delta_psnr_db=median(delta),
                    spec_ids=[a['spec_id'] for a,b in pairs])
    search_variants = ['G_shared', 'S1', 'S3', 'S3_R', 'Full']
    search_quality = {}
    for arm in ['04', '06']:
        search_quality[arm] = {}
        for variant in search_variants:
            accepted = [index[variant, arm, r['spec_id']] for r in cohort
                        if index[variant, arm, r['spec_id']]['status'] == 'accepted']
            assert accepted and len(accepted) == metrics[variant]['union'][arm]['accepted']
            assert len(accepted) == len({r['spec_id'] for r in accepted})
            search_quality[arm][variant] = dict(n=len(accepted),
                psnr_db=mean(r['verdict']['psnr_live'] for r in accepted))
    report = dict(complete=True, measured_at=audit['updated_at'], primary_metric='live request satisfaction',
                  paired_representatives_verified=True, overlapping_cohorts=17,
                  reporting_cohort='union', reporting_requests=924,
                  reporting_variants=['S1', 'S3', 'Full', 'Grid'],
                  reporting_margins=[0.04],
                  grid_reporting_metric='paired embedding PSNR on joint live successes',
                  search_psnr_conditioning="Each search variant's own live-accepted requests within each margin; equal weight per distinct request",
                  search_own_success_quality=search_quality,
                  metrics=metrics, paired_quality=paired_quality, source_sha256=hashes)
    (PAPER / 'data/ablation_results_20260920.json').write_text(json.dumps(report, indent=2) + '\n')
    def ranked(value, compared):
        ranks = []
        for candidate in sorted(compared, reverse=True):
            if not ranks or abs(candidate - ranks[-1]) > 1e-10:
                ranks.append(candidate)
        text = f'{value:.2f}'
        return (r'\textbf{' + text + '}' if abs(value - ranks[0]) <= 1e-10 else
                r'\underline{' + text + '}' if len(ranks) > 1 and abs(value - ranks[1]) <= 1e-10 else text)
    names = [('SMT-top1', 'S1'),
             ('SMT-top3', 'S3'),
             (r'\sysfull{}', 'Full')]
    lines = [r'\centering',
        r'\caption{Request satisfaction and PSNR comparison.}',
        r'\label{tab:ablation-process}', r'\small',
        r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{1.08}',
        r'\begin{tabular}{@{}l c c@{}}', r'\toprule', r'\toprule',
        r'& \textit{Request satisfaction} & \textit{Image fidelity} \\',
        r'\cmidrule(lr){2-2} \cmidrule(lr){3-3}',
        r'Method & Rate (\%) $\uparrow$ & PSNR $\uparrow$ \\', r'\midrule']
    arm = '04'
    quality = {v:search_quality[arm][v]['psnr_db'] for _,v in names}
    for label, variant in names:
        satisfaction = ranked(metrics[variant]['union'][arm]['percent'],
                              [metrics[v]['union'][arm]['percent'] for _,v in names])
        psnr = ranked(quality[variant], quality.values())
        lines.append(' & '.join([label, satisfaction, psnr]) + r' \\')
    lines += [r'\bottomrule', r'\bottomrule', r'\end{tabular}']
    (PAPER / 'tab/ablation_process.tex').write_text('\n'.join(lines) + '\n')

    lines = [r'\centering',
        r'\caption{PSNR comparison.}',
        r'\label{tab:ablation-model}', r'\small',
        r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{1.08}',
        r'\begin{tabular}{@{}l c@{}}', r'\toprule', r'\toprule',
        r'& \textit{Image fidelity} \\', r'\cmidrule(lr){2-2}',
        r'Method & PSNR $\uparrow$ \\', r'\midrule']
    q = paired_quality['Grid']['union'][arm]
    values = [q['full_psnr_db'], q['variant_psnr_db']]
    for label, value in zip([r'\sysfull{}', 'Grid strengths'], values):
        formatted = (r'\textbf{' if value >= max(values) - 1e-10 else r'\underline{') + f'{value:.2f}' + '}'
        lines.append(' & '.join([label, formatted]) + r' \\')
    lines += [r'\bottomrule', r'\bottomrule', r'\end{tabular}']
    (PAPER / 'tab/ablation_model.tex').write_text('\n'.join(lines) + '\n')
    print('Audited 924 distinct requests; exported pooled search rates and paired Grid PSNR.')


if __name__ == '__main__':
    main()
