"""Add audited Enumeration and optional-geometry controls to the completed main table.

Reads frozen measurements only. The original SMT/greedy results and their
paired-quality evidence remain unchanged; pending fallback/editing runs are
not inputs to this exporter. All generated artifacts are staged at --output.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from export_paired_quality import write_main_table


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paper = Path(__file__).resolve().parents[1]
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    specs = read(paper / 'data/request_specifications_6532.json')['rows']
    index = {r['spec_id']: r for r in specs}
    assert len(specs) == len(index) == 6532
    summary = read(paper / 'data/paired_quality_20260921.json')
    original = json.loads(json.dumps(summary))
    shared_path = args.run_root / 'shared_table_20260916/priority_20260917/manifest.json'
    shared = read(shared_path)
    expected_requests = {r['spec_id']: dict(spec_id=r['spec_id'], cls=r['cls'], i=r['i'], request=r['request'])
                         for r in shared['targets'] if r['variant'] == 'G_shared' and r['arm'] == '04'}
    assert len(expected_requests) == 6532
    campaigns = {}
    additions = {}
    for directory, cell_count in [('fixed1_20260918', 147), ('single_geo_20260921', 231)]:
        root = args.run_root / directory
        audit = read(root / 'audit.json')
        sources = read(root / 'sources.json')
        if directory.startswith('fixed1'):
            assert sources['request_source_sha256'] == inputs[str(shared_path)]
        progress = read(root / 'progress.json')
        results = read(root / 'results.json')
        requests = read(root / 'inputs/requests.json')
        assert audit['passed'] and audit['final_required'] is False
        assert progress['complete'] and results['status'] == progress
        assert audit['source_binding'] == inputs[str(root / 'sources.json')]
        assert audit['results_sha256'] == inputs[str(root / 'results.json')]
        assert progress['known_config_attack_cells'] == progress['possible_config_attack_cells'] == cell_count
        for path, expected in sources['hashes'].items():
            assert digest(path) == expected, ('changed source', path)
        assert len(audit['measurement_evidence']) == cell_count
        for path, expected in audit['measurement_evidence'].items():
            assert digest(path) == expected, ('changed live evidence', path)
        if directory.startswith('single_geo'):
            policy = read(root / 'policy_validation.json')
            assert policy['passed'] and policy['source_binding'] == audit['source_binding']
            assert digest(root / 'frontend_results.json') == audit['frontend_results_sha256']
        assert len(requests) == len({r['spec_id'] for r in requests}) == 6532
        for row in requests:
            spec = index[row['spec_id']]
            assert row == expected_requests[row['spec_id']]
            assert row['cls'] == spec['cls'] and row['i'] in spec['original_indices']
            request = dict(row['request'], attacks=sorted(row['request']['attacks']))
            assert request == spec['inputs']
        groups = {'Fixed1': results['rows']} if directory.startswith('fixed1') else {
            fragment + 'Geo': [r for r in results['rows'] if r['fragment'] == fragment]
            for fragment in ['VINE', 'TrustMark', 'VideoSeal']}
        assert sum(map(len, groups.values())) == len(results['rows'])
        for method, rows in groups.items():
            assert len(rows) == len({r['spec_id'] for r in rows}) == 6532
            counts = Counter(r['status'] for r in rows)
            assert set(counts) <= {'accepted', 'live_exhausted'}
            recorded = progress['counts'] if method == 'Fixed1' else progress['counts'][method[:-3]]
            assert counts == recorded
            for row in rows:
                spec = index[row['spec_id']]
                assert row['cls'] == spec['cls'] and row['i'] == expected_requests[row['spec_id']]['i']
                assert all(not step['pending'] for step in row['walk'])
                if row['status'] == 'accepted':
                    verdict = row['verdict']
                    assert verdict['pass'] and not verdict['pending']
                    assert all(verdict[k] for k in ['psnr_ok', 'latency_ok', 'cols_ok'])
                    assert math.isfinite(verdict['psnr_live'])
                    assert verdict['psnr_live'] >= spec['inputs']['min_psnr']
                    assert row['walk'][-1]['rank'] == row['rank'] and row['walk'][-1]['passed']
                    assert all(not step['passed'] for step in row['walk'][:-1])
                else:
                    assert all(not step['passed'] for step in row['walk'])
            scenarios = {}
            for i in range(1, 6):
                subset = [r for r in rows if r['cls'] == f'C{i}']
                accepted = sum(r['status'] == 'accepted' for r in subset)
                scenarios[f'S{i}'] = dict(n=len(subset), accepted=accepted,
                                           percent=100 * accepted / len(subset))
            psnr = mean(r['verdict']['psnr_live'] for r in rows if r['status'] == 'accepted')
            summary['accepted'][method] = counts['accepted']
            summary['live_percent'][method] = 100 * counts['accepted'] / len(rows)
            summary['scenario_results'][method] = dict(scenarios=scenarios,
                average_percent=mean(r['percent'] for r in scenarios.values()))
            summary['own_success_quality'][method] = dict(n=counts['accepted'], psnr_mean=psnr)
            summary['method_display_names'][method] = 'Enumeration' if method == 'Fixed1' else method[:-3] + ' + Geo.'
            additions[method] = dict(counts=dict(counts), **summary['scenario_results'][method],
                                     psnr_mean=psnr, completed_at=progress['updated_at'])
        campaigns[directory] = dict(completed_at=audit['completed_at'],
            source_binding=audit['source_binding'], results_sha256=audit['results_sha256'],
            source_files_verified=len(sources['hashes']), live_evidence_files_verified=cell_count)
    for key in ['accepted', 'live_percent', 'scenario_results', 'own_success_quality', 'method_display_names']:
        assert all(summary[key][method] == value for method, value in original[key].items())
    assert summary['pairs'] == original['pairs']
    report = dict(schema='completed_main_with_baselines_v1', complete=True,
        request_denominator=6532, original_campaign_retained=True,
        representative_rule='Frozen shared-greedy representatives; four input fields and original-index membership verified',
        shared_representative_aliases=sum(r['i'] != index[sid]['representative_i'] for sid,r in expected_requests.items()),
        live_failure_fallback_included=False, editing_request_extension_included=False,
        reported_methods=[m for m in summary['scenario_results'] if m != 'VideoSealGeo'],
        omitted_main_methods={'VideoSealGeo': 'Same accepted requests and selected no-frontend configurations as plain VideoSeal; measurements retained'},
        satisfaction_aggregation=summary['main_table_satisfaction_aggregation'],
        quality_conditioning=summary['main_table_quality_conditioning'],
        accepted=summary['accepted'], scenario_results=summary['scenario_results'],
        own_success_quality={k:dict(n=v['n'], psnr_mean=v['psnr_mean'])
                             for k,v in summary['own_success_quality'].items()},
        added_baselines=additions, validated_campaigns=campaigns, inputs=inputs,
        exporter_sha256=digest(__file__), table_writer_sha256=digest(paper/'scripts/export_paired_quality.py'))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'main_baselines_completed_20260921.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_main_table(args.output, summary)
    for name, row in additions.items():
        print(f"{name}: accepted={row['counts']['accepted']}/6532, Avg={row['average_percent']:.2f}%, PSNR={row['psnr_mean']:.2f} dB")
    print('Validated all source/evidence hashes and retained every original main-table value.')


if __name__ == '__main__':
    main()
