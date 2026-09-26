"""Publish audited Enumeration + Geo. and completed component ablations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

PAPER = Path(__file__).resolve().parents[1]


def export(root):
    hashes = {}

    def read(path, expected=None):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert expected is None or digest == expected, ('Changed source', str(path))
        hashes[str(path)] = digest
        return json.loads(raw)

    ref = read(PAPER / 'data/s4_reduced_evaluation_20260922.json')
    additions = read(PAPER / 'data/completed_additions_20260923.json')
    keep, abkeep = set(ref['retained_spec_ids']), set(ref['ablation_spec_ids'])
    assert len(keep) == 7414 and len(abkeep) == 1111

    def frozen(path):
        expected = ref['source_sha256'][str(path)]
        return read(path, expected)

    def quality(r):
        return r['psnr_live'] if r.get('psnr_live') is not None else r['verdict']['psnr_live']

    def summary(rows, ids):
        assert len(rows) == len({r['spec_id'] for r in rows}) == len(ids)
        assert {r['spec_id'] for r in rows} == ids
        assert all(r['status'] in ('accepted', 'model_unsat', 'live_exhausted') for r in rows)
        ok = [r for r in rows if r['status'] == 'accepted']
        assert all(r['verdict']['pass'] and not r['verdict']['pending'] for r in ok)
        scenes = {}
        for c in ('C1', 'C2', 'C3', 'C4', 'C5'):
            part = [r for r in rows if r['cls'] == c]
            n = sum(r['status'] == 'accepted' for r in part)
            scenes['S' + c[1:]] = dict(n=len(part), accepted=n, percent=100*n/len(part))
        return dict(n=len(rows), accepted=len(ok), percent=100*len(ok)/len(rows),
                    psnr=mean(quality(r) for r in ok), scenarios=scenes,
                    macro_percent=mean(r['percent'] for r in scenes.values()),
                    status_counts=dict(Counter(r['status'] for r in rows)))

    verified = {}
    for run, filename in (('enumeration_geo_20260923', 'audit.json'),
                          ('component_ablation_20260923', 'pipeline_audit.json')):
        folder = root / run
        audit = read(folder / filename)
        binding = read(folder / 'sources.json', audit['source_binding'])
        for name, expected in binding['hashes'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
        verified[run] = dict(source_files=len(binding['hashes']))
        if run.startswith('enumeration'):
            assert audit['passed']
            for name, expected in audit['measurement_evidence'].items():
                assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
            verified[run]['measurement_files'] = len(audit['measurement_evidence'])
            enum_raw = read(folder / 'results.json', audit['results_sha256'])
            assert enum_raw['status']['complete']
            assert enum_raw['status']['known_config_attack_cells'] == enum_raw['status']['possible_config_attack_cells'] == 609
        else:
            assert audit['complete'] and audit['pending_cpu'] == audit['pending_gpu_configs'] == 0
            component = read(folder / 'results.json')
            assert component['status']['complete'] and component['status']['counts'] == audit['counts']

    enum = summary(enum_raw['rows'], keep)
    enum_report = read(root / 'enumeration_geo_20260923/main_table_row.json')
    assert enum['accepted'] == enum_report['accepted']
    assert abs(enum['psnr'] - enum_report['psnr']) < 1e-9
    assert abs(enum['macro_percent'] - enum_report['macro_percent']) < 1e-9
    assert all(abs(float(s)-1) < 1e-9 for r in enum_raw['rows'] if r['status'] == 'accepted'
               for s in r['candidate']['s'].values())
    components = {}
    for name in ('SingleOptimized', 'UnitStrength'):
        rows = [r for r in component['rows'] if r['variant'] == name]
        assert all(r['arm'] == '04' for r in rows)
        for r in rows:
            if r['status'] != 'accepted': continue
            c = r['candidate']
            if name == 'SingleOptimized': assert len(c['order']) == 1
            else: assert all(abs(float(v)-1) < 1e-9 for v in c['s'].values())
        components[name] = summary(rows, abkeep)

    amended = frozen(root / 'live_failure_fallback_20260921/amended_results.json')['rows']
    editing = frozen(root / 'editing_request_extension_20260921/results.json')['rows']
    pairs = {}
    enum_ok = {r['spec_id']: r for r in enum_raw['rows'] if r['status'] == 'accepted'}
    for arm in ('04', '06'):
        rows = [r for r in amended if r['cohort'] == 'main' and r['arm'] == arm]
        rows += [r for r in editing if r['variant'] == 'Full' and r['arm'] == arm]
        rows = [r for r in rows if r['spec_id'] in keep]
        assert len(rows) == len({r['spec_id'] for r in rows}) == len(keep)
        full_ok = {r['spec_id']: r for r in rows if r['status'] == 'accepted'}
        assert len(full_ok) == ref['reduced']['SMT' + arm]['accepted']
        joint = sorted(full_ok.keys() & enum_ok.keys())
        pairs[arm] = dict(n=len(joint),
            baseline_psnr=mean(quality(enum_ok[i]) for i in joint),
            full_psnr=mean(quality(full_ok[i]) for i in joint),
            joint_spec_ids_sha256=hashlib.sha256(json.dumps(joint).encode()).hexdigest())
        pairs[arm]['delta_psnr'] = pairs[arm]['full_psnr'] - pairs[arm]['baseline_psnr']

    methods = dict(ref['reduced'])
    for name, r in additions['single_baselines'].items():
        methods[name] = dict(n=r['n'], accepted=r['accepted'], percent=r['overall_percent'],
            psnr=r['psnr'], macro_percent=r['macro_percent'],
            scenarios={'S'+c[1:]:v for c,v in r['classes'].items()})
    methods['EnumGeo'] = enum
    order = [('StegaStamp','StegaStamp'), ('MaskWM','MaskWM'), ('VINE','VINE'),
             ('TrustMark','TrustMark'), ('VideoSeal','VideoSeal'), ('VINEGeo','VINE + Geo.'),
             ('TrustMarkGeo','TrustMark + Geo.'), ('Fixed1','Enumeration'), ('EnumGeo','Enumeration + Geo.'),
             ('Greedy04',r'\sysgreedy{}'), ('SMT04',r'\sysfull{}'),
             ('Greedy06',r'\sysgreedy{}'), ('SMT06',r'\sysfull{}')]

    def rank(value, candidates):
        if value is None: return '-'
        values = sorted(set(round(v, 9) for v in candidates if v is not None), reverse=True)
        s = f'{value:.2f}'
        if abs(value-values[0]) < 1e-8: return r'\textbf{' + s + '}'
        if len(values)>1 and abs(value-values[1]) < 1e-8: return r'\underline{' + s + '}'
        return s

    def replace_body(filename, body):
        p = PAPER / 'tab' / filename
        s = p.read_text(); start = s.index('\\midrule\n')+len('\\midrule\n')
        end = s.index('\\bottomrule', start)
        p.write_text(s[:start]+'\n'.join(body)+'\n'+s[end:])

    metrics = {k:[*[v['scenarios'][f'S{i}']['percent'] for i in range(1,6)],v['macro_percent'],v['psnr']]
               for k,v in methods.items()}
    body = []
    for key,label in order:
        if key in ('VINEGeo','Greedy04','Greedy06'): body.append(r'\midrule')
        values = [rank(v,[metrics[k][j] for k,_ in order]) for j,v in enumerate(metrics[key])]
        values[5] = r'\cellcolor{orange!8}' + values[5]
        arm = '0.'+key[-2:] if key.startswith(('Greedy','SMT')) else '-'
        body.append(' & '.join([label,arm,*values])+r' \\')
    replace_body('main_results.tex', body)
    replace_body('eval_current_main.tex', [
        ' & '.join([label,'0.'+key[-2:] if key.startswith(('Greedy','SMT')) else '-',
                    f"{methods[key]['percent']:.2f}",*[f'{v:.2f}' for v in [metrics[key][5],*metrics[key][:5]]]])+r' \\'
        for key,label in order])
    path = PAPER / 'tab/paired_quality.tex'; lines = path.read_text().splitlines(); i = 0
    for j,line in enumerate(lines):
        if line.startswith('Enumeration + Geo. &'):
            p = pairs[('04','06')[i]]; i += 1
            b,f = p['baseline_psnr'],p['full_psnr']
            lines[j] = ' & '.join(['Enumeration + Geo.',f"{enum['percent']:.2f}",f"{p['n']:,}",
                rank(b,[b,f]),rank(f,[b,f])])+r' \\'
    assert i == 2
    path.write_text('\n'.join(lines)+'\n')
    ablations = dict(ref['ablations']['04'], **components)
    aborder = [('SingleOptimized','Single-fragment'),('UnitStrength','Unit-strength'),
               ('S1','SMT-top1'),('S3','SMT-top3'),('Full',r'\sysfull{}')]
    replace_body('ablation_process.tex', [' & '.join([label,*[
        rank(ablations[k][metric],[ablations[x][metric] for x,_ in aborder])
        for metric in ('percent','psnr')]])+r' \\' for k,label in aborder])

    report = dict(schema='completed_final_additions_v1', complete=True, main_requests=len(keep),
        ablation_requests=len(abkeep), enumeration_geo=enum, enumeration_geo_paired=pairs,
        component_ablations=components, full_ablation_reference=ablations['Full'],
        source_sha256=hashes, verified_evidence=verified,
        main_table_aggregation='unweighted mean of five scenario rates; own-accepted-request PSNR',
        ablation_table_aggregation='equal-request satisfaction; own-accepted-request PSNR',
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    additions.update(pending=[], single_fragment=components['SingleOptimized'], enumeration_geo=enum,
                     final_completion_report='data/final_additions_20260923.json')
    additions['source_sha256'][str(root/'component_ablation_20260923/results.json')] = hashes[str(root/'component_ablation_20260923/results.json')]
    (PAPER / 'data/completed_additions_20260923.json').write_text(json.dumps(additions,indent=2)+'\n')
    hashes[str(PAPER / 'data/completed_additions_20260923.json')] = hashlib.sha256(
        (PAPER / 'data/completed_additions_20260923.json').read_bytes()).hexdigest()
    (PAPER / 'data/final_additions_20260923.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(enumeration_geo=enum,paired=pairs,components=components),indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    export(p.parse_args().run_root)


if __name__ == '__main__': main()
