"""Update completed new controls without publishing unfinished experiments."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

PAPER = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    final_runs = [root / name / 'results.json' for name in
                  ('enumeration_geo_20260923', 'component_ablation_20260923')]
    if all(p.exists() and json.loads(p.read_text())['status']['complete'] for p in final_runs):
        from export_final_additions import export
        export(root)
        return
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    single = root / 'single_extra_20260923'
    audit = read(single / 'audit.json')
    records = read(single / 'results.json')
    assert audit['passed'] and records['status']['complete']
    assert audit['result_sha256'] == hashes[str(single / 'results.json')]
    report = read(single / 'main_table_rows.json')
    amended = read(root / 'live_failure_fallback_20260921/amended_results.json')['rows']
    extension = read(root / 'editing_request_extension_20260921/results.json')['rows']
    ids = {r['spec_id'] for r in records['rows'] if r['method'] == 'MaskWM'}
    assert len(ids) == 7414
    full = {}
    for arm in ('04', '06'):
        rows = [r for r in amended if r['cohort'] == 'main' and r['arm'] == arm]
        rows += [r for r in extension if r['variant'] == 'Full' and r['arm'] == arm]
        full[arm] = {r['spec_id']: r for r in rows if r['spec_id'] in ids}
        assert set(full[arm]) == ids
    pairs = {}
    for method in ('MaskWM', 'StegaStamp'):
        rows = [r for r in records['rows'] if r['method'] == method]
        assert len(rows) == len({r['spec_id'] for r in rows}) == 7414
        assert all(not r['status'].startswith('pending') for r in rows)
        accepted = {r['spec_id']: r for r in rows if r['status'] == 'accepted'}
        assert len(accepted) == report[method]['accepted']
        report[method]['capacity_bits'] = mean(r['verdict']['capacity_bits_estimate'] for r in accepted.values()) if accepted else None
        pairs[method] = {}
        for arm in ('04', '06'):
            joint = sorted(i for i in accepted if full[arm][i]['status'] == 'accepted')
            pairs[method][arm] = dict(n=len(joint), spec_ids=joint,
                baseline_psnr=mean(accepted[i]['verdict']['psnr_live'] for i in joint) if joint else None,
                full_psnr=mean(full[arm][i]['psnr_live'] if 'psnr_live' in full[arm][i]
                    else full[arm][i]['verdict']['psnr_live'] for i in joint) if joint else None)

    component = read(root / 'component_ablation_20260923/results.json')
    unit = [r for r in component['rows'] if r['variant'] == 'UnitStrength']
    assert len(unit) == len({r['spec_id'] for r in unit}) == 1111
    assert all(not r['status'].startswith('pending') for r in unit)
    passed = [r for r in unit if r['status'] == 'accepted']
    assert all(all(abs(float(s)-1) < 1e-9 for s in r['candidate']['s'].values()) for r in passed)
    unit_summary = dict(n=len(unit), accepted=len(passed), complete=True,
        percent=100*len(passed)/len(unit), psnr=mean(r['verdict']['psnr_live'] for r in passed))

    def replace(filename, method, replacements):
        path = PAPER / 'tab' / filename
        lines = path.read_text().splitlines()
        count = 0
        for i, line in enumerate(lines):
            if line.startswith(method + ' &'):
                lines[i] = replacements[count]
                count += 1
        assert count == len(replacements), (filename, method, count)
        path.write_text('\n'.join(lines) + '\n')

    fmt = lambda x: '-' if x is None else f'{x:.2f}'
    for method, row in report.items():
        rates = [row['classes'][f'C{i}']['percent'] for i in range(1,6)]
        gray = lambda x: r'\textcolor{tblgray}{' + x + '}'
        replace('main_results.tex',gray(method), [' & '.join([gray(method),gray('-'),
            *[gray(fmt(x)) for x in rates],r'\cellcolor{orange!8}'+gray(fmt(row['macro_percent'])),gray(fmt(row['psnr']))])+r' \\'])
        replace('eval_current_main.tex',method,[' & '.join([method,'-',fmt(row['overall_percent']),fmt(row['macro_percent']),*[fmt(x) for x in rates]])+r' \\'])
        replace('eval_current_quality.tex',method,[' & '.join([method,f"{row['accepted']:,}",fmt(row['psnr']),fmt(row['capacity_bits'])])+r' \\'])
        pair_rows = []
        for arm in ('04','06'):
            pair = pairs[method][arm]
            b, f = pair['baseline_psnr'], pair['full_psnr']
            def quality(x, other):
                if x is None: return '-'
                return (r'\textbf{' if x >= other else r'\underline{') + fmt(x) + '}'
            pair_rows.append(' & '.join([method,fmt(row['overall_percent']),f"{pair['n']:,}",quality(b,f),quality(f,b)])+r' \\')
        replace('paired_quality.tex',method,pair_rows)
    replace('ablation_process.tex','Unit-strength',[
        f"Unit-strength & {unit_summary['percent']:.2f} & {unit_summary['psnr']:.2f}" + r' \\'])
    (PAPER / 'data/completed_additions_20260923.json').write_text(json.dumps(dict(
        single_baselines=report, paired_quality=pairs, unit_strength=unit_summary,
        pending=['Enumeration + Geo.', 'Single-fragment'], source_sha256=hashes),indent=2)+'\n')
    print(json.dumps(dict(single_baselines=report,paired_quality={k:{a:{x:y for x,y in v.items() if x!='spec_ids'} for a,v in p.items()} for k,p in pairs.items()},unit_strength=unit_summary),indent=2))


if __name__ == '__main__':
    main()
