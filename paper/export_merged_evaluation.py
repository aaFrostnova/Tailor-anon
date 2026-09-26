"""Merge completed editing requests and amended fallback into the paper tables.

Usage: python -B scripts/export_merged_evaluation.py --report .../all_results.json
       --run-root .../unified_ba_20260914 --output /data/tailor/staged
Writes staged tables and compact provenance only; never changes measurements.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from export_paired_quality import write_main_table, write_paired_table

PAPER=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    hashes={}
    def read(p):
        raw=p.read_bytes();hashes[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    report=read(args.report)
    assert report['complete'] and report['combined_requests']==7532
    for p,digest in report['source_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,('changed source',p)
    u=args.run_root;e=u/'editing_request_extension_20260921';a=u/'live_failure_fallback_20260921'
    for root in [e,a]:
        audit=read(root/'pipeline_audit.json');read(root/'sources.json')
        assert audit['complete'] and audit['pending_cpu']==audit['pending_gpu_configs']==0
        assert audit['source_binding']==hashes[str(root/'sources.json')]
    merged=report['combined_summary']
    methods=['VINE','TrustMark','VideoSeal','VINEGeo','TrustMarkGeo','Fixed1','Greedy04','SMT04','Greedy06','SMT06']
    d=dict(accepted={k:v['accepted'] for k,v in merged.items()},
           live_percent={k:v['percent'] for k,v in merged.items()},
           scenario_results={k:dict(scenarios={'S'+c[1:]:r for c,r in v['scenes'].items()},average_percent=v['macro_percent']) for k,v in merged.items()},
           own_success_quality={k:dict(n=v['accepted'],psnr_mean=v['psnr']) for k,v in merged.items()})
    write_main_table(out,d)
    paired=[]
    labelkeys={report['tables']['main_7532']['rows'][i][0]:k for i,k in enumerate(methods)}
    for row in report['tables']['paired_main']['rows']:
        scope,arm,label,n,ours,other,delta,med=row
        if scope!='merged7532':continue
        paired.append(dict(arm=arm,baseline=labelkeys[label],n=n,psnr_ours=ours,psnr_baseline=other,
                           delta_psnr_mean=delta,delta_psnr_median=med))
    d['pairs']=paired;write_paired_table(out,d)

    base=read(u/'shared_table_20260916/priority_20260917/results.json')['rows']
    amended=read(a/'amended_results.json')['rows'];editing=read(e/'results.json')['rows']
    oldmanifest=read(PAPER/'data/ablation_unique_reporting.json')['rows']
    oldids={r['spec_id'] for r in oldmanifest};newids=set(read(e/'manifest.json')['priority_ablation_specs'])
    assert len(oldids)==924 and len(newids)==200 and not oldids&newids
    index={(r['variant'],r['arm'],r['spec_id']):r for r in base}
    for r in amended:
        if r['cohort']=='ablation':index[r['variant'],r['arm'],r['spec_id']]=r
    for r in editing:
        if r['spec_id'] in newids:index[r['variant'],r['arm'],r['spec_id']]=r
    def psnr(r):return r.get('psnr_live') if r.get('psnr_live') is not None else r['verdict']['psnr_live']
    variants=['G_shared','S1','S3','S3_R','Full','Grid','NoFrontend','NoRecoveryInteraction']
    ablations={};pairs={}
    for arm in ['04','06']:
        ablations[arm]={};success={}
        for v in variants:
            group=[index[v,arm,i] for i in sorted(oldids|newids)]
            assert len(group)==1124 and not any(r['status'].startswith('pending') for r in group)
            passed={r['spec_id']:r for r in group if r['status']=='accepted'};success[v]=passed
            ablations[arm][v]=dict(n=1124,accepted=len(passed),percent=100*len(passed)/1124,
                                   psnr_db=mean(psnr(r) for r in passed.values()))
        pairs[arm]={}
        for v in ['G_shared','Grid','NoFrontend','NoRecoveryInteraction']:
            ids=sorted(success['Full'].keys()&success[v].keys())
            ours=[psnr(success['Full'][i]) for i in ids];other=[psnr(success[v][i]) for i in ids]
            delta=[x-y for x,y in zip(ours,other)]
            pairs[arm][v]=dict(n=len(ids),full_psnr_db=mean(ours),variant_psnr_db=mean(other),
                mean_delta_psnr_db=mean(delta),median_delta_psnr_db=median(delta),spec_ids=ids)
    def rank(value,values):
        distinct=[]
        for v in sorted(values,reverse=True):
            if not distinct or abs(v-distinct[-1])>1e-10:distinct.append(v)
        s=f'{value:.2f}'
        return r'\textbf{'+s+'}' if abs(value-distinct[0])<1e-10 else (r'\underline{'+s+'}' if len(distinct)>1 and abs(value-distinct[1])<1e-10 else s)
    def replace_rows(filename,rows):
        source=(PAPER/'tab'/filename).read_text();start=source.index('\\midrule\n')+len('\\midrule\n');end=source.index('\\bottomrule',start)
        (out/filename).write_text(source[:start]+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n'+source[end:])
    shown=[('S1','SMT-top1'),('S3','SMT-top3'),('Full',r'\sysfull{}')]
    replace_rows('ablation_process.tex',[[label,*[rank(ablations['04'][v][metric],[ablations['04'][other][metric] for other,_ in shown]) for metric in ['percent','psnr_db']]] for v,label in shown])
    grid=pairs['04']['Grid'];values=[grid['full_psnr_db'],grid['variant_psnr_db']]
    replace_rows('ablation_model.tex',[[label,rank(value,values)] for label,value in zip([r'\sysfull{}','Grid strengths'],values)])

    labels={'VINE':'VINE only','TrustMark':'TrustMark only','VideoSeal':'VideoSeal only','VINEGeo':'VINE + Geo.',
            'TrustMarkGeo':'TrustMark + Geo.','Fixed1':'Enumeration','Greedy04':r'\sysgreedy{}','Greedy06':r'\sysgreedy{}',
            'SMT04':r'\sysfull{}','SMT06':r'\sysfull{}'}
    rows=[]
    for k in methods:
        v=merged[k];arm='0.'+k[-2:] if k.startswith(('Greedy','SMT')) else '-'
        rows.append([labels[k],arm,f"{v['percent']:.2f}",f"{v['macro_percent']:.2f}",*[f"{v['scenes'][f'C{i}']['percent']:.2f}" for i in range(1,6)]])
    replace_rows('eval_current_main.tex',rows)
    p=out/'eval_current_main.tex';s=p.read_text().replace('6,532','7,532').replace('data/main_live_completed_20260918.json','data/merged_evaluation_20260922.json');p.write_text(s)
    controls=read(PAPER/'data/evaluation_deduplicated_20260917.json')['fixed_control_live']
    capacity={}
    for k in methods[:3]:
        original=controls[k]['own_success_means']
        assert original['n']==report['original_main_summary'][k]['accepted']
        added=[r for r in editing if r['variant']=='Single_'+k and r['status']=='accepted']
        assert original['n']+len(added)==merged[k]['accepted']
        capacity[k]=(original['n']*original['capacity_bits']+sum(r['verdict']['capacity_bits_estimate'] for r in added))/merged[k]['accepted']
    lines=[r'\begin{table}[!htbp]',r'\centering',r'\small',r"\caption{Descriptive PSNR and BSC capacity estimates on each fixed control's live-accepted requests.}",
           r'\label{tab:main-quality}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Method & $n$ & PSNR (dB) & BSC estimate (bits) \\',r'\midrule']
    lines += [' & '.join([labels[k],f"{merged[k]['accepted']:,}",f"{merged[k]['psnr']:.2f}",f"{capacity[k]:.2f}"])+r' \\' for k in methods[:3]]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(out/'eval_current_quality.tex').write_text('\n'.join(lines)+'\n')

    oldspecs=read(PAPER/'data/request_specifications_6532.json')['rows'];newspecs=read(e/'inputs/requests.json')
    specs=[dict(spec_id=r['spec_id'],cls=r['cls'],inputs=r['inputs'],cohort='original') for r in oldspecs]
    specs += [dict(spec_id=r['spec_id'],cls=r['cls'],inputs=r['request'],cohort='editing',source_spec_id=r['source_spec_id']) for r in newspecs]
    def key(r):
        x=r['inputs'];return tuple(sorted(x['attacks'])),float(x['fpr']),float(x['min_psnr']),float(x['max_ms'])
    assert len(specs)==len({key(r) for r in specs})==len({r['spec_id'] for r in specs})==7532
    scenes={};fprs=[.1,.01,.0001,.000001,.000000001,2**-37]
    scene_labels=['Signal / re-encoding','Geometry','Regeneration / editing','Watermark removal','Broad attack coverage']
    for i in range(1,6):
        part=[r for r in specs if r['cls']==f'C{i}'];scenes[f'C{i}']=dict(n=len(part),share=100*len(part)/7532,
            attack_sets=len({key(r)[0] for r in part}),fpr_counts=[sum(r['inputs']['fpr']==v for r in part) for v in fprs],
            editing=sum('editing_ip2p_s20_v1' in r['inputs']['attacks'] for r in part))
    nsets=len({key(r)[0] for r in specs});assert nsets==256
    replace_rows('request_overview.tex',[[f'S{i}',scene_labels[i-1],f"{scenes[f'C{i}']['n']:,}",f"{scenes[f'C{i}']['share']:.2f}",str(scenes[f'C{i}']['attack_sets'])] for i in range(1,6)]+[['Total','', '7,532','100.00',str(nsets)]])
    text=(PAPER/'tab/request_distribution.tex').read_text().replace('6,532','7,532')
    start=text.index('\\midrule',text.index('\\textbf{Scenario / FPR}'))+len('\\midrule\n');end=text.index('\\bottomrule',start)
    counts=[scenes[f'C{i}']['fpr_counts'] for i in range(1,6)];totals=[sum(row[j] for row in counts) for j in range(6)]
    rows=[[f'S{i}',*[f'{n:,}' for n in counts[i-1]]] for i in range(1,6)]+[[r'\textbf{Total}',*[f'{n:,}' for n in totals]]]
    text=text[:start]+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n'+text[end:]
    start=text.index('\n',text.index(r'\begin{minipage}{\linewidth}\footnotesize'))+1;end=text.index(' S4 selects',start)
    text=text[:start]+f"Editing appears in 600 S3 requests ({100*600/scenes['C3']['n']:.2f}\\% of S3) and 500 S5 requests ({100*500/scenes['C5']['n']:.2f}\\% of S5), totaling 1,100 requests ({100*1100/7532:.2f}\\% overall)."+text[end:]
    (out/'request_distribution.tex').write_text(text)
    attacks=collections.Counter(a for r in specs for a in set(r['inputs']['attacks']))
    dist=dict(n=7532,attack_settings=len(attacks),attack_sets=nsets,scenes=scenes,fpr_values=fprs,fpr_counts=totals,attack_coverage=dict(attacks))
    abmanifest=[dict(spec_id=i,cls=index['Full','04',i]['cls'],cohort='original' if i in oldids else 'editing') for i in sorted(oldids|newids)]
    combined=dict(schema='merged_live_evaluation_20260922',complete=True,main_requests=7532,ablation_requests=1124,
        main=d,summary=merged,ablations=ablations,ablation_paired_quality=pairs,distribution=dist,fixed_control_bsc_estimates=capacity,
        ablation_scenarios=dict(collections.Counter(r['cls'] for r in abmanifest)),
        conditioning=dict(main_psnr='own accepted requests',search_psnr='own accepted requests',grid_psnr='joint Full/Grid accepted requests'),
        fallback_policy='completed_live_failure_margin_backoff_v2',source_sha256=hashes,upstream_source_sha256=report['source_sha256'],
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'merged_evaluation_20260922.json').write_text(json.dumps(combined,indent=2)+'\n')
    (out/'editing_requests_1000_20260922.json').write_text(json.dumps([r for r in specs if r['cohort']=='editing'],indent=2)+'\n')
    (out/'ablation_merged_requests_1124.json').write_text(json.dumps(abmanifest,indent=2)+'\n')
    print('MERGED',json.dumps(dict(ablations=ablations,grid={arm:{k:v for k,v in pairs[arm]['Grid'].items() if k!='spec_ids'} for arm in pairs},distribution=dist,ablation_scenarios=combined['ablation_scenarios'])))
    for p in paired:
        if p['baseline'].startswith('Greedy'):print('PAIRED_GREEDY',p)

if __name__=='__main__':main()
