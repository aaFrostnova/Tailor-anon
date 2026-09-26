"""Filter the frozen 7,414-request cohort; do not restore earlier exclusions.

Drop every request containing ctrlregen_s07. No measurement or decision changes.
The separately frozen 1,111-request ablation cohort is unchanged.
"""
import argparse, collections, hashlib, importlib.util, json, math
from pathlib import Path
from statistics import mean
PAPER=Path(__file__).resolve().parents[1]
ORDER=['StegaStamp','MaskWM','VINE','TrustMark','VideoSeal','VINEGeo','TrustMarkGeo','Fixed1','EnumGeo','Greedy04','SMT04','Greedy06','SMT06']
LABELS=dict(zip(ORDER,['StegaStamp','MaskWM','VINE','TrustMark','VideoSeal','VINE + Geo.','TrustMark + Geo.','Enumeration','Enumeration + Geo.',r'\sysgreedy{}',r'\sysfull{}',r'\sysgreedy{}',r'\sysfull{}']))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();u=a.run_root;out=a.output;out.mkdir(parents=True,exist_ok=True)
    hashes={}
    def read(p):
        raw=p.read_bytes();hashes[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    ref=read(PAPER/'data/s4_reduced_evaluation_20260922.json');old=set(ref['retained_spec_ids'])
    specs=read(PAPER/'data/request_specifications_6532.json')['rows']+read(PAPER/'data/editing_requests_1000_20260922.json')
    kept=sorted((r for r in specs if r['spec_id'] in old and 'ctrlregen_s07' not in r['inputs']['attacks']),key=lambda r:r['spec_id']);ids={r['spec_id'] for r in kept}
    assert len(old)==7414 and len(ids)==len(kept)==7321 and len(old-ids)==93
    base=read(u/'quality_paired_20260921/requests.json')['methods'];methods={k:list(v.values()) for k,v in base.items()}
    methods['Fixed1']=read(u/'fixed1_20260918/results.json')['rows'];geo=read(u/'single_geo_20260921/results.json')['rows']
    for f in ['VINE','TrustMark']:methods[f+'Geo']=[r for r in geo if r['fragment']==f]
    amended=read(u/'live_failure_fallback_20260921/amended_results.json')['rows'];editing=read(u/'editing_request_extension_20260921/results.json')['rows']
    mapping={'SMT04':('Full','04'),'SMT06':('Full','06'),'Greedy04':('G_shared','04'),'Greedy06':('G_shared','06'),'Fixed1':('Fixed1','na'),'VINE':('Single_VINE','na'),'TrustMark':('Single_TrustMark','na'),'VideoSeal':('Single_VideoSeal','na'),'VINEGeo':('SingleGeo_VINE','na'),'TrustMarkGeo':('SingleGeo_TrustMark','na')}
    for arm in ['04','06']:methods['SMT'+arm]=[r for r in amended if r['cohort']=='main' and r['arm']==arm]
    methods={k:methods[k]+[r for r in editing if (r['variant'],r['arm'])==v] for k,v in mapping.items()}
    extra=read(u/'single_extra_20260923/results.json');ea=read(u/'single_extra_20260923/audit.json');assert ea['passed'] and ea['result_sha256']==hashes[str(u/'single_extra_20260923/results.json')]
    for k in ['MaskWM','StegaStamp']:methods[k]=[r for r in extra['rows'] if r['method']==k]
    en=read(u/'enumeration_geo_20260923/results.json');audit=read(u/'enumeration_geo_20260923/audit.json');assert audit['passed'] and audit['results_sha256']==hashes[str(u/'enumeration_geo_20260923/results.json')];methods['EnumGeo']=en['rows']
    for p,h in hashes.items():
        if p in ref['source_sha256']:assert h==ref['source_sha256'][p],p
    failure={'live_failed','live_exhausted','model_unsat','unsat','greedy_stalled','fixed_live_exhausted','fallback_live_exhausted','zero_margin_model_unsat'}
    def psnr(r):
        v=r.get('psnr_live');v=r['verdict']['psnr_live'] if v is None else v;assert math.isfinite(v);return v
    def summary(rows):
        assert all(r['status']=='accepted' or r['status'] in failure for r in rows)
        ok=[r for r in rows if r['status']=='accepted'];scenes={}
        for i in range(1,6):
            part=[r for r in rows if r['cls']=='C'+str(i)];ac=[r for r in part if r['status']=='accepted'];scenes['S'+str(i)]=dict(n=len(part),accepted=len(ac),percent=100*len(ac)/len(part),psnr=mean(map(psnr,ac)) if ac else None)
        return dict(n=len(rows),accepted=len(ok),percent=100*len(ok)/len(rows),psnr=mean(map(psnr,ok)) if ok else None,macro_percent=mean(s['percent'] for s in scenes.values()),scenarios=scenes)
    previous={};index={};summaries={}
    for k in ORDER:
        rows=[r for r in methods[k] if r['spec_id'] in old];assert len(rows)==len({r['spec_id'] for r in rows})==7414
        previous[k]=summary(rows);index[k]={r['spec_id']:r for r in rows if r['spec_id'] in ids};assert set(index[k])==ids;summaries[k]=summary(list(index[k].values()))
        if k in ref['reduced']:assert previous[k]['accepted']==ref['reduced'][k]['accepted'] and abs(previous[k]['psnr']-ref['reduced'][k]['psnr'])<1e-8
    assert all(previous[k]['accepted']==summaries[k]['accepted'] for k in ORDER)
    pairs=[]
    for arm in ['04','06']:
        for k in ['Greedy'+arm,'Fixed1','EnumGeo','VINE','TrustMark','VideoSeal','MaskWM','StegaStamp','VINEGeo','TrustMarkGeo']:
            joint=sorted(i for i in ids if index['SMT'+arm][i]['status']==index[k][i]['status']=='accepted');b=mean(psnr(index[k][i]) for i in joint) if joint else None;f=mean(psnr(index['SMT'+arm][i]) for i in joint) if joint else None
            pairs.append(dict(arm=arm,baseline=k,n=len(joint),psnr_baseline=b,psnr_ours=f,delta_psnr_mean=f-b if joint else None))
    inputs=[dict(spec_id=r['spec_id'],scenario=r['cls'].replace('C','S'),**r['inputs'],attack_count=len(r['inputs']['attacks'])) for r in kept]
    assert len({(tuple(sorted(r['attacks'])),r['fpr'],r['min_psnr'],r['max_ms']) for r in inputs})==7321
    states={k:[int(index[k][r['spec_id']]['status']=='accepted') for r in kept] for k in ORDER}
    snapshot=dict(n=len(kept),inputs=inputs,states=states,totals={k:dict(n=7321,accepted=sum(states[k]),pending=0) for k in ORDER},source_sha256=hashes,notes=['Frozen 7,414 cohort minus 93 CtrlRegen+ 0.7 requests; no earlier exclusions restored.','Main pipelines only; identical request IDs for every method.'])
    attacks=collections.Counter(a for r in inputs for a in r['attacks']);assert len(attacks)==20 and attacks['ctrlregen_s03']==548 and attacks['ctrlregen_s05']==274
    distribution={s:dict(n=sum(r['scenario']==s for r in inputs),attack_sets=len({tuple(sorted(r['attacks'])) for r in inputs if r['scenario']==s}),fpr_counts=dict(collections.Counter(str(r['fpr']) for r in inputs if r['scenario']==s))) for s in ['S1','S2','S3','S4','S5']}
    report=dict(schema='main_no_ctrlregen07_v1',n_requests=7321,previous_requests=7414,removed_requests=93,restored_requests=0,retained_spec_ids=sorted(ids),removed_spec_ids=sorted(old-ids),original_exclusions_preserved=118,selection='Remove every ctrlregen_s07 request from the frozen 7,414 cohort; do not restore earlier exclusions.',interpretation='Post-hoc reporting-cohort change, not an algorithm or measurement improvement.',summary=summaries,previous=previous,paired=pairs,distribution=distribution,attack_counts=dict(attacks),ablation_requests=1111,ablation_cohort_changed=False,source_sha256=hashes)
    def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
    dump(PAPER/'data/main_no_ctrlregen07_20260924.json',report);dump(out/'plot_snapshot.json',snapshot)
    quality=dict(n_requests=7321,overall=[],scenarios=[],source_sha256=hashes)
    for k in ORDER:
        for scene,v in [(None,summaries[k]),*summaries[k]['scenarios'].items()]:
            row=dict(method=k,n_requests=v['n'],accepted=v['accepted'],psnr_n_requests=v['accepted'],psnr_mean_db=v['psnr'],satisfaction_percent=v['percent'])
            if scene:row['scenario']=scene
            quality['scenarios' if scene else 'overall'].append(row)
    dump(PAPER/'data/quality_satisfaction_20260924.json',quality)
    def body(file,lines):
        p=PAPER/'tab'/file;s=p.read_text();start=s.index('\\midrule\n')+len('\\midrule\n');end=s.index('\\bottomrule',start);p.write_text(s[:start]+'\n'.join(lines)+'\n'+s[end:])
    def rank(v,vs):
        if v is None:return '-'
        vals=sorted({round(x,9) for x in vs if x is not None},reverse=True);t=f'{v:.2f}'
        return (r'\textbf{'+t+'}') if abs(v-vals[0])<1e-8 else ((r'\underline{'+t+'}') if len(vals)>1 and abs(v-vals[1])<1e-8 else t)
    arm=lambda k:'0.'+k[-2:] if k.startswith(('SMT','Greedy')) else '-'
    metrics={k:[*[v['scenarios'][f'S{i}']['percent'] for i in range(1,6)],v['macro_percent'],v['psnr']] for k,v in summaries.items()};lines=[]
    for k in ORDER:
        if k in ['VINEGeo','Greedy04','Greedy06']:lines.append(r'\midrule')
        vals=[rank(v,[metrics[x][j] for x in ORDER]) for j,v in enumerate(metrics[k])];vals[5]=r'\cellcolor{orange!8}'+vals[5];lines.append(' & '.join([LABELS[k],arm(k),*vals])+r' \\')
    body('main_results.tex',lines)
    body('eval_current_main.tex',[' & '.join([LABELS[k],arm(k),f"{summaries[k]['percent']:.2f}",*[f'{v:.2f}' for v in [metrics[k][5],*metrics[k][:5]]]])+r' \\' for k in ORDER])
    lines=[]
    for ar in ['04','06']:
        if ar=='06':lines.append(r'\midrule')
        lines.append(r'\multicolumn{5}{l}{\sysfull{}: $m_0=0.'+ar+r'$} \\')
        for p in [p for p in pairs if p['arm']==ar]:
            k=p['baseline'];b,f=p['psnr_baseline'],p['psnr_ours'];lines.append(' & '.join([LABELS[k],f"{p['n']:,}",rank(b,[b,f]),rank(f,[b,f]),f"{p['delta_psnr_mean']:+.2f}" if p['n'] else '-'])+r' \\')
    body('paired_quality.tex',lines)
    scene_labels=['Signal / re-encoding','Geometry','Regeneration / editing','Watermark removal','Broad attack coverage']
    # One table now carries the counts, the attack sets and the FPR split, so one row
    # per scenario holds all eleven fields.
    fprs=[1e-1,1e-2,1e-4,1e-6,1e-9,2**-37];counts=[[sum(r['scenario']==s and r['fpr']==f for r in inputs) for f in fprs] for s in distribution];assert sum(map(sum,counts))==7321
    nsets=len({tuple(sorted(r['attacks'])) for r in inputs})
    lines=[' & '.join([s,scene_labels[i],f"{v['n']:,}",f"{100*v['n']/7321:.2f}",str(v['attack_sets']),*[f'{n:,}' for n in counts[i]]])+r' \\' for i,(s,v) in enumerate(distribution.items())]
    lines.append(r'\midrule');lines.append(' & '.join(['Total','','7,321','100.00',str(nsets),*[f'{sum(c[j] for c in counts):,}' for j in range(6)]])+r' \\')
    body('request_overview.tex',lines)
    path=PAPER/'tab/request_overview.tex';text=path.read_text();text=text.replace('7,414','7,321').replace('14.84','15.03').replace('274 at 0.5, and 93 at 0.7; 290 distinct requests additionally include UnMarker.',f"274 at 0.5; {attacks['unmarker']} distinct requests additionally include UnMarker.");path.write_text(text)
    dump(out/'report.json',report)
    print(json.dumps({k:{x:v[x] for x in ['accepted','percent','macro_percent','psnr']} for k,v in summaries.items()},indent=2));print('DISTRIBUTION',json.dumps(distribution));print('PAIRED',json.dumps(pairs))
if __name__=='__main__':main()
