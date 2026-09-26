"""Compact manuscript and appendix plots from audited quality summaries.

Use --output-dir on scratch to keep PNG/SVG exports outside the paper tree.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

METHODS=[
 ('StegaStamp','StegaStamp','#80858e','X',False),
 ('MaskWM','MaskWM','#a89466','v',False),
 ('VINE','VINE','#5294a1','o',False),
 ('TrustMark','TrustMark','#9582aa','s',False),
 ('VideoSeal','VideoSeal','#7c90a9','D',False),
 ('VINEGeo','VINE + Geo.','#5294a1','o',True),
 ('TrustMarkGeo','TrustMark + Geo.','#9582aa','s',True),
 ('Fixed1','Enumeration','#ad8b5d','p',False),
 ('EnumGeo','Enumeration + Geo.','#ad8b5d','p',True),
 ('Greedy04','TAILOR-G (0.04)','#6baed6','^',False),
 ('SMT04','TAILOR-F (0.04)','#6baed6','*',False),
 ('Greedy06','TAILOR-G (0.06)','#084594','^',False),
 ('SMT06','TAILOR-F (0.06)','#084594','*',False),
]
SCENES=[('S1','Signal / re-encoding'),('S2','Geometry'),
        ('S3','Regeneration / editing'),('S4','Watermark removal'),('S5','Broad attack coverage')]
INK=MUTED='#000000'


def display_methods(margin):
    if margin=='all':
        return METHODS
    return [(m[0],m[1].replace(' (0.06)',''),*m[2:]) for m in METHODS
            if m[0] not in {'Greedy04','SMT04'}]


def validate(d):
    assert d['n_requests']>0 and len(d['overall'])==13 and len(d['scenarios'])==65
    for total in d['overall']:
        rs=[r for r in d['scenarios'] if r['method']==total['method']]
        assert {r['scenario'] for r in rs}=={s for s,_ in SCENES}
        assert sum(r['n_requests'] for r in rs)==total['n_requests']==d['n_requests']
        assert sum(r['accepted'] for r in rs)==total['accepted']==total['psnr_n_requests']
        assert abs(total['satisfaction_percent']-100*total['accepted']/d['n_requests'])<1e-10
        if total['accepted']:
            weighted=sum(r['accepted']*r['psnr_mean_db'] for r in rs if r['accepted'])/total['accepted']
            assert abs(total['psnr_mean_db']-weighted)<1e-9
        else:assert total['psnr_mean_db'] is None
        for r in rs:
            assert r['accepted']==r['psnr_n_requests']
            assert abs(r['satisfaction_percent']-100*r['accepted']/r['n_requests'])<1e-10
            assert (r['psnr_mean_db'] is None)==(r['accepted']==0)
    assert {r['method'] for r in d['overall']}=={m[0] for m in METHODS}


def symbol(m,size):
    return dict(marker=m[3],markersize=size*(1.58 if m[3]=='*' or m[4] else 1),
        markerfacecolor='none' if m[4] else m[2],markeredgecolor=m[2],
        markeredgewidth=1.25 if m[4] else .55,color=m[2],linestyle='None',
        zorder=4 if m[4] else 5)


def handles(size,methods):
    return [Line2D([],[],label=m[1],**symbol(m,size)) for m in methods]


def panel(ax,rows,methods,axis_font=12,tick_font=11,marker_size=6.9):
    by={r['method']:r for r in rows};assert len(by)==13
    xs=[r['psnr_mean_db'] for r in rows if r['psnr_mean_db'] is not None]
    lo=2*math.floor((min(xs)-.35)/2);hi=2*math.ceil((max(xs)+.35)/2)
    for m in methods:
        r=by[m[0]]
        if r['psnr_mean_db'] is not None:
            ax.plot(r['psnr_mean_db'],r['satisfaction_percent'],**symbol(m,marker_size))
    ax.set_xlim(lo,hi);ax.set_ylim(-6,108)
    # Fewer ticks leave room for readable type at manuscript width.
    step=2 if hi-lo<=8 else 4
    ax.set_xticks(np.arange(lo,hi+.1,step))
    ax.set_yticks([0,25,50,75,100])
    ax.tick_params(axis='both',length=0,pad=5,labelsize=tick_font,labelcolor=MUTED)
    ax.set_xlabel('PSNR (dB)',fontsize=axis_font,labelpad=6)
    ax.set_ylabel('Request satisfaction (%)',fontsize=axis_font,labelpad=6)
    ax.grid(axis='y',color='#dce2e8',lw=.65,linestyle=(0,(3,3)),zorder=0)
    ax.grid(axis='x',color='#eef1f4',lw=.5,zorder=0);ax.set_axisbelow(True)
    for side in ('top','right'):ax.spines[side].set_visible(False)
    for side in ('left','bottom'):
        ax.spines[side].set_color('#9faab5');ax.spines[side].set_linewidth(.7)
    return [m for m in methods if by[m[0]]['psnr_mean_db'] is None]


def na_row(fig,missing,left,y,size=9.6,spacing=.026,marker_start=None):
    fig.text(left,y,'N/A PSNR (0%):',ha='left',va='center',fontsize=size,color=MUTED)
    start=left+.195 if marker_start is None else marker_start
    for i,m in enumerate(missing):
        fig.add_artist(Line2D([start+i*spacing],[y],transform=fig.transFigure,**symbol(m,size*.50)))


def save(fig,out,stem):
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'{stem}.{ext}',dpi=400,facecolor='white')
    fig.savefig(out/f'{stem}_preview.png',dpi=160,facecolor='white')
    plt.close(fig)


def draw_overall(d,out,methods):
    # A compact three-column legend keeps the half-width panel under 5.3 cm
    # high while retaining roughly 7--8 pt type at manuscript scale.
    extra=max(0,(len(methods)+2)//3-4)*.15
    height=2.50+extra
    width=3.50 if len(methods)>11 else 3.25
    fig=plt.figure(figsize=(width,height))
    ax=fig.add_axes([.195,(1.17+extra)/height,.775,1.27/height])
    ax.set_facecolor('#f2f3f5')
    missing=panel(ax,d['overall'],methods,axis_font=9.4,tick_font=9.0,marker_size=6.0)
    ax.tick_params(axis='both',pad=2)
    ax.xaxis.labelpad=3
    ax.yaxis.label.set_y(.39)
    legend_methods=[(m[0],m[1].replace('Enumeration','Enum.'),*m[2:]) for m in methods]
    lg=fig.legend(handles=handles(4.4,legend_methods),loc='lower center',bbox_to_anchor=(.500,.025/height),
        ncol=3,frameon=False,fontsize=8.5,handletextpad=.26,labelspacing=.20,
        borderaxespad=0,handlelength=1.0,columnspacing=.40)
    lg._legend_box.align='left'
    if missing:
        fig.text(.510,(.720+extra)/height,', '.join(m[1] for m in missing)+': PSNR N/A (0%)',
            ha='center',va='center',fontsize=8.0,color=MUTED)
    save(fig,out,'quality_satisfaction_overall')


def draw_scenarios(d,out,methods):
    # Two columns provide substantially larger lettering than a three-column montage.
    fig=plt.figure(figsize=(6.6,8.35))
    rects=[(.105,.738,.380,.206),(.610,.738,.370,.206),
           (.105,.422,.380,.206),(.610,.422,.370,.206),
           (.105,.106,.380,.206)]
    for rect,(scene,title) in zip(rects,SCENES):
        ax=fig.add_axes(rect)
        part=[r for r in d['scenarios'] if r['scenario']==scene]
        missing=panel(ax,part,methods,axis_font=10.1,tick_font=9.1,marker_size=5.0)
        ax.set_title(scene+': '+title,loc='left',fontsize=10.5,weight='bold',pad=9)
        x,y,w,h=rect
        na_row(fig,missing,x,y-.064,size=8.0,spacing=.023,marker_start=x+.192)
    la=fig.add_axes([.605,.053,.39,.277]);la.axis('off')
    lg=la.legend(handles=handles(4.9,methods),title='Methods',loc='center left',ncol=1,
        frameon=False,fontsize=8.9,title_fontsize=10,labelspacing=.55,
        handlelength=1.7,handletextpad=.6,borderaxespad=0)
    lg._legend_box.align='left'
    save(fig,out,'quality_satisfaction_scenarios')
    for scene,title in SCENES:
        fig=plt.figure(figsize=(7.4,4.1))
        ax=fig.add_axes([.108,.215,.580,.682])
        part=[r for r in d['scenarios'] if r['scenario']==scene]
        missing=panel(ax,part,methods,axis_font=13.0,tick_font=11.7,marker_size=6.9)
        ax.set_title(scene+': '+title,loc='left',fontsize=13.2,weight='bold',pad=10)
        fig.legend(handles=handles(5.8,methods),loc='upper left',bbox_to_anchor=(.710,.968),
            frameon=False,fontsize=10.6,handletextpad=.42,labelspacing=.43,borderaxespad=0,
            handlelength=1.45)
        na_row(fig,missing,.108,.041,size=10.0,spacing=.035,marker_start=.332)
        save(fig,out,scene.lower()+'_quality_satisfaction')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/quality_satisfaction_20260924.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--margin',choices=['06','all'],default='06',
        help='Displayed TAILOR settings; source validation always covers both margins.')
    p.add_argument('--only-overall',action='store_true',help='Skip appendix and scenario exports.')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    d=json.loads(a.data.read_text());validate(d)
    plt.rcParams.update({'font.family':'Nimbus Roman','mathtext.fontset':'stix',
        'font.size':11,'text.color':INK,'axes.labelcolor':INK,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none'})
    methods=display_methods(a.margin)
    draw_overall(d,a.output_dir,methods)
    if not a.only_overall:draw_scenarios(d,a.output_dir,methods)
    (a.output_dir/'plot_validation.json').write_text(json.dumps(dict(
        data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        overall_records=13,scenario_records=65,overall_request_count=d['n_requests'],
        displayed_margin=a.margin,displayed_methods=[m[0] for m in methods],
        pooled_satisfaction=True,psnr_own_accepted_requests=True,
        no_numeric_jitter=True,weighted_counts_and_psnr_verified=True),indent=2)+'\n')
    print('Saved overall figure.' if a.only_overall else
          'Saved overall, appendix, and individual scenario figures.')


if __name__=='__main__':main()
