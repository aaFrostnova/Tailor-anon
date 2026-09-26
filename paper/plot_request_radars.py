"""Scenario and attack-membership radars for all complete main-table pipelines."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea

BASE=Path('/data/tailor/workspace/home_offload_20260919')
METHODS=[
 ('StegaStamp','StegaStamp','#80858e','x',':',False),
 ('MaskWM','MaskWM','#a89466','v','-',False),
 ('VINE','VINE','#56a4b7','o','-',False),
 ('TrustMark','TrustMark','#ae80aa','s','-',False),
 ('VideoSeal','VideoSeal','#879b83','D','-',False),
 ('VINEGeo','VINE + Geo.','#56a4b7','o','--',True),
 ('TrustMarkGeo','TrustMark + Geo.','#ae80aa','s','--',True),
 ('Fixed1','Enumeration','#c39452','p','-',False),
 ('EnumGeo','Enumeration + Geo.','#c39452','p','--',True),
 ('Greedy04','TAILOR-G (0.04)','#74aaa2','^','--',True),
 ('SMT04','TAILOR-F (0.04)','#80a8cf','*','-',False),
 ('Greedy06','TAILOR-G (0.06)','#27847d','^','--',True),
 ('SMT06','TAILOR-F (0.06)','#234f80','*','-',False)]
SCENES=[('S1','S1\nSignal / re-encoding'),('S2','S2\nGeometry'),
 ('S3','S3\nRegeneration\n/ editing'),('S4','S4\nWatermark removal'),('S5','S5\nBroad attack\ncoverage')]
ATTACKS=[('bright','Brightness'),('contrast','Contrast'),('jpeg25','JPEG Q25'),('blur','Gaussian blur'),
 ('noise','Gaussian noise'),('crop75','Crop 75%'),('crop50','Crop 50%'),('rot9','Rotation 9°'),
 ('crop_jpeg','Crop + JPEG'),('rs256','Resize 256'),('hflip','Horizontal flip'),('border20','Border-20'),
 ('vaeB','VAE-B'),('vaeC','VAE-C'),('regen','Regen'),('rinse2x','Rinse-2x'),
 ('ctrlregen_s03','CtrlRegen+ 0.3'),('ctrlregen_s05','CtrlRegen+ 0.5'),
 ('editing_ip2p_s20_v1','Image editing'),('unmarker','UnMarker')]
INK=MUTED='#000000'


def display_methods(margin):
    if margin=='all':
        return METHODS
    return [(m[0],m[1].replace(' (0.06)',''),*m[2:]) for m in METHODS
            if m[0] not in {'Greedy04','SMT04'}]


def aggregate(snapshot):
    inputs=snapshot['inputs'];result=[]
    assert len(inputs)==len({r['spec_id'] for r in inputs})==7321
    assert set(a for r in inputs for a in r['attacks'])==set(a for a,_ in ATTACKS)
    for kind,axes in [('scenario',SCENES),('attack_membership',ATTACKS)]:
        for axis,label in axes:
            ix=[i for i,r in enumerate(inputs) if (r['scenario']==axis if kind=='scenario' else axis in r['attacks'])]
            assert ix
            for method,*_ in METHODS:
                states=np.array(snapshot['states'][method])[ix];assert set(states)<={0,1}
                accepted=int(states.sum())
                result.append(dict(grouping=kind,axis=axis,label=label.replace('\n',' / '),method=method,
                    n=len(ix),accepted=accepted,rate_percent=100*accepted/len(ix)))
    for method,*_ in METHODS:
        cells=[r for r in result if r['grouping']=='scenario' and r['method']==method]
        assert sum(r['n'] for r in cells)==7321
        assert sum(r['accepted'] for r in cells)==snapshot['totals'][method]['accepted']
    return result


def plot(out,data,kind,methods):
    is_attack=kind=='attack_membership';axes=ATTACKS if is_attack else SCENES
    n_axes=len(axes);theta=np.linspace(0,2*np.pi,n_axes,endpoint=False);closed=np.r_[theta,theta[0]]
    # The attack radar shares a paper row with Table 2. Keep its legend below
    # the plot so the manuscript width is spent on the radar and its labels.
    fig=plt.figure(figsize=(6.8,6.3) if is_attack else (10.8,6.4))
    ax=fig.add_axes([.285,.295,.43,.540] if is_attack else [.100,.120,.570,.750],projection='polar')
    ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1)
    ax.set_ylim(0,105);ax.set_yticks([20,40,60,80,100]);ax.set_yticklabels([])
    # A polar axis draws its tick labels under the series whatever zorder they carry, so
    # the radial scale is placed as text on an opaque plate instead. On the attack radar,
    # which has twenty spokes, only every other ring is labeled.
    rings=((20,'20'),(60,'60'),(100,'100%')) if is_attack else (
        (20,'20'),(40,'40'),(60,'60'),(80,'80'),(100,'100%'))
    for value,text in rings:
        ax.text(np.deg2rad(9 if is_attack else 20),value,text,
            fontsize=12.5 if is_attack else 10.2,color=MUTED,ha='center',va='center',
            zorder=20,bbox=dict(boxstyle='round,pad=0.20',facecolor='white',edgecolor='none'))
    ax.yaxis.grid(True,color='#d3dde7',lw=.65);ax.xaxis.grid(True,color='#e2e8ef',lw=.65)
    ax.set_facecolor('#fafbfd');ax.spines['polar'].set_color('#d0dbe6');ax.spines['polar'].set_linewidth(.7)
    ax.set_xticks(theta)
    counts={r['axis']:r['n'] for r in data if r['grouping']==kind}
    labels=[f'{label}\n(n={counts[key]:,})' for key,label in axes]
    ax.set_xticklabels([] if is_attack else labels,fontsize=12.2,color=INK)
    ax.tick_params(axis='x',pad=13 if is_attack else 19)
    if is_attack:
        label_boxes=[]
        for angle,(key,label) in zip(theta,axes):
            side=np.sin(angle)
            align='left' if side>.15 else ('right' if side<-.15 else 'center')
            # Keep the complete attack name and strength together on one line.
            name=TextArea(label,textprops=dict(fontsize=17.5,color=INK,
                ha=align,multialignment=align,linespacing=1.0))
            # Attack membership counts are shown in Figure 3 and retained in
            # the source exports; here the labels identify the response axes.
            y_shift={'bright':10,'hflip':-6,'rs256':-8,'border20':-8}.get(key,0)
            box=AnnotationBbox(name,(angle,113),xybox=(0,y_shift),
                xycoords='data',boxcoords='offset points',frameon=False,pad=0,
                box_alignment=(0 if align=='left' else (1 if align=='right' else .5),.5),
                annotation_clip=False)
            ax.add_artist(box);label_boxes.append(box)
        for label in ax.get_yticklabels():label.set_va('top')
        # Resolve only actual text-box collisions, keeping every label at its
        # own spoke's horizontal position.
        for _ in range(30):
            fig.canvas.draw();renderer=fig.canvas.get_renderer();moved=False
            bounds=[box.get_window_extent(renderer) for box in label_boxes]
            for i in range(len(label_boxes)):
                for j in range(i+1,len(label_boxes)):
                    a,b=bounds[i],bounds[j]
                    if min(a.x1,b.x1)-max(a.x0,b.x0)<=1:continue
                    overlap=min(a.y1,b.y1)-max(a.y0,b.y0)
                    if overlap<1:continue
                    dy=(overlap+5)*72/fig.dpi/2
                    sign=1 if (a.y0+a.y1)>(b.y0+b.y1) else -1
                    for k,direction in ((i,sign),(j,-sign)):
                        x,y=label_boxes[k].xybox;label_boxes[k].xybox=(x,y+direction*dy)
                    moved=True
            if not moved:break
    # Paint the broadest regions first with lighter fills; internal regions remain visible.
    fill_layers=[]
    for method,label,color,marker,style,hollow in methods:
        by={r['axis']:r for r in data if r['grouping']==kind and r['method']==method}
        vals=np.array([by[key]['rate_percent'] for key,_ in axes])
        area=float(np.sum(vals*np.roll(vals,-1)))
        alpha=.022 if method.startswith('SMT') else (.035 if method.startswith('Greedy') else .075)
        fill_layers.append((area,color,alpha,np.r_[vals,vals[0]]))
    for _,color,alpha,values in sorted(fill_layers,key=lambda item:item[0],reverse=True):
        ax.fill(closed,values,color=color,alpha=alpha,linewidth=0,zorder=1)
    handles=[]
    for j,(method,label,color,marker,style,hollow) in enumerate(methods):
        by={r['axis']:r for r in data if r['grouping']==kind and r['method']==method}
        vals=np.array([by[key]['rate_percent'] for key,_ in axes]);values=np.r_[vals,vals[0]]
        full=method.startswith('SMT');width=2.05 if full else (1.55 if method.startswith('Greedy') else 1.1)
        size=7.4 if full else (4.8 if hollow else 4.0)
        kw=dict(color=color,linestyle=style,linewidth=width,marker=marker,markersize=size,
                markerfacecolor='white' if hollow else color,markeredgecolor=color,
                markeredgewidth=.85,alpha=.94 if full else .90)
        # Every axis is an observed rate; connecting lines are a categorical visual guide.
        ax.plot(closed,values,**kw,zorder=5 if full else 3)
        legend_label=label.replace('Enumeration','Enum.') if is_attack else label
        handles.append(Line2D([],[],label=legend_label,**kw))
    # All-zero StegaStamp coincides at the origin; keep its actual location visible.
    ax.scatter([0],[0],s=33,marker='x',c=methods[0][2],linewidths=1.1,zorder=8)
    if is_attack:
        # Matplotlib fills legend columns top to bottom. Reorder only the
        # handles to retain the main-table order when read across each row.
        cols=3
        legend_handles=[handles[i] for col in range(cols) for i in range(col,len(handles),cols)]
        leg=fig.legend(handles=legend_handles,loc='lower center',bbox_to_anchor=(.50,.035),
            ncol=cols,frameon=False,fontsize=17.5,labelspacing=.28,handlelength=1.15,
            handletextpad=.34,columnspacing=.65,borderaxespad=0)
    else:
        leg=fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.735,.50),
            frameon=False,fontsize=11.2,labelspacing=.65,handlelength=2.3,handletextpad=.55)
    leg._legend_box.align='left'
    stem='radar_attack_membership_20' if is_attack else 'radar_scenarios_5'
    for ext in ('pdf','svg','png'):fig.savefig(out/f'{stem}.{ext}',dpi=350,facecolor='white',bbox_inches='tight',pad_inches=.10)
    fig.savefig(out/f'{stem}_preview.png',dpi=150,facecolor='white',bbox_inches='tight',pad_inches=.10);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshot',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--include-attacks',action='store_true')
    ap.add_argument('--only-attacks',action='store_true',help='Skip the scenario radar.')
    ap.add_argument('--margin',choices=['06','all'],default='06',
        help='Displayed TAILOR settings; source aggregation always covers both margins.')
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    snap=json.loads(a.snapshot.read_text());data=aggregate(snap)
    methods=display_methods(a.margin)
    plt.rcParams.update({'font.family':'Nimbus Roman','mathtext.fontset':'stix','font.size':12,'text.color':INK,'axes.labelcolor':INK,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    if not a.only_attacks:plot(out,data,'scenario',methods)
    if a.include_attacks or a.only_attacks:plot(out,data,'attack_membership',methods)
    with (out/'radar_data.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'radar_data.json').write_text(json.dumps(data,indent=2)+'\n')
    (out/'plot_snapshot.json').write_bytes(a.snapshot.read_bytes())
    audit=dict(n_requests=7321,n_methods=13,scenario_axes=5,attack_axes=20,scenario_records=65,
        attack_records=260,all_method_denominators_identical_within_axis=True,
        attack_metric='P(request accepted by all gates | request contains attack)',
        single_attack_detection_rate=False,attack_groups_overlap=True,
        displayed_margin=a.margin,displayed_methods=[m[0] for m in methods],
        baseline_order=[m[0] for m in METHODS],source_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest())
    (out/'radar_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
