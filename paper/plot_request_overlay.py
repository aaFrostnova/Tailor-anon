"""Overlay complete pipeline outcomes on the already fitted input-only request map."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

BASE=Path('/data/tailor/workspace/home_offload_20260919')
METHODS=['StegaStamp','MaskWM','VINE','TrustMark','VideoSeal','VINEGeo',
 'TrustMarkGeo','Fixed1','EnumGeo','Greedy04','SMT04','Greedy06','SMT06']
LABELS=['StegaStamp','MaskWM','VINE','TrustMark','VideoSeal','VINE + Geo.',
 'TrustMark + Geo.','Enumeration','Enumeration + Geo.',
 'TAILOR-G (0.04)','TAILOR-F (0.04)','TAILOR-G (0.06)','TAILOR-F (0.06)']
COLORS=['#80858e','#99804e','#5294a1','#9582aa','#7c90a9','#357f90',
 '#715388','#b17c42','#b7925d','#79aaa4','#689ac9','#51969e','#154d82']
SYMBOLS=['cross','diamond','circle','square','diamond','circle-open',
 'square-open','circle','circle-open','diamond','circle','diamond','circle']
INK=MUTED='#000000'


def static_plot(out,z,report):
    xy=z['xy'];rates=z['rates100'];columns=[8,11,12]
    fig=plt.figure(figsize=(8.8,6.5))
    ax=fig.add_axes([.015,.015,.92,.91],projection='3d')
    # No coordinate jitter or vertical offsets: a request has the exact same x/y for every method.
    ax.scatter(xy[:,0],xy[:,1],np.zeros(len(xy)),s=.5,c='#bfcad6',alpha=.15,
               linewidths=0,depthshade=False,rasterized=True)
    markers=['o','D','o'];sizes=[4.8,4.0,2.8];alphas=[.50,.50,.50]
    for col,mark,size,alpha in zip(columns,markers,sizes,alphas):
        ax.scatter(xy[:,0],xy[:,1],rates[:,col],s=size,marker=mark,c=COLORS[col],
                   alpha=alpha,linewidths=0,depthshade=False,rasterized=True)
    ax.set_zlim(0,103);ax.set_zticks([0,25,50,75,100]);ax.set_xticks([]);ax.set_yticks([])
    ax.set_xlim(xy[:,0].min()-4,xy[:,0].max()+4);ax.set_ylim(xy[:,1].min()-4,xy[:,1].max()+4)
    ax.set_xlabel('t-SNE 1',labelpad=-9,fontsize=13,color=MUTED)
    ax.set_ylabel('t-SNE 2',labelpad=-9,fontsize=13,color=MUTED)
    ax.set_zlabel('Local satisfaction (%)',labelpad=9,fontsize=13,color=MUTED)
    ax.tick_params(axis='z',labelsize=12,pad=2,colors=MUTED)
    ax.view_init(elev=23,azim=-57);ax.set_box_aspect((1.04,1,.82));ax.computed_zorder=True
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.set_facecolor((.97,.98,.99,.72));axis.pane.set_edgecolor('#e2e8ee')
        axis.line.set_color('#c3ced9');axis._axinfo['grid']['color']='#e2e7ed'
        axis._axinfo['grid']['linewidth']=.6
    handles=[Line2D([],[],ls='None',marker=mark,markersize=6.5,color=COLORS[col],
        label=LABELS[col])
        for col,mark in zip(columns,markers)]
    legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.49,.98),ncol=3,
        frameon=False,fontsize=11,handlelength=1.1,handletextpad=.4,columnspacing=1.5)
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'request_success_overlay.{ext}',dpi=360,facecolor='white',bbox_inches='tight',pad_inches=.10,bbox_extra_artists=(legend,ax.xaxis.label,ax.yaxis.label,ax.zaxis.label))
    fig.savefig(out/'request_success_overlay_preview.png',dpi=155,facecolor='white',bbox_inches='tight',pad_inches=.10,bbox_extra_artists=(legend,ax.xaxis.label,ax.yaxis.label,ax.zaxis.label))
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    z=np.load(args.source/'atlas_arrays.npz')
    report=json.loads((args.source/'analysis.json').read_text())
    plt.rcParams.update({'font.family':'Nimbus Roman','mathtext.fontset':'stix',
        'font.size':12,'text.color':INK,'axes.labelcolor':INK,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none'})
    static_plot(args.output,z,report)

if __name__=='__main__':main()
