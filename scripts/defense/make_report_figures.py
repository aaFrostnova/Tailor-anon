"""Generate all statistical figures for the VideoSeal-upgrade report from the measured
result JSONs. Figures are English-labelled (shared by the CN and EN report versions) ->
results/defense/figures/fig{1..7}_*.png. Single source of truth = the result JSONs."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom
R="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/"
FIG=R+"figures/"; os.makedirs(FIG,exist_ok=True)
def L(f): return json.load(open(R+f))
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150})
C={"vine":"#1f77b4","tm":"#ff7f0e","vs":"#2ca02c","head":"#d62728","eq3":"#7f7f7f","best":"#111111",
   "V+TM":"#7f7f7f","V+VS":"#ff7f0e","V+TM+VS":"#d62728"}

# ---------- FIG 1: 18-attack RAVEN composite detection (3 configs) ----------
D={"V+TM":L("official_vine_tm.json"),"V+VS":L("official_vine_videoseal.json"),"V+TM+VS":L("official_vine_tm_videoseal.json")}
order=["clean","jpeg","blur","noise","bright","contrast","bm3d","vae_b","vae_c","rs256","regen","rinse2x","rinse4x","hflip","crop75","crop_jpeg","rot9","crop50"]
x=np.arange(len(order)); w=0.27
fig,ax=plt.subplots(figsize=(15,5))
for i,(k,d) in enumerate(D.items()):
    vals=[d["attacks"][a]["composite_or"] for a in order]
    ax.bar(x+(i-1)*w,vals,w,label=k,color=C[k],edgecolor="white",linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(order,rotation=45,ha="right")
ax.set_ylabel("Composite detection rate (n=64)"); ax.set_ylim(0,1.05)
ax.axhline(1.0,color="gray",ls=":",lw=0.8)
ax.set_title("RAVEN 18-attack robustness: VINE+TM vs VINE+VideoSeal vs VINE+TM+VideoSeal (3-fragment)")
ax.legend(title="composite",loc="lower left")
for sp in ["rot9","crop_jpeg","crop75"]:
    ax.annotate("geometry",xy=(order.index(sp),0.02),ha="center",fontsize=7,color="purple")
fig.tight_layout(); fig.savefig(FIG+"fig1_raven18.png"); plt.close(fig)

# ---------- FIG 2: per-fragment division of labour (bit-acc heatmap) ----------
d3=D["V+TM+VS"]["attacks"]
M=np.array([[d3[a].get("vine",np.nan),d3[a].get("trustmark",np.nan),d3[a].get("videoseal",np.nan)] for a in order]).T
fig,ax=plt.subplots(figsize=(15,3.2))
im=ax.imshow(M,aspect="auto",cmap="RdYlGn",vmin=0.5,vmax=1.0)
ax.set_yticks(range(3)); ax.set_yticklabels(["VINE\n(latent)","TrustMark\n(pixel)","VideoSeal\n(pixel)"])
ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=45,ha="right")
for i in range(3):
    for j in range(len(order)):
        ax.text(j,i,"%.2f"%M[i,j],ha="center",va="center",fontsize=7,
                color="black" if M[i,j]>0.62 else "white")
ax.set_title("Fragment division of labour: per-fragment bit-accuracy (0.50=dead, 1.0=alive)")
fig.colorbar(im,ax=ax,fraction=0.015,pad=0.01,label="bit-acc")
fig.tight_layout(); fig.savefig(FIG+"fig2_division_of_labour.png"); plt.close(fig)

# ---------- FIG 3: CtrlRegen+ fusion comparison (deployed detector, n=200) ----------
S=[0.3,0.5,0.7]
det={k:[L("ctrlregen200_s0%d.json"%int(s*10))[k] for s in S] for k in ["VINE_only","eq3","HEAD","bestpath_oracle"]}
x=np.arange(3); w=0.2
fig,ax=plt.subplots(figsize=(8,5))
lbl={"VINE_only":("VINE-only",C["vine"]),"eq3":("equal-MRC (3-frag)",C["eq3"]),"HEAD":("learned HEAD (deployed)",C["head"]),"bestpath_oracle":("best-path",C["best"])}
for i,k in enumerate(["eq3","VINE_only","HEAD","bestpath_oracle"]):
    ax.bar(x+(i-1.5)*w,det[k],w,label=lbl[k][0],color=lbl[k][1],edgecolor="white",lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(["s=0.3","s=0.5","s=0.7"]); ax.set_ylim(0,1.05)
ax.set_ylabel("Composite detection rate (n=200)"); ax.set_xlabel("CtrlRegen+ strength")
ax.set_title("CtrlRegen+ : equal-MRC dilution vs learned-head recovery (deployed detector)")
ax.legend(loc="upper right",fontsize=9)
fig.tight_layout(); fig.savefig(FIG+"fig3_ctrlregen_fusion.png"); plt.close(fig)

# ---------- FIG 4: WAVES Performance-Quality curve + Q@P ----------
q=L("sweep_qp.json"); rows=sorted(q["rows"],key=lambda r:r["s"])
Q=[r["Q"] for r in rows]
fig,ax=plt.subplots(figsize=(8,6))
for pk,nm,c in [("P_head","HEAD (deployed)",C["head"]),("P_vine","VINE",C["vine"]),
                ("P_eq3","equal-MRC",C["eq3"]),("P_bestpath","best-path",C["best"])]:
    ax.plot(Q,[r[pk] for r in rows],"-o",color=c,label=nm,ms=4)
for T in [0.95,0.70]:
    ax.axhline(T,color="purple",ls="--",lw=0.8)
    ax.text(0.01,T+0.01,"P=%.2f"%T,color="purple",fontsize=8)
res=q["results"]
for nm,key,c in [("HEAD","HEAD",C["head"])]:
    for T,lab in [("Q@0.95P","Q@0.95P"),("Q@0.70P","Q@0.70P")]:
        qq=res[key][T]; ax.plot([qq],[0.95 if "95" in T else 0.70],"*",color=c,ms=16,zorder=5)
        ax.annotate("%s=%.3f"%(lab,qq),xy=(qq,0.95 if "95" in T else 0.70),xytext=(qq+0.03,(0.95 if "95" in T else 0.70)-0.07),
                    fontsize=9,color=c,arrowprops=dict(arrowstyle="->",color=c,lw=0.8))
ax.set_xlabel("Quality degradation Q  (WAVES quantile-norm of {PSNR,SSIM,LPIPS})")
ax.set_ylabel("Performance P = TPR@0.1%FPR"); ax.set_ylim(0,1.05); ax.set_xlim(0,0.78)
ax.set_title("WAVES Performance-vs-Quality curve under CtrlRegen+ (n=100)\nUnMarker: P=1.0 at all strengths -> Q@P = inf (off-chart)")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(FIG+"fig4_qp_curve.png"); plt.close(fig)

# ---------- FIG 5: CtrlRegen+ strength sweep: P(s) and quality(s) ----------
s=[r["s"] for r in rows]
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
for pk,nm,c in [("P_head","HEAD",C["head"]),("P_vine","VINE",C["vine"]),("P_eq3","equal-MRC",C["eq3"]),("P_bestpath","best-path",C["best"])]:
    a1.plot(s,[r[pk] for r in rows],"-o",color=c,label=nm,ms=4)
a1.axhline(0.95,color="purple",ls="--",lw=0.7); a1.axhline(0.70,color="purple",ls="--",lw=0.7)
a1.set_xlabel("CtrlRegen+ strength s"); a1.set_ylabel("P = TPR@0.1%FPR"); a1.set_ylim(0,1.05)
a1.set_title("Performance vs strength"); a1.legend(fontsize=9)
a2.plot(s,[r["ssim"] for r in rows],"-o",color="#1f77b4",label="SSIM")
a2.plot(s,[r["lpips"] for r in rows],"-s",color="#d62728",label="LPIPS")
a2.plot(s,Q,"-^",color="black",label="Q (norm. degradation)")
a2b=a2.twinx(); a2b.plot(s,[min(r["psnr"],40) for r in rows],"-d",color="#2ca02c",label="PSNR (dB, capped40)")
a2b.set_ylabel("PSNR (dB)",color="#2ca02c")
a2.set_xlabel("CtrlRegen+ strength s"); a2.set_ylabel("SSIM / LPIPS / Q"); a2.set_ylim(0,1.05)
a2.set_title("Image quality vs strength"); a2.legend(loc="center right",fontsize=9)
fig.tight_layout(); fig.savefig(FIG+"fig5_ctrlregen_sweep.png"); plt.close(fig)

# ---------- FIG 6: UnMarker strength sweep (n=100) ----------
us=["200_100","300_150","400_200"]; ulab=["200/100","300/150\n(orig)","400/200\n(bumped)"]
ud=[L("unmarker100_s%s.json"%u) for u in us]
fig,ax=plt.subplots(figsize=(8,5))
xx=np.arange(3)
ax.plot(xx,[d["HEAD"] for d in ud],"-o",color=C["head"],label="composite detection",ms=8,lw=2)
ax.plot(xx,[d["head_fused_ba"] for d in ud],"-s",color=C["vine"],label="fused bit-acc (head)",ms=7)
ax.axhline(0.66,color="purple",ls="--",lw=1,label="0.1%FPR threshold (0.66)")
ax.axhline(0.63,color="gray",ls=":",lw=1,label="zero-bit threshold tau=0.63")
ax.axhline(0.5,color="red",ls=":",lw=0.8,label="chance (0.50)")
ax.set_xticks(xx); ax.set_xticklabels(ulab); ax.set_ylim(0.45,1.05)
ax.set_xlabel("UnMarker strength (stage1/stage2 iterations)"); ax.set_ylabel("rate / bit-acc (n=100)")
ax.set_title("UnMarker pure-spectral: detection stays 1.0 at every strength\n(fused bit-acc 0.98->0.97, huge margin above threshold -> Q@P=inf)")
ax.legend(loc="center left",fontsize=8)
fig.tight_layout(); fig.savefig(FIG+"fig6_unmarker_sweep.png"); plt.close(fig)

# ---------- FIG 7: FPR (two-tier) ----------
f=L("fpr_test.json"); n=100
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
xb=np.arange(0,n+1)
a1.bar(xb/n,binom.pmf(xb,n,0.5),width=1/n,color="#cccccc",label="analytic Binom(100,.5)/100")
mu,sd=f["null_head_mean"],f["null_head_std"]
xx=np.linspace(0.3,0.75,300); a1.plot(xx,binom.pmf(np.round(xx*n).astype(int),n,0.5),color=C["head"],lw=0)
a1.axvline(mu,color=C["head"],lw=2,label="HEAD null mean=%.3f (std=%.3f)"%(mu,sd))
a1.axvline(f["tau"],color="purple",ls="--",lw=1.5,label="tau=0.63 (zero-bit)")
a1.axvline(0.66,color="green",ls="--",lw=1.2,label="0.66 (0.1%FPR)")
a1.set_xlim(0.3,0.75); a1.set_xlabel("fused bit-accuracy (unwatermarked)"); a1.set_ylabel("density")
a1.set_title("Null distribution: learned HEAD = analytic null\n(no distortion -> FPR-safe)"); a1.legend(fontsize=8)
paths=["fver_fpr","bestpath_fpr","zerobit_head_fpr","composite_or_fpr"]
plab=["fused crypto\n(identity)","best-path crypto\n(identity)","zero-bit\n(presence)","composite_or\n(deployed)"]
vals=[f[p] for p in paths]
bars=a2.bar(range(4),vals,color=[C["best"],C["best"],"#ff7f0e",C["head"]],edgecolor="white")
a2.axhline(0.006,color="purple",ls="--",lw=1,label="analytic zero-bit 0.6%")
for i,v in enumerate(vals): a2.text(i,v+0.0002,("%.4f"%v) if v>0 else "0",ha="center",fontsize=9)
a2.set_xticks(range(4)); a2.set_xticklabels(plab,fontsize=8); a2.set_ylim(0,0.008)
a2.set_ylabel("False positive rate"); a2.set_title("Two-tier FPR: identity crypto ~0 (2^-37),\npresence zero-bit at designed 0.6%  (N*K=60k/crypto 6k trials)")
a2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG+"fig7_fpr.png"); plt.close(fig)

print("FIGURES:",sorted(os.listdir(FIG)))
print("MAKE_FIGURES_DONE")
