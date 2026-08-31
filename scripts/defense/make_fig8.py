"""fig8: multi-attack WAVES Performance-Quality curves (deployed 3-frag+head)."""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/"
d=json.load(open(R+"sweep_qp_multi.json"))
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150})
STY={"jpeg":("#1f77b4","o"),"blur":("#17becf","s"),"noise":("#9467bd","^"),"bright":("#8c564b","v"),
     "contrast":("#7f7f7f","D"),"rotate":("#2ca02c","P"),"crop_zoom":("#ff7f0e","X"),"ctrlregen":("#d62728","*")}
fig,ax=plt.subplots(figsize=(9,6.5))
for a in ["jpeg","blur","noise","bright","contrast","rotate","crop_zoom","ctrlregen"]:
    v=d["per_attack"][a]; cur=sorted(v["curve"],key=lambda c:c["Q"])
    Q=[0.0]+[c["Q"] for c in cur]; P=[1.0]+[c["P"] for c in cur]   # anchor clean at (0,1)
    c,m=STY[a]; q95=v["Q@0.95P"]
    lab="%s (Q@0.95P=%s)"%(a,("inf" if isinstance(q95,str) else "%.2f"%q95))
    ax.plot(Q,P,"-",marker=m,color=c,label=lab,ms=7 if a=="ctrlregen" else 5,lw=2 if a in("ctrlregen","crop_zoom","rotate") else 1.2,
            alpha=1.0 if a in("ctrlregen","crop_zoom","rotate") else 0.6)
for T in [0.95,0.70]:
    ax.axhline(T,color="purple",ls="--",lw=0.8); ax.text(0.005,T+0.01,"P=%.2f"%T,color="purple",fontsize=8)
ax.set_xlabel("Quality degradation Q  (WAVES quantile-norm of {PSNR,SSIM,LPIPS}, pooled over all attacks)")
ax.set_ylabel("Performance P = TPR@0.1%FPR"); ax.set_ylim(0,1.05); ax.set_xlim(0,0.95)
ax.set_title("WAVES Performance-vs-Quality across attacks (deployed 3-frag+head, n=100)\n"
             "signal-processing -> P=1.0 at all Q (Q@P=inf); only regen/zoom/rotation break it")
ax.legend(loc="lower left",fontsize=8,ncol=2)
ax.annotate("regen = most efficient\n(lowest Q@0.95P=0.37)",xy=(0.37,0.95),xytext=(0.45,0.80),
            fontsize=9,color="#d62728",arrowprops=dict(arrowstyle="->",color="#d62728"))
ax.annotate("rotation non-monotone:\nrot90 recovered by VideoSeal",xy=(0.91,0.99),xytext=(0.50,0.55),
            fontsize=8,color="#2ca02c",arrowprops=dict(arrowstyle="->",color="#2ca02c"))
fig.tight_layout(); fig.savefig(R+"figures/fig8_qp_multi.png"); plt.close(fig)
print("fig8 saved")
