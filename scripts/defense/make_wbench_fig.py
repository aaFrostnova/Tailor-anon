"""W-Bench results figure (scaled up, official dataset) for the deployed 3-frag+head composite.
3 panels: per-axis detection summary | regeneration strength curve | local-editing mask-size curve."""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/"
def L(f): return json.load(open(R+f))
plt.rcParams.update({"font.size":10.5,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150})
dist=L("wbench_distortion_n1000.json")["rows"]; sw=L("wbench_sto_sweep_n500.json")["sweep"]
ins=L("wbench_instruct.json"); svd=L("wbench_svd.json")
loc={"10-20":L("wbench_local_10-20.json"),"30-40":L("wbench_local.json"),"50-60":L("wbench_local_50-60.json")}

fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(18,5.3))
# Panel A: per-axis detection summary
bars=[("clean",dist["clean"]["det"]),("jpeg10",dist["jpeg10"]["det"]),("noise",dist["noise0.05"]["det"]),
      ("crop75",dist["crop75"]["det"]),("rot9",dist["rot9"]["det"]),
      ("regen\ns=0.3",sw["0.3"]["det"]),("regen\ns=0.5",sw["0.5"]["det"]),
      ("global\nedit",ins["det"]),("local\n30-40%",loc["30-40"]["det"]),("img2video\nSVD",svd["frame_det"])]
labs=[b[0] for b in bars]; vals=[b[1] for b in bars]
cols=["#7f7f7f"]*5+["#d62728","#d62728","#9467bd","#2ca02c","#ff7f0e"]
a1.bar(range(len(bars)),vals,color=cols,edgecolor="white")
for i,v in enumerate(vals): a1.text(i,v+0.01,"%.2f"%v,ha="center",fontsize=8.5)
a1.set_xticks(range(len(bars))); a1.set_xticklabels(labs,fontsize=8); a1.set_ylim(0,1.08)
a1.set_ylabel("Composite detection rate"); a1.axhline(1.0,color="gray",ls=":",lw=0.7)
a1.set_title("W-Bench OFFICIAL: detection across 6 axes\n(distortion n=1000, regen n=500, global n=450, local n=500, SVD n=100)")
# Panel B: regen strength curve
s=[float(k) for k in sw]; det=[sw[k]["det"] for k in sw]; ssim=[sw[k]["ssim"] for k in sw]
a2.plot(s,det,"-o",color="#d62728",ms=7,lw=2,label="detection")
a2.set_xlabel("stochastic-regeneration strength"); a2.set_ylabel("detection rate",color="#d62728"); a2.set_ylim(0,1.05)
a2.axhline(0.5,color="gray",ls=":",lw=0.7)
a2b=a2.twinx(); a2b.plot(s,ssim,"-s",color="#1f77b4",ms=6); a2b.set_ylabel("SSIM",color="#1f77b4"); a2b.set_ylim(0.35,0.75)
for x,y in zip(s,det): a2.text(x,y+0.03,"%.2f"%y,ha="center",fontsize=8,color="#d62728")
a2.set_title("Regeneration: detection vs strength (n=500)\ngraceful degradation, VINE-carried")
# Panel C: local mask-size curve
ms=[15,35,55]; ld=[loc["10-20"]["det"],loc["30-40"]["det"],loc["50-60"]["det"]]; lss=[loc["10-20"]["ssim"],loc["30-40"]["ssim"],loc["50-60"]["ssim"]]
a3.plot(ms,ld,"-o",color="#2ca02c",ms=8,lw=2,label="detection")
a3.set_xlabel("edited-region size (% of image area)"); a3.set_ylabel("detection rate",color="#2ca02c"); a3.set_ylim(0.9,1.02)
a3b=a3.twinx(); a3b.plot(ms,lss,"-s",color="#1f77b4",ms=6); a3b.set_ylabel("SSIM",color="#1f77b4"); a3b.set_ylim(0.5,0.95)
for x,y in zip(ms,ld): a3.text(x,y+0.003,"%.3f"%y,ha="center",fontsize=9,color="#2ca02c")
a3.set_xticks(ms); a3.set_xticklabels(["10-20%","30-40%","50-60%"])
a3.set_title("Local editing: detection vs mask size (n=500)\nrobust even when 50-60% is repainted")
fig.tight_layout(); fig.savefig(R+"figures/fig9_wbench.png"); plt.close(fig)
print("fig9_wbench (3-panel, scaled up) saved")
print("SVD: mean-frame-det=%.3f any-frame=%.3f"%(svd["frame_det"],svd["any_frame_det"]))
