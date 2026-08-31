"""VINE border-cause ablation figure. Measured (checkpoint-3000). Clean signal = GAN-off configs (B,D)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FG="results/defense/final_figs"
# (name, border/center, PSNR, gan_on)
rows=[("VINE-B\nbaseline",4.83,42.35,None),("A full\nlosses",2.11,35.24,True),
      ("B no-perc\n(secret only)",1.04,9.50,False),("C crop\naug",1.76,33.79,True),
      ("D no-GAN\n(L2+LPIPS)",3.05,36.86,False),("E no-LPIPS\n(L2+GAN)",3.44,35.66,True)]
INK,INK2="#0b0b0b","#52514e"
plt.rcParams.update({"font.size":11,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
fig,ax=plt.subplots(figsize=(11,4.6)); x=np.arange(len(rows))
# color: baseline gray, GAN-off (clean) blue, GAN-on (confounded) hatched orange
cols=["#8a8a8a" if g is None else ("#2a78d6" if g is False else "#eb6834") for _,_,_,g in rows]
bars=ax.bar(x,[r[1] for r in rows],color=cols,width=0.62,edgecolor="white")
for b,(_,br,ps,g) in zip(bars,rows):
    if g: b.set_hatch("//")
    ax.annotate(f"{br:.2f}",(b.get_x()+b.get_width()/2,br+0.06),ha="center",fontsize=10,color=INK)
    ax.annotate(f"PSNR\n{ps:.1f}",(b.get_x()+b.get_width()/2,0.15),ha="center",fontsize=8,color="white",weight="bold")
ax.axhline(1.0,color=INK2,ls=":",lw=1); ax.text(len(rows)-0.5,1.05,"uniform (=1)",color=INK2,fontsize=8,ha="right")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows],fontsize=9)
ax.set_ylabel("payload border/center energy ratio"); ax.set_ylim(0,5.2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#8a8a8a",label="baseline (not fine-tuned)"),Patch(fc="#2a78d6",label="GAN-off (clean)"),
                   Patch(fc="#eb6834",hatch="//",label="GAN-on (random-disc confounded)")],frameon=False,fontsize=9,loc="upper right")
ax.set_title("Ablation: which loss pushes VINE's watermark to the border?\nClean comparison D vs B (both GAN-off): L2+LPIPS perceptual loss keeps it at the border (3.05) vs uniform without it (1.04)",
             fontsize=10.5,loc="left",color=INK)
fig.savefig(f"{FG}/fig_border_ablation.png",dpi=150,bbox_inches="tight"); print("saved fig_border_ablation.png")
