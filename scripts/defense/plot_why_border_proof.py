import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FG="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/final_figs"
INK,INK2="#0b0b0b","#52514e"; VINE="#2a78d6"; RED="#e34948"; GRN="#008300"
plt.rcParams.update({"font.size":11,"axes.edgecolor":INK2,"axes.linewidth":0.8,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
fig,axs=plt.subplots(1,2,figsize=(12,4.3))
# LEFT: real VINE relocate cliff (measured, why_border_decode)
sh=[0,8,16,32,64]; ba=[1.000,0.502,0.509,0.508,0.508]
axs[0].plot(sh,ba,"-o",color=VINE,lw=2.2,ms=8); axs[0].axhline(0.5,ls=":",color=INK2,lw=1)
axs[0].set_ylim(0.45,1.05); axs[0].set_xlabel("payload displacement off the border (px)"); axs[0].set_ylabel("VINE decode bit-acc")
for x,y in zip(sh,ba): axs[0].annotate(f"{y:.2f}",(x,y),textcoords="offset points",xytext=(0,8),fontsize=8,ha="center",color=INK2)
axs[0].set_title("(a) The PHENOMENON — real VINE is razor border-locked\nmove the payload 8px inward -> decode collapses to random",fontsize=10.5,loc="left",color=INK)
axs[0].spines["top"].set_visible(False); axs[0].spines["right"].set_visible(False); axs[0].grid(axis="y",color="#eee")
# RIGHT: controlled ablation @ 20 distractors — the CAUSE
labs=["border\n(no pos-enc)","CENTER\n(no pos-enc)","CENTER\n(+ pos-enc)"]; vals=[0.918,0.496,0.983]; cols=[VINE,RED,GRN]
b=axs[1].bar(range(3),vals,color=cols,width=0.6); axs[1].axhline(0.5,ls=":",color=INK2,lw=1)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(labs,fontsize=9.5); axs[1].set_ylim(0,1.08); axs[1].set_ylabel("decoder bit-acc")
for bar,v in zip(b,vals): axs[1].annotate(f"{v:.2f}",(bar.get_x()+bar.get_width()/2,v+0.02),ha="center",fontsize=10,color=INK)
axs[1].set_title("(b) The CAUSE — controlled decoder, hard localization (20 distractors)\nno pos-enc: only the border is readable; add pos-enc: interior readable too",fontsize=10.5,loc="left",color=INK)
axs[1].spines["top"].set_visible(False); axs[1].spines["right"].set_visible(False); axs[1].grid(axis="y",color="#eee")
axs[1].text(1.5,0.30,"same center code, same task —\nonly difference = positional encoding",ha="center",fontsize=9,color=INK2,style="italic")
fig.suptitle("Why VINE's payload is at the border: zero-padding is the only content-independent position anchor absent positional encoding",fontsize=11.5,y=1.02,color=INK)
fig.savefig(f"{FG}/fig_why_border_proof.png",dpi=150,bbox_inches="tight"); print("saved")
