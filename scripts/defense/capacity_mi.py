"""Reliable-bits (usable capacity) per fragment per attack = native_bits x per-bit mutual information.
MI(embedded_bit ; decoder_soft_output) estimated NON-parametrically (calibration-free) from the
cached per-bit soft outputs in hidden_big.npz (n=3000 imgs x 6 attacks, VINE/TM/VideoSeal).
Also reports hard-BSC capacity (1-H(BER)) for comparison -> the soft-vs-hard gap = soft-decoding gain."""
import sys, numpy as np
d = np.load(sys.argv[1] if len(sys.argv)>1 else "/scratch/workspace/mingzhel_umass_edu-ablator/hidden_big.npz", allow_pickle=True)
CLAMP=15.0
def prob2llr(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
PV=d["PV"].astype(np.float64); PT=d["PT"].astype(np.float64); AS=d["AS"].astype(np.float64)
SV=d["SV"].astype(np.int64); MV=d["MV"].astype(np.float64); ST=d["ST"].astype(np.int64); MT=d["MT"].astype(np.float64)
TX=d["TX"].astype(np.int64); ATK=d["ATK"].astype(str)
def npalign(raw,sig,M): return np.take_along_axis(M,sig,1)*np.take_along_axis(raw,sig,1)
FV=npalign(np.clip(prob2llr(PV),-CLAMP,CLAMP),SV,MV)   # VINE codeword LLR
FT=npalign(np.clip(PT,-CLAMP,CLAMP),ST,MT)             # TM codeword LLR
FS=np.clip(AS,-CLAMP,CLAMP)                            # VideoSeal codeword LLR
def Hb(p): p=np.clip(p,1e-12,1-1e-12); return -(p*np.log2(p)+(1-p)*np.log2(1-p))
def mi_per_bit(truth, llr, K=25):
    """Non-parametric I(bit;llr) in bits/position, pooled over all (img,pos). Calibration-free via quantile bins."""
    b=truth.ravel().astype(np.float64); s=llr.ravel()
    Hprior=Hb(b.mean())                                # ~1.0 (balanced codeword)
    edges=np.quantile(s,np.linspace(0,1,K+1)); edges[0]-=1e-9; edges[-1]+=1e-9
    idx=np.clip(np.digitize(s,edges)-1,0,K-1)
    Hcond=0.0
    for k in range(K):
        m=idx==k
        if not m.any(): continue
        Hcond += m.mean()*Hb(b[m].mean())
    return max(Hprior-Hcond,0.0), Hprior
def hard_cap(truth, llr):
    ba=((llr>0).astype(int)==truth).mean(); ber=1-ba
    return ba, max(1-Hb(ber),0.0)
NAT={"VINE":100,"TrustMark":100,"VideoSeal":100}   # embedded bits used; VideoSeal native=256
FR={"VINE":FV,"TrustMark":FT,"VideoSeal":FS}
atts=["clean","jpeg","blur","noise","crop","regen"]
print(f"=== reliable bits per fragment per attack  (n={ (ATK=='clean').sum() } imgs/attack, native=100 embedded bits) ===")
print(f"{'attack':7s} | " + " | ".join(f"{f:^26s}" for f in FR))
print(f"{'':7s} | " + " | ".join(f"{'ba   hardbits softbits':^26s}" for _ in FR))
rows={}
for a in atts:
    m=ATK==a
    line=f"{a:7s} | "
    rows[a]={}
    for f,F in FR.items():
        ba,hc=hard_cap(TX[m],F[m]); mi,_=mi_per_bit(TX[m],F[m])
        hb=hc*NAT[f]; sb=mi*NAT[f]; rows[a][f]=(ba,hb,sb)
        line+=f"{ba:4.2f} {hb:6.1f}   {sb:6.1f}   | "
    print(line)
print("\n=== soft reliable-bits summary (of 100 embedded) + VideoSeal-native(256) potential ===")
print(f"{'attack':7s} {'VINE':>7s} {'TrustM':>7s} {'VidSeal':>8s} {'VidSeal256':>11s} {'3-frag SUM':>11s} {'best-single':>12s}")
for a in atts:
    v=rows[a]["VINE"][2]; t=rows[a]["TrustMark"][2]; s=rows[a]["VideoSeal"][2]
    s256=s/100*256
    print(f"{a:7s} {v:7.1f} {t:7.1f} {s:8.1f} {s256:11.1f} {v+t+s:11.1f} {max(v,t,s):12.1f}")
print("\nNote: SUM=concatenation capacity (non-robust); best-single=robust-repetition capacity (any 1 frag survives).")
print("CAP_DONE")
