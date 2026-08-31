"""Reliable CAPACITY of fragment COMBINATIONS (bit), from cached soft outputs (hidden_big.npz).
For each combo (non-empty subset of VINE/TrustMark/VideoSeal) x attack, report:
  best-single = max_f MI(bit;LLR_f)            robust, "any one fragment survives" (what SMT --min_bits uses)
  fused       = MI(bit; sum_f LLR_f)           robust, SAME codeword soft-combined across frags (>= best-single)
  concat      = sum_f MI(bit;LLR_f)             NON-robust, different bits per frag (dies if a frag dies)
+ per-combo robust-ID capacity = min over in-scope attacks (all-6, and no-regen)."""
import os, sys, itertools, numpy as np
CLAMP=15.0
def prob2llr(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
SC="/scratch/workspace/mingzhel_umass_edu-ablator"
paths=sys.argv[1:] if len(sys.argv)>1 else [f"{SC}/hidden_big.npz", f"{SC}/hidden_ext.npz"]
paths=[p for p in paths if os.path.exists(p)]
def loadcat(key): return np.concatenate([np.load(p,allow_pickle=True)[key] for p in paths],axis=0)
print(f"loaded caches: {[os.path.basename(p) for p in paths]}")
PV=loadcat("PV").astype(np.float64); PT=loadcat("PT").astype(np.float64); AS=loadcat("AS").astype(np.float64)
SV=loadcat("SV").astype(np.int64); MV=loadcat("MV").astype(np.float64); ST=loadcat("ST").astype(np.int64); MT=loadcat("MT").astype(np.float64)
TX=loadcat("TX").astype(np.int64); ATK=loadcat("ATK").astype(str)
def npalign(raw,sig,M): return np.take_along_axis(M,sig,1)*np.take_along_axis(raw,sig,1)
# aligned-to-codeword per-bit LLRs (same as capacity_mi.py). All aligned to the SAME TX -> summable.
F={"VINE":     npalign(np.clip(prob2llr(PV),-CLAMP,CLAMP),SV,MV),
   "TrustMark":npalign(np.clip(PT,-CLAMP,CLAMP),ST,MT),
   "VideoSeal":np.clip(AS,-CLAMP,CLAMP)}
def Hb(p): p=np.clip(p,1e-12,1-1e-12); return -(p*np.log2(p)+(1-p)*np.log2(1-p))
def mi(truth, llr, K=25):
    b=truth.ravel().astype(np.float64); s=llr.ravel()
    Hprior=Hb(b.mean())
    edges=np.quantile(s,np.linspace(0,1,K+1)); edges[0]-=1e-9; edges[-1]+=1e-9
    idx=np.clip(np.digitize(s,edges)-1,0,K-1)
    Hc=sum((idx==k).mean()*Hb(b[(idx==k)].mean()) for k in range(K) if (idx==k).any())
    return max(Hprior-Hc,0.0)*100.0   # x100 embedded bits -> reliable bits
FRAGS=["VINE","TrustMark","VideoSeal"]
atts=[a for a in ["clean","jpeg","blur","noise","crop","rot9","rot30","vaeB","vaeC","regen","rinse"] if (ATK==a).any()]
combos=[c for r in range(1,4) for c in itertools.combinations(FRAGS,r)]
def cap(combo, a, mode):
    m=ATK==a
    if mode=="best":  return max(mi(TX[m],F[f][m]) for f in combo)
    if mode=="fused": return mi(TX[m], sum(F[f][m] for f in combo))
    if mode=="concat":return sum(mi(TX[m],F[f][m]) for f in combo)
short={"VINE":"V","TrustMark":"T","VideoSeal":"S"}
def name(c): return "+".join(short[f] for f in c)
print(f"=== FUSED reliable capacity (bit) — same codeword soft-combined across combo (n=3000/attack) ===")
print(f"{'combo':8s} "+" ".join(f"{a:>7s}" for a in atts)+f"   {'min6':>6s} {'minNoReg':>9s}")
rows={}
for c in combos:
    vals={a:cap(c,a,"fused") for a in atts}; rows[c]=vals
    min6=min(vals.values()); minnr=min(vals[a] for a in atts if a!="regen")
    print(f"{name(c):8s} "+" ".join(f"{vals[a]:7.1f}" for a in atts)+f"   {min6:6.1f} {minnr:9.1f}")
print("\n=== fusion GAIN: best-single vs fused vs concat (3-frag V+T+S) ===")
allc=tuple(FRAGS)
print(f"{'attack':7s} {'best-single':>12s} {'fused':>7s} {'concat':>7s}")
for a in atts:
    print(f"{a:7s} {cap(allc,a,'best'):12.1f} {cap(allc,a,'fused'):7.1f} {cap(allc,a,'concat'):7.1f}")
print("\nkey: min6=robust-ID capacity if ALL 6 attacks in scope; minNoReg=if regen excluded.")
print("     best-single=any 1 frag survives; fused=soft-combine same ID (>=best); concat=split bits (non-robust).")
print("COMBOS_DONE")
