import json, os
R="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/"
def L(f):
    try: return json.load(open(R+f))
    except Exception as e: return {"_err":str(e)}
nan=float("nan")
print("==== 18-ATTACK (n=64) composite_or: V+TM | V+VS | V+TM+VS ====")
cfgs={"V+TM":"official_vine_tm.json","V+VS":"official_vine_videoseal.json","V+TM+VS":"official_vine_tm_videoseal.json"}
D={k:L(v) for k,v in cfgs.items()}
print("n_images:",{k:D[k].get("n_images") for k in D})
atks=list(D["V+TM+VS"]["attacks"].keys())
for a in atks:
    print("  %-10s %s"%(a," ".join("%s=%.3f"%(k,D[k]["attacks"][a]["composite_or"]) for k in D)))
print("\n==== per-frag bit-acc (3-frag) ====")
for a in atks:
    v=D["V+TM+VS"]["attacks"][a]
    print("  %-10s vine=%.3f tm=%.3f vs=%.3f"%(a,v.get("vine",nan),v.get("trustmark",nan),v.get("videoseal",nan)))

print("\n==== CtrlRegen+ n=200 breakdown (VINE/eq3/HEAD/best) ====")
for t in ["03","05","07"]:
    d=L("ctrlregen200_s%s.json"%t)
    print("  s0.%s VINE=%.3f eq3=%.3f HEAD=%.3f best=%.3f head_ba=%.3f n=%d"%(t[1],d["VINE_only"],d["eq3"],d["HEAD"],d["bestpath_oracle"],d["head_fused_ba"],d["n"]))

print("\n==== UnMarker n=100 sweep ====")
for s in ["200_100","300_150","400_200"]:
    d=L("unmarker100_s%s.json"%s)
    print("  s%s VINE=%.3f HEAD=%.3f best=%.3f head_ba=%.3f n=%d"%(s,d["VINE_only"],d["HEAD"],d["bestpath_oracle"],d["head_fused_ba"],d["n"]))

print("\n==== sweep_qp.json (P-Q curve rows + Q@P) ====")
q=L("sweep_qp.json")
print("  rows:")
for r in q["rows"]:
    print("    s=%.1f Q=%.3f SSIM=%.3f PSNR=%.1f LPIPS=%.3f | P_head=%.3f P_vine=%.3f P_eq3=%.3f P_best=%.3f"%(
        r["s"],r["Q"],r["ssim"],r["psnr"],r["lpips"],r["P_head"],r["P_vine"],r["P_eq3"],r["P_bestpath"]))
print("  results:")
for k,v in q["results"].items():
    print("    %-9s Q@0.95P=%s Q@0.70P=%s AvgP=%.3f AvgQ=%.3f"%(k,v["Q@0.95P"],v["Q@0.70P"],v["AvgP"],v["AvgQ"]))

print("\n==== fpr_test.json ====")
f=L("fpr_test.json")
print("  ",{k:f[k] for k in ["N","K","KC","tau","zerobit_head_fpr","zerobit_eq_fpr","null_head_mean","null_head_std","fver_fpr","bestpath_fpr","composite_or_fpr"]})

print("\n==== fidelity (from official jsons clean_coexist) ====")
for k in D: print("  %s clean_coexist=%s"%(k,D[k].get("clean_coexist")))
