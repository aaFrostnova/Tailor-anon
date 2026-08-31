"""Fetch N=10k UltraEdit source images + instructions to /scratch (for main-table eval).
FreeForm split of BleachNick/UltraEdit_500k. Retries with backoff on HF 429 rate-limit."""
import os, sys, json, time
os.environ["HF_HOME"]="/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache"
os.environ["HF_HUB_CACHE"]="/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache/hub"
tok=open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
os.environ["HF_TOKEN"]=tok; os.environ["HUGGING_FACE_HUB_TOKEN"]=tok
from datasets import load_dataset
from PIL import Image
N=int(sys.argv[1]) if len(sys.argv)>1 else 10000
OUT="/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k"
os.makedirs(os.path.join(OUT,"source"),exist_ok=True)
def load_retry():
    for a in range(30):
        try: return load_dataset("BleachNick/UltraEdit_500k", split="FreeForm", streaming=True)
        except Exception as e:
            if "429" in str(e) or "Too Many" in str(e):
                w=min(60*(a+1),330); print(f"[429] attempt {a}, sleep {w}s",flush=True); time.sleep(w)
            else: print("ERR:",repr(e)[:200],flush=True); raise
    raise RuntimeError("giving up after retries")
ds=load_retry()
man=open(os.path.join(OUT,"manifest.jsonl"),"w")
img_keys=None; txt_key=None; n=0
for s in ds:
    if img_keys is None:  # detect schema on first sample
        img_keys=[k for k,v in s.items() if isinstance(v,Image.Image)]
        txt_key="edit_prompt"  # UltraEdit edit instruction
        print("SCHEMA img_keys=",img_keys," txt_key=",txt_key," all=",list(s.keys()),flush=True)
    src_key = "source_image" if "source_image" in img_keys else (img_keys[0] if img_keys else None)
    if src_key is None: print("no image field!",flush=True); break
    img=s[src_key].convert("RGB")
    fn=f"{n:06d}.png"; _p=os.path.join(OUT,"source",fn)  # RESUME
    if not os.path.exists(_p): img.save(_p)
    man.write(json.dumps({"id":fn,"instruction":s.get(txt_key,"") if txt_key else "",
                          "edited_key":("edited_image" if "edited_image" in img_keys else "")})+"\n")
    n+=1
    if n%500==0: print(f"  saved {n}/{N}",flush=True)
    if n>=N: break
man.close()
print(f"FETCH_DONE saved {n} -> {OUT}",flush=True)
