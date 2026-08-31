"""Measure StegaStamp (Tancik CVPR2020, TF SavedModel) as an SMT candidate fragment.
Runs in the aiwm env (TF 2.9). 100-bit, 400x400. Decode path (image->decoded) is independent
of the encoder, so we decode any attacked image directly.
Modes:
  main   : embed N imgs, apply self-contained attacks (SIG/GEO), decode, print bit-acc;
           ALSO save wm PNGs + msgs.npy to WORK for the regen phase (done in fingerprint env).
  decode : decode pre-attacked PNGs in a dir (WORK/<tag>_{i}.png) vs msgs.npy -> bit-acc.
Prior: spatial CNN like TrustMark -> expected regen-dead."""
import os,sys,io,glob,time,json,numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
from PIL import Image, ImageFilter
import tensorflow as tf
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
STEGA=CF+"/external/ai-watermark/pretrained_models/watermarkers/stega/models"
WORK="/scratch/workspace/mingzhel_umass_edu-ablator/stega_work"; os.makedirs(WORK,exist_ok=True)
S=400
g=tf.compat.v1.Graph(); sess=tf.compat.v1.InteractiveSession(graph=g)
tf.compat.v1.saved_model.loader.load(sess,[tf.compat.v1.saved_model.tag_constants.SERVING],STEGA)
G=tf.compat.v1.get_default_graph()
secret_t=G.get_tensor_by_name("input_prep:0"); image_t=G.get_tensor_by_name("input_hide:0")
stega_t=G.get_tensor_by_name("clip_by_value:0"); decoded_t=G.get_tensor_by_name("Round:0")
def to400(pil): return pil.convert("RGB").resize((S,S),Image.BILINEAR)
def embed(pil,msg):
    x=(np.asarray(to400(pil),np.float32)/255.)[None]
    wm=sess.run(stega_t,{secret_t:msg[None].astype(np.float32),image_t:x})[0]
    return Image.fromarray((np.clip(wm,0,1)*255).astype(np.uint8),"RGB")
def decode(pil,msg):
    x=(np.asarray(to400(pil),np.float32)/255.)[None]
    d=sess.run(decoded_t,{image_t:x})[0]
    return float((np.round(d).astype(int)==msg.astype(int)).mean())
def psnr(a,b):
    a=np.asarray(to400(a),np.float64); b=np.asarray(to400(b),np.float64)
    m=np.mean((a-b)**2); return 99. if m<1e-9 else 10*np.log10(255**2/m)
def attack(name,im):
    if name=="clean": return im
    if name=="jpeg50": b=io.BytesIO(); im.save(b,"JPEG",quality=50); b.seek(0); return Image.open(b).convert("RGB")
    if name=="jpeg25": b=io.BytesIO(); im.save(b,"JPEG",quality=25); b.seek(0); return Image.open(b).convert("RGB")
    if name=="blur": return im.filter(ImageFilter.GaussianBlur(2))
    if name=="noise":
        x=np.asarray(im,np.float32)/255+np.random.RandomState(0).randn(*np.asarray(im).shape).astype(np.float32)*0.05
        return Image.fromarray((np.clip(x,0,1)*255).astype('uint8'))
    if name=="crop90":
        W,H=im.size; c=int(min(W,H)*0.9); l=(W-c)//2; t=(H-c)//2; return im.crop((l,t,l+c,t+c)).resize((W,H),Image.BILINEAR)
    if name=="crop75":
        W,H=im.size; c=int(min(W,H)*0.75); l=(W-c)//2; t=(H-c)//2; return im.crop((l,t,l+c,t+c)).resize((W,H),Image.BILINEAR)
    if name=="rot9": return im.rotate(9,resample=Image.BILINEAR)
    if name=="rot30": return im.rotate(30,resample=Image.BILINEAR)
    return im
SELF=["clean","jpeg50","jpeg25","blur","noise","crop90","crop75","rot9","rot30"]
N=int(sys.argv[1]) if len(sys.argv)>1 else 50
mode=sys.argv[2] if len(sys.argv)>2 else "main"
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
if mode=="main":
    np.random.seed(0); msgs=np.random.randint(0,2,(N,100))
    np.save(WORK+"/msgs.npy",msgs)
    acc={a:[] for a in SELF}; ps=[]; te=[]; td=[]
    for i,fp in enumerate(files):
        orig=to400(Image.open(fp)); msg=msgs[i]
        t0=time.time(); wm=embed(orig,msg); te.append(time.time()-t0)
        ps.append(psnr(orig,wm)); wm.save(WORK+f"/wm_{i:04d}.png")   # for regen phase
        for a in SELF:
            atk=attack(a,wm); t0=time.time(); ba=decode(atk,msg)
            if a=="clean": td.append(time.time()-t0)
            acc[a].append(ba)
        if (i+1)%10==0: print(f"  [{i+1}/{N}]",flush=True)
    print(f"\n=== StegaStamp (100-bit,400px) solo self-contained attacks, n={N} ===")
    for a in SELF: print(f"{a:10s} {np.mean(acc[a]):8.3f}")
    print(f"\nclean PSNR {np.mean(ps):.2f} dB · embed {np.mean(te)*1000:.0f} ms · decode {np.mean(td)*1000:.0f} ms")
    out={a:round(float(np.mean(acc[a])),4) for a in SELF}
    out.update(psnr=round(float(np.mean(ps)),2),embed_ms=round(float(np.mean(te)*1000),1),decode_ms=round(float(np.mean(td)*1000),1),n_bits=100)
    json.dump(out,open(CF+"/results/defense/stega_frag.json","w"),indent=2)
    print("saved wm PNGs + stega_frag.json ; now run regen phase in fingerprint env\nSTEGA_MAIN_DONE")
elif mode=="decode":
    tag=sys.argv[3]  # e.g. regen or rinse
    msgs=np.load(WORK+"/msgs.npy"); ba=[]
    for i in range(N):
        p=WORK+f"/{tag}_{i:04d}.png"
        if not os.path.exists(p): continue
        ba.append(decode(Image.open(p),msgs[i]))
    print(f"StegaStamp {tag} bit-acc (n={len(ba)}): {np.mean(ba):.3f}")
    # merge into stega_frag.json
    fj=CF+"/results/defense/stega_frag.json"; d=json.load(open(fj)); d[tag]=round(float(np.mean(ba)),4); json.dump(d,open(fj,"w"),indent=2)
    print(f"STEGA_{tag.upper()}_DONE")
