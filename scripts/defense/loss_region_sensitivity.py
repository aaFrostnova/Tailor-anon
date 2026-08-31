"""Where is the imperceptibility loss most sensitive to a FIXED-strength, FIXED-area perturbation?
User's idea: perturb the image region-by-region (same sigma, equal-area tiles), see which location costs
the most loss. If the border is the perceptually-CHEAPEST place, that explains why VINE hides the mark there.

Two conditions per tile:
  (A) perturb the CLEAN image directly           -> intrinsic spatial sensitivity of L2 & LPIPS
  (B) perturb the VAE-ROUND-TRIPPED image (which already has the zero-padded border predistortion),
      loss measured vs the clean cover, MINUS the round-trip baseline -> MARGINAL cost in VINE's real setting.

Key expectation: L2 is spatially UNIFORM (fixed sigma * equal area => identical MSE regardless of location),
so L2 cannot drive the border. Any border preference must come from LPIPS (perceptual) and/or the VAE
predistortion making the border 'already sacrificed' (cheap to add more error there).

Outputs: radial profiles (border-ring -> center) + G x G heatmaps + border/center ratios. Uses VINE-B's trained VAE."""
import sys, glob, numpy as np, torch, lpips
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo

dev="cuda"; IMG=256; G=8; TILE=IMG//G; SIGMA=0.10; K=4   # 8x8 tiles (32px), fixed sigma, 6 noise draws/tile
torch.manual_seed(0)
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))
covers=[np.full((IMG,IMG,3),0.5,np.float32)]+[np.asarray(Image.open(p).convert("RGB").resize((IMG,IMG)),np.float32)/255 for p in NAT[:4]]
def to_t(a01): return (torch.from_numpy(a01).permute(2,0,1)[None].float()*2-1).to(dev)   # [0,1]->[-1,1]

print("[load] VINE-B-Enc VAE + LPIPS-vgg", flush=True)
enc=VINE_Turbo.from_pretrained("Shilin-LU/VINE-B-Enc").to(dev).eval()
vae=enc.vae_a2b; sf=getattr(vae.config,"scaling_factor",0.18215)
lp=lpips.LPIPS(net='vgg').to(dev)
def vae_rt(x):  # x in [-1,1]
    with torch.no_grad():
        lat=vae.encode(x).latent_dist.mode()*sf
        rec=vae.decode(lat/sf).sample
    return rec.clamp(-1,1)

# accumulators over covers
dL2_A=np.zeros((G,G)); dLP_A=np.zeros((G,G)); dLP_B=np.zeros((G,G)); recerr=np.zeros((G,G)); nc=0
for cov in covers:
    x=to_t(cov)
    x_rt=vae_rt(x)
    with torch.no_grad(): base_B=lp(x_rt,x).item()
    # reconstruction-error map (predistortion), pooled to GxG
    re=(x_rt-x).abs().mean(1)[0].cpu().numpy()   # HxW
    for gi in range(G):
        for gj in range(G):
            recerr[gi,gj]+=re[gi*TILE:(gi+1)*TILE, gj*TILE:(gj+1)*TILE].mean()
    for gi in range(G):
        for gj in range(G):
            l2a=lpa=lpb=0.0
            for _ in range(K):
                noise=torch.zeros_like(x); n=torch.randn(1,3,TILE,TILE,device=dev)*(SIGMA*2)  # *2: [-1,1] scale
                noise[:,:,gi*TILE:(gi+1)*TILE, gj*TILE:(gj+1)*TILE]=n
                xA=(x+noise).clamp(-1,1); xB=(x_rt+noise).clamp(-1,1)
                with torch.no_grad():
                    l2a+=((xA-x)**2).mean().item()
                    lpa+=lp(xA,x).item()
                    lpb+=(lp(xB,x).item()-base_B)
            dL2_A[gi,gj]+=l2a/K; dLP_A[gi,gj]+=lpa/K; dLP_B[gi,gj]+=lpb/K
    nc+=1; print(f"  cover {nc}/{len(covers)} done", flush=True)
for M in (dL2_A,dLP_A,dLP_B,recerr): M/=nc

# radial profile: distance-to-nearest-border ring index
ring=np.minimum(np.minimum(np.arange(G)[:,None],G-1-np.arange(G)[:,None]),
                np.minimum(np.arange(G)[None,:],G-1-np.arange(G)[None,:]))
def radial(M):
    return [float(M[ring==r].mean()) for r in range(G//2)]
print("\n=== RADIAL PROFILES (ring 0 = BORDER  ->  ring 3 = CENTER) ===", flush=True)
for name,M in [("VAE recon-err (predistortion)",recerr),("dL2 (A: clean)",dL2_A),
               ("dLPIPS (A: clean)",dLP_A),("dLPIPS (B: post-VAE marginal)",dLP_B)]:
    prof=radial(M); b,c=prof[0],prof[-1]
    print(f"  {name:34s}: "+"  ".join(f"{v:.4f}" for v in prof)+f"   | border/center = {b/max(c,1e-9):.2f}", flush=True)

# figure: 4 heatmaps
fig,ax=plt.subplots(1,4,figsize=(18,4.3))
for a_,(name,M) in zip(ax,[("VAE recon-err",recerr),("ΔL2 (clean)",dL2_A),("ΔLPIPS (clean)",dLP_A),("ΔLPIPS (post-VAE marginal)",dLP_B)]):
    im=a_.imshow(M,cmap="viridis"); a_.set_title(name,fontsize=11); plt.colorbar(im,ax=a_,fraction=0.046)
    a_.set_xticks([]); a_.set_yticks([])
plt.suptitle(f"Where does a fixed-strength (σ={SIGMA}) equal-area perturbation cost most loss?  (8×8 tiles, {len(covers)} covers)",fontsize=12)
plt.tight_layout(); out="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/fig_loss_region_sensitivity.png"
plt.savefig(out,dpi=130); print(f"\nSAVED {out}\nLOSS_REGION_DONE",flush=True)
