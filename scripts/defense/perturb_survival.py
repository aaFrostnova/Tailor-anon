"""DECISIVE test for 'why does injected signal survive at the border': place the SAME perturbation block at
different positions, push through the VAE round-trip (null-secret embed), and measure how much survives.
survival = ||embed(x+delta,null) - embed(x,null)|| / ||delta||, per placement. If border > interior, the
zero-padded VAE round-trip genuinely preserves border perturbations more than interior ones."""
import sys, numpy as np
from PIL import Image
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0,"scripts/defense")
import torch
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper
KEY=b"v5_key_encoder_master"; sb=ShortenedBCH(); n=sb.n
V=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device="cuda")
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def emb(arr): return np.asarray(to512(V.embed_with_target(Image.fromarray(np.clip(arr,0,255).astype(np.uint8)),np.zeros(n,dtype=np.uint8))),np.float64)
gray=np.full((512,512,3),128.0)
base=emb(gray)                                   # VAE round-trip of clean gray (null secret)
rng=np.random.RandomState(0)
blk=rng.randn(64,64,3)*8.0                        # fixed perturbation block, amplitude ~8/255
S=64
# placements: (row0,col0) top-left of the 64x64 block. dist-to-edge of block center.
places={"corner(TL)":(0,0),"edge-mid(top)":(0,224),"quarter-in":(96,96),"center":(224,224)}
print(f"block energy ||delta||={np.sqrt((blk**2).sum()):.1f}")
for name,(r0,c0) in places.items():
    x=gray.copy(); x[r0:r0+S,c0:c0+S,:]+=blk
    prop=emb(x)-base                              # how the input block propagated through the round-trip
    din=min(r0,511-(r0+S-1),c0,511-(c0+S-1))      # block's distance to nearest image edge (px)
    surv_total=np.sqrt((prop**2).sum())/np.sqrt((blk**2).sum())
    # energy that stayed inside the block footprint vs leaked out
    m=np.zeros((512,512),bool); m[r0:r0+S,c0:c0+S]=True
    e_in=(prop[m]**2).sum(); e_out=(prop[~m]**2).sum()
    print(f"  {name:14} block-dist-to-edge={din:3d}px  survival(||prop||/||delta||)={surv_total:.3f}  "
          f"in-footprint frac={e_in/(e_in+e_out):.3f}")
print("PERTURB_DONE")
