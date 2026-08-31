"""DECISIVE causal test (avoids the joint-AE cold-start). Place a code patch at a FIXED location + random
distractors elsewhere; train ONLY the decoder to read the code at the true location. To read the RIGHT code the
decoder must LOCALIZE that location (distractors defeat 'read-anywhere' global pooling).
  border+no-coord : can it read a code at the frame border without positional encoding?  (expect YES: padding cue)
  center+no-coord : can it read a code at the CENTER without positional encoding?          (expect NO: no cue there)
  center+CoordConv: does adding positional encoding let it read the CENTER code?           (expect YES -> the proof)
  corner control + no-distractor sanity."""
import sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
torch.manual_seed(0); np.random.seed(0)
dev = "cuda"; IMG = 64; P = 16; SEC = 16; AMP = 0.15
paths = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:400]
X = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG)), np.float32) / 255).permute(2, 0, 1) for p in paths]).to(dev)
print(f"[data] {tuple(X.shape)}", flush=True)
def code_of(sec): return F.interpolate((2 * sec - 1).view(-1, 1, 4, 4), size=(P, P), mode="nearest") * AMP  # B,1,P,P
def coordch(b):
    ys = torch.linspace(-1, 1, IMG, device=dev).view(1, 1, IMG, 1).expand(b, 1, IMG, IMG)
    xs = torch.linspace(-1, 1, IMG, device=dev).view(1, 1, 1, IMG).expand(b, 1, IMG, IMG)
    return torch.cat([xs, ys], 1)
def batch(B, Lrc, ndist=4):
    idx = torch.randint(0, X.size(0), (B,), device=dev); orig = X[idx]; img = orig.clone()
    for _ in range(ndist):                                    # distractors force localization
        r = np.random.randint(0, IMG - P); c = np.random.randint(0, IMG - P)
        img[:, :, r:r + P, c:c + P] = (img[:, :, r:r + P, c:c + P] + code_of(torch.randint(0, 2, (B, SEC), device=dev).float())).clamp(0, 1)
    sec = torch.randint(0, 2, (B, SEC), device=dev).float(); r0, c0 = Lrc
    img[:, :, r0:r0 + P, c0:c0 + P] = (orig[:, :, r0:r0 + P, c0:c0 + P] + code_of(sec)).clamp(0, 1)  # true code LAST, clean
    return img, sec
class Dec(nn.Module):
    def __init__(s, coord, pad="zeros"):
        super().__init__(); s.coord = coord; cin = 3 + (2 if coord else 0)
        s.net = nn.Sequential(nn.Conv2d(cin, 64, 3, 2, 1, padding_mode=pad), nn.ReLU(),
                              nn.Conv2d(64, 128, 3, 2, 1, padding_mode=pad), nn.ReLU(),
                              nn.Conv2d(128, 128, 3, 2, 1, padding_mode=pad), nn.ReLU()); s.fc = nn.Linear(128, SEC)
    def forward(s, x):
        if s.coord: x = torch.cat([x, coordch(x.size(0))], 1)
        return s.fc(s.net(x).mean([2, 3]))
def run(Lrc, coord, ndist, tag, steps=3000):
    torch.manual_seed(1); Dm = Dec(coord).to(dev); opt = torch.optim.Adam(Dm.parameters(), 2e-3)
    for it in range(steps):
        img, sec = batch(64, Lrc, ndist); loss = F.binary_cross_entropy_with_logits(Dm(img), sec)
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) in (500, 3000):
            with torch.no_grad(): ba = ((Dm(img) > 0).float() == sec).float().mean().item()
            print(f"    [{tag}] step {it + 1}: bitacc={ba:.3f}", flush=True)
    Dm.eval()
    with torch.no_grad():
        accs = [((Dm(batch(128, Lrc, ndist)[0]) > 0).float() == batch(128, Lrc, ndist)[1]).float().mean().item() for _ in range(6)]
    # note: fresh batches for the eval mean below (paired)
    with torch.no_grad():
        accs = []
        for _ in range(8):
            img, sec = batch(128, Lrc, ndist); accs.append(((Dm(img) > 0).float() == sec).float().mean().item())
    return float(np.mean(accs))
CONDS = [("border(corner)+no-coord", ((0, 0), False, 4)), ("CENTER+no-coord", ((24, 24), False, 4)),
         ("CENTER+CoordConv", ((24, 24), True, 4)), ("border(corner)+CoordConv", ((0, 0), True, 4)),
         ("CENTER+no-coord NO-distractor(sanity)", ((24, 24), False, 0))]
print("RESULT: can the decoder READ a code at each location? (bit-acc; 0.5=random=cannot localize)")
for tag, (Lrc, coord, ndist) in CONDS:
    print(f"[{tag:42}] final bit-acc = {run(Lrc, coord, ndist, tag[:20]):.3f}", flush=True)
print("POSENC_READABILITY_DONE", flush=True)
