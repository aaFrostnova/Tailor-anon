"""SHARPENED causal test: harder settings so the border-advantage (no positional encoding) is stark.
Larger image (128), smaller code patch (10), more distractors (8), deeper decoder (128->8, 4 stride-2 convs) ->
the zero-padding position cue is relatively far from the center, so localizing an interior code without
positional encoding should be much harder. Prediction: CENTER+no-coord collapses toward random while
CENTER+CoordConv stays high -> positional encoding is decisively what enables interior reading."""
import glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
torch.manual_seed(0); np.random.seed(0)
dev = "cuda"; IMG = 128; P = 10; SEC = 16; AMP = 0.18
paths = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:400]
X = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG)), np.float32) / 255).permute(2, 0, 1) for p in paths]).to(dev)
print(f"[data] {tuple(X.shape)} P={P} distractors=8", flush=True)
def code_of(sec): return F.interpolate((2 * sec - 1).view(-1, 1, 4, 4), size=(P, P), mode="nearest") * AMP
def coordch(b):
    ys = torch.linspace(-1, 1, IMG, device=dev).view(1, 1, IMG, 1).expand(b, 1, IMG, IMG)
    xs = torch.linspace(-1, 1, IMG, device=dev).view(1, 1, 1, IMG).expand(b, 1, IMG, IMG)
    return torch.cat([xs, ys], 1)
def batch(B, Lrc, ndist=8):
    idx = torch.randint(0, X.size(0), (B,), device=dev); orig = X[idx]; img = orig.clone()
    for _ in range(ndist):
        r = np.random.randint(0, IMG - P); c = np.random.randint(0, IMG - P)
        img[:, :, r:r + P, c:c + P] = (img[:, :, r:r + P, c:c + P] + code_of(torch.randint(0, 2, (B, SEC), device=dev).float())).clamp(0, 1)
    sec = torch.randint(0, 2, (B, SEC), device=dev).float(); r0, c0 = Lrc
    img[:, :, r0:r0 + P, c0:c0 + P] = (orig[:, :, r0:r0 + P, c0:c0 + P] + code_of(sec)).clamp(0, 1)
    return img, sec
class Dec(nn.Module):
    def __init__(s, coord, pad="zeros"):
        super().__init__(); s.coord = coord; cin = 3 + (2 if coord else 0)
        s.net = nn.Sequential(nn.Conv2d(cin, 48, 3, 2, 1, padding_mode=pad), nn.ReLU(),
                              nn.Conv2d(48, 96, 3, 2, 1, padding_mode=pad), nn.ReLU(),
                              nn.Conv2d(96, 128, 3, 2, 1, padding_mode=pad), nn.ReLU(),
                              nn.Conv2d(128, 128, 3, 2, 1, padding_mode=pad), nn.ReLU()); s.fc = nn.Linear(128, SEC)
    def forward(s, x):
        if s.coord: x = torch.cat([x, coordch(x.size(0))], 1)
        return s.fc(s.net(x).mean([2, 3]))
def run(Lrc, coord, ndist, tag, steps=3500):
    torch.manual_seed(1); Dm = Dec(coord).to(dev); opt = torch.optim.Adam(Dm.parameters(), 2e-3)
    for it in range(steps):
        img, sec = batch(64, Lrc, ndist); loss = F.binary_cross_entropy_with_logits(Dm(img), sec)
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) in (1000, 3500):
            with torch.no_grad(): ba = ((Dm(img) > 0).float() == sec).float().mean().item()
            print(f"    [{tag}] step {it + 1}: bitacc={ba:.3f}", flush=True)
    Dm.eval()
    with torch.no_grad():
        accs = []
        for _ in range(8):
            img, sec = batch(128, Lrc, ndist); accs.append(((Dm(img) > 0).float() == sec).float().mean().item())
    return float(np.mean(accs))
CEN = (IMG // 2 - P // 2, IMG // 2 - P // 2)
CONDS = [("border(corner)+no-coord", ((0, 0), False, 8)), ("CENTER+no-coord", (CEN, False, 8)),
         ("CENTER+CoordConv", (CEN, True, 8)), ("border(corner)+CoordConv", ((0, 0), True, 8)),
         ("CENTER+no-coord NO-distractor(sanity)", (CEN, False, 0))]
print("RESULT (sharpened): bit-acc per location (0.5=random=cannot localize)")
for tag, (Lrc, coord, ndist) in CONDS:
    print(f"[{tag:42}] final bit-acc = {run(Lrc, coord, ndist, tag[:20]):.3f}", flush=True)
print("POSENC_READABILITY2_DONE", flush=True)
