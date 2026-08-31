"""Surgical sharpening: keep v1's TRAINABLE base (64px, P=16, 3-conv) and vary ONLY localization difficulty
(number of distractors). Reading stays easy (big high-amp patch); only LOCALIZING the true code gets harder.
Prediction if the mechanism holds:
  border+no-coord  : stays HIGH across all distractor counts (border is localizable via zero-padding)
  CENTER+no-coord  : DEGRADES as distractors rise (no positional cue at the interior)
  CENTER+CoordConv : stays HIGH (positional encoding localizes the interior regardless)
=> the border-vs-center gap (no-coord) widens with distractors, but positional encoding closes it."""
import glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
torch.manual_seed(0); np.random.seed(0)
dev = "cuda"; IMG = 64; P = 16; SEC = 16; AMP = 0.15
paths = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:400]
X = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG)), np.float32) / 255).permute(2, 0, 1) for p in paths]).to(dev)
def code_of(sec): return F.interpolate((2 * sec - 1).view(-1, 1, 4, 4), size=(P, P), mode="nearest") * AMP
def coordch(b):
    ys = torch.linspace(-1, 1, IMG, device=dev).view(1, 1, IMG, 1).expand(b, 1, IMG, IMG)
    xs = torch.linspace(-1, 1, IMG, device=dev).view(1, 1, 1, IMG).expand(b, 1, IMG, IMG)
    return torch.cat([xs, ys], 1)
def batch(B, Lrc, ndist):
    idx = torch.randint(0, X.size(0), (B,), device=dev); orig = X[idx]; img = orig.clone()
    for _ in range(ndist):
        r = np.random.randint(0, IMG - P); c = np.random.randint(0, IMG - P)
        img[:, :, r:r + P, c:c + P] = (img[:, :, r:r + P, c:c + P] + code_of(torch.randint(0, 2, (B, SEC), device=dev).float())).clamp(0, 1)
    sec = torch.randint(0, 2, (B, SEC), device=dev).float(); r0, c0 = Lrc
    img[:, :, r0:r0 + P, c0:c0 + P] = (orig[:, :, r0:r0 + P, c0:c0 + P] + code_of(sec)).clamp(0, 1)
    return img, sec
class Dec(nn.Module):
    def __init__(s, coord):
        super().__init__(); s.coord = coord; cin = 3 + (2 if coord else 0)
        s.net = nn.Sequential(nn.Conv2d(cin, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
                              nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU()); s.fc = nn.Linear(128, SEC)
    def forward(s, x):
        if s.coord: x = torch.cat([x, coordch(x.size(0))], 1)
        return s.fc(s.net(x).mean([2, 3]))
def run(Lrc, coord, ndist, steps=3000):
    torch.manual_seed(1); Dm = Dec(coord).to(dev); opt = torch.optim.Adam(Dm.parameters(), 2e-3)
    for it in range(steps):
        img, sec = batch(64, Lrc, ndist); loss = F.binary_cross_entropy_with_logits(Dm(img), sec)
        opt.zero_grad(); loss.backward(); opt.step()
    Dm.eval()
    with torch.no_grad():
        accs = [((Dm(batch(128, Lrc, ndist)[0]) > 0).float() == s2).float().mean().item()
                for _ in range(6) for s2 in [None] if True for img2, s2 in [batch(128, Lrc, ndist)]]
    with torch.no_grad():
        accs = []
        for _ in range(8):
            img, sec = batch(128, Lrc, ndist); accs.append(((Dm(img) > 0).float() == sec).float().mean().item())
    return float(np.mean(accs))
CEN = (24, 24)
print("RESULT: bit-acc vs number of distractors (localization difficulty). 0.5 = cannot localize.")
print(f"{'ndist':>6} | border+no-coord | CENTER+no-coord | CENTER+CoordConv")
for nd in [2, 6, 12, 20]:
    b = run((0, 0), False, nd); c = run(CEN, False, nd); cc = run(CEN, True, nd)
    print(f"{nd:>6} |      {b:.3f}     |      {c:.3f}     |      {cc:.3f}", flush=True)
print("POSENC_SWEEP_DONE", flush=True)
