"""Faithfulness check: redo the with/without positional-encoding ablation using VINE's ACTUAL decoder
architecture -- torchvision ConvNeXt (VINE uses convnext_base; here convnext_tiny, same family: all convs
zero-padded, global-avg-pool, NO positional encoding). Same read-task (code at a fixed location + 20 distractors)
that forces localization. If ConvNeXt (no-coord) can read the BORDER code but NOT the CENTER code, and adding
CoordConv rescues the center -> the mechanism holds on VINE's real decoder architecture, not just a toy CNN."""
import glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as M
from PIL import Image
torch.manual_seed(0); np.random.seed(0)
dev = "cuda"; IMG = 128; P = 20; SEC = 16; AMP = 0.15; NDIST = 20
paths = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:400]
X = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG)), np.float32) / 255).permute(2, 0, 1) for p in paths]).to(dev)
print(f"[data] {tuple(X.shape)}  decoder=ConvNeXt-tiny (VINE family)  distractors={NDIST}", flush=True)
def code_of(sec): return F.interpolate((2 * sec - 1).view(-1, 1, 4, 4), size=(P, P), mode="nearest") * AMP
def coordch(b):
    ys = torch.linspace(-1, 1, IMG, device=dev).view(1, 1, IMG, 1).expand(b, 1, IMG, IMG)
    xs = torch.linspace(-1, 1, IMG, device=dev).view(1, 1, 1, IMG).expand(b, 1, IMG, IMG)
    return torch.cat([xs, ys], 1)
def batch(B, Lrc):
    idx = torch.randint(0, X.size(0), (B,), device=dev); orig = X[idx]; img = orig.clone()
    for _ in range(NDIST):
        r = np.random.randint(0, IMG - P); c = np.random.randint(0, IMG - P)
        img[:, :, r:r + P, c:c + P] = (img[:, :, r:r + P, c:c + P] + code_of(torch.randint(0, 2, (B, SEC), device=dev).float())).clamp(0, 1)
    sec = torch.randint(0, 2, (B, SEC), device=dev).float(); r0, c0 = Lrc
    img[:, :, r0:r0 + P, c0:c0 + P] = (orig[:, :, r0:r0 + P, c0:c0 + P] + code_of(sec)).clamp(0, 1)
    return img, sec
class DecCNX(nn.Module):
    def __init__(s, coord):
        super().__init__(); s.coord = coord
        net = M.convnext_tiny(weights=None)                       # VINE decoder family; all convs zero-pad, GAP, no posenc
        if coord:
            stem = net.features[0][0]                             # Conv2d(3,96,4,4) patchify stem
            net.features[0][0] = nn.Conv2d(5, stem.out_channels, kernel_size=stem.kernel_size, stride=stem.stride)
        s.features = net.features; s.fc = nn.Linear(768, SEC)     # convnext_tiny final dim = 768
    def forward(s, x):
        if s.coord: x = torch.cat([x, coordch(x.size(0))], 1)
        return s.fc(s.features(x).mean([2, 3]))                   # global average pool -> read
def run(Lrc, coord, tag, steps=2500):
    torch.manual_seed(1); Dm = DecCNX(coord).to(dev)
    opt = torch.optim.AdamW(Dm.parameters(), 4e-4, weight_decay=0.05)
    for it in range(steps):
        img, sec = batch(32, Lrc); loss = F.binary_cross_entropy_with_logits(Dm(img), sec)
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) in (800, 2500):
            with torch.no_grad(): ba = ((Dm(img) > 0).float() == sec).float().mean().item()
            print(f"    [{tag}] step {it + 1}: bitacc={ba:.3f}", flush=True)
    Dm.eval()
    with torch.no_grad():
        accs = []
        for _ in range(8):
            img, sec = batch(64, Lrc); accs.append(((Dm(img) > 0).float() == sec).float().mean().item())
    return float(np.mean(accs))
CEN = (IMG // 2 - P // 2, IMG // 2 - P // 2)
print("RESULT (VINE's ConvNeXt decoder architecture): bit-acc, 20 distractors. 0.5 = cannot localize.")
for tag, (Lrc, coord) in {"border+no-coord": ((0, 0), False), "CENTER+no-coord": (CEN, False), "CENTER+CoordConv": (CEN, True)}.items():
    print(f"[{tag:20}] final bit-acc = {run(Lrc, coord, tag[:18]):.3f}", flush=True)
print("POSENC_CONVNEXT_DONE", flush=True)
