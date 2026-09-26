"""3-way context-gated calibrated MRC fusion head over VINE + TrustMark + VideoSeal
aligned codeword LLRs. Trained by scripts/defense/train_frag3_head.py (N=250); recovers
the CtrlRegen+ double-dead-fragment dilution (eq-MRC 0.92/0.67/0.42 -> 0.96/0.79/0.58 =
VINE-only/best-path) at FPR 0. Output is a codeword LLR -> still Chase-BCH + exact 37-bit
match, so crypto-ID 2^-37 FPR is preserved.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLAMP = 15.0


class Head3(nn.Module):
    def __init__(self, hid: int = 24):
        super().__init__()
        self.cal = nn.ModuleList([nn.Sequential(nn.Linear(1, hid), nn.ReLU(), nn.Linear(hid, 1)) for _ in range(3)])
        self.gate = nn.Sequential(nn.Linear(15, hid), nn.ReLU(), nn.Linear(hid, 3))

    @staticmethod
    def feats(a):
        aa = a.abs()
        return torch.stack([aa.mean(1), aa.std(1), (aa < 0.5).float().mean(1), torch.tanh(aa).mean(1)], 1)

    def ctx(self, av, at, as_):
        agr = lambda x, y: (torch.sign(x) == torch.sign(y)).float().mean(1, keepdim=True)
        return torch.cat([self.feats(av), self.feats(at), self.feats(as_),
                          agr(av, at), agr(av, as_), agr(at, as_)], 1)

    def forward(self, av, at, as_):
        B, m = av.shape
        w = F.softplus(self.gate(self.ctx(av, at, as_)))
        cs = [self.cal[i](x.reshape(-1, 1)).reshape(B, m) for i, x in enumerate([av, at, as_])]
        return w[:, 0:1] * cs[0] + w[:, 1:2] * cs[1] + w[:, 2:3] * cs[2]


def load_head3(path: str, device: str = "cuda") -> Head3:
    h = Head3().to(device)
    h.load_state_dict(torch.load(path, map_location=device))
    h.eval()
    return h


def head3_llr(head: Head3, av, at, as_, device: str = "cuda") -> np.ndarray:
    """av/at/as_ : codeword LLRs, numpy [n] (single image) or [B,n]. Returns the fused
    codeword LLR (clamped to the training range). Same leading shape as input."""
    single = (np.ndim(av) == 1)
    AV = np.clip(np.atleast_2d(av), -CLAMP, CLAMP).astype(np.float32)
    AT = np.clip(np.atleast_2d(at), -CLAMP, CLAMP).astype(np.float32)
    AS = np.clip(np.atleast_2d(as_), -CLAMP, CLAMP).astype(np.float32)
    with torch.no_grad():
        f = head(torch.tensor(AV, device=device), torch.tensor(AT, device=device),
                 torch.tensor(AS, device=device)).cpu().numpy()
    return f[0] if single else f
