"""Extract pre-readout hidden states from VINE + TrustMark decoders via forward-pre hooks.
Robustly locates each decoder's final readout Linear and taps its INPUT (the hidden)."""
import torch, numpy as np
def _last_readout_linear(module, out_features):
    cand=[(n,m) for n,m in module.named_modules() if type(m).__name__=="Linear"]
    exact=[(n,m) for n,m in cand if getattr(m,'out_features',None)==out_features]
    return (exact[-1] if exact else cand[-1])
class HiddenTap:
    def __init__(self, vine, tm, n_bits=100):
        self.vine=vine; self.tm=tm; self.h={}
        self.vdec=vine._load_decoder() if hasattr(vine,"_load_decoder") else vine._dec
        nv,self.vfc=_last_readout_linear(self.vdec, n_bits)
        nt,self.tfc=_last_readout_linear(self.tm.tm.decoder, n_bits)
        print(f"[HiddenTap] VINE readout={nv} in={self.vfc.in_features} | TM readout={nt} in={self.tfc.in_features}",flush=True)
        self.vfc.register_forward_pre_hook(lambda m,i: self.h.__setitem__("v", i[0].detach()))
        self.tfc.register_forward_pre_hook(lambda m,i: self.h.__setitem__("t", i[0].detach()))
    def extract(self, pil):
        self.h.clear()
        pv=self.vine.raw_probs(pil); pt=self.tm.raw_logits(pil)
        return self.h["v"].reshape(-1).cpu().numpy(), self.h["t"].reshape(-1).cpu().numpy(), pv, pt
