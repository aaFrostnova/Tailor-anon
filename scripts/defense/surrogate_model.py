import numpy as np
import z3

RANGES = {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6), "VideoSeal": (0.5, 1.5)}
PHASE1_ATTACKS = ["jpeg25","blur","noise","bright","contrast","crop75","crop50","rot9","vaeB","vaeC"]

class PWL:
    """Piecewise-linear curve through knots (xs, ys), clamped outside [xs[0], xs[-1]]."""
    def __init__(self, xs, ys):
        assert len(xs) == len(ys) >= 2
        assert all(xs[i] < xs[i+1] for i in range(len(xs)-1)), "xs must be strictly increasing"
        self.xs = [float(x) for x in xs]
        self.ys = [float(y) for y in ys]

    def eval(self, x):
        return float(np.interp(x, self.xs, self.ys))  # np.interp clamps to end values

    def add_to_z3(self, x_var, name):
        """Return (y_var, constraints) encoding y_var == PWL(x_var), CLAMPED outside
        [xs[0], xs[-1]] to the endpoint y-values -- matching np.interp's behavior in .eval()
        (an out-of-range x_var is no longer unsat). One boolean per interior segment plus a
        `low` and `high` boundary case; exactly one of {low, seg_0 .. seg_{n-2}, high} is
        active. Boundary overlap at a knot (e.g. x_var == xs[0]) is fine -- more than one case
        yields the same y there, z3 just picks one."""
        y = z3.Real(f"y_{name}")
        segs = [z3.Bool(f"seg_{name}_{i}") for i in range(len(self.xs) - 1)]
        low = z3.Bool(f"low_{name}")
        high = z3.Bool(f"high_{name}")
        cons = [z3.PbEq([(b, 1) for b in ([low] + segs + [high])], 1)]  # exactly one case active
        cons.append(z3.Implies(low, z3.And(x_var <= self.xs[0], y == self.ys[0])))
        cons.append(z3.Implies(high, z3.And(x_var >= self.xs[-1], y == self.ys[-1])))
        for i, b in enumerate(segs):
            x0, x1, y0, y1 = self.xs[i], self.xs[i+1], self.ys[i], self.ys[i+1]
            slope = (y1 - y0) / (x1 - x0)
            cons.append(z3.Implies(b, z3.And(x_var >= x0, x_var <= x1,
                                             y == y0 + slope * (x_var - x0))))
        return y, cons

class Surrogate:
    def __init__(self, fragments, attacks, ranges, base, delta, d, e):
        self.fragments=list(fragments); self.attacks=list(attacks); self._ranges=dict(ranges)
        self._base=base; self._delta=delta; self._d=d; self._e=e   # dict-keyed PWLs
    def range(self,f): return self._ranges[f]
    def base(self,f,a): return self._base[(f,a)]
    def delta(self,g,f,a): return self._delta[(g,f,a)]
    def d(self,f): return self._d[f]
    def e(self,f,g): return self._e[tuple(sorted((f,g)))]
    def to_dict(self):
        pk=lambda p:{"xs":p.xs,"ys":p.ys}
        return {"fragments":self.fragments,"attacks":self.attacks,"ranges":self._ranges,
                "base":{f"{f}|{a}":pk(self._base[(f,a)]) for (f,a) in self._base},
                "delta":{f"{g}|{f}|{a}":pk(self._delta[(g,f,a)]) for (g,f,a) in self._delta},
                "d":{f:pk(self._d[f]) for f in self._d},
                "e":{f"{p[0]}|{p[1]}":pk(self._e[p]) for p in self._e}}
    @staticmethod
    def from_dict(dd):
        mk=lambda o:PWL(o["xs"],o["ys"])
        base={(k.split("|")[0],k.split("|")[1]):mk(v) for k,v in dd["base"].items()}
        delta={(k.split("|")[0],k.split("|")[1],k.split("|")[2]):mk(v) for k,v in dd["delta"].items()}
        d={k:mk(v) for k,v in dd["d"].items()}
        e={tuple(k.split("|")):mk(v) for k,v in dd["e"].items()}
        return Surrogate(dd["fragments"],dd["attacks"],dd["ranges"],base,delta,d,e)

def synthetic_surrogate(fragments=("VINE","TrustMark","VideoSeal"), attacks=tuple(PHASE1_ATTACKS)):
    import numpy as np
    fragments=list(fragments); attacks=list(attacks); ranges={f:RANGES[f] for f in fragments}
    base,delta,d,e={},{},{},{}
    for fi,f in enumerate(fragments):
        lo,hi=ranges[f]; xs=list(np.linspace(lo,hi,5))
        for a in attacks:
            # monotone-increasing saturating base in [0.5,1.0], attack-specific offset
            off=0.04*(hash((f,a))%5)
            ys=[min(1.0,0.55+0.4*((x-lo)/(hi-lo))-off) for x in xs]
            base[(f,a)]=PWL(xs,ys)
        d[f]=PWL(xs,[ (0.5+ ((x-lo)/(hi-lo)) )*1.0 for x in xs])  # distortion rises with strength
    for gi,g in enumerate(fragments):
        for fi,f in enumerate(fragments):
            if g==f: continue
            lo,hi=ranges[g]; xs=list(np.linspace(lo,hi,5))
            for a in attacks:
                delta[(g,f,a)]=PWL(xs,[0.02*((x-lo)/(hi-lo)) for x in xs])  # small positive
    for i in range(len(fragments)):
        for j in range(i+1,len(fragments)):
            p=tuple(sorted((fragments[i],fragments[j])))
            xs=[0.6,1.25,1.9,2.55,3.2]; e[p]=PWL(xs,[0.1*x for x in xs])   # 1-D in s_f+s_g; spans up to the max pairwise strength sum (TrustMark.hi+VideoSeal.hi=3.1)
    return Surrogate(fragments,attacks,ranges,base,delta,d,e)
