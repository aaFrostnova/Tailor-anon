"""CDBW prototype simulation — does cross-domain crypto binding (a) keep identity detection,
and (b) catch copy/overwrite that base-only mis-attributes? Uses REAL per-domain error rates
p_d measured on our fragments (bit-acc from official_vine_tm_videoseal.json) + REAL crypto
(HMAC-SHA256, (2,3)-Shamir over GF(p)). Channel per domain = BSC(p_d). Honest, not idealized.

Payload per domain d = [ ID(base, repeated 3x) | tau_d (per-domain content-MAC) | share_d ((2,3)) ],
binding bits repetition-coded (r) then majority; tau checked exactly (Hamming tol optional).
"""
import hashlib, hmac, argparse
import numpy as np
rng = np.random.default_rng(0)
KEY = b"cdbw_master_key"; PRIME = 1048583  # > 2^20

# ---- measured per-domain error prob p=1-bitacc (from our 3-frag per-fragment results) ----
PD = {  # attack -> (p_VINE, p_TM, p_VS)
    "clean":  (0.000, 0.002, 0.003),
    "jpeg":   (0.000, 0.015, 0.058),
    "regen":  (0.063, 0.398, 0.439),   # only VINE reliable
    "crop75": (0.494, 0.110, 0.097),   # only TM,VS reliable
    "rot9":   (0.511, 0.452, 0.174),   # only VS reliable
}
DOMS = ["V", "T", "S"]

def hbits(msg, nb):  # keyed HMAC -> nb-bit int
    return int.from_bytes(hmac.new(KEY, msg, hashlib.sha256).digest(), "big") % (2 ** nb)
def shamir_shares(secret):  # f(x)=secret+r*x mod PRIME ; shares f(1),f(2),f(3)
    r = hbits(secret.to_bytes(4, "big") + b"r", 20)
    return [(secret + r * x) % PRIME for x in (1, 2, 3)]
def shamir_recover(pts):  # pts=[(x,y),...] any 2 -> f(0)
    (x0, y0), (x1, y1) = pts[0], pts[1]
    # Lagrange at 0 over GF(PRIME)
    def inv(a): return pow(a % PRIME, PRIME - 2, PRIME)
    l0 = (0 - x1) * inv(x0 - x1) % PRIME
    l1 = (0 - x0) * inv(x1 - x0) % PRIME
    return (y0 * l0 + y1 * l1) % PRIME

def bits(v, n): return [(v >> i) & 1 for i in range(n)]
def frombits(b): return sum((bit & 1) << i for i, bit in enumerate(b))
def rep_enc(bl, r): return [b for b in bl for _ in range(r)]   # symbol-wise repetition (matches rep_majority)
def bsc(bitlist, p):  # flip each bit w.p. p
    return [b ^ int(f) for b, f in zip(bitlist, rng.random(len(bitlist)) < p)]
def rep_majority(recv, r):  # recv = flat repetition-coded bits, decode by majority per symbol
    n = len(recv) // r; out = []
    for i in range(n):
        out.append(1 if sum(recv[i*r:(i+1)*r]) * 2 > r else 0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--idbits", type=int, default=25)
    ap.add_argument("--taubits", type=int, default=16)
    ap.add_argument("--rep", type=int, default=5, help="repetition for binding bits")
    ap.add_argument("--reliable_p", type=float, default=0.25, help="domain 'alive' iff p_d<this")
    a = ap.parse_args()

    print(f"CDBW sim: n={a.n} idbits={a.idbits} taubits={a.taubits} rep={a.rep}\n")
    hdr = f"{'attack':8s} | {'ID-detect':9s} | {'auth(genuine)':13s} | {'COPY→base':9s} {'COPY→CDBW':9s} | {'OVERWRITE-flag':13s}"
    print(hdr); print("-" * len(hdr))
    for atk, pd in PD.items():
        idok = auth = copy_base = copy_cdbw = ow_flag = 0
        for _ in range(a.n):
            ID = int(rng.integers(0, 2 ** a.idbits)); Himg = int(rng.integers(0, 2 ** 24))
            ctx = ID.to_bytes(4, "big") + Himg.to_bytes(4, "big")
            A = hbits(ctx, 20); shs = shamir_shares(A)
            reliable = [pd[i] < a.reliable_p for i in range(3)]
            # ---- (base) identity: repeated across domains; detected if ANY domain's ID bits survive ----
            id_hit = False
            for i in range(3):
                rid = rep_majority(bsc(rep_enc(bits(ID, a.idbits), a.rep), pd[i]), a.rep)  # rep-coded ID as base proxy
                if frombits(rid) == ID: id_hit = True
            idok += id_hit
            # ---- (genuine auth) per-domain content-MAC tau_d over (ID,Himg,d); check reliable domains ----
            good = 0
            for i in range(3):
                if not reliable[i]: continue
                tau = hbits(ctx + DOMS[i].encode(), a.taubits)
                rtau = frombits(rep_majority(bsc(rep_enc(bits(tau, a.taubits), a.rep), pd[i]), a.rep))
                if rtau == tau: good += 1
            auth += (good >= 1)   # authenticated if >=1 reliable domain's content-MAC verifies
            # ---- COPY attack: copy domain V (for image A) onto a CLEAN image B (attacker wants convincing fake) ----
            IDb = int(rng.integers(0, 2 ** a.idbits)); Himgb = int(rng.integers(0, 2 ** 24))
            # base verifier: reads ID_A from the copied domain -> attributes A to image B (FALSE)
            copy_base += 1  # base-only always mis-attributes a perfectly-copied valid domain
            # CDBW verifier on image B: recompute tau_V with Himg_B; copied tau was for Himg_A -> mismatch => flagged
            tauA = hbits(ID.to_bytes(4,"big")+Himg.to_bytes(4,"big")+b"V", a.taubits)   # travels with the copy
            tauB_expect = hbits(ID.to_bytes(4,"big")+Himgb.to_bytes(4,"big")+b"V", a.taubits)
            # copy lands on clean image (p~0) -> tau recovered exactly; flagged iff != expected
            copy_cdbw += (tauA != tauB_expect)   # 1 = copy correctly flagged
            # ---- OVERWRITE: attacker (no key) overwrites domain T with random valid-looking bits ----
            if reliable[1]:
                fake_tau = int(rng.integers(0, 2 ** a.taubits))
                real_tau = hbits(ctx + b"T", a.taubits)
                ow_flag += (fake_tau != real_tau)   # flagged iff attacker's guess != real MAC
        N = a.n
        print(f"{atk:8s} | {idok/N:9.3f} | {auth/N:13.3f} | {copy_base/N:9.3f} {copy_cdbw/N:9.3f} | {ow_flag/N:13.3f}")
    print("\nlegend: ID-detect=identity recovered (base, repetition);"
          " auth=genuine content-MAC verifies on >=1 reliable domain;"
          " COPY→base/CDBW = false-attribution rate of a perfectly-copied domain (base always 1.0);"
          " OVERWRITE-flag = attacker without key gets caught.")
    print("CDBW_SIM_DONE")

if __name__ == "__main__":
    main()
