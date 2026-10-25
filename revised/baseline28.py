"""Reference method [28] (Kurniawan, Yuliana, Sudarsono, IES 2025) re-implemented on the 22 sessions.

Pipeline as stated in [28]: scalar Kalman filter on each node's RSSI -> uniform multibit quantization
(Algorithm 2: N = ceil(sqrt(Range)) bits if Range < 16 dB, else N = 3) with Gray labels -> BCH reconciliation
-> universal hash. Choices this file makes where [28] is silent or inconsistent (stated in Table 5):
  * Kalman: a = h = 1, Q = 0.01, R = 1 (the authors' code), x0 = first sample, p0 = 1.
  * N follows Algorithm 2 (the text says 5 and Table IV 12 bits per sample); Alice announces her N (4 bits).
  * BCH(255,7) is not a BCH code; the BCH code with the stated t = 63 at n = 255 is BCH(255,9), used here.
  * Reconciliation uses the code-offset construction of Section 3.5 (random codeword, 16-bit tag), one
    attempt, so leakage is accounted for exactly as for the proposed method.
  * The whole session is one block ([28] has no partitioning and no guard band).
Writes baseline28_results.csv.   Self-check: python -I baseline28.py --check
"""
import csv
import functools
import math
import os
import secrets
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # -I drops the script dir
import numpy as np

import a2_extract as a2
import jurnal_revisi as j
from run_a1_scenarios import cli, load_sessions, read_rssi

N_CODE, T_CODE, M_GF, PRIM_GF = 255, 63, 8, 0x11D  # GF(256), x^8 + x^4 + x^3 + x^2 + 1
KF = dict(a=1.0, h=1.0, Q=0.01, R=1.0, p0=1.0)
N_ANNOUNCE_BITS = 4


@functools.lru_cache(maxsize=None)
def bch255_codebook():
    """Narrow-sense BCH(255, k) with designed distance 2t+1 = 127; returns (k, codebook rows MSB first)."""
    exp, x = [], 1
    for _ in range(N_CODE):
        exp.append(x)
        x <<= 1
        if x & 0x100:
            x ^= PRIM_GF
    log = {v: i for i, v in enumerate(exp)}
    mul = lambda a, b: 0 if a == 0 or b == 0 else exp[(log[a] + log[b]) % N_CODE]
    g, done = 1, set()
    for i in range(1, 2 * T_CODE + 1):  # roots alpha^1 .. alpha^126
        if i in done:
            continue
        coset = {(i << s) % N_CODE for s in range(M_GF)}
        done |= coset
        poly = [1]
        for r in coset:
            ar = exp[r]
            poly = [(poly[s - 1] if s > 0 else 0) ^ (mul(poly[s], ar) if s < len(poly) else 0)
                    for s in range(len(poly) + 1)]
        g = j._gf2_mul(g, sum(c << s for s, c in enumerate(poly)))
    k = N_CODE - (g.bit_length() - 1)
    words = [(r << (N_CODE - k)) | j._gf2_mod(r << (N_CODE - k), g) for r in range(1 << k)]
    if len(set(words)) != 1 << k or min(w.bit_count() for w in words if w) < 2 * T_CODE + 1:
        raise RuntimeError("BCH(255,k) codebook check failed")
    book = np.array([[(w >> (N_CODE - 1 - i)) & 1 for i in range(N_CODE)] for w in words], dtype=np.uint8)
    book.flags.writeable = False
    return k, book


def decode(y):
    _, book = bch255_codebook()
    dist = (book ^ y).sum(axis=1)
    i = int(np.argmin(dist))
    return i if dist[i] <= T_CODE else None


def reconcile(KA, KB, randbits=secrets.randbits, tag_bits=16):
    """One code-offset attempt over 255-bit segments; returns (KB' or None, leak bits)."""
    k, book = bch255_codebook()
    L = len(KB)
    S = -(-L // N_CODE)
    pad = lambda v: np.concatenate([np.asarray(v, np.uint8), np.zeros(S * N_CODE - L, np.uint8)]).reshape(S, N_CODE)
    sketch = [seg ^ book[randbits(k)] for seg in pad(KB)]
    leak = (N_CODE - k) * S + 1  # sketch + ACK/NACK
    out = []
    for y, s in zip(pad(KA), sketch):
        i = decode(y ^ s)
        if i is None:
            return None, leak
        out.append(s ^ book[i])
    KBp = np.concatenate(out)[:L]
    leak += tag_bits + 1  # tag after ACK + accept/reject
    tag = lambda v: j.InformationReconciliation(tag_bits=tag_bits).tag_of(v, b"b28", 0, 0)
    return (KBp if tag(KBp) == tag(KB) else None), leak


def kalman(z):
    x, p, out = float(z[0]), KF["p0"], []
    for zi in z:
        xp, pp = KF["a"] * x, KF["a"] * p * KF["a"] + KF["Q"]
        g = pp * KF["h"] / (KF["h"] * pp * KF["h"] + KF["R"])
        x, p = xp + g * (zi - KF["h"] * xp), (1 - g * KF["h"]) * pp
        out.append(x)
    return np.array(out)


def n_bits(r):
    rng = r.max() - r.min()
    return math.ceil(math.sqrt(rng)) if rng < 16 else 3


def quantize(r, N):
    """Algorithm 2 of [28]: 2^N equal intervals over [min, max], Gray label per sample."""
    lo, levels = r.min(), 1 << N
    width = (r.max() - lo) / levels
    idx = np.full(len(r), levels - 1) if width == 0 else np.clip(np.floor((r - lo) / width), 0, levels - 1).astype(int)
    idx[r == r.max()] = levels - 1
    return [format(i ^ (i >> 1), f"0{N}b") for i in idx]


def session(A, B, randbits=secrets.randbits):
    a, b, n_raw = j.filter_valid_pairs(A, B)
    fa, fb = kalman(np.array(a, float)), kalman(np.array(b, float))
    N = n_bits(fa)  # Alice announces N; Bob quantizes with the same N
    la, lb = quantize(fa, N), quantize(fb, N)
    KA, KB = j.to_bits("".join(la)), j.to_bits("".join(lb))
    KBp, leak = reconcile(KA, KB, randbits)
    leak += N_ANNOUNCE_BITS
    ok = KBp is not None
    H = a2.mcv_by_alphabet(lb)[1] if ok else 0.0
    rec_bits = len(KB) if ok else 0
    l = {e: a2.extractable_length(H, leak if ok else 0, 2.0 ** -e) for e in a2.EPS_EXPONENTS}
    T_obs = a2.PROBE_INTERVAL_S * n_raw
    raw = np.array(a, float)
    lr = quantize(raw, n_bits(raw))  # same quantizer on unfiltered RSSI, for the label-dependence comparison
    repeat = lambda lab: float(np.mean([x == y for x, y in zip(lab[1:], lab[:-1])]))
    return {"label_repeat_kalman": repeat(la), "label_repeat_raw": repeat(lr), "n_raw": n_raw, "n_valid": len(a), "r_kalman": float(np.corrcoef(fa, fb)[0, 1]),
            "N_alice": N, "N_bob_own": n_bits(fb), "bits": len(KB), "kdr_pre": float((KA != KB).mean()),
            "reconciled": int(ok), "reconciled_bits": rec_bits, "leak_bits": leak if ok else 0, "H_total": H,
            "l_eps10": l[10], "l_eps20": l[20], "l_eps32": l[32], "SKR": max(l[20], 0) / T_obs}


def selfcheck():
    k, book = bch255_codebook()
    assert k == 9 and book.shape == (512, 255)
    rng = np.random.default_rng(1)
    KB = rng.integers(0, 2, 300, dtype=np.uint8)
    for e, expect in ((0, True), (63, True)):
        KA = KB.copy()
        KA[rng.choice(255, e, replace=False)] ^= 1  # errors in the first segment only
        KBp, _ = reconcile(KA, KB)
        assert (KBp is not None and np.array_equal(KBp, KB)) == expect, e
    KA = KB.copy()
    KA[:80] ^= 1
    assert reconcile(KA, KB)[0] is None  # 80 > t errors: rejected, never a wrong key
    assert quantize(np.array([0.0, 1, 2, 3]), 2) == ["00", "01", "11", "10"]
    print("baseline28 self-check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        selfcheck()
        sys.exit(0)
    args = cli()
    rows = []
    for s in load_sessions(args.manifest, args.base):
        A, B = read_rssi(s["ab_path"])
        r = {"scenario": s["scenario"], "site": s["site"], "link": s["link"]}
        r.update(session(A, B))
        rows.append(r)
        print({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(HERE, "baseline28_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    med = lambda k: float(np.median([r[k] for r in rows]))
    print(f"median share of labels equal to the previous one: Kalman {med('label_repeat_kalman'):.2f}, raw {med('label_repeat_raw'):.2f}")
