"""Passive-eavesdropper evaluation for the sessions whose Eve file has one row per probing round.

Row i of the Eve file and row i of the Alice/Bob trace are the same probing round (confirmed by
the authors). Eve's column alice_rssi is her RSSI of Alice's packets (the packets Bob also
measures), bob_rssi her RSSI of Bob's packets. Eve follows the public protocol: she uses the
public block layout, m_k and alpha_k, quantizes her own samples with her own thresholds, keeps
Bob's accepted indices, and runs Alice's decoder on Bob's public sketches with her bits.
Writes eve_results.csv.
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import numpy as np
import pandas as pd

import jurnal_revisi as j
from run_a1_scenarios import cli, load_sessions, read_rssi


def valid(*xs):
    return all(not pd.isna(x) and x != 0 for x in xs)


def agreement(lab_x, lab_y, kept):
    """Bitwise agreement of two label lists at the kept indices where both labels exist."""
    same = tot = 0
    for i in kept:
        if lab_x[i] is None or lab_y[i] is None:
            continue
        same += sum(a == b for a, b in zip(lab_x[i], lab_y[i]))
        tot += len(lab_x[i])
    return same, tot


def session(r, seed=1):
    A, B = read_rssi(r["ab_path"])
    Ea, Eb = read_rssi(r["eve_path"], cols=("alice_rssi", "bob_rssi"))
    assert len(Ea) == len(A), "Eve file must have one row per probing round"
    rows3 = [i for i in range(len(A)) if valid(A[i], B[i], Ea[i], Eb[i])]
    corr = lambda x, y: float(np.corrcoef([x[i] for i in rows3], [y[i] for i in rows3])[0, 1])
    out = {"scenario": r["scenario"], "site": r["site"], "link": r["link"], "rounds": len(A),
           "rounds_all_valid": len(rows3), "r_alice_bob": corr(A, B), "r_bob_eveA": corr(B, Ea),
           "r_alice_eveA": corr(A, Ea), "r_alice_eveB": corr(A, Eb), "r_bob_eveB": corr(B, Eb)}

    # legitimate layout exactly as in the pipeline: valid Alice/Bob rounds only
    idx = [i for i in range(len(A)) if valid(A[i], B[i])]
    n = len(idx)
    qbs = j.compute_qbs(n)
    ranges = [(k * qbs, (k + 1) * qbs) for k in range(n // qbs)]
    if n - (n // qbs) * qbs >= 10:
        ranges.append(((n // qbs) * qbs, n))
    ir = j.InformationReconciliation(randbits=random.Random(seed).getrandbits)
    nonce = b"eve-eval"
    agree_eb = tot_eb = agree_ab = tot_ab = 0
    rec_blocks = eve_exact = one_seg = 0
    for k, (s, e) in enumerate(ranges):
        a = [A[idx[t]] for t in range(s, e)]
        b = [B[idx[t]] for t in range(s, e)]
        ev = [Ea[idx[t]] for t in range(s, e)]
        c = j.calculate_complexity_for_block(a)
        m = j.determine_m_from_complexity(c)
        alpha = j.guard_band(c, m)
        ev_ok = [x for x in ev if valid(x)]
        for att in range(2):  # N_c = 1
            la = j.kuantisasi_per_block(a, alpha, m)
            lb = j.kuantisasi_per_block(b, alpha, m)
            # Eve quantizes the samples she has; a missing round gives no label
            le_valid = j.kuantisasi_per_block(ev_ok, alpha, m) if len(ev_ok) >= 2 else []
            it = iter(le_valid)
            le = [next(it) if valid(x) else None for x in ev]
            _, kb, kept = j.mutual_accept(la, lb)
            if att == 0:
                x, t = agreement(le, lb, kept); agree_eb += x; tot_eb += t
                x, t = agreement(la, lb, kept); agree_ab += x; tot_ab += t
            KA = j.to_bits(''.join(la[i] for i in kept))
            KB = j.to_bits(kb)
            KE = j.to_bits(''.join(le[i] if le[i] is not None else '0' * len(lb[i]) for i in kept))
            if len(KB) == 0:
                alpha += 0.2
                continue
            public, tag = ir.bob_sketch(KB, nonce, k, att)
            KA2 = ir.alice_recover(KA, public)
            if KA2 is not None and ir.tag_of(KA2, nonce, k, att) == tag:
                rec_blocks += 1
                one_seg += len(KB) <= j.BCH_N
                KE2 = ir.alice_recover(KE, public)  # Eve: same decoder, her own bits
                eve_exact += KE2 is not None and np.array_equal(KE2, KB)
                break
            alpha += 0.2
    out.update({"bob_bits_first_attempt": tot_ab, "alice_bob_agreement": agree_ab / tot_ab if tot_ab else float("nan"),
                "eve_bits_compared": tot_eb, "eve_bob_agreement": agree_eb / tot_eb if tot_eb else float("nan"),
                "reconciled_blocks": rec_blocks, "eve_recovered_with_own_bits": eve_exact,
                "one_segment_blocks_recoverable_from_transcript": one_seg})
    return out


if __name__ == "__main__":
    args = cli()
    rows = [session(r) for r in load_sessions(args.manifest, args.base) if r["eve_rows_match"] == "yes"]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eve_results.csv"), "w", newline="", encoding="utf-8") as f:
        f.write("# Eve rows aligned with Alice/Bob by probing round (confirmed by the authors). "
                "eveA = Eve's RSSI of Alice's packets, eveB = of Bob's packets. Agreement at Bob's accepted "
                "indices, first attempt. Recovery: Alice's decoder run on Bob's public sketch with Eve's bits.\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(r["scenario"], f"rAB={r['r_alice_bob']:.3f} rB-EveA={r['r_bob_eveA']:.3f} rA-EveA={r['r_alice_eveA']:.3f}",
              f"agrAB={r['alice_bob_agreement']:.3f} agrEB={r['eve_bob_agreement']:.3f} ({r['eve_bits_compared']} bits)",
              f"rec={r['reconciled_blocks']} eve_exact={r['eve_recovered_with_own_bits']} one_seg={r['one_segment_blocks_recoverable_from_transcript']}")
