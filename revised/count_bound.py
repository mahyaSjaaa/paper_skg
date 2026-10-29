"""Size bound on the extractable length, from message sizes and reconciled lengths only -> count_bound_results.csv.

For any source distribution and any eavesdropper, H_min(X|E) <= N, the number of reconciled bits. Replacing the
min-entropy in the extractable length l by N therefore gives an upper bound on l that needs no entropy estimate:

    l <= N - leak - 2 log2(1/eps)

where leak is the number of bits charged to the leakage (exact counts of the public messages, transcripts.py).
The script evaluates this bound per session for the three leakage totals and the three values of eps, and checks
per reconciled block k the counting inequality  L_k - leak_k <= 6 G_k - 22,  where L_k <= 31 G_k is the block
length, G_k its number of 31-bit segments, and leak_k >= 25 G_k + 4 + 1 + 17 (sketch, LZ-76 count, ACK, tag and
decision bit). Every block holds at most 45 bits (G_k <= 2), so each reconciled block contributes at most
-16 bits (G_k = 1) or -27 bits (G_k = 2), and the bound is negative for every session, whatever the data.

Also checks N and leak against a2_results.csv. Usage: python -I revised/count_bound.py
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcripts import HERE, MANIFEST, REPO, Recorder, load_sessions, read_rssi  # noqa: E402

EPS = (10, 20, 32)
COLS = (["scenario", "reconciled_bits", "reconciled_blocks", "segments", "leak_key", "leak_all", "leak_bitmap_incl"]
        + [f"size_bound_eps{e}" for e in EPS] + ["size_bound_eps20_allblocks", "size_bound_eps20_bitmap_incl",
                                                 "max_block_margin"])


def session(path):
    rec = Recorder(1)
    st = rec.run(*read_rssi(path))
    N = leak_key = segs = 0
    margins = []
    for b in st["blocks_data"]:
        if b["status"] != "SUCCESS":
            continue
        k = b["block_id"]
        rows = [r for r in rec.rows if r["block"] == k]
        L = len(b["key_after"])
        leak_k = sum(r["charged_to_leak_EC"] for r in rows)
        # segments of the accepted attempt: the last attempt of the block
        G = sum(r["message"] == "code-offset sketch segment" and r["attempt"] == b["calibrations"] for r in rows)
        assert L <= 31 * G and G <= 2, (k, L, G)
        assert L - leak_k <= 6 * G - 22, (k, L, leak_k, G)
        assert L - leak_k <= min(31 * G, 45) - 25 * G - 22 <= -16, (k, L, leak_k, G)  # G=1: -16, G=2: -27
        N, leak_key, segs = N + L, leak_key + leak_k, segs + G
        margins.append(L - leak_k)
    bitmap = sum(r["bitmap_bits"] for r in rec.rows)
    out = {"reconciled_bits": N, "reconciled_blocks": len(margins), "segments": segs, "leak_key": leak_key,
           "leak_all": st["leak_EC_bits"], "leak_bitmap_incl": st["leak_EC_bits"] + bitmap,
           "max_block_margin": max(margins) if margins else ""}
    for e in EPS:
        out[f"size_bound_eps{e}"] = N - leak_key - 2 * e
    out["size_bound_eps20_allblocks"] = N - st["leak_EC_bits"] - 40
    out["size_bound_eps20_bitmap_incl"] = N - out["leak_bitmap_incl"] - 40
    return out


def main():
    with open(os.path.join(HERE, "a2_results.csv"), newline="", encoding="utf-8") as f:
        ref = {r["scenario"]: r for r in csv.DictReader(f)}
    rows = []
    for s in load_sessions(MANIFEST, REPO):
        r = {"scenario": s["scenario"], **session(s["ab_path"])}
        a = ref[s["scenario"]]
        assert r["reconciled_bits"] == int(a["reconciled_bits"]), (s["scenario"], r["reconciled_bits"])
        assert r["leak_key"] == int(a["leak_key"]), (s["scenario"], r["leak_key"], a["leak_key"])
        assert all(r[f"size_bound_eps{e}"] < 0 for e in EPS), s["scenario"]
        # the MCV-based l can never exceed the size bound
        assert math.floor(float(a["l_eps20"])) <= r["size_bound_eps20"], s["scenario"]
        rows.append(r)
    with open(os.path.join(HERE, "count_bound_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    with_bits = [r for r in rows if r["reconciled_bits"] > 0]
    print(f"{len(rows)} sessions, {len(with_bits)} with reconciled bits, "
          f"{sum(r['reconciled_blocks'] for r in rows)} reconciled blocks")
    for e in EPS:
        v = [r[f"size_bound_eps{e}"] for r in with_bits]
        print(f"size bound at eps = 2^-{e}: {max(v)} to {min(v)} bits (sessions with reconciled bits)")
    print("before the 2 log2(1/eps) term:", max(r["reconciled_bits"] - r["leak_key"] for r in with_bits), "bits at most")
    print("max block margin L_k - leak_k:", max(r["max_block_margin"] for r in with_bits), "bits")
    print("all-block, bitmap-inclusive at 2^-20:",
          max(r["size_bound_eps20_allblocks"] for r in rows), max(r["size_bound_eps20_bitmap_incl"] for r in rows))


if __name__ == "__main__":
    main()
