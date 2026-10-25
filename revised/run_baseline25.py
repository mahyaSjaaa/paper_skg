"""Runs the reference method [25] (paper_skg_repo/baseline.py, as pushed by the authors) on the 22 sessions.

Three runs per session:
  pushed   baseline.py unchanged on the raw trace (zeros kept), block 40, NC = 3
  filtered baseline.py unchanged after valid-pair filtering (same input as the proposed method)
  R0       [25] configuration (block 40, no residual, NC = 3) on the revised code-offset
           reconciliation of jurnal_revisi.py, i.e. the ablation starting point
For the pushed reconciliation, Eve's attack uses public data only: Alice's decoder run on Bob's
public chunks with an all-zero guess. Writes baseline25_results.csv.
"""
import contextlib
import csv
import io
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # -I drops the script dir
import numpy as np

import jurnal_revisi as j
from run_a1_scenarios import REPO, cli, load_sessions, read_rssi

with contextlib.chdir(REPO), contextlib.redirect_stdout(io.StringIO()):
    sys.path.insert(0, REPO)
    import baseline as bl  # module top level reads one CSV from the repo root


def run_pushed(A, B):
    """baseline.py reconciliation; returns stats or the failure reason."""
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            key, ok, st = bl.reconciliation_per_block_with_calibration(A, B, delta_alpha=0.2, NC=3, block_size=40)
        except ZeroDivisionError:
            return None, "no output (fewer than 40 samples)"
    # Eve, public data only (no RSSI), on Bob's first-attempt bits of every block:
    #   zero guess: Alice's decoder on Bob's public chunks with an all-zero guess
    #   CRC search: each <= 6-bit chunk is the unique candidate matching its public CRC32
    ir, zero, crc = bl.InformationReconciliation(), 0, 0
    for kb in st["keys_B_all"]:
        KB = bl.to_bits(kb)
        chunks = ir.bob_process_chunked(KB)
        with contextlib.redirect_stdout(io.StringIO()):
            KE, flag = ir.alice_process_chunked(np.zeros_like(KB), chunks, verbose=False)
        zero += flag == 1 and np.array_equal(KE, KB)
        guess = []
        for c in chunks:
            L = c["original_length"]
            hits = [x for x in range(2 ** L)
                    if zlib.crc32(np.array([(x >> (L - 1 - i)) & 1 for i in range(L)], np.uint8).tobytes()) == c["crc"]]
            guess += [(hits[0] >> (L - 1 - i)) & 1 for i in range(L)] if len(hits) == 1 else [2] * L
        crc += np.array_equal(np.array(guess, np.uint8), KB)
    st["eve_zero_guess"], st["eve_crc_search"] = zero, crc
    st["bits"] = len(key)
    return st, ""


def run_r0(A, B, NC=3):
    """[25] layout (40-sample blocks, residual dropped) on the revised reconciliation."""
    a, b, _ = j.filter_valid_pairs(A, B)
    n = len(a) // 40 * 40
    if n == 0:
        return None, "no output (fewer than 40 samples)"
    qbs = j.compute_qbs
    j.compute_qbs = lambda _n: 40
    try:
        key, ok, st = j.reconciliation_per_block_with_calibration(a[:n], b[:n], delta_alpha=0.2, NC=NC)
    finally:
        j.compute_qbs = qbs
    st["bits"] = len(key)
    return st, ""


if __name__ == "__main__":
    args = cli()
    rows = []
    for s in load_sessions(args.manifest, args.base):
        A, B = read_rssi(s["ab_path"])
        a, b, _ = j.filter_valid_pairs(A, B)
        r = {"scenario": s["scenario"], "site": s["site"], "link": s["link"], "n_raw": len(A), "n_valid": len(a)}
        for tag, (st, err) in (("pushed", run_pushed(list(A), list(B))), ("filtered", run_pushed(list(a), list(b)))):
            r[f"{tag}_blocks"] = st["total_blocks"] if st else 0
            r[f"{tag}_ok"] = st["successful_blocks"] if st else 0
            r[f"{tag}_kdr_pre"] = f"{st['overall_kdr']:.4f}" if st else ""
            r[f"{tag}_bits"] = st["bits"] if st else 0
            r[f"{tag}_post_err"] = st["total_error_after"] if st else ""
            r[f"{tag}_eve_zero_guess"] = st["eve_zero_guess"] if st else ""
            r[f"{tag}_eve_crc_search"] = st["eve_crc_search"] if st else ""
            r[f"{tag}_note"] = err
        st, err = run_r0(A, B)
        r.update({"r0_blocks": st["total_blocks"] if st else 0, "r0_ok": st["successful_blocks"] if st else 0,
                  "r0_kdr_pre": f"{st['overall_kdr']:.4f}" if st else "", "r0_bits": st["bits"] if st else 0,
                  "r0_note": err})
        rows.append(r)
    with open(os.path.join(HERE, "baseline25_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["scenario"], r["n_raw"], r["n_valid"],
              f"pushed {r['pushed_ok']}/{r['pushed_blocks']} kdr {r['pushed_kdr_pre']} bits {r['pushed_bits']} post {r['pushed_post_err']} eve0 {r['pushed_eve_zero_guess']} eveCRC {r['pushed_eve_crc_search']} {r['pushed_note']}|",
              f"filt {r['filtered_ok']}/{r['filtered_blocks']} kdr {r['filtered_kdr_pre']} bits {r['filtered_bits']} eve {r['filtered_eve_zero_guess']}|",
              f"R0 {r['r0_ok']}/{r['r0_blocks']} kdr {r['r0_kdr_pre']} bits {r['r0_bits']} {r['r0_note']}")
