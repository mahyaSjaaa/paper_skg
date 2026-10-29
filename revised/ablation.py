"""Component ablation (Sec. 5.3) and configuration R0 of Chen et al. [25] (Sec. 5.4) -> ablation_results.csv, r0_results.csv.

R0 is the block layout of [25] (40-sample blocks, remainder dropped) on the revised code-offset reconciliation;
CSP is the proposed partitioning. Both run with NC = 1 and NC = 3. The bound on l treats every reconciled bit
as full-entropy (reconciled bits - leakage - 40); r0_results.csv adds l from the MCV estimate (extractable-length bound l).
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # -I drops the script dir
import a2_extract as a2
import jurnal_revisi as j
from run_a1_scenarios import cli, load_sessions, read_rssi


def r0(A, B, NC):
    """[25] layout: 40-sample blocks, residual dropped; None if fewer than 40 valid pairs."""
    a, b, _ = j.filter_valid_pairs(A, B)
    n = len(a) // 40 * 40
    if n == 0:
        return None
    qbs, j.compute_qbs = j.compute_qbs, lambda _n: 40
    try:
        _, _, st = j.reconciliation_per_block_with_calibration(a[:n], b[:n], NC=NC)
    finally:
        j.compute_qbs = qbs
    st["n_raw"] = len(A)  # T_obs of Eq. (5) uses the raw session length
    return st


def csp(A, B, NC):
    return j.reconciliation_per_block_with_calibration(A, B, NC=NC)[2]


CONFIGS = [("R0", 3, r0), ("R0", 1, r0), ("CSP", 1, csp), ("CSP", 3, csp)]

if __name__ == "__main__":
    args = cli()
    data = [(s["scenario"], *read_rssi(s["ab_path"])) for s in load_sessions(args.manifest, args.base)]
    rows, r0rows = [], []
    for name, NC, fn in CONFIGS:
        for sc, A, B in data:
            st = fn(A, B, NC)
            if st is None:
                rows.append({"config": name, "NC": NC, "scenario": sc, "note": "not applicable"})
                continue
            bits = len(st["final_key_bob"])
            leak = st["leak_EC_bits_key_blocks"]
            rows.append({"config": name, "NC": NC, "scenario": sc, "blocks": st["total_blocks"],
                         "ok": st["successful_blocks"], "quantized_bits": st["total_bits_quantization"],
                         "errors_pre": st["total_errors_quantization"], "reconciled_bits": bits,
                         "leak_key_blocks": leak, "l_bound_full_entropy": bits - leak - 40, "note": ""})
            if (name, NC) == ("R0", 3):
                pa = a2.privacy_amplification_eq5(st)
                r0rows.append({"scenario": sc, "blocks": st["total_blocks"], "ok": st["successful_blocks"],
                               "bits": bits, "leak": leak, "H": round(pa["H_total"], 2),
                               "l10": pa["l"][10], "l20": pa["l"][20], "l32": pa["l"][32]})
    for path, rs in (("ablation_results.csv", rows), ("r0_results.csv", r0rows)):
        keys = list(dict.fromkeys(k for r in rs for k in r))
        with open(os.path.join(HERE, path), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)
    for name, NC, _ in CONFIGS:
        rs = [r for r in rows if r["config"] == name and r["NC"] == NC and not r["note"]]
        E, Q = sum(r["errors_pre"] for r in rs), sum(r["quantized_bits"] for r in rs)
        print(f"{name} NC={NC}: sessions {len(rs)}, KDR_pre {100 * E / Q:.2f}%, "
              f"R_rec {sum(r['ok'] for r in rs)}/{sum(r['blocks'] for r in rs)}, bits {sum(r['reconciled_bits'] for r in rs)}, "
              f"sessions with bits {sum(r['reconciled_bits'] > 0 for r in rs)}, leak {sum(r['leak_key_blocks'] for r in rs)}, "
              f"max bound {max(r['l_bound_full_entropy'] for r in rs)}")
    print("R0 NC=3 l(2^-20) range:", min(r["l20"] for r in r0rows), "to", max(r["l20"] for r in r0rows))
