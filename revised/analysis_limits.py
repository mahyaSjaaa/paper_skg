"""Phase A2 limits analysis -> limits_results.csv.

(a) per session: oracle best case over a fixed (m, alpha) grid (optimistic, chosen after seeing the data);
(b) multi-session accumulation per site and over all manifest sessions, one global config at a time;
(c) minimum samples for a 128-bit key at eps = 2^-20 from PRE-DECLARED global configs: primary = the Chen et al. rule as
    implemented, sensitivity = fixed (4, 0.0) and (8, 0.0); per-session rate median/min/max; oracle row kept only
    as a labeled optimistic best case.
"""
import contextlib
import csv
import io
import math
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a2_extract as a2  # noqa: E402
import jurnal_revisi as j  # noqa: E402
from run_a1_scenarios import cli, load_sessions, read_rssi  # noqa: E402

PENALTY = 40  # 2 log2(1/eps), eps = 2^-20
KEY_LEN = 128
TAG = 16
GRID = [(m, a) for m in (2, 4, 8) for a in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)]
CONFIGS = [None] + GRID  # None = complexity rule of Chen et al. (m, alpha per block)
PREDECLARED = [None, (4, 0.0), (8, 0.0)]  # primary first, then sensitivity
HEADER = f"""# Phase A2 limits analysis (eps = 2^-20, penalty 2*log2(1/eps) = {PENALTY} bits).
# Multi-session accumulation: (b), (c) pool bits of several sessions (traces) as if they were one session.
# Bounds are asymptotic Slepian-Wolf (n*h(KDR) leakage, no finite-blocklength penalty); Eve is ignored beyond the
# public reconciliation transcript. Bits are mutually accepted labels at the first attempt (NC=0) of the A1 pipeline
# with m and alpha fixed, or the Chen et al. rule; no tag, control or bitmap leakage in the ideal bounds.
# Column prefixes: fe_ = full-entropy bits (1 bit of min-entropy per bit; OPTIMISTIC);
#   mcv_ = MCV entropy H_MCV of Bob's pooled labels (a2_extract.mcv_by_alphabet: itself an optimistic estimate of
#   H_min(X|E): unconditional, IID assumed, not validated by SP 800-90B IID tests, N far below 1e6).
# ideal = H - n*h(KDR) - {PENALTY} (fe: H = n).
# bch = pooled bits cut into 31-bit segments (tail < 31 dropped), real BCH(31,6) code-offset per segment:
#   per successful segment H_seg - 25 sketch bits (fe: H_seg = 31; mcv: H_seg = 31*H_MCV/n), minus 1 ACK bit per segment,
#   minus 1 more bit per ACKed segment, one {TAG}-bit tag per pool, minus {PENALTY}.
# rate = secret bits per sample before the fixed costs; n_min = ceil((128 + {PENALTY} [+ {TAG} for bch]) / rate);
# t_obs = 2 s x n_min (nominal probing interval). inf = rate <= 0 (no key at any length).
"""
COLS = ["section", "label", "m", "alpha", "samples", "bits", "kdr", "H_mcv", "seg_ok", "segments",
        "fe_bound_ideal", "fe_bound_bch", "mcv_bound_ideal", "mcv_bound_bch",
        "fe_rate_ideal", "fe_rate_bch", "mcv_rate_ideal", "mcv_rate_bch",
        "fe_n_min_ideal", "fe_n_min_bch", "mcv_n_min_ideal", "mcv_n_min_bch", "fe_t_obs_min_ideal_s"]
RATES = ("fe_rate_ideal", "fe_rate_bch", "mcv_rate_ideal", "mcv_rate_bch")


def h2(p):
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def run_cfg(A, B, cfg):
    """A1 pipeline at NC=0 with (m, alpha) fixed for every block; cfg None keeps the Chen et al. rule."""
    orig = j.determine_m_from_complexity, j.guard_band
    if cfg is not None:
        j.determine_m_from_complexity = lambda c: cfg[0]
        j.guard_band = lambda c, m: cfg[1]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, st = j.reconciliation_per_block_with_calibration(
                list(A), list(B), NC=0, verbose=False)
    finally:
        j.determine_m_from_complexity, j.guard_band = orig
    return st


def n_min(rate, overhead):
    return math.ceil((KEY_LEN + overhead) / rate) if rate > 0 else float("inf")


def with_n_min(r):
    r["fe_n_min_ideal"] = n_min(r["fe_rate_ideal"], PENALTY)
    r["fe_n_min_bch"] = n_min(r["fe_rate_bch"], PENALTY + TAG)
    r["mcv_n_min_ideal"] = n_min(r["mcv_rate_ideal"], PENALTY)
    r["mcv_n_min_bch"] = n_min(r["mcv_rate_bch"], PENALTY + TAG)
    r["fe_t_obs_min_ideal_s"] = 2.0 * r["fe_n_min_ideal"]
    return r


def pooled(stats_list):
    """Pool first-attempt bits of several sessions: ideal and real BCH(31,6) bounds, full-entropy and MCV."""
    A = np.array([int(b) for st in stats_list for k in st["keys_A_all"] for b in k], dtype=np.uint8)
    B = np.array([int(b) for st in stats_list for k in st["keys_B_all"] for b in k], dtype=np.uint8)
    # Bob's per-sample labels: block k of each session holds labels of log2(m_k) bits
    syms = [kb[i:i + L] for st in stats_list for kb, m in zip(st["keys_B_all"], st["m_values"], strict=True)
            for L in [m.bit_length() - 1] for i in range(0, len(kb), L)]
    H = a2.mcv_by_alphabet(syms)[1]
    n = sum(st["n_valid"] for st in stats_list)
    nb = len(B)
    kdr = float(np.mean(A != B)) if nb else float("nan")
    leak_sw = nb * h2(kdr) if nb else 0.0
    ir = j.InformationReconciliation()
    S = nb // j.BCH_N
    ok = 0
    for s in range(S):
        seg = slice(s * j.BCH_N, (s + 1) * j.BCH_N)
        pub, _ = ir.bob_sketch(B[seg], b"limits", 0, s)
        r = ir.alice_recover(A[seg], pub)
        ok += r is not None and np.array_equal(r, B[seg])
    sketch = (j.BCH_N - j.BCH_K) * ok
    acks = S + ok
    fe_bch = j.BCH_N * ok - sketch - acks
    mcv_bch = (j.BCH_N * ok * H / nb if nb else 0.0) - sketch - acks
    fe_ideal = nb - leak_sw
    mcv_ideal = H - leak_sw
    return with_n_min({
        "samples": n, "bits": nb, "kdr": kdr, "H_mcv": H, "seg_ok": ok, "segments": S,
        "fe_bound_ideal": fe_ideal - PENALTY, "fe_bound_bch": fe_bch - TAG - PENALTY,
        "mcv_bound_ideal": mcv_ideal - PENALTY, "mcv_bound_bch": mcv_bch - TAG - PENALTY,
        "fe_rate_ideal": fe_ideal / n, "fe_rate_bch": fe_bch / n,
        "mcv_rate_ideal": mcv_ideal / n, "mcv_rate_bch": mcv_bch / n})


def cfg_cols(cfg):
    return {"m": "rule_chen" if cfg is None else cfg[0], "alpha": "rule_chen" if cfg is None else cfg[1]}


def fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else ("inf" if math.isinf(v) else f"{v:.4f}")
    return str(v)


def main():
    sessions = load_sessions(**vars(cli()))
    TRACES = [s["scenario"] for s in sessions]
    GROUPS = {site: [s["scenario"] for s in sessions if s["site"] == site] for site in dict.fromkeys(s["site"] for s in sessions)}
    GROUPS[f"all{len(TRACES)}"] = TRACES
    rssi = {s["scenario"]: read_rssi(s["ab_path"]) for s in sessions}
    cache = {(t, cfg): run_cfg(*rssi[t], cfg) for t in TRACES for cfg in CONFIGS}
    rows = []

    # (a) per-session oracle best case over the fixed grid (optimistic: config picked after seeing the data)
    for t in TRACES:
        best = max(((pooled([cache[(t, cfg)]]), cfg) for cfg in GRID), key=lambda x: x[0]["fe_bound_ideal"])
        rows.append({"section": "a_session_oracle_best_optimistic", "label": t, **cfg_cols(best[1]), **best[0]})

    # (b) multi-session accumulation, one global config at a time
    for cfg in CONFIGS:
        for g, ts in GROUPS.items():
            rows.append({"section": "b_group", "label": g, **cfg_cols(cfg), **pooled([cache[(t, cfg)] for t in ts])})

    # (c) pre-declared global configs: pooled rows, per-session rate spread, then the oracle best case
    for i, cfg in enumerate(PREDECLARED):
        sec = "c_predeclared_primary" if i == 0 else "c_predeclared_sensitivity"
        rows += [{**r, "section": sec} for r in rows if r["section"] == "b_group" and r["m"] == cfg_cols(cfg)["m"]
                 and r["alpha"] == cfg_cols(cfg)["alpha"]]
        per = [pooled([cache[(t, cfg)]]) for t in TRACES]
        for stat, f in (("session median", statistics.median), ("session min", min), ("session max", max)):
            rows.append(with_n_min({"section": f"{sec}_session_rates", "label": stat, **cfg_cols(cfg),
                                    **{k: f(p[k] for p in per) if f is not statistics.median else f([p[k] for p in per])
                                       for k in RATES}}))
    oracle = max((r for r in rows if r["section"] in ("a_session_oracle_best_optimistic", "b_group")),
                 key=lambda r: r["fe_rate_ideal"])
    rows.append({**oracle, "section": "c_oracle_best_case_optimistic", "label": f"oracle best case (optimistic): {oracle['label']}"})

    out = os.path.join(HERE, "limits_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write(HEADER)
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: fmt(r.get(k, "")) for k in COLS})
    print(HEADER + ",".join(COLS))
    for r in rows:
        if r["section"].startswith("c_"):
            print(",".join(fmt(r.get(k, "")) for k in COLS))
    print(f"\nwrote {out} ({len(rows)} rows; console shows section c only)")


if __name__ == "__main__":
    main()
