"""Phase A2: Eq. 5 extractable length, SKR and NIST diagnostics (pre-extraction bits) on the manifest sessions -> a2_results.csv."""
import contextlib
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a2_extract as a2  # noqa: E402
import jurnal_revisi as j  # noqa: E402
from run_a1_scenarios import cli, fmt, load_sessions, read_rssi  # noqa: E402

COLS = ["scenario", "site", "link", "n_raw", "reconciled_bits", "symbols", "h_per_symbol", "H_total", "leak_key", "leak_cons",
        "l_eps10", "l_eps20", "l_eps32", "l20_allblocks", "l_cons_eps20", "key_bits", "SKR_eps20", "T_obs", "nist_applicable",
        "nist_pvalues", "error"]


def run(path):
    A, B = read_rssi(path)
    with contextlib.redirect_stdout(io.StringIO()):
        key, _, st = j.reconciliation_per_block_with_calibration(A, B, NC=1, verbose=False)
    pa = a2.privacy_amplification_eq5(st)
    nist = a2.nist_diagnostics(key)
    app = [k for k, v in nist.items() if v['applicable']]
    return {
        "n_raw": st["n_raw"], "reconciled_bits": len(key), "symbols": len(st["final_symbols_bob"]),
        "h_per_symbol": ";".join(f"m{m}:{g['h']:.3f}(N={g['N']})" for m, g in pa['h_per_symbol'].items()),
        "H_total": pa['H_total'], "leak_key": pa['leak_EC_bits_key_blocks'],
        "leak_cons": pa['leak_EC_bits_conservative'],
        "l_eps10": pa['l'][10], "l_eps20": pa['l'][20], "l_eps32": pa['l'][32],
        "l20_allblocks": pa['l_allblocks'][20], "l_cons_eps20": pa['l_conservative'][20], "key_bits": len(pa['key_bits']),
        "SKR_eps20": pa['SKR'], "T_obs": pa['T_obs'], "nist_applicable": f"{len(app)}/{len(nist)}",
        "nist_pvalues": ";".join(f"{k}={nist[k]['p_value']:.4f}" for k in app),
    }


def main():
    args = cli()
    rows = []
    for s in load_sessions(args.manifest, args.base):
        row = {k: s[k] for k in ("scenario", "site", "link")}
        try:
            row.update(run(s["ab_path"]))
        except ValueError as e:
            row["error"] = f"ValueError: {e}"
        rows.append(row)

    print("H_total: MCV, optimistic estimate of H_min(X|E) (unconditional, IID assumed, not validated by "
          "SP 800-90B IID tests; N far below 1e6); l <= 0 means no key")
    w = [8, 13, 4, 5, 5, 4, 22, 7, 5, 5, 6, 6, 6, 6, 5, 7, 6]
    print(" ".join(h.ljust(x) for h, x in zip(COLS[:17], w)))
    for r in rows:
        if r.get("error"):
            print(f"{r['scenario']:<8} {r['error']}")
        else:
            print(" ".join(fmt(r[k]).ljust(x) for k, x in zip(COLS[:17], w)) + f" {r['nist_applicable']} {r['nist_pvalues']}")

    out = os.path.join(HERE, "a2_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: fmt(r.get(k)) for k in COLS})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
