"""Phase A1: run reconciliation (NC=1) on the sessions of a manifest (default sessions.csv); print a table, write a1_results.csv."""
import argparse
import contextlib
import csv
import io
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jurnal_revisi as j  # noqa: E402

MANIFEST = os.path.join(HERE, "sessions.csv")
# manifest paths are relative to the data repository: the parent folder when this code sits inside it, else a sibling clone
_PARENT = os.path.dirname(HERE)
REPO = _PARENT if os.path.isdir(os.path.join(_PARENT, "data_mentah_mashabib")) else os.path.join(_PARENT, "paper_skg_repo")
COLS = ["scenario", "site", "link", "n_raw", "n_valid", "qbs", "total_blocks", "residual_block", "kdr_pre", "r_rec",
        "sample_discard_ratio", "successful_blocks", "reconciled_bits", "post_kdr", "leak_EC_bits", "leak_EC_bits_conservative",
        "leak_EC_bits_key_blocks", "naive_upper_bound", "error"]


def cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST, help="CSV: scenario,site,link,...,ab_file,eve_file,eve_rows_match")
    ap.add_argument("--base", default=REPO, help="directory the manifest file paths are relative to")
    return ap.parse_args()


def load_sessions(manifest=MANIFEST, base=REPO):
    """Manifest rows with absolute ab_path / eve_path added."""
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["ab_path"], r["eve_path"] = os.path.join(base, r["ab_file"]), os.path.join(base, r["eve_file"])
    return rows


def read_rssi(path, cols=("rssi_alice", "rssi_bob")):
    """Only the named columns are used; extra Unnamed columns are ignored. Some files carry a BOM."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    assert set(cols) <= set(df.columns), f"{path}: columns {list(df.columns)}"
    return tuple(df[c].values for c in cols)


def run(path):
    A, B = read_rssi(path)
    with contextlib.redirect_stdout(io.StringIO()):
        key, _, st = j.reconciliation_per_block_with_calibration(A, B, NC=1, verbose=False)
    bits = len(key)
    return {
        "n_raw": st["n_raw"], "n_valid": st["n_valid"], "qbs": st["qbs"],
        "total_blocks": st["total_blocks"], "residual_block": st["residual_block"],
        "kdr_pre": st["overall_kdr"], "r_rec": st["r_rec"],
        "sample_discard_ratio": st["sample_discard_ratio"], "successful_blocks": st["successful_blocks"],
        "reconciled_bits": bits, "post_kdr": st["post_kdr"], "leak_EC_bits": st["leak_EC_bits"],
        "leak_EC_bits_conservative": st["leak_EC_bits_conservative"],
        "leak_EC_bits_key_blocks": st["leak_EC_bits_key_blocks"],
        "naive_upper_bound": bits - st["leak_EC_bits_key_blocks"] - 40,
    }


def fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.4f}"
    return "" if v is None else str(v)


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

    hdr = ["scenario", "site", "link", "n_raw", "n_valid", "qbs", "blocks", "resid", "kdr_pre", "r_rec", "discard", "ok",
           "bits", "post_kdr", "leak", "leak_cons", "leak_key", "UPPER_BOUND(bits-leak_key-40)"]
    keys = COLS[:-1]
    w = [8, 13, 4, 6, 7, 4, 6, 5, 7, 7, 7, 4, 6, 8, 6, 9, 8, 29]
    print("naive_upper_bound is an UPPER BOUND only (quantized bits are not full-entropy)")
    print(" ".join(h.ljust(x) for h, x in zip(hdr, w)))
    for r in rows:
        if r.get("error"):
            print(f"{r['scenario']:<8} {r['error']}")
        else:
            print(" ".join(fmt(r[k]).ljust(x) for k, x in zip(keys, w)))

    out = os.path.join(HERE, "a1_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: fmt(r.get(k)) for k in COLS})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
