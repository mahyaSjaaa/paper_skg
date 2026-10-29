"""Public-channel transcript of every session: one row per message that Alice or Bob sends in the open.

Writes transcripts/<scenario>.csv. Each row gives the sender, the message, the block and attempt it belongs to,
its size, the bits charged to leak_EC, the bits counted only in the bitmap-inclusive leakage, and the value sent.
The random choices (session nonce, code-offset codewords) come from a seeded generator, so a rerun with the same
--seed writes the same files; the pass/fail outcome of every block does not depend on the seed.

Self-check (always run): per session, the charged bits equal leak_EC_bits of a1_results.csv, the bitmap bits equal
leak_EC_bits_conservative - leak_EC_bits, and the number of accepted blocks equals successful_blocks.

Usage: python -I revised/transcripts.py [--seed 1]
"""
import argparse
import contextlib
import csv
import io
import os
import random
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_a1_scenarios import HERE, MANIFEST, REPO, j, load_sessions, read_rssi

OUT = os.path.join(HERE, "transcripts")
COLS = ["seq", "sender", "message", "block", "attempt", "size_bits", "charged_to_leak_EC", "bitmap_bits", "value"]


def bitstr(a):
    return "".join(str(int(b)) for b in a)


class Recorder:
    """Wraps the public steps of jurnal_revisi without changing what they compute."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.rows, self.block, self.attempt = [], -1, -1

    def emit(self, sender, message, size, charged=0, bitmap=0, value=""):
        self.rows.append({"seq": len(self.rows) + 1, "sender": sender, "message": message,
                          "block": "" if self.block < 0 else self.block,
                          "attempt": "" if self.attempt < 0 else self.attempt,
                          "size_bits": size, "charged_to_leak_EC": charged, "bitmap_bits": bitmap, "value": value})

    def run(self, A, B):
        rec, orig = self, {k: getattr(j, k) for k in
                           ("filter_valid_pairs", "calculate_complexity_for_block", "mutual_accept",
                            "InformationReconciliation", "secrets")}

        def filter_valid_pairs(rA, rB):
            out = orig["filter_valid_pairs"](rA, rB)
            bad = [i for i, (a, b) in enumerate(zip(rA, rB)) if not (a == a and b == b and a != 0 and b != 0)]
            rec.emit("Both", "invalid round indices", "variable", value=";".join(map(str, bad)))
            return out

        def token_bytes(n):
            v = rec.rng.randbytes(n)
            rec.emit("Alice", "session nonce", 8 * n, value=v.hex())
            return v

        def complexity(block):
            rec.block, rec.attempt = rec.block + 1, -1
            c = orig["calculate_complexity_for_block"](block)
            nu = len(j.lz76_decomposition(block)[1])
            rec.emit("Alice", "LZ-76 component count", 4, charged=4, value=nu)
            return c

        def mutual_accept(lA, lB):
            rec.attempt += 1
            for who, lab in (("Alice", lA), ("Bob", lB)):
                rec.emit(who, "acceptance bitmap", len(lab), bitmap=len(lab),
                         value="".join("0" if x is None else "1" for x in lab))
            return orig["mutual_accept"](lA, lB)

        class IR(orig["InformationReconciliation"]):
            def __init__(self):
                super().__init__(randbits=rec.rng.getrandbits)

            def reconcile(self, KA, KB, nonce, block_idx, attempt):
                # same steps and leakage as InformationReconciliation.reconcile, with each message recorded
                KB = np.asarray(KB, dtype=np.uint8)
                S = -(-len(KB) // j.BCH_N)
                if S == 0:
                    return None, {'segments': 0, 'sketch': 0, 'ack': 0, 'tag': 0}
                public, tag = self.bob_sketch(KB, nonce, block_idx, attempt)
                for s in public["s"]:
                    rec.emit("Bob", "code-offset sketch segment", j.BCH_N, charged=j.BCH_N - j.BCH_K, value=bitstr(s))
                KB_prime = self.alice_recover(KA, public)
                leak = {'segments': S, 'sketch': (j.BCH_N - j.BCH_K) * S, 'ack': 1, 'tag': 0}
                if KB_prime is None:
                    rec.emit("Alice", "NACK", 1, charged=1, value=0)
                    return None, leak
                rec.emit("Alice", "ACK", 1, charged=1, value=1)
                rec.emit("Bob", "verification tag", self.tag_bits, charged=self.tag_bits, value=f"{tag:04x}")
                ok = self.tag_of(KB_prime, nonce, block_idx, attempt) == tag
                rec.emit("Alice", "accept" if ok else "reject", 1, charged=1, value=int(ok))
                leak['tag'], leak['ack'] = self.tag_bits, 2
                return (KB_prime if ok else None), leak

        patch = {"filter_valid_pairs": filter_valid_pairs, "calculate_complexity_for_block": complexity,
                 "mutual_accept": mutual_accept, "InformationReconciliation": IR,
                 "secrets": types.SimpleNamespace(token_bytes=token_bytes)}
        try:
            for k, v in patch.items():
                setattr(j, k, v)
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, st = j.reconciliation_per_block_with_calibration(A, B, NC=1, verbose=False)
        finally:
            for k, v in orig.items():
                setattr(j, k, v)
        return st


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--base", default=REPO)
    args = ap.parse_args()
    with open(os.path.join(HERE, "a1_results.csv"), newline="", encoding="utf-8") as f:
        ref = {r["scenario"]: r for r in csv.DictReader(f)}
    os.makedirs(OUT, exist_ok=True)
    for s in load_sessions(args.manifest, args.base):
        rec = Recorder(f"{args.seed}:{s['scenario']}")
        st = rec.run(*read_rssi(s["ab_path"]))
        charged = sum(r["charged_to_leak_EC"] for r in rec.rows)
        bitmap = sum(r["bitmap_bits"] for r in rec.rows)
        accepted = sum(r["message"] == "accept" for r in rec.rows)
        r = ref[s["scenario"]]
        assert charged == st["leak_EC_bits"] == int(r["leak_EC_bits"]), (s["scenario"], charged, r["leak_EC_bits"])
        assert bitmap == int(r["leak_EC_bits_conservative"]) - int(r["leak_EC_bits"]), (s["scenario"], bitmap)
        assert accepted == int(r["successful_blocks"]), (s["scenario"], accepted)
        with open(os.path.join(OUT, f"{s['scenario']}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rec.rows)
        print(f"{s['scenario']}: {len(rec.rows)} messages, {charged} bits charged, {bitmap} bitmap bits, "
              f"{accepted} accepted blocks (matches a1_results.csv)")


if __name__ == "__main__":
    main()
