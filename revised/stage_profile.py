"""Share of processing time per pipeline stage (Sec. 5.6), over 30 runs of every session.   python -I stage_profile.py"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # -I drops the script dir
import a2_extract as a2
import jurnal_revisi as j
from run_a1_scenarios import load_sessions, read_rssi

if __name__ == "__main__":
    tot = dict.fromkeys(("complexity", "quantization", "reconciliation", "pa_eq5", "total"), 0.0)
    for s in load_sessions():
        A, B = read_rssi(s["ab_path"])
        for _ in range(30):
            t0 = time.perf_counter()
            _, _, st = j.reconciliation_per_block_with_calibration(A, B, NC=1)
            t1 = time.perf_counter()
            a2.privacy_amplification_eq5(st)
            t2 = time.perf_counter()
            tot["complexity"] += st["time_complexity"]
            tot["quantization"] += st["time_quantization"]
            tot["reconciliation"] += st["time_reconciliation"]
            tot["pa_eq5"] += t2 - t1
            tot["total"] += t2 - t0
    print({k: f"{100 * v / tot['total']:.1f}%" for k, v in tot.items()})
