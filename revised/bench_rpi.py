"""Processing time and energy estimate per session on a Raspberry Pi 5 (Sec. 4.5.3) -> bench_rpi.csv.

T_proc is the wall-clock time from the aligned RSSI arrays (file already read) to the extraction
decision: reconciliation with NC = 1 followed by Eq. (5). NIST diagnostics are not timed.
u_CPU = CPU time / wall time x 100 over all reps (400 % = four busy cores); E_session = P_dyn x T_proc (Eq. (12)),
with P_dyn = 1.93 W measured by power_pmic.py (pipeline minus idle board power, Eq. (X1)).
The reference method [25] is timed as configuration R0 (40-sample blocks, residual dropped,
NC = 3, same reconciliation and Eq. (5)) on the sessions it applies to; [28] runs baseline28.session.

Run on an otherwise idle Pi:   python -I bench_rpi.py --reps 30
"""
import argparse
import contextlib
import csv
import io
import os
import platform
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # -I drops the script dir
import a2_extract as a2
import baseline28 as b28
import jurnal_revisi as j
from run_a1_scenarios import load_sessions, read_rssi


P_DYN_W = 1.93  # measured board power, pipeline minus idle (power_pmic.json), Eq. (X1)


def sysfile(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip("\x00\n ")
    except OSError:
        return ""


def system_info():
    mem = next((l.split()[1] for l in sysfile("/proc/meminfo").splitlines() if l.startswith("MemTotal")), "")
    osr = next((l.split("=", 1)[1].strip('"') for l in sysfile("/etc/os-release").splitlines()
                if l.startswith("PRETTY_NAME=")), platform.platform())
    return {"model": sysfile("/proc/device-tree/model") or platform.machine(),
            "ram_GB": f"{int(mem) / 2**20:.1f}" if mem else "", "os": osr, "kernel": platform.release(),
            "python": platform.python_version(),
            "governor": sysfile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")}


def proposed(A, B):
    _, _, st = j.reconciliation_per_block_with_calibration(A, B, NC=1)
    a2.privacy_amplification_eq5(st)


def ref25(A, B):
    """[25] as configuration R0: 40-sample blocks, remainder dropped, NC = 3."""
    n = len(A) // 40 * 40
    if n == 0:
        raise ValueError("fewer than 40 valid pairs")
    qbs, j.compute_qbs = j.compute_qbs, lambda _n: 40
    try:
        _, _, st = j.reconciliation_per_block_with_calibration(A[:n], B[:n], NC=3)
    finally:
        j.compute_qbs = qbs
    a2.privacy_amplification_eq5(st)


def bench(fn, A, B, reps):
    with contextlib.redirect_stdout(io.StringIO()):
        fn(A, B)  # warm-up, also raises for traces the method cannot process
        wall, c0 = [], time.process_time()
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(A, B)
            wall.append(time.perf_counter() - t0)
        cpu = time.process_time() - c0  # one CPU reading over all reps: a single run is shorter than some timer ticks
    return wall, 100 * cpu / sum(wall)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    info = system_info()
    print(info)
    methods = [("proposed", proposed), ("ref25_R0", ref25), ("ref28", b28.session)]
    rows = []
    for s in load_sessions():
        A, B = read_rssi(s["ab_path"])
        a, b, _ = j.filter_valid_pairs(A, B)  # same valid-pair filtering for every method
        for name, fn in methods:
            try:
                wall, util = bench(fn, list(a), list(b), args.reps)
            except (ValueError, ZeroDivisionError):
                rows.append({"method": name, "scenario": s["scenario"], "n_valid": len(a), "note": "not applicable"})
                continue
            t, u = statistics.mean(wall), util
            p = P_DYN_W
            rows.append({"method": name, "scenario": s["scenario"], "n_valid": len(a), "reps": args.reps,
                         "T_mean_ms": f"{1e3 * t:.3f}", "T_sd_ms": f"{1e3 * statistics.stdev(wall):.3f}",
                         "u_cpu_pct": f"{u:.1f}", "P_dyn_W": f"{p:.3f}", "E_session_mJ": f"{1e3 * p * t:.3f}", "note": ""})
            print(rows[-1])
    out = os.path.join(HERE, "bench_rpi.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("# " + "; ".join(f"{k}={v}" for k, v in info.items()) + "\n")
        w = csv.DictWriter(f, fieldnames=["method", "scenario", "n_valid", "reps", "T_mean_ms", "T_sd_ms",
                                          "u_cpu_pct", "P_dyn_W", "E_session_mJ", "note"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
