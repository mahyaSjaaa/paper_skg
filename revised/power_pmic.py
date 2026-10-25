"""Board power of the Raspberry Pi 5 from its PMIC rail sensors (vcgencmd pmic_read_adc) -> power_pmic.json.

P = sum over rails of V x I. Phases: idle, full load (one busy process per core), and the proposed pipeline
looping over the 22 sessions in one process. The PMIC rails exclude USB peripherals and the input stage,
so the values are board-rail power, not wall power.   Run: python -I power_pmic.py [--secs 60]
"""
import json
import multiprocessing as mp
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def power():
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True).stdout
    amps = dict(re.findall(r"(\S+)_A current\(\d+\)=([\d.]+)A", out))
    volts = dict(re.findall(r"(\S+)_V volt\(\d+\)=([\d.]+)V", out))
    return sum(float(a) * float(volts[k]) for k, a in amps.items() if k in volts)


def sample(secs, dt=0.5):
    xs, end = [], time.time() + secs
    while time.time() < end:
        xs.append(power())
        time.sleep(dt)
    return {"mean_W": statistics.mean(xs), "sd_W": statistics.stdev(xs), "n": len(xs)}


def burn(stop):
    while not stop.is_set():
        pass


def pipeline(stop):
    import contextlib, io
    import a2_extract as a2
    import jurnal_revisi as j
    from run_a1_scenarios import load_sessions, read_rssi
    data = [read_rssi(s["ab_path"]) for s in load_sessions()]
    with contextlib.redirect_stdout(io.StringIO()):
        while not stop.is_set():
            for A, B in data:
                _, _, st = j.reconciliation_per_block_with_calibration(A, B, NC=1)
                a2.privacy_amplification_eq5(st)


def loaded(target, workers, secs):
    stop = mp.Event()
    ps = [mp.Process(target=target, args=(stop,)) for _ in range(workers)]
    for p in ps:
        p.start()
    time.sleep(5)  # ramp-up excluded
    r = sample(secs)
    stop.set()
    for p in ps:
        p.join()
    return r


def temp():
    return subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    secs = int(sys.argv[sys.argv.index("--secs") + 1]) if "--secs" in sys.argv else 60
    while os.getloadavg()[0] > 0.3:  # wait for boot activity to settle
        time.sleep(10)
    res = {"load_before": os.getloadavg()[0], "temp_start": temp()}
    res["idle"] = sample(secs)
    res["full_4cores"] = loaded(burn, os.cpu_count(), secs)
    time.sleep(20)
    res["pipeline_1proc"] = loaded(pipeline, 1, secs)
    time.sleep(20)
    res["idle_after"] = sample(secs // 2)
    res["temp_end"] = temp()
    res["throttled"] = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    json.dump(res, open(os.path.join(HERE, "power_pmic.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
