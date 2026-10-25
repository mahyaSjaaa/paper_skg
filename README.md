# Adaptive Secret Key Generation from Sparse RSSI Observations for LoRa-Based Vehicular Communications

Code, RSSI traces, and per-session results for the paper. Everything reported in the paper is produced by the scripts in `revised/` from the traces listed in `revised/sessions.csv`.

## Data

| Folder | Sessions | Content |
|---|---|---|
| `data_mentah_mashabib/Pakuwon/` | A to D (V2I), K to N (V2V) | Alice/Bob traces and Eve recordings |
| `data_mentah_mashabib/PondokCandra/` | E, F (V2I), O to R (V2V) | Alice/Bob traces and Eve recordings (identical copies in `pocan/`) |
| `darmahusada/` | G to J (V2I) | Alice/Bob traces (the Eve files used are in `data_mentah_mashabib/Dharmahusada/`) |
| `data_mentah_mashabib/Dharmahusada/` | S to V (V2V); Eve recordings for G to J | Alice/Bob traces and Eve recordings |

`revised/sessions.csv` maps every scenario (A to V) to its site, link type, mobility pattern, speed, trace file, and Eve file. Each trace is one pass of the vehicles through the site.

Trace format (`combined_*.csv`): one row per probing round, written after Alice sends a probe and Bob replies. Columns `rssi_alice` and `rssi_bob` hold the RSSI in dBm; a zero marks a packet that was not received. Rows with a zero or missing value are removed before processing. Eve files (`filtered_eve_*.csv`) hold Eve's RSSI of Alice's packets (`alice_rssi`) and of Bob's packets (`bob_rssi`).

Notes on the data:
- Only the Eve files of sessions B, C, D, L, M, and N have one row per probing round of the Alice/Bob trace; the Eve analysis uses these six.
- `data_mentah_mashabib/Pakuwon/filtered_eve_combined_V2V_searah20.csv` is a copy of the `searah30` file (session L); the original Eve recording of session K is not available.
- The `combined_*.csv` and `filtered_eve_*.csv` files in the repository root, `jurnalgithub.py`, `baseline.py`, the per-scenario subfolders of `pocan/` and `darmahusada/` (outputs of the earlier code), and `pocan/venv/` are earlier working files. They are kept for reference and are not used by the scripts in `revised/`.

## Reproducing the results

Python 3.11 or later:

```
python -m venv .venv
.venv/bin/activate            (Windows: .venv\Scripts\activate)
pip install -r revised/requirements.txt
python -I revised/test_a1.py
python -I revised/test_a2.py
```

Each script reads the traces through `revised/sessions.csv` and writes its CSV next to itself in `revised/`.

| Script | Output | Paper |
|---|---|---|
| `run_a1_scenarios.py` | `a1_results.csv` | Reliability and rates per session (Sections 5.1, 5.2) |
| `run_a2_scenarios.py` | `a2_results.csv` | Min-entropy, extractable length, SKR, NIST tests (Section 5.5) |
| `analysis_limits.py` | `limits_results.csv` | Pooled entropy limits and observation budget (Section 5.5) |
| `eve_analysis.py` | `eve_results.csv` | Passive eavesdropper (Sections 5.1, 5.5) |
| `ablation.py` | `ablation_results.csv`, `r0_results.csv` | Component ablation and reference method [25] as configuration R0 (Sections 5.3, 5.4) |
| `baseline28.py` | `baseline28_results.csv` | Reference method [28] (Section 5.4); `--check` runs its self-test |
| `run_baseline25.py` | `baseline25_results.csv` | Runs `baseline.py` unchanged; needs the optional `bchlib` |
| `bench_rpi.py` | `bench_rpi.csv` | Processing time per session on the Raspberry Pi 5 (Section 5.6) |
| `power_pmic.py` | `power_pmic.json` | Board power from the Raspberry Pi 5 PMIC (Section 4.1) |
| `stage_profile.py` | printed | Share of processing time per stage (Section 5.6) |

`bench_rpi.py` and `power_pmic.py` are meant to run on a Raspberry Pi 5; `power_pmic.py` uses `vcgencmd pmic_read_adc`. The CSV files in `revised/` are the results reported in the paper; rerunning the scripts reproduces them exactly, except the timing and power files, which depend on the device.

Core modules: `jurnal_revisi.py` (valid-pair filtering, conditional signal partitioning, LZ-76 complexity, quantization, code-offset BCH(31,6) reconciliation, leakage accounting) and `a2_extract.py` (MCV min-entropy estimate, extractable length of Eq. (5), Toeplitz extractor, NIST SP 800-22 tests).
