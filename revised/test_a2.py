"""Phase A2 checks: MCV entropy, Eq. 5 length, Toeplitz extractor, NIST SP 800-22 known answers.
Plain asserts; prints PASS/FAIL per check; nonzero exit on any failure."""
import math
import os
import random
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jurnal_revisi as j  # noqa: E402
import a2_extract as a2  # noqa: E402

from run_a1_scenarios import REPO  # noqa: E402
TRACE = os.path.join(REPO, "darmahusada", "combined_V2Ibelok20.csv")


def bits_of(s):
    return [int(c) for c in s]


def close(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol


def p_u_ref(p, n):
    return min(1.0, p + 2.576 * math.sqrt(p * (1 - p) / (n - 1)))


def check_mcv_uniform():
    syms = [f"{i:03b}" for i in range(8)] * 1000
    h, p, pu, n = a2.mcv_min_entropy(syms)
    assert n == 8000 and close(p, 0.125), (n, p)
    assert close(pu, p_u_ref(0.125, 8000)) and close(h, -math.log2(pu)), (pu, h)
    assert 2.85 < h < 3.0, h


def check_mcv_constant():
    h, p, pu, n = a2.mcv_min_entropy(["01"] * 100)
    assert (p, pu, n) == (1.0, 1.0, 100) and h == 0.0, (h, p, pu, n)


def check_mcv_known_counts():
    h, p, pu, n = a2.mcv_min_entropy(["a"] * 60 + ["b"] * 30 + ["c"] * 10)
    assert n == 100 and close(p, 0.6)
    assert close(pu, 0.6 + 2.576 * math.sqrt(0.24 / 99)), pu
    assert close(h, 0.460305, 1e-5), h
    # p_u capped at 1 for tiny samples
    h, _, pu, _ = a2.mcv_min_entropy(["a"] * 6 + ["b"] * 3 + ["c"])
    assert pu == 1.0 and h == 0.0, (pu, h)


def check_mcv_too_short():
    assert a2.mcv_min_entropy(["1"])[0] == 0.0
    assert a2.mcv_min_entropy([])[0] == 0.0


def check_extractable_length():
    assert a2.extractable_length(100, 30, 2 ** -20) == 30
    assert a2.extractable_length(100.7, 30, 2 ** -20) == 30
    assert a2.extractable_length(100, 30, 2 ** -10) == 50
    assert a2.extractable_length(100, 30, 2 ** -32) == 6
    assert a2.extractable_length(36.5, 0, 2 ** -20) == -4


def check_toeplitz_length_and_empty():
    rng = random.Random(1)
    x = [rng.getrandbits(1) for _ in range(50)]
    seed = [rng.getrandbits(1) for _ in range(50 + 20 - 1)]
    y = a2.toeplitz_extract(x, 20, seed)
    assert len(y) == 20 and set(np.asarray(y).tolist()) <= {0, 1}
    assert len(a2.toeplitz_extract(x, 0, seed)) == 0
    assert len(a2.toeplitz_extract(x, -5, seed)) == 0
    assert len(a2.toeplitz_extract(x, 20)) == 20  # seed drawn from secrets
    try:
        a2.toeplitz_extract(x, 20, seed[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("short seed accepted")


def check_toeplitz_linear():
    rng = random.Random(2)
    N, L = 64, 24
    seed = [rng.getrandbits(1) for _ in range(N + L - 1)]
    for _ in range(50):
        x = np.array([rng.getrandbits(1) for _ in range(N)], dtype=np.uint8)
        y = np.array([rng.getrandbits(1) for _ in range(N)], dtype=np.uint8)
        lhs = np.asarray(a2.toeplitz_extract(x ^ y, L, seed))
        rhs = np.asarray(a2.toeplitz_extract(x, L, seed)) ^ np.asarray(a2.toeplitz_extract(y, L, seed))
        assert np.array_equal(lhs, rhs)


def check_toeplitz_structure():
    rng = random.Random(3)
    N, L = 12, 7
    seed = [rng.getrandbits(1) for _ in range(N + L - 1)]
    cols = [np.asarray(a2.toeplitz_extract(np.eye(N, dtype=np.uint8)[k], L, seed)) for k in range(N)]
    T = np.stack(cols, axis=1)
    assert all(T[i, k] == T[i + 1, k + 1] for i in range(L - 1) for k in range(N - 1)), "not Toeplitz"


def check_nist_frequency_kat():
    assert close(j.nist_frequency_test(bits_of("1011010101"), min_n=1), 0.527089)


def check_nist_runs_kat():
    assert close(j.nist_runs_test(bits_of("1001101011"), min_n=1), 0.147232)


EPS100 = "1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000"


def check_nist_cusum_kat():
    b = bits_of(EPS100)
    assert close(j.nist_cumulative_sums_test(b, mode='forward'), 0.219194), j.nist_cumulative_sums_test(b, mode='forward')
    assert close(j.nist_cumulative_sums_test(b, mode='reverse'), 0.114866), j.nist_cumulative_sums_test(b, mode='reverse')


def check_nist_serial_kat():
    p1, p2 = j.nist_serial_test(bits_of("0011011101"), m=3, min_n=1)
    assert close(p1, 0.808792) and close(p2, 0.670320), (p1, p2)


def check_nist_block_frequency_kat():
    assert close(j.nist_block_frequency_test(bits_of(EPS100), m=10), 0.706438)


EPS128 = ("11001100000101010110110001001100111000000000001001001101010100010001001111010110"
          "100000001101011111001100111001101101100010110010")


def check_nist_longest_run_kat():
    assert close(j.nist_longest_runs_test(bits_of(EPS128)), 0.180609), j.nist_longest_runs_test(bits_of(EPS128))


def check_nist_apen_kat():
    assert close(j.nist_approximate_entropy_test(bits_of(EPS100), m=2, min_n=1), 0.235301)
    assert close(j.nist_approximate_entropy_test(bits_of("0100110101"), m=3, min_n=1), 0.261961)


def check_nist_int8_overflow():
    # >127 ones: int8 accumulation wraps; results must equal those on plain Python ints
    import hashlib
    hx = hashlib.sha256(b"a2-2").hexdigest()  # 137 ones
    ref = [int(c) for c in bin(int(hx, 16))[2:].zfill(256)]
    bits = np.array(ref, dtype=np.int8)
    assert j.nist_runs_test(bits) == j.nist_runs_test(ref) > 0, (j.nist_runs_test(bits), j.nist_runs_test(ref))
    assert j.nist_block_frequency_test(bits, m=128) == j.nist_block_frequency_test(ref, m=128)
    assert j.nist_runs_test(np.ones(256, dtype=np.int8)) == 0.0  # frequency prerequisite fails: P = 0, not N/A


def check_dead_pa_removed():
    for name in ("privacy_amplification_multi_key", "hex_to_bit_array", "NIST_ENTROPY_THRESHOLD"):
        assert not hasattr(j, name), name


def check_nist_not_applicable():
    short = bits_of("10" * 25)
    assert j.nist_frequency_test(short) is None
    assert j.nist_runs_test(short) is None
    assert j.nist_cumulative_sums_test(short[:40]) is None
    assert j.nist_fft_test(bits_of("10" * 100)) is None
    assert j.nist_longest_runs_test(short) is None
    assert j.nist_block_frequency_test(short) is None
    assert j.nist_approximate_entropy_test(bits_of("10" * 100)) is None  # needs n >= 2^(m+6) = 256
    assert j.nist_serial_test(bits_of("1101"), m=2) == (None, None)
    d = a2.nist_diagnostics(short[:20])  # Serial m=2 needs n >= 32
    assert d and all(not v['applicable'] and v['p_value'] is None and v['pass'] is None for v in d.values()), d


def check_nist_diagnostics_applicable():
    rng = random.Random(4)
    bits = [rng.getrandbits(1) for _ in range(300)]
    d = a2.nist_diagnostics(bits)
    for name in ("Frequency", "Runs", "CuSum forward", "CuSum reverse", "Longest Run", "Block Frequency",
                 "Approximate Entropy", "Serial 1", "Serial 2"):
        assert d[name]['applicable'] and d[name]['n'] == 300 and 0 <= d[name]['p_value'] <= 1, (name, d[name])
        assert d[name]['pass'] == (d[name]['p_value'] >= 0.01)
    assert not d['FFT']['applicable']


def check_final_symbols_bob():
    df = pd.read_csv(TRACE, encoding="utf-8-sig")
    key, idx, st = j.reconciliation_per_block_with_calibration(df["rssi_alice"].values, df["rssi_bob"].values, NC=1)
    syms = st['final_symbols_bob']
    assert ''.join(syms) == st['final_key_bob'] == key and len(key) > 0
    assert all(isinstance(s, str) and len(s) in (1, 2, 3) for s in syms)


def check_eq5_synthetic():
    syms = ["000", "001", "011", "010", "110", "111", "101", "100"] * 40 + ["10", "11", "01", "00"] * 25
    key = ''.join(syms)
    stats = {'final_symbols_bob': syms, 'final_key_bob': key, 'leak_EC_bits_key_blocks': 100,
             'leak_EC_bits': 200, 'leak_EC_bits_conservative': 300, 'n_raw': 420}
    r = a2.privacy_amplification_eq5(stats)
    g = r['h_per_symbol']
    assert set(g) == {8, 4} and g[8]['N'] == 320 and g[4]['N'] == 100, g
    H = sum(v['h'] * v['N'] for v in g.values())
    assert close(r['H_total'], H)
    for k, e in ((10, 2 ** -10), (20, 2 ** -20), (32, 2 ** -32)):
        assert r['l'][k] == a2.extractable_length(H, 100, e), (k, r['l'])
        assert r['l_conservative'][k] == a2.extractable_length(H, 300, e)
        assert r['l_allblocks'][k] == a2.extractable_length(H, 200, e)
    l = r['l'][20]
    assert l > 0 and len(r['key_bits']) == l and set(r['key_bits']) <= {'0', '1'}
    assert r['seed_len'] == len(key) + l - 1 == len(r['seed'])
    assert r['T_obs'] == 840.0 and close(r['SKR'], l / 840.0)


def check_eq5_no_key():
    stats = {'final_symbols_bob': ["0", "1"] * 10, 'final_key_bob': "01" * 10, 'leak_EC_bits_key_blocks': 50,
             'leak_EC_bits': 60, 'leak_EC_bits_conservative': 80, 'n_raw': 15}
    r = a2.privacy_amplification_eq5(stats)
    assert r['l'][20] < 0 and r['key_bits'] == '' and r['seed_len'] == 0 and r['SKR'] == 0.0, r


CHECKS = [
    check_mcv_uniform, check_mcv_constant, check_mcv_known_counts, check_mcv_too_short,
    check_extractable_length,
    check_toeplitz_length_and_empty, check_toeplitz_linear, check_toeplitz_structure,
    check_nist_frequency_kat, check_nist_runs_kat, check_nist_cusum_kat, check_nist_serial_kat,
    check_nist_block_frequency_kat, check_nist_longest_run_kat, check_nist_apen_kat,
    check_nist_int8_overflow, check_dead_pa_removed, check_nist_not_applicable, check_nist_diagnostics_applicable,
    check_final_symbols_bob, check_eq5_synthetic, check_eq5_no_key,
]


def main():
    failed = []
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # report every failure, keep going
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3, file=sys.stdout)
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
