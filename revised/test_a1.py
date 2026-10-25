"""Phase A1 self-check for jurnal_revisi (written from the plan/contract, not the implementation).

Run: .venv\\Scripts\\python -I test_a1.py   (exit code != 0 on any failure)
"""
import contextlib
import io
import math
import os
import random
import secrets
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.stats import chisquare

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jurnal_revisi as j  # noqa: E402

from run_a1_scenarios import REPO as DATA  # noqa: E402  data repository (root holds the trace used here)
NONCE = bytes(range(16))


def rb(seed):
    return random.Random(seed).getrandbits


def rand_bits(rng, n):
    return np.array([rng.getrandbits(1) for _ in range(n)], dtype=np.uint8)


def flip(bits, positions):
    out = bits.copy()
    out[list(positions)] ^= 1
    return out


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ---------------------------------------------------------------- 1. codebook
def check_constants():
    assert (j.BCH_N, j.BCH_K, j.BCH_T, j.PRIM_POLY) == (31, 6, 7, 37), \
        (j.BCH_N, j.BCH_K, j.BCH_T, j.PRIM_POLY)


def check_codebook():
    cb = np.asarray(j.bch31_6_codebook())
    assert cb.shape == (64, 31), cb.shape
    assert cb.dtype == np.uint8, cb.dtype
    rows = {tuple(r) for r in cb}
    assert len(rows) == 64, f"distinct rows {len(rows)}"
    for a in cb:
        for b in cb:
            assert tuple(a ^ b) in rows, "codebook not XOR-closed"
    w = sorted(int(r.sum()) for r in cb)
    assert w[0] == 0 and w[1] == 15, f"weights {w[:3]} (expected 0 then min nonzero 15)"
    for r in range(64):
        prefix = int("".join(str(int(b)) for b in cb[r, :6]), 2)
        assert prefix == r, f"row {r} systematic prefix {prefix}"


def check_decode():
    cb = np.asarray(j.bch31_6_codebook())
    rng = random.Random(11)
    for r in range(64):
        idx, d = j.bch31_6_decode(cb[r].copy())
        assert (idx, d) == (r, 0), (r, idx, d)
        e = rng.sample(range(31), 7)
        idx, d = j.bch31_6_decode(flip(cb[r], e))
        assert (idx, d) == (r, 7), (r, idx, d)
    for _ in range(300):
        y = rand_bits(rng, 31)
        idx, d = j.bch31_6_decode(y)
        dmin = int(min((cb ^ y).sum(axis=1)))
        assert d == dmin, f"distance {d} != nearest {dmin}"
        if idx is None:
            assert d > 7, f"None with distance {d}"
        else:
            assert d <= 7 and int((cb[idx] ^ y).sum()) == d, (idx, d)


# ------------------------------------------------------------- 2. roundtrip
def check_roundtrip():
    rng = random.Random(2)
    for L in (1, 6, 31, 32, 62, 100):
        ir = j.InformationReconciliation(randbits=rb(100 + L))
        KB = rand_bits(rng, L)
        KBp, leak = ir.reconcile(KB.copy(), KB.copy(), NONCE, 0, 0)
        assert KBp is not None, f"L={L}: reconcile failed with KA == KB"
        assert np.array_equal(np.asarray(KBp), KB), f"L={L}: KB' != KB"
        S = math.ceil(L / 31)
        exp = {'segments': S, 'sketch': 25 * S, 'ack': 2, 'tag': 16}  # ACK + decision bit after tag
        assert leak == exp, f"L={L}: leak {leak} != {exp}"


def check_empty_block():
    ir = j.InformationReconciliation(randbits=rb(3))
    e = np.zeros(0, dtype=np.uint8)
    KBp, leak = ir.reconcile(e, e, NONCE, 0, 0)
    assert KBp is None, KBp
    assert all(v == 0 for v in leak.values()), leak


# ----------------------------------------------------------- 3. error patterns
def check_le7_errors_succeed():
    rng = random.Random(4)
    ir = j.InformationReconciliation(randbits=rb(5))
    for L in (31, 62):
        for w in range(8):
            for t in range(25):
                KB = rand_bits(rng, L)
                pos = []
                for s in range(L // 31):  # w errors in every segment
                    pos += [31 * s + p for p in rng.sample(range(31), w)]
                KBp, leak = ir.reconcile(flip(KB, pos), KB, NONCE, w, t)
                assert KBp is not None and np.array_equal(np.asarray(KBp), KB), \
                    f"L={L} weight {w}/segment trial {t}: expected success with KB' == KB"
                assert (leak['ack'], leak['tag']) == (2, 16), leak


def check_ge8_errors_never_wrong():
    rng = random.Random(6)
    ir = j.InformationReconciliation(randbits=rb(7))
    fails = 0
    for w in range(8, 32):
        for t in range(40):
            KB = rand_bits(rng, 31)
            KBp, leak = ir.reconcile(flip(KB, rng.sample(range(31), w)), KB, NONCE, w, t)
            assert (leak['ack'], leak['tag']) in ((1, 0), (2, 16)), f"ack/tag rule: {leak}"
            if KBp is None:
                fails += 1
            else:
                assert np.array_equal(np.asarray(KBp), KB), f"weight {w} trial {t}: WRONG key returned"
    print(f"    (weight>=8: {fails}/{24 * 40} rejected, rest returned exactly KB)")


def check_codeword_error_caught_by_tag():
    cb = np.asarray(j.bch31_6_codebook())
    rng = random.Random(8)
    ir = j.InformationReconciliation(randbits=rb(9))
    for r in range(1, 64):
        KB = rand_bits(rng, 31)
        KA = KB ^ cb[r]
        KBp, leak = ir.reconcile(KA, KB, NONCE, r, 0)
        assert KBp is None, f"e = codeword {r} (weight {int(cb[r].sum())}): wrong key accepted, tag missed it"
        assert (leak['ack'], leak['tag']) == (2, 16), f"Alice decoded all segments, tag must be counted: {leak}"


# -------------------------------------------------------------------- 4. attack
def check_attack_zero_guess():
    rng = random.Random(10)
    ir = j.InformationReconciliation(randbits=rb(12))
    hits = decoded = 0
    for i in range(1000):
        KB = rand_bits(rng, 31)
        pub, _ = ir.bob_sketch(KB, NONCE, i, 0)
        rec = ir.alice_recover(np.zeros(31, dtype=np.uint8), pub)
        # ponytail: success = exact KB; non-None alone is ~10.6% (64 radius-7 spheres), mostly wrong codewords
        decoded += rec is not None
        hits += rec is not None and np.array_equal(np.asarray(rec), KB)
    print(f"    (Eve zero-guess exact recovery {hits}/1000, theory ~0.17%; decoded-to-any-codeword {decoded}/1000, theory ~10.6%)")
    assert hits < 10, f"Eve recover rate {hits / 10:.1f}% >= 1%"


def check_leak_ge_L_one_segment():
    rng = random.Random(13)
    ir = j.InformationReconciliation(randbits=rb(14))
    for L in range(1, 32):
        KB = rand_bits(rng, L)
        _, leak = ir.reconcile(KB.copy(), KB, NONCE, 0, 0)
        total = leak['sketch'] + leak['ack'] + leak['tag']
        assert total >= L, f"L={L}: leak {total} < L"


# ------------------------------------------------------------ 5. randomness
def check_seed_independence():
    rng = random.Random(15)
    a = j.InformationReconciliation(randbits=rb(1))
    b = j.InformationReconciliation(randbits=rb(2))
    KB = rand_bits(rng, 62)
    sa = a.bob_sketch(KB, NONCE, 0, 0)[0]['s']
    sb = b.bob_sketch(KB, NONCE, 0, 0)[0]['s']
    assert any(not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(sa, sb)), \
        "sketches identical under different seeds (no randomness)"
    for w in range(0, 16):
        for t in range(10):
            KA = flip(KB, rng.sample(range(62), w))
            ra, _ = a.reconcile(KA, KB, NONCE, w, t)
            rb_, _ = b.reconcile(KA, KB, NONCE, w, t)
            assert (ra is None) == (rb_ is None), f"weight {w} trial {t}: status depends on seed"
            if ra is not None:
                assert np.array_equal(np.asarray(ra), np.asarray(rb_))


def check_tag_of():
    KB = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    t = j.InformationReconciliation().tag_of(KB, NONCE, 3, 1)
    assert isinstance(t, int) and 0 <= t < 2 ** 16, t
    assert t == j.InformationReconciliation().tag_of(KB, NONCE, 3, 1), "tag not deterministic"


# ---------------------------------------- 5b. sketch randomness (default RNG)
def _cb():
    return np.asarray(j.bch31_6_codebook())


def recover_Rs(pub, KB):
    """R per segment from s XOR KB_seg (must be a codebook row; systematic, so row index = R)."""
    rows = {tuple(r): i for i, r in enumerate(_cb())}
    S = len(pub['s'])
    padded = np.zeros(31 * S, dtype=np.uint8)
    padded[:len(KB)] = KB
    out = []
    for i, s in enumerate(pub['s']):
        c = tuple(np.asarray(s, dtype=np.uint8) ^ padded[31 * i:31 * i + 31])
        assert c in rows, f"segment {i}: s XOR KB is not a codeword"
        out.append(rows[c])
    return out


def check_default_rng():
    assert j.InformationReconciliation().randbits is secrets.randbits, "default randbits is not secrets.randbits"


def check_sketch_uniform():
    KB = rand_bits(random.Random(60), 62)
    ir = j.InformationReconciliation()
    counts = np.zeros((2, 64), dtype=int)
    for i in range(2000):
        for seg, R in enumerate(recover_Rs(ir.bob_sketch(KB, NONCE, i, 0)[0], KB)):
            counts[seg, R] += 1
    for seg in range(2):
        assert (counts[seg] > 0).all(), f"segment {seg}: only {(counts[seg] > 0).sum()}/64 R values seen"
        p = chisquare(counts[seg]).pvalue
        assert p > 0.001, f"segment {seg}: R not uniform, chi-square p={p:.2e}"


def check_segments_independent():
    rng = random.Random(61)
    ir = j.InformationReconciliation()
    diff = 0
    for i in range(500):
        KB = rand_bits(rng, 62)
        R = recover_Rs(ir.bob_sketch(KB, NONCE, i, 0)[0], KB)
        diff += R[0] != R[1]
    assert diff > 450, f"R1 != R2 in only {diff}/500 sketches (expect ~492)"


def check_attempts_independent():
    rng = random.Random(62)
    ir = j.InformationReconciliation()
    diff = np.zeros(2, dtype=int)
    for i in range(500):
        KB = rand_bits(rng, 62)
        R0 = recover_Rs(ir.bob_sketch(KB, NONCE, i, 0)[0], KB)
        R1 = recover_Rs(ir.bob_sketch(KB, NONCE, i, 1)[0], KB)
        diff += np.array(R0) != np.array(R1)
    assert (diff > 450).all(), f"attempt 0 vs 1 R differs in only {diff.tolist()}/500 per segment"


def check_public_keys():
    rng = random.Random(63)
    ir = j.InformationReconciliation()
    for L in (1, 6, 31, 32, 62, 100):
        KB = rand_bits(rng, L)
        pub, tag = ir.bob_sketch(KB, NONCE, 5, 1)
        assert isinstance(pub, dict) and set(pub) == {'s', 'L'}, f"public fields {sorted(pub)} != ['L', 's']"
        assert pub['L'] == L and len(pub['s']) == math.ceil(L / 31), (pub['L'], len(pub['s']))
        assert all(len(s) == 31 for s in pub['s'])
        assert tag == ir.tag_of(KB, NONCE, 5, 1), "returned tag != tag_of(KB, nonce, block, attempt)"


def eve_bruteforce(ir, pub, tag, nonce, b, a):
    """Eve: try every R per segment, drop non-zero padding, keep candidates matching the public tag."""
    cb = _cb()
    L, S = pub['L'], len(pub['s'])
    per_seg = [[np.asarray(s, dtype=np.uint8) ^ cb[r] for r in range(64)] for s in pub['s']]
    hits = []
    for combo in np.ndindex(*([64] * S)):
        full = np.concatenate([per_seg[i][r] for i, r in enumerate(combo)])
        if full[L:].any():
            continue
        if ir.tag_of(full[:L], nonce, b, a) == tag:
            hits.append(full[:L])
    return hits


def check_eve_bruteforce():
    rng = random.Random(64)
    ir = j.InformationReconciliation()
    ok = n = 0
    for L in (6, 20, 31):
        for t in range(100):
            KB = rand_bits(rng, L)
            pub, tag = ir.bob_sketch(KB, NONCE, t, 0)
            hits = eve_bruteforce(ir, pub, tag, NONCE, t, 0)
            assert any(np.array_equal(h, KB) for h in hits), "true KB not among Eve's tag matches"
            ok += len(hits) == 1
            n += 1
    print(f"    (Eve 1-segment brute force, 64 candidates + tag: exact unique recovery {ok}/{n})")
    assert ok >= 0.97 * n, f"Eve recovered only {ok}/{n}: leak >= L should be a real attack"
    ok2 = 0
    for t in range(20):
        KB = rand_bits(rng, 62)
        pub, tag = ir.bob_sketch(KB, NONCE, 1000 + t, 0)
        hits = eve_bruteforce(ir, pub, tag, NONCE, 1000 + t, 0)
        ok2 += len(hits) == 1 and np.array_equal(hits[0], KB)
    print(f"    (Eve 2-segment brute force, 4096 candidates + tag: unique exact {ok2}/20, informational)")


# ------------------------------------------- 5c. mutation self-test of the suite
def _mutant(Rs_of, extra=None):
    def bob_sketch(self, KB, nonce, block_idx, attempt):
        KB = np.asarray(KB, dtype=np.uint8)
        L = len(KB)
        S = math.ceil(L / 31)
        padded = np.zeros(31 * S, dtype=np.uint8)
        padded[:L] = KB
        cb = _cb()
        Rs = Rs_of(self, padded, nonce, block_idx, attempt, S)
        pub = {'s': [padded[31 * i:31 * i + 31] ^ cb[R] for i, R in enumerate(Rs)], 'L': L}
        if extra:
            pub.update(extra(self, KB, nonce, block_idx, attempt))
        return pub, self.tag_of(KB, nonce, block_idx, attempt)
    return bob_sketch


_ATTEMPT_CACHE = {}
MUTANTS = {
    "s = KB in clear": _mutant(lambda self, p, n, b, a, S: [0] * S),
    "R = int(KB_seg[:6])": _mutant(lambda self, p, n, b, a, S: [int("".join(map(str, p[31 * i:31 * i + 6])), 2)
                                                                for i in range(S)]),
    "R shared across segments": _mutant(lambda self, p, n, b, a, S: [self.randbits(6)] * S),
    "R reused across attempts": _mutant(lambda self, p, n, b, a, S: _ATTEMPT_CACHE.setdefault(
        (n, b, S), [self.randbits(6) for _ in range(S)])),
    "extra 'crc' field": _mutant(lambda self, p, n, b, a, S: [self.randbits(6) for _ in range(S)],
                                 extra=lambda self, KB, n, b, a: {'crc': self.tag_of(KB, n, b, a + 1)}),
}


def check_mutation_self_test():
    detectors = [check_sketch_uniform, check_segments_independent, check_attempts_independent, check_public_keys]
    orig = j.InformationReconciliation.bob_sketch
    missed = []
    try:
        for name, mut in MUTANTS.items():
            j.InformationReconciliation.bob_sketch = mut
            _ATTEMPT_CACHE.clear()
            caught = []
            for d in detectors:
                try:
                    quiet(d)
                except Exception:
                    caught.append(d.__name__)
            print(f"    (mutant '{name}': caught by {caught or 'NOTHING'})")
            if not caught:
                missed.append(name)
    finally:
        j.InformationReconciliation.bob_sketch = orig
    assert not missed, f"suite misses leaky variants: {missed}"


# --------------------------------------------------------- 6. leak accounting
class Recorder:
    """Wrap InformationReconciliation.reconcile to collect every per-attempt leak."""

    def __enter__(self):
        self.calls = []
        self.orig = j.InformationReconciliation.reconcile
        rec = self

        def wrapped(self_, KA, KB, nonce, block_idx, attempt):
            out = rec.orig(self_, KA, KB, nonce, block_idx, attempt)
            rec.calls.append((int(block_idx), int(attempt), dict(out[1])))
            return out

        j.InformationReconciliation.reconcile = wrapped
        return self

    def __exit__(self, *exc):
        j.InformationReconciliation.reconcile = self.orig


def block_sizes(n_valid, qbs):
    sizes = [qbs] * (n_valid // qbs)
    if n_valid % qbs >= 10:
        sizes.append(n_valid % qbs)
    return sizes


def verify_stats(key, idx, st, calls, n_raw_expected=None):
    lb = st['leak_breakdown']
    assert set(lb) == {'sketch', 'ack', 'tag', 'lz_ctrl', 'bitmap'}, lb
    if n_raw_expected is not None:
        assert st['n_raw'] == n_raw_expected, (st['n_raw'], n_raw_expected)
    assert st['n_valid'] <= st['n_raw']
    assert st['qbs'] == j.compute_qbs(st['n_valid'])
    sizes = block_sizes(st['n_valid'], st['qbs'])
    assert st['total_blocks'] == len(sizes), (st['total_blocks'], sizes)
    assert bool(st['residual_block']) == (st['n_valid'] % st['qbs'] >= 10)
    assert st['successful_blocks'] == len(idx)
    assert st['final_key_bob'] == key, "final_key_bob != Alice final key"
    if idx:
        assert st['post_kdr'] == 0, f"post_kdr {st['post_kdr']} on successful blocks"
    else:
        assert math.isnan(st['post_kdr']), f"post_kdr {st['post_kdr']} with 0 successful blocks (expected nan)"
    if st['total_bits_quantization'] == 0:
        assert math.isnan(st['overall_kdr']), f"overall_kdr {st['overall_kdr']} with 0 bits (expected nan)"
    assert 0.0 <= st['sample_discard_ratio'] <= 1.0, st['sample_discard_ratio']
    assert math.isclose(st['r_rec'], len(idx) / st['total_blocks']), st['r_rec']
    assert st['total_blocks'] <= st['attempts_total'] <= 2 * st['total_blocks'], st['attempts_total']
    assert 0.0 <= st['discarded_bit_fraction'] <= 1.0

    # contract-level identities
    assert lb['sketch'] % 25 == 0, lb
    assert lb['lz_ctrl'] == 4 * st['total_blocks'], (lb['lz_ctrl'], st['total_blocks'])
    assert st['leak_EC_bits'] == lb['sketch'] + lb['ack'] + lb['tag'] + lb['lz_ctrl']
    assert st['leak_EC_bits_conservative'] == st['leak_EC_bits'] + lb['bitmap']
    assert st['leak_EC_bits_key_blocks'] <= st['leak_EC_bits']

    # item-by-item recompute from every reconcile() call (failed attempts included)
    assert calls, "reconcile() was never called by the loop"
    assert len(calls) == st['attempts_total'], (len(calls), st['attempts_total'])
    for item in ('sketch', 'ack', 'tag'):
        s = sum(c[2][item] for c in calls)
        assert lb[item] == s, f"{item}: stats {lb[item]} != sum over attempts {s}"
    bitmap = sum(2 * sizes[b] for b, _, _ in calls)
    assert lb['bitmap'] == bitmap, f"bitmap: stats {lb['bitmap']} != 2*samples per attempt {bitmap}"
    ok = set(idx)
    kb = sum(c[2]['sketch'] + c[2]['ack'] + c[2]['tag'] for c in calls if c[0] in ok)
    assert st['leak_EC_bits_key_blocks'] in (kb + 4 * len(ok), kb), \
        f"key-block leak {st['leak_EC_bits_key_blocks']} != {kb + 4 * len(ok)} (or {kb} w/o lz_ctrl)"


def synth(n, seed, zeros_at=()):
    rng = np.random.default_rng(seed)
    A = np.round(-65 + 8 * np.sin(np.arange(n) / 3.0) + rng.normal(0, 2, n))
    B = A + rng.choice([-1, 0, 0, 1], n)
    for i in zeros_at:
        A[i] = 0
    return A.astype(float), B.astype(float)


def check_leak_formula_synthetic():
    A, B = synth(300, 21)
    with Recorder() as r:
        key, idx, st = quiet(j.reconciliation_per_block_with_calibration, A, B, NC=1, verbose=False)
    verify_stats(key, idx, st, r.calls, n_raw_expected=300)


# ------------------------------------------------------- 7. loop plumbing
def check_mutual_accept():
    la = ['10', None, '11', '01', '00', None]
    lb = ['10', '00', None, '00', '00', None]
    a, b, kept = j.mutual_accept(la, lb)
    assert list(kept) == [0, 3, 4], kept
    assert (a, b) == ('100100', '100000'), (a, b)


def check_filter_valid_pairs():
    A = [-60, 0, -61, float('nan'), -62, -63, None]
    B = [-70, -71, 0, -72, float('nan'), -73, -74]
    a, b, n_raw = j.filter_valid_pairs(A, B)
    assert n_raw == 7, n_raw
    assert [float(x) for x in a] == [-60, -63], a
    assert [float(x) for x in b] == [-70, -73], b


def check_compute_qbs():
    assert j.compute_qbs(15) == 15 and j.compute_qbs(39) == 15
    assert j.compute_qbs(40) == 10 and j.compute_qbs(1000) == 10
    try:
        j.compute_qbs(14)
    except ValueError:
        return
    raise AssertionError("compute_qbs(14) did not raise ValueError")


def check_quantizer_lengths():
    rng = random.Random(30)
    for n in (10, 15, 11):
        block = [rng.randint(-80, -50) for _ in range(n)]
        for m, nb in ((2, 1), (4, 2), (8, 3)):
            for a in (0.0, 0.3, 1.0):
                out = j.kuantisasi_per_block(block, a, m)
                assert isinstance(out, list) and len(out) == n, f"m={m} a={a}: len {len(out)} != {n}"
                assert all(x is None or (isinstance(x, str) and len(x) == nb and set(x) <= {'0', '1'})
                           for x in out), (m, out)


def check_residual_processed():
    # 28 raw rows, 2 invalid -> n_valid 26 -> qbs 15 -> 1 full + residual 11
    A, B = synth(28, 31, zeros_at=(4,))
    B[20] = np.nan
    with Recorder() as r:
        key, idx, st = quiet(j.reconciliation_per_block_with_calibration, A, B, NC=1, verbose=False)
    assert (st['n_raw'], st['n_valid'], st['qbs']) == (28, 26, 15), (st['n_raw'], st['n_valid'], st['qbs'])
    assert st['total_blocks'] == 2 and st['residual_block'], (st['total_blocks'], st['residual_block'])
    assert {b for b, _, _ in r.calls} == {0, 1}, "residual block never reconciled"
    verify_stats(key, idx, st, r.calls, n_raw_expected=28)


def check_residual_boundaries():
    for n, blocks, resid in ((24, 1, False), (25, 2, True), (40, 4, False), (49, 4, False), (50, 5, False)):
        A, B = synth(n, 40 + n)
        _, _, st = quiet(j.reconciliation_per_block_with_calibration, A, B, NC=1, verbose=False)
        assert (st['total_blocks'], bool(st['residual_block'])) == (blocks, resid), \
            f"n={n}: got {(st['total_blocks'], st['residual_block'])}, expected {(blocks, resid)}"


def check_too_short_raises():
    A, B = synth(16, 50, zeros_at=(0, 1))  # 14 valid
    try:
        quiet(j.reconciliation_per_block_with_calibration, A, B, NC=1, verbose=False)
    except ValueError:
        return
    raise AssertionError("n_valid=14 did not raise ValueError")


# ---------------------------------------------------------------- smoke
def check_smoke_real_trace():
    df = pd.read_csv(os.path.join(DATA, "combined_v2i lurus 20.csv"))
    assert {'rssi_alice', 'rssi_bob'} <= set(df.columns), df.columns
    with Recorder() as r:
        key, idx, st = quiet(j.reconciliation_per_block_with_calibration,
                             df['rssi_alice'].values, df['rssi_bob'].values, NC=1, verbose=False)
    verify_stats(key, idx, st, r.calls, n_raw_expected=len(df))
    lb = st['leak_breakdown']
    print(f"    (real trace: n_valid={st['n_valid']} blocks={st['total_blocks']} ok={len(idx)} "
          f"bits={len(key)} leak={st['leak_EC_bits']} cons={st['leak_EC_bits_conservative']} {lb})")


CHECKS = [
    check_constants, check_codebook, check_decode,
    check_roundtrip, check_empty_block,
    check_le7_errors_succeed, check_ge8_errors_never_wrong, check_codeword_error_caught_by_tag,
    check_attack_zero_guess, check_leak_ge_L_one_segment,
    check_seed_independence, check_tag_of,
    check_default_rng, check_sketch_uniform, check_segments_independent, check_attempts_independent,
    check_public_keys, check_eve_bruteforce, check_mutation_self_test,
    check_leak_formula_synthetic,
    check_mutual_accept, check_filter_valid_pairs, check_compute_qbs, check_quantizer_lengths,
    check_residual_processed, check_residual_boundaries, check_too_short_raises,
    check_smoke_real_trace,
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
