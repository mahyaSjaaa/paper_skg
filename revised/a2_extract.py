"""Phase A2: entropy-bounded privacy amplification (paper Eq. 5) and NIST diagnostics on pre-extraction bits.

Pipeline per session: Bob's reconciled per-sample labels -> MCV min-entropy (SP 800-90B 6.3.1) per label
alphabet -> H_total -> l = floor(H_total - leak_EC - 2 log2(1/eps)) -> Toeplitz universal hash with a public
seed -> l key bits (no key when l <= 0). SKR = l / T_obs.
"""
import math
import secrets
from collections import Counter

import numpy as np
from scipy.linalg import toeplitz

import jurnal_revisi as j

EPS_EXPONENTS = (10, 20, 32)  # eps = 2^-10, 2^-20 (primary), 2^-32
PROBE_INTERVAL_S = 2.0  # nominal probing interval: T_obs = 2.0 s x n_raw (traces carry no timestamps)
NIST_ALPHA = 0.01


def mcv_min_entropy(symbols):
    """NIST SP 800-90B sec 6.3.1 most-common-value estimate: (h per symbol, p_hat, p_u, N).

    p_u = min(1, p_hat + 2.576 sqrt(p_hat (1 - p_hat) / (N - 1))) is the 99% upper bound on the most likely
    symbol probability; h = -log2(p_u). N < 2 gives h = 0.
    """
    N = len(symbols)
    if N < 2:
        return 0.0, 1.0, 1.0, N
    p_hat = Counter(symbols).most_common(1)[0][1] / N
    p_u = min(1.0, p_hat + 2.576 * math.sqrt(p_hat * (1 - p_hat) / (N - 1)))
    return -math.log2(p_u), p_hat, p_u, N


def extractable_length(H_total, leak_bits, eps):
    """Paper Eq. 5 (leftover hash lemma): l = floor(H_total - leak_EC - 2 log2(1/eps)); l <= 0 means no key."""
    return math.floor(H_total - leak_bits - 2 * math.log2(1 / eps))


def toeplitz_extract(bits, l, seed_bits=None):
    """Binary Toeplitz universal hash: l x N matrix from N + l - 1 seed bits (public), output T x mod 2.

    seed_bits=None draws the seed from `secrets`. l <= 0 returns an empty array.
    """
    x = np.asarray(bits, dtype=np.int64)
    if l <= 0:
        return np.zeros(0, dtype=np.uint8)
    need = len(x) + l - 1
    if seed_bits is None:
        seed_bits = np.unpackbits(np.frombuffer(secrets.token_bytes(-(-need // 8)), dtype=np.uint8))[:need]
    seed = np.asarray(seed_bits, dtype=np.int64)
    if len(seed) < need:
        raise ValueError(f"Toeplitz seed needs {need} bits, got {len(seed)}")
    # first column = seed[0:l], first row = seed[0], seed[l:l+N-1]
    T = toeplitz(seed[:l], np.concatenate((seed[:1], seed[l:need])))
    return ((T @ x) % 2).astype(np.uint8)


def mcv_by_alphabet(symbols):
    """MCV per alphabet size m = 2^len(label): ({m: {h, p_hat, p_u, N}}, H_total = sum_m h_m * N_m).

    Optimistic estimate of H_min(X|E): unconditional (Eve's traces are not aligned with the legitimate pair),
    IID assumed and not validated by the SP 800-90B IID tests, N far below the 10^6 samples of SP 800-90B.
    """
    groups = {}
    for s in symbols:
        groups.setdefault(2 ** len(s), []).append(s)
    h_per_symbol = {}
    for m, syms in sorted(groups.items()):
        h, p_hat, p_u, N = mcv_min_entropy(syms)
        h_per_symbol[m] = {'h': h, 'p_hat': p_hat, 'p_u': p_u, 'N': N}
    return h_per_symbol, sum(g['h'] * g['N'] for g in h_per_symbol.values())


def privacy_amplification_eq5(stats, eps=2 ** -20):
    """Entropy-bounded PA over the pooled reconciled bits of one session.

    Symbols are Bob's per-sample labels of successful blocks (stats['final_symbols_bob']); H_total comes from
    mcv_by_alphabet, an optimistic estimate of H_min(X|E) (see there), so l is optimistic too.

    leak = stats['leak_EC_bits_key_blocks'] (primary); l is also reported with leak_EC_bits (all blocks,
    no bitmap) and leak_EC_bits_conservative (all blocks + bitmap).
    The key at `eps` is Toeplitz-hashed from stats['final_key_bob'] with a fresh public seed.
    """
    h_per_symbol, H_total = mcv_by_alphabet(stats['final_symbols_bob'])
    leak, leak_cons = stats['leak_EC_bits_key_blocks'], stats['leak_EC_bits_conservative']

    l_key = extractable_length(H_total, leak, eps)
    x = [int(b) for b in stats['final_key_bob']]
    seed = ''
    key = ''
    if l_key > 0:
        seed_len = len(x) + l_key - 1
        seed_arr = np.unpackbits(np.frombuffer(secrets.token_bytes(-(-seed_len // 8)), dtype=np.uint8))[:seed_len]
        seed = ''.join(map(str, seed_arr))
        key = ''.join(map(str, toeplitz_extract(x, l_key, seed_arr)))
    T_obs = PROBE_INTERVAL_S * stats['n_raw']
    return {
        'h_per_symbol': h_per_symbol,
        'H_total': H_total,
        'leak_EC_bits_key_blocks': leak,
        'leak_EC_bits_conservative': leak_cons,
        'l': {k: extractable_length(H_total, leak, 2.0 ** -k) for k in EPS_EXPONENTS},
        'l_allblocks': {k: extractable_length(H_total, stats['leak_EC_bits'], 2.0 ** -k) for k in EPS_EXPONENTS},
        'l_conservative': {k: extractable_length(H_total, leak_cons, 2.0 ** -k) for k in EPS_EXPONENTS},
        'eps': eps,
        'l_key': l_key,
        'key_bits': key,
        'seed': seed,
        'seed_len': len(seed),
        'T_obs': T_obs,
        'SKR': max(l_key, 0) / T_obs if T_obs > 0 else 0.0,
    }


def nist_diagnostics(bits):
    """SP 800-22 tests on pre-extraction bits: {test: {applicable, n, p_value, pass}} at alpha = 0.01.

    Tests below their input-size recommendation are not applicable (p_value None, pass None).
    Diagnostic only: a pass does not certify entropy.
    """
    bits = [int(b) for b in bits]
    n = len(bits)
    s1, s2 = j.nist_serial_test(bits, m=2)
    p = {
        'Frequency': j.nist_frequency_test(bits),
        'Block Frequency': j.nist_block_frequency_test(bits, m=max(20, n // 99 + 1)),  # M >= 20, M > n/100, N < 100
        'Runs': j.nist_runs_test(bits),
        'Longest Run': j.nist_longest_runs_test(bits),
        'CuSum forward': j.nist_cumulative_sums_test(bits, mode='forward'),
        'CuSum reverse': j.nist_cumulative_sums_test(bits, mode='reverse'),
        'FFT': j.nist_fft_test(bits),
        'Approximate Entropy': j.nist_approximate_entropy_test(bits, m=2),
        'Serial 1': s1,
        'Serial 2': s2,
    }
    return {k: {'applicable': v is not None, 'n': n, 'p_value': v,
                'pass': None if v is None else v >= NIST_ALPHA} for k, v in p.items()}
