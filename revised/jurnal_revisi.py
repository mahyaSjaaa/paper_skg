import math
import os
import sys
import functools
import secrets
import pandas as pd
import numpy as np
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import binascii
import time
import csv
from datetime import datetime
from scipy.special import erfc, gammaincc
from scipy.stats import norm
from numpy.fft import fft

PA_CHUNK_SIZE = 0



#=============================================================
# =================== COMPLEXITY TEST =======================
#=============================================================

def lz76_decomposition(sequence):
    n = len(sequence)
    i = 0
    components = []
    while i < n:
        length = 1
        while i + length <= n:
            substring = sequence[i:i+length]
            found = False
            for j in range(0, i):
                if j + length > i:
                    break
                if sequence[j:j+length] == substring:
                    found = True
                    break
            if not found or (i + length == n):
                components.append(substring)
                i += length
                break
            length += 1
    complexity = len(components) / n if n > 0 else 0
    return complexity, components

def lz76_decomposition_ir(sequence):
    n = len(sequence)
    print(f"Performing LZ76 decomposition on sequence of length {n}...")
    i = 0
    components = []
    while i < n:
        length = 1
        while i + length <= n:
            substring = sequence[i:i+length]
            found = False
            for j in range(0, i):
                if j + length > i:
                    break
                if sequence[j:j+length] == substring:
                    found = True
                    break
            if not found or (i + length == n):
                components.append(substring)
                i += length
                break
            length += 1
    complexity = len(components) / (n/math.log2(n)) if n > 0 else 0
    return complexity, components

def calculate_complexity_for_block(rssi_block):
    complexity, components = lz76_decomposition(rssi_block)
    return complexity

def determine_m_from_complexity(complexity):
    if complexity < 0.3:
        return 2
    elif complexity < 0.675:
        return 4
    else:
        return 8

def guard_band(complexity, m):
    a = 0
    if complexity <= 0.275 and m == 2:
        a = 1
        return a
    elif complexity > 0.275 and m == 2:
        a = (-5.83 * complexity) + 2.57
        return a
    elif complexity <= 0.33 and m == 4:
        a = (1.085 * complexity) - 0.082
        return a
    elif complexity > 0.33 and complexity <= 0.46 and m == 4:
        a = (-3.47 * complexity) + 1.6
        return a
    elif complexity > 0.46 and m == 4:
        a = 0
        return a
    elif m == 8:
        a = 0
        return a
    return a

#===============================================
#============== QUANTIZATION =====================
#===============================================

def kuantisasi_4_block(rssi_block, a):
    q0 = min(rssi_block)
    q4 = max(rssi_block)
    q2_paper = sum(rssi_block)/len(rssi_block)
    q1_arr = [rssi for rssi in rssi_block if rssi >= q0 and rssi <= q2_paper]
    q3_arr = [rssi for rssi in rssi_block if rssi > q2_paper and rssi <= q4]
    q1_paper = sum(q1_arr)/len(q1_arr) if len(q1_arr) > 0 else q0
    q3_paper = sum(q3_arr)/len(q3_arr) if len(q3_arr) > 0 else q4
    q1 = q1_paper
    q3 = q3_paper
    q2 = q2_paper
    temp_s1_Arr = [rssi for rssi in rssi_block if rssi > q0 and rssi <= q2]
    temp_s2_Arr = [rssi for rssi in rssi_block if rssi > q1 and rssi <= q3]
    temp_s3_Arr = [rssi for rssi in rssi_block if rssi > q2 and rssi <= q4]
    mean_s1 = sum(temp_s1_Arr)/len(temp_s1_Arr) if len(temp_s1_Arr) > 0 else 0
    s1 = math.sqrt(sum((rssi - mean_s1)**2 for rssi in temp_s1_Arr) / len(temp_s1_Arr)) if len(temp_s1_Arr) > 0 else 0
    mean_s2 = sum(temp_s2_Arr)/len(temp_s2_Arr) if len(temp_s2_Arr) > 0 else 0
    s2 = math.sqrt(sum((rssi - mean_s2)**2 for rssi in temp_s2_Arr) / len(temp_s2_Arr)) if len(temp_s2_Arr) > 0 else 0
    mean_s3 = sum(temp_s3_Arr)/len(temp_s3_Arr) if len(temp_s3_Arr) > 0 else 0
    s3 = math.sqrt(sum((rssi - mean_s3)**2 for rssi in temp_s3_Arr) / len(temp_s3_Arr)) if len(temp_s3_Arr) > 0 else 0
    Q1 = [q0, q1 - (a * s1)]
    Q2 = [q1 + (a * s1), q2 - (a * s2)]
    Q3 = [q2 + (a * s2), q3 - (a * s3)]
    Q4 = [q3 + (a * s3), q4]
    labels = []
    for rssi in rssi_block:
        if rssi >= Q1[0] and rssi <= Q1[1]:
            labels.append('10')
        elif rssi >= Q2[0] and rssi <= Q2[1]:
            labels.append('11')
        elif rssi >= Q3[0] and rssi <= Q3[1]:
            labels.append('01')
        elif rssi >= Q4[0] and rssi <= Q4[1]:
            labels.append('00')
        else:
            labels.append(None)  # guard band
    return labels

def kuantisasi_2_block(rssi_block, a):
    q0 = min(rssi_block)
    q2 = max(rssi_block)
    q1_paper = sum(rssi_block)/len(rssi_block)
    q1 = q1_paper
    temp_s1_Arr = [rssi for rssi in rssi_block if rssi > q0 and rssi <= q2]
    mean_s1 = sum(temp_s1_Arr)/len(temp_s1_Arr) if len(temp_s1_Arr) > 0 else 0
    s1 = math.sqrt(sum((rssi - mean_s1)**2 for rssi in temp_s1_Arr) / len(temp_s1_Arr)) if len(temp_s1_Arr) > 0 else 0
    Q1 = [q0, q1 - (a * s1)]
    Q2 = [q1 + (a * s1), q2]
    labels = []
    for rssi in rssi_block:
        if rssi >= Q1[0] and rssi <= Q1[1]:
            labels.append('1')
        elif rssi >= Q2[0] and rssi <= Q2[1]:
            labels.append('0')
        else:
            labels.append(None)  # guard band
    return labels

def kuantisasi_8_block(rssi_block, a):
    q0 = min(rssi_block)
    q8 = max(rssi_block)
    q4_paper = sum(rssi_block)/len(rssi_block)
    q2_arr = [rssi for rssi in rssi_block if rssi >= q0 and rssi <= q4_paper]
    q2_paper = sum(q2_arr)/len(q2_arr) if len(q2_arr) > 0 else q0
    q6_arr = [rssi for rssi in rssi_block if rssi > q4_paper and rssi <= q8]
    q6_paper = sum(q6_arr)/len(q6_arr) if len(q6_arr) > 0 else q8
    q1_arr = [rssi for rssi in rssi_block if rssi >= q0 and rssi <= q2_paper]
    q1_paper = sum(q1_arr)/len(q1_arr) if len(q1_arr) > 0 else q0
    q3_arr = [rssi for rssi in rssi_block if rssi > q2_paper and rssi <= q4_paper]
    q3_paper = sum(q3_arr)/len(q3_arr) if len(q3_arr) > 0 else q2_paper
    q5_arr = [rssi for rssi in rssi_block if rssi > q4_paper and rssi <= q6_paper]
    q5_paper = sum(q5_arr)/len(q5_arr) if len(q5_arr) > 0 else q4_paper
    q7_arr = [rssi for rssi in rssi_block if rssi > q6_paper and rssi <= q8]
    q7_paper = sum(q7_arr)/len(q7_arr) if len(q7_arr) > 0 else q6_paper
    q2 = q2_paper
    q4 = q4_paper
    q6 = q6_paper
    q1 = q1_paper
    q3 = q3_paper
    q5 = q5_paper
    q7 = q7_paper
    regions = [
        [rssi for rssi in rssi_block if rssi > q0 and rssi <= q2],
        [rssi for rssi in rssi_block if rssi > q1 and rssi <= q3],
        [rssi for rssi in rssi_block if rssi > q2 and rssi <= q4],
        [rssi for rssi in rssi_block if rssi > q3 and rssi <= q5],
        [rssi for rssi in rssi_block if rssi > q4 and rssi <= q6],
        [rssi for rssi in rssi_block if rssi > q5 and rssi <= q7],
        [rssi for rssi in rssi_block if rssi > q6 and rssi <= q8]
    ]
    std_devs = []
    for region in regions:
        if len(region) > 0:
            mean = sum(region)/len(region)
            std = math.sqrt(sum((rssi - mean)**2 for rssi in region) / len(region))
            std_devs.append(std)
        else:
            std_devs.append(0)
    s1, s2, s3, s4, s5, s6, s7 = std_devs
    Q1 = [q0, q1 - (a * s1)]
    Q2 = [q1 + (a * s1), q2 - (a * s2)]
    Q3 = [q2 + (a * s2), q3 - (a * s3)]
    Q4 = [q3 + (a * s3), q4 - (a * s4)]
    Q5 = [q4 + (a * s4), q5 - (a * s5)]
    Q6 = [q5 + (a * s5), q6 - (a * s6)]
    Q7 = [q6 + (a * s6), q7 - (a * s7)]
    Q8 = [q7 + (a * s7), q8]
    labels = []
    for rssi in rssi_block:
        if rssi >= Q1[0] and rssi <= Q1[1]:
            labels.append('000')
        elif rssi >= Q2[0] and rssi <= Q2[1]:
            labels.append('001')
        elif rssi >= Q3[0] and rssi <= Q3[1]:
            labels.append('011')
        elif rssi >= Q4[0] and rssi <= Q4[1]:
            labels.append('010')
        elif rssi >= Q5[0] and rssi <= Q5[1]:
            labels.append('110')
        elif rssi >= Q6[0] and rssi <= Q6[1]:
            labels.append('111')
        elif rssi >= Q7[0] and rssi <= Q7[1]:
            labels.append('101')
        elif rssi >= Q8[0] and rssi <= Q8[1]:
            labels.append('100')
        else:
            labels.append(None)  # guard band
    return labels

def kuantisasi_per_block(rssi_block, a, m):
    if m == 2:
        return kuantisasi_2_block(rssi_block, a)
    elif m == 4:
        return kuantisasi_4_block(rssi_block, a)
    elif m == 8:
        return kuantisasi_8_block(rssi_block, a)
    else:
        raise ValueError(f"Invalid m value: {m}")

#=========================================
#=========== RECONCILIATION ================
#=========================================

def to_bits(bit_string):
    return np.array([int(b) for b in bit_string], dtype=np.uint8)

def count_bit_errors(bits_a, bits_b):
    bits_a = np.array(bits_a, dtype=int)
    bits_b = np.array(bits_b, dtype=int)
    min_len = min(len(bits_a), len(bits_b))
    return int(np.sum(bits_a[:min_len] != bits_b[:min_len]))

def count_chunk_errors(key_str_a, key_str_b, chunk_idx, chunk_size):
    start = chunk_idx * chunk_size
    end = start + chunk_size
    chunk_a = key_str_a[start:end]
    chunk_b = key_str_b[start:end]
    bits_a = to_bits(chunk_a)
    bits_b = to_bits(chunk_b)
    return count_bit_errors(bits_a, bits_b)

#=========================================
#=========== BCH(31,6) CODE ================
#=========================================

BCH_N, BCH_K, BCH_T = 31, 6, 7
PRIM_POLY = 37  # x^5 + x^2 + 1, primitive over GF(2)


def _gf2_mul(a, b):
    """Carry-less product of two GF(2)[x] polynomials stored as ints (bit i = coeff of x^i)."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def _gf2_mod(a, m):
    """Remainder of a(x) / m(x) over GF(2)."""
    dm = m.bit_length()
    while a.bit_length() >= dm:
        a ^= m << (a.bit_length() - dm)
    return a


def _bch31_generator():
    """g(x) of degree 25: product of the minimal polynomials of alpha^1, 3, 5, 7, 11 in GF(32)."""
    exp = []
    x = 1
    for _ in range(BCH_N):
        exp.append(x)
        x <<= 1
        if x & 0b100000:
            x ^= PRIM_POLY
    log = {v: i for i, v in enumerate(exp)}

    def gf_mul(a, b):
        return 0 if a == 0 or b == 0 else exp[(log[a] + log[b]) % BCH_N]

    g = 1
    # Cosets of 1, 3, 5, 7, 11 cover alpha^1..alpha^14, so the designed distance is 15 (t = 7)
    for i in (1, 3, 5, 7, 11):
        poly = [1]  # coefficients, lowest degree first
        for j in {(i << k) % BCH_N for k in range(5)}:
            aj = exp[j]
            poly = [(poly[k - 1] if k > 0 else 0) ^ (gf_mul(poly[k], aj) if k < len(poly) else 0)
                    for k in range(len(poly) + 1)]
        assert set(poly) <= {0, 1}
        g = _gf2_mul(g, sum(c << k for k, c in enumerate(poly)))
    assert g.bit_length() - 1 == BCH_N - BCH_K
    return g


@functools.lru_cache(maxsize=None)
def bch31_6_codebook():
    """64 x 31 systematic codebook: row r = [r as 6 bits MSB first | 25 parity bits]."""
    g = _bch31_generator()
    words = []
    for r in range(1 << BCH_K):
        shifted = r << (BCH_N - BCH_K)
        words.append(shifted | _gf2_mod(shifted, g))
    # Explicit raises (not assert) so the self-check survives python -O
    word_set = set(words)
    if len(word_set) != 1 << BCH_K:
        raise RuntimeError("BCH(31,6) codebook: codewords are not distinct")
    if not all(a ^ b in word_set for a in words for b in words):
        raise RuntimeError("BCH(31,6) codebook: not closed under XOR")
    if min(w.bit_count() for w in words if w) != 2 * BCH_T + 1:
        raise RuntimeError("BCH(31,6) codebook: minimum distance is not 15")
    book = np.array([[(w >> (BCH_N - 1 - i)) & 1 for i in range(BCH_N)] for w in words], dtype=np.uint8)
    book.flags.writeable = False
    return book


def bch31_6_decode(y):
    """Bounded-distance decoding: (row, distance) of the nearest codeword if within t, else (None, distance)."""
    dist = (bch31_6_codebook() ^ np.asarray(y, dtype=np.uint8).reshape(BCH_N)).sum(axis=1)
    idx = int(np.argmin(dist))
    d = int(dist[idx])
    return (idx, d) if d <= BCH_T else (None, d)


class InformationReconciliation:
    """Code-offset reconciliation with BCH(31,6): Bob publishes KB XOR c for a random codeword c."""

    def __init__(self, randbits=secrets.randbits, tag_bits=16):
        self.randbits = randbits
        self.tag_bits = tag_bits
        self.codebook = bch31_6_codebook()

    @staticmethod
    def _segments(bits, L):
        """First L bits (zero padded if shorter), zero padded to whole 31-bit segments."""
        out = np.zeros(-(-L // BCH_N) * BCH_N, dtype=np.uint8)
        src = np.asarray(bits, dtype=np.uint8)[:L]
        out[:len(src)] = src
        return out.reshape(-1, BCH_N)

    def tag_of(self, bits, nonce, block_idx, attempt):
        """Truncated SHA-256 over (nonce, block, attempt, one byte per key bit)."""
        h = hashlib.sha256(nonce + block_idx.to_bytes(4, 'big') + attempt.to_bytes(4, 'big')
                           + np.asarray(bits, dtype=np.uint8).tobytes())
        return int.from_bytes(h.digest(), 'big') >> (256 - self.tag_bits)

    # Bob side: returns (public sketch, tag); the tag is kept until Alice ACKs
    def bob_sketch(self, KB, nonce, block_idx, attempt):
        KB = np.asarray(KB, dtype=np.uint8)
        L = len(KB)
        s = [seg ^ self.codebook[self.randbits(BCH_K)] for seg in self._segments(KB, L)]
        return {'s': s, 'L': L}, self.tag_of(KB, nonce, block_idx, attempt)

    # Alice side: KA XOR s = c XOR (KA XOR KB); decode c, then KB' = s XOR c
    def alice_recover(self, KA, public):
        L = public['L']
        out = []
        for y_seg, s in zip(self._segments(KA, L), public['s'], strict=True):
            idx, _ = bch31_6_decode(y_seg ^ s)
            if idx is None:
                return None
            out.append(s ^ self.codebook[idx])
        return np.array(out, dtype=np.uint8).reshape(-1)[:L]

    def reconcile(self, KA, KB, nonce, block_idx, attempt):
        """One attempt: sketch, Alice decodes and sends ACK/NACK, Bob sends the tag after ACK."""
        KB = np.asarray(KB, dtype=np.uint8)
        S = -(-len(KB) // BCH_N)
        if S == 0:
            return None, {'segments': 0, 'sketch': 0, 'ack': 0, 'tag': 0}
        public, tag = self.bob_sketch(KB, nonce, block_idx, attempt)
        KB_prime = self.alice_recover(KA, public)
        leak = {'segments': S, 'sketch': (BCH_N - BCH_K) * S, 'ack': 1, 'tag': 0}
        if KB_prime is None:
            return None, leak
        # Bob sent the tag; Alice's accept/reject after the tag check is one more public bit
        leak['tag'] = self.tag_bits
        leak['ack'] += 1
        if self.tag_of(KB_prime, nonce, block_idx, attempt) != tag:
            return None, leak
        return KB_prime, leak


def filter_valid_pairs(rssi_A, rssi_B):
    """Drop row i when either side is 0, None or NaN; pairs stay aligned by row index."""
    A, B = [], []
    for a, b in zip(rssi_A, rssi_B, strict=True):
        if pd.isna(a) or pd.isna(b) or a == 0 or b == 0:
            continue
        A.append(a)
        B.append(b)
    return A, B, len(rssi_A)


def compute_qbs(n):
    """Quantization block size from the number of valid paired samples."""
    if n < 15:
        raise ValueError(f"Need at least 15 valid paired samples, got {n}")
    return 15 if n < 40 else 10


def mutual_accept(labels_A, labels_B):
    """Keep sample indices accepted by both sides (order preserved) and concatenate their labels."""
    kept = [i for i, (a, b) in enumerate(zip(labels_A, labels_B, strict=True))
            if a is not None and b is not None]
    return ''.join(labels_A[i] for i in kept), ''.join(labels_B[i] for i in kept), kept


def reconciliation_per_block_with_calibration(rssi_A_full, rssi_B_full, delta_alpha=0.2, NC=1, verbose=False):
    log = print if verbose else (lambda *args, **kwargs: None)

    _tc = time.time()
    rssi_A_valid, rssi_B_valid, n_raw = filter_valid_pairs(rssi_A_full, rssi_B_full)
    n_valid = len(rssi_A_valid)
    qbs = compute_qbs(n_valid)
    full_blocks = n_valid // qbs
    block_ranges = [(k * qbs, (k + 1) * qbs) for k in range(full_blocks)]
    residual_block = n_valid - full_blocks * qbs >= 10
    if residual_block:
        block_ranges.append((full_blocks * qbs, n_valid))
    time_csp = time.time() - _tc

    ir = InformationReconciliation()
    nonce = secrets.token_bytes(16)  # session nonce: binds every tag to this run
    final_key = ""
    final_key_bob = ""
    final_symbols_bob = []  # Bob's per-sample labels of successful blocks (A2 entropy estimate)
    success_indices = []
    blocks_data = []
    keys_A_all = []
    keys_B_all = []

    total_blocks = len(block_ranges)
    total_error_before = 0
    total_error_after = 0
    dropped_blocks = 0
    calibration_counts = []
    attempts_total = 0
    leak = {'sketch': 0, 'ack': 0, 'tag': 0, 'lz_ctrl': 0, 'bitmap': 0}
    leak_EC_bits_key_blocks = 0

    total_time_complexity = 0.0
    total_time_quantization = 0.0
    total_time_reconciliation = 0.0

    complexity_values = []
    m_values = []
    alpha_values = []

    log(f"\n{'='*70}")
    log(f"RECONCILIATION WITH PER-BLOCK COMPLEXITY ESTIMATION")
    log(f"{'='*70}")
    log(f"Samples: {n_raw} raw, {n_valid} valid | QBS: {qbs}")
    log(f"Total blocks: {total_blocks} | Residual block: {residual_block}")
    log(f"Δα={delta_alpha} | NC={NC}")
    log(f"{'='*70}\n")

    for block_idx, (start_idx, end_idx) in enumerate(block_ranges):
        rssi_A_block = rssi_A_valid[start_idx:end_idx]
        rssi_B_block = rssi_B_valid[start_idx:end_idx]

        log(f"{'='*70}")
        log(f"BLOCK {block_idx} (valid samples {start_idx}-{end_idx-1})")
        log(f"{'='*70}")

        # C_k, m_k and alpha_k come from Alice's block only and are public
        _t0 = time.time()
        complexity_block = calculate_complexity_for_block(rssi_A_block)
        m_block = determine_m_from_complexity(complexity_block)
        alpha_init_block = guard_band(complexity_block, m_block)
        total_time_complexity += time.time() - _t0

        # Alice announces her LZ-76 component count once per block (4 bits); later alphas follow from it
        leak['lz_ctrl'] += 4
        block_leak = 4

        log(f"Complexity: {complexity_block:.4f} | m: {m_block} | Initial α: {alpha_init_block:.4f}")
        complexity_values.append(complexity_block)
        m_values.append(m_block)
        alpha_values.append(alpha_init_block)

        success = False
        alpha_k = alpha_init_block
        kdr_block = 0
        block_info = {
            'block_id': block_idx,
            'residual': residual_block and block_idx == total_blocks - 1,
            'complexity': f"{complexity_block:.4f}",
            'm': m_block,
            'alpha_init': f"{alpha_init_block:.4f}"
        }

        for nc in range(NC + 1):
            log(f"\nCalibration {nc}: α={alpha_k:.4f}")
            _t1 = time.time()
            labels_A = kuantisasi_per_block(rssi_A_block, alpha_k, m_block)
            labels_B = kuantisasi_per_block(rssi_B_block, alpha_k, m_block)
            total_time_quantization += time.time() - _t1

            # Both sides exchange accepted-sample bitmaps and keep the intersection
            key_A_block, key_B_block, kept = mutual_accept(labels_A, labels_B)
            leak['bitmap'] += 2 * len(rssi_A_block)

            KA_k = to_bits(key_A_block)
            KB_k = to_bits(key_B_block)
            err_before = count_bit_errors(KA_k, KB_k)
            kdr_block = err_before / len(KB_k) if len(KB_k) > 0 else 0

            log(f"  Bits: {len(key_A_block)} | Errors: {err_before} | KDR: {kdr_block:.4f} ({kdr_block:.2%})")

            if nc == 0:
                total_error_before += err_before
                keys_A_all.append(key_A_block)
                keys_B_all.append(key_B_block)
                block_info['key_before'] = key_A_block
                block_info['key_length'] = len(key_A_block)
                block_info['errors_before'] = err_before
                block_info['kdr_before'] = f"{kdr_block:.4f}"

            log(f"  → Trying BCH(31,6) code-offset reconciliation...")
            _t2 = time.time()
            KB_prime, attempt_leak = ir.reconcile(KA_k, KB_k, nonce, block_idx, nc)
            total_time_reconciliation += time.time() - _t2
            attempts_total += 1
            for item in ('sketch', 'ack', 'tag'):
                leak[item] += attempt_leak[item]
            block_leak += attempt_leak['sketch'] + attempt_leak['ack'] + attempt_leak['tag']

            if KB_prime is not None:
                err_after = count_bit_errors(KB_prime, KB_k)
                total_error_after += err_after
                key_after = "".join(KB_prime.astype(str))
                final_key += key_after
                final_key_bob += key_B_block
                final_symbols_bob += [labels_B[i] for i in kept]
                leak_EC_bits_key_blocks += block_leak
                success_indices.append(block_idx)
                calibration_counts.append(nc)
                success = True
                block_info['status'] = 'SUCCESS'
                block_info['key_after'] = key_after
                block_info['key_after_bob'] = key_B_block
                block_info['errors_after'] = err_after
                block_info['calibrations'] = nc
                block_info['alpha_final'] = f"{alpha_k:.4f}"
                block_info['kdr_final'] = f"{kdr_block:.4f}"
                block_info['IRCk'] = 1
                log(f"  ✓ SUCCESS - IRC=1 | KDR: {kdr_block:.2%} | Calibrations: {nc}")
                break

            log(f"  ✗ IRC=0")
            if nc < NC:
                alpha_k += delta_alpha
                log(f"  → Will try calibration {nc+1} with α={alpha_k:.4f}")
            else:
                log(f"  ✗ No more calibrations, DROPPING block")

        if not success:
            dropped_blocks += 1
            block_info['status'] = 'DROPPED'
            block_info['IRCk'] = 0
            block_info['key_after'] = ''
            block_info['key_after_bob'] = ''
            block_info['kdr_final'] = f"{kdr_block:.4f}"
            log(f"\n✗ Result: DROPPED (IRC=0 after {NC+1} attempts)")

        log()
        blocks_data.append(block_info)

    total_bits_quantization = sum(len(k) for k in keys_A_all)
    total_errors_quantization = sum(
        count_bit_errors(to_bits(keys_A_all[i]), to_bits(keys_B_all[i]))
        for i in range(len(keys_A_all))
    )
    overall_kdr = total_errors_quantization / total_bits_quantization if total_bits_quantization > 0 else float('nan')

    reconciled_bits = len(final_key)
    r_rec = len(success_indices) / total_blocks if total_blocks > 0 else 0.0
    post_kdr = total_error_after / reconciled_bits if success_indices else float('nan')
    samples_processed = sum(end - start for start, end in block_ranges)
    sample_discard_ratio = (n_raw - samples_processed) / n_raw if n_raw > 0 else float('nan')
    discarded_bit_fraction = 1 - reconciled_bits / total_bits_quantization if total_bits_quantization > 0 else 0.0
    leak_EC_bits = leak['sketch'] + leak['ack'] + leak['tag'] + leak['lz_ctrl']

    log(f"\n{'='*70}")
    log(f"QUANTIZATION SUMMARY")
    log(f"{'='*70}")
    log(f"Total blocks  : {total_blocks}")
    log(f"Total bits    : {total_bits_quantization} bits")
    log(f"Total errors  : {total_errors_quantization}")
    log(f"Overall KDR   : {overall_kdr:.4f} ({overall_kdr:.2%})")

    log(f"\n{'='*70}")
    log(f"RECONCILIATION SUMMARY")
    log(f"{'='*70}")
    log(f"Success : {len(success_indices)}/{total_blocks} blocks ({r_rec:.1%}) in {attempts_total} attempts")
    log(f"Dropped : {dropped_blocks}/{total_blocks} blocks")
    log(f"Final key (before PA): {reconciled_bits} bits | post-KDR: {post_kdr:.4f}")
    log(f"Leak EC: {leak_EC_bits} bits (with bitmap: {leak_EC_bits + leak['bitmap']}) | key blocks: {leak_EC_bits_key_blocks} bits")

    return final_key, success_indices, {
        'total_blocks': total_blocks,
        'successful_blocks': len(success_indices),
        'dropped_blocks': dropped_blocks,
        'drop_rate': dropped_blocks / total_blocks if total_blocks > 0 else 0.0,
        'total_error_before': total_error_before,
        'total_error_after': total_error_after,
        'calibration_counts': calibration_counts,
        'blocks_data': blocks_data,
        'keys_A_all': keys_A_all,
        'keys_B_all': keys_B_all,
        'overall_kdr': overall_kdr,
        'total_errors_quantization': total_errors_quantization,
        'total_bits_quantization': total_bits_quantization,
        'complexity_values': complexity_values,
        'm_values': m_values,
        'alpha_values': alpha_values,
        'time_complexity': total_time_complexity,
        'time_quantization': total_time_quantization,
        'time_reconciliation': total_time_reconciliation,
        'n_raw': n_raw,
        'n_valid': n_valid,
        'qbs': qbs,
        'residual_block': residual_block,
        'r_rec': r_rec,
        'attempts_total': attempts_total,
        'post_kdr': post_kdr,
        'discarded_bit_fraction': discarded_bit_fraction,
        'sample_discard_ratio': sample_discard_ratio,
        'final_key_bob': final_key_bob,
        'final_symbols_bob': final_symbols_bob,
        'leak_breakdown': leak,
        'leak_EC_bits': leak_EC_bits,
        'leak_EC_bits_conservative': leak_EC_bits + leak['bitmap'],
        'leak_EC_bits_key_blocks': leak_EC_bits_key_blocks,
        'time_csp': time_csp,
    }


#=================== NIST SP 800-22 TESTS ===============

NIST_APEN_M   = 2

# All tests return None when the input is shorter than the SP 800-22 rev1a input-size recommendation
# (not applicable: excluded from pass counts). min_n overrides it only for the short worked examples.

def nist_frequency_test(bits, min_n=100):
    n = len(bits)
    if n < max(min_n, 1):
        return None
    s = sum(1 if b == 1 else -1 for b in bits)
    s_obs = abs(s) / math.sqrt(n)
    return float(erfc(s_obs / math.sqrt(2.0)))

def nist_block_frequency_test(bits, m=128):
    bits = np.asarray(bits, dtype=int)
    n = len(bits)
    if m <= 0 or n < max(100, m):
        return None
    N = n // m
    cs = 0.0
    for i in range(N):
        block = bits[i*m:(i+1)*m]
        pi = int(block.sum()) / m
        cs += 4.0 * m * (pi - 0.5)**2
    return float(gammaincc(N/2.0, cs/2.0))

def nist_runs_test(bits, min_n=100):
    bits = np.asarray(bits, dtype=int)
    n = len(bits)
    if n < max(min_n, 2):
        return None
    pi = int(bits.sum()) / n
    tau = 2.0 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return 0.0  # frequency prerequisite failed: the test is failed (sec 2.3.4), not inapplicable
    v_obs = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
    num = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    return float(erfc(num / den))

def nist_longest_runs_test(bits):
    n = len(bits)
    if n < 128:
        return None
    if n < 6272:
        M = 8; v_values = [1,2,3,4]; pi_vals = [0.21484375,0.3671875,0.23046875,0.1875]
    elif n < 750000:
        M = 128; v_values = [4,5,6,7,8,9]
        pi_vals = [0.1174035788,0.242955959,0.249363483,0.17517706,0.102701071,0.112398847]
    else:
        M = 10000; v_values = [10,11,12,13,14,15,16]
        pi_vals = [0.0882,0.2092,0.2483,0.1933,0.1208,0.0675,0.0727]
    N = n // M
    v_counts = [0] * len(v_values)
    for i in range(N):
        block = bits[i*M:(i+1)*M]
        longest = 0; run = 0
        for b in block:
            if b == 1:
                run += 1; longest = max(longest, run)
            else:
                run = 0
        if longest <= v_values[0]:
            v_counts[0] += 1
        elif longest >= v_values[-1]:
            v_counts[-1] += 1
        else:
            for j in range(1, len(v_values)):
                if longest == v_values[j]:
                    v_counts[j] += 1; break
    cs = 0.0
    for j in range(len(v_values)):
        exp = N * pi_vals[j]
        if exp > 0:
            cs += (v_counts[j] - exp) ** 2 / exp
    return float(gammaincc((len(pi_vals) - 1) / 2.0, cs / 2.0))

def nist_cumulative_sums_test(bits, mode='forward', min_n=100):
    """SP 800-22 rev1a sec 2.13.4 (summation bounds truncated toward zero as in the NIST reference code)."""
    n = len(bits)
    if n < max(min_n, 1):
        return None
    x = np.array([1 if b == 1 else -1 for b in bits], dtype=int)
    if mode == 'reverse':
        x = x[::-1]
    z = int(np.max(np.abs(np.cumsum(x))))
    sq = math.sqrt(n)
    sum1 = sum(norm.cdf((4*k+1)*z/sq) - norm.cdf((4*k-1)*z/sq)
               for k in range(int((-n/z + 1)/4), int((n/z - 1)/4) + 1))
    sum2 = sum(norm.cdf((4*k+3)*z/sq) - norm.cdf((4*k+1)*z/sq)
               for k in range(int((-n/z - 3)/4), int((n/z - 1)/4) + 1))
    return float(1.0 - sum1 + sum2)

def nist_fft_test(bits):
    n = len(bits)
    if n < 1000:
        return None
    x = np.array([2*b - 1 for b in bits], dtype=float)
    s = np.abs(fft(x))[:n//2]
    T = math.sqrt(math.log(1.0/0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = np.sum(s < T)
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return float(erfc(abs(d) / math.sqrt(2.0)))

def _pattern_counts(bits, mm):
    """Overlapping mm-bit pattern counts with wrap-around (augmented sequence of SP 800-22 sec 2.11/2.12)."""
    n = len(bits)
    counts = {}
    for i in range(n):
        pattern = tuple(int(bits[(i+j) % n]) for j in range(mm))
        counts[pattern] = counts.get(pattern, 0) + 1
    return counts

def nist_approximate_entropy_test(bits, m=NIST_APEN_M, min_n=None):
    n = len(bits)
    if n < (2**(m + 6) if min_n is None else max(min_n, m + 1)):  # sec 2.12.7: m < floor(log2 n) - 5
        return None
    def _phi(mm):
        return sum((c / n) * math.log(c / n) for c in _pattern_counts(bits, mm).values())
    ap_en  = _phi(m) - _phi(m + 1)
    chi_sq = 2.0 * n * (math.log(2.0) - ap_en)
    return float(gammaincc(2**(m-1), chi_sq / 2.0))

def nist_serial_test(bits, m=2, min_n=None):
    """SP 800-22 rev1a sec 2.11.4; m >= 2. Default applicability m < floor(log2 n) - 2 (sec 2.11.7)."""
    n = len(bits)
    if n < (2**(m + 3) if min_n is None else max(min_n, m)):
        return (None, None)
    def psi(mm):
        if mm <= 0:
            return 0.0
        return sum(v*v for v in _pattern_counts(bits, mm).values()) * (2.0**mm) / n - n
    psi_m, psi_m1, psi_m2 = psi(m), psi(m - 1), psi(m - 2)
    del1 = psi_m - psi_m1
    del2 = psi_m - 2.0 * psi_m1 + psi_m2
    return (float(gammaincc(2**(m-2), del1/2.0)), float(gammaincc(2**(m-3), del2/2.0)))

#==================== AES =================================

def alice_encrypt(pesan_asli, key_hex):
    key_bytes = binascii.unhexlify(key_hex)
    cipher = AES.new(key_bytes, AES.MODE_CBC)
    iv = cipher.iv
    pesan_padded = pad(pesan_asli.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(pesan_padded)
    return binascii.hexlify(iv).decode(), binascii.hexlify(ciphertext).decode()

def bob_decrypt(iv_hex, ciphertext_hex, key_hex):
    key_bytes = binascii.unhexlify(key_hex)
    iv_bytes = binascii.unhexlify(iv_hex)
    ciphertext_bytes = binascii.unhexlify(ciphertext_hex)
    cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
    pesan_decrypted = unpad(cipher.decrypt(ciphertext_bytes), AES.block_size)
    return pesan_decrypted.decode('utf-8')

#==================== SAVE FUNCTIONS ======================

def save_reconciliation_results(blocks_data, filename="reconciliation_results.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Bit_Alice', 'Bit_Bob'])
        for block in blocks_data:
            for bit_a, bit_b in zip(block.get('key_after', ''), block.get('key_after_bob', ''), strict=True):
                writer.writerow([bit_a, bit_b])
    print(f"Reconciliation results saved to: {filename}")

def save_final_key(final_key_bits, filename="final_key.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Bit'])
        for bit in final_key_bits:
            writer.writerow([bit])
    print(f"✓ Final key (reconciled bits) saved to: {filename}")

def estimate_program_power(cpu_percent):
    idle = 3.5
    full = 9.0
    max_cpu = 400.0
    total = idle + (cpu_percent / max_cpu) * (full - idle)
    return total - idle

#===============================================================
#========================== MAIN PROGRAM =======================
#===============================================================

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')  # symbols like Δα and ✓ fail on cp1252 when piped
    DEFAULT_AB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extracted",
                              "paper_skg-main", "darmahusada", "combined_V2Ilurus30.csv")
    ab_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AB
    df = pd.read_csv(ab_path)
    rssi_A_raw = df["rssi_alice"].to_list()
    rssi_B_raw = df["rssi_bob"].to_list()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_wall = time.time()
    start_cpu = time.process_time()

    # PER-BLOCK PROCESSING + RECONCILIATION

    kunci_ir_alice, blok_sukses, stats = reconciliation_per_block_with_calibration(
        rssi_A_full=rssi_A_raw,
        rssi_B_full=rssi_B_raw,
        delta_alpha=0.2,
        NC=1,
        verbose=True
    )

    total_bits_after_reconciliation = len(kunci_ir_alice)
    time_proc = stats['time_csp'] + stats['time_complexity'] + stats['time_quantization'] + stats['time_reconciliation']

    # PRIVACY AMPLIFICATION: Toeplitz universal hash, output length from Eq. 5 (a2_extract)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I omits the script dir
    import a2_extract as a2
    pa = a2.privacy_amplification_eq5(stats)

    print(f"\n{'='*70}")
    print("PRIVACY AMPLIFICATION (Eq. 5, Toeplitz universal hash, public seed)")
    print(f"{'='*70}")
    for m, g in pa['h_per_symbol'].items():
        print(f"  m={m}: N={g['N']} p_hat={g['p_hat']:.4f} p_u={g['p_u']:.4f} h={g['h']:.4f} bit/symbol")
    print(f"  H_total (MCV, optimistic H_min(X|E))   : {pa['H_total']:.2f} bits")
    print(f"  Bits after quantization / reconciliation: {stats['total_bits_quantization']} / {total_bits_after_reconciliation}")
    print(f"  leak_EC key blocks / conservative       : {pa['leak_EC_bits_key_blocks']} / {pa['leak_EC_bits_conservative']} bits")
    for k in a2.EPS_EXPONENTS:
        print(f"  l at eps=2^-{k:<2}                         : {pa['l'][k]} (all blocks: {pa['l_allblocks'][k]}, "
              f"conservative: {pa['l_conservative'][k]})")
    print(f"  Key bits at eps=2^-20                   : {len(pa['key_bits'])} (seed {pa['seed_len']} bits, public)")
    print(f"  T_obs = {a2.PROBE_INTERVAL_S} s x {stats['n_raw']} samples = {pa['T_obs']:.0f} s | SKR = {pa['SKR']:.4f} bit/s")
    print(f"  T_proc (CSP + complexity + quantization + reconciliation): {time_proc:.6f} s")

    # NIST SP 800-22 DIAGNOSTICS ON PRE-EXTRACTION RECONCILED BITS

    print(f"\n{'='*70}")
    print(f"NIST SP 800-22 DIAGNOSTICS (reconciled bits before extraction, alpha={a2.NIST_ALPHA})")
    print(f"{'='*70}")
    for name, r in a2.nist_diagnostics(kunci_ir_alice).items():
        if r['applicable']:
            print(f"  {name:<20} n={r['n']:<5} p={r['p_value']:.6f}  {'PASS' if r['pass'] else 'FAIL'}")
        else:
            print(f"  {name:<20} n={r['n']:<5} N/A (below input-size recommendation)")

    # AES ENCRYPTION DEMO: only with an Eq. 5 key of at least 128 bits

    if len(pa['key_bits']) >= 128:
        kunci_aes_128 = f"{int(pa['key_bits'][:128], 2):032x}"
        print(f"\nKey (hex, AES-128): {kunci_aes_128.upper()}")
        pesan_awal = "Halo Bob! Ini pesan rahasia kita berdua."
        print(f"Plaintext : '{pesan_awal}'")
        iv, pesan_terenkripsi = alice_encrypt(pesan_awal, kunci_aes_128)
        print(f"Ciphertext: {pesan_terenkripsi}")
        print(f"IV        : {iv}")
        pesan_akhir = bob_decrypt(iv, pesan_terenkripsi, kunci_aes_128)
        print(f"Decrypted : '{pesan_akhir}'")
        print("✓ Status: SUCCESS" if pesan_awal == pesan_akhir else "✗ Status: FAILED")
    else:
        print(f"\nNo secret key: Eq. 5 gives {len(pa['key_bits'])} bits (< 128), AES demo skipped.")

    print(f"\n{'='*70}")
    print(f"PROGRAM COMPLETED")
    print(f"{'='*70}\n")
    end_cpu = time.process_time()
    end_wall = time.time()
    cpu_time = end_cpu - start_cpu
    wall_time = end_wall - start_wall

    cpu_usage = (cpu_time / wall_time) * 100
    power = estimate_program_power(cpu_usage)

    print(f"CPU Usage (real): {cpu_usage:.2f}%")
    print(cpu_time)
    print(f"Estimated Power: {power:.2f} W")