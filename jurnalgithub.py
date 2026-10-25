import math
import pandas as pd
import numpy as np
import zlib
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import binascii
import bchlib         
import time
import csv
from datetime import datetime
from scipy.special import erfc, gammaincc
from numpy.fft import fft

BLOCK_SIZE = 0
PA_CHUNK_SIZE = 0 

# baca file csv
# df = pd.read_csv("combined_V2V_searah30.csv")
# df = pd.read_csv("combined_v2i lurus 30.csv")
# df = pd.read_csv("darmahusada/combined_V2Ibelok30.csv")
df = pd.read_csv("darmahusada/combined_V2Ilurus30.csv")
# df = pd.read_csv("pocan/combined_v2vlawan30.csv")
# df = pd.read_csv("pocan/combined_v2vsearah30.csv")
# df = pd.read_csv("pocan/combined_v2ilurus30km.csv")
# # # df = pd.read_csv("combined_v2i lurus 20.csv")
rssi_A_raw = df["rssi_alice"].to_list()
rssi_B_raw = df["rssi_bob"].to_list()

#eve
# df = pd.read_csv("pocan/filtered_evev2vsearah20.csv")
# df = pd.read_csv("filtered_eve_combined_v2i lurus 30.csv")
#df = pd.read_csv("filtered_eve_combined_V2V_searah30.csv")
# df = pd.read_csv("pocan/filtered_evev2vlawan30.csv")
# df = pd.read_csv("pocan/filtered_evev2vsearah30.csv")
# df = pd.read_csv("pocan/filtered_evev2ilurus30km.csv")
# df = pd.read_csv("darmahusada/filtered_eve_V2Ibelok30.csv")
# df = pd.read_csv("darmahusada/filtered_eve_V2Ilurus30.csv")
# rssi_A_raw = df["alice_rssi"].to_list()
# rssi_B_raw = df["bob_rssi"].to_list()
# print(f"Data loaded - Alice: {len(rssi_A_raw)} samples, Bob: {len(rssi_B_raw)} samples")

start_csp = time.time()
if len(rssi_A_raw) >= 15 and len(rssi_A_raw) < 25:
    BLOCK_SIZE = 15
elif len(rssi_A_raw) >= 25 and len(rssi_A_raw) < 40:
    BLOCK_SIZE = 15
elif len(rssi_A_raw) >= 40:
    BLOCK_SIZE = 10

time_csp = time.time() - start_csp


#=============================================================
# =================== COMPLEXITY TEST =======================
#=============================================================

def lz76_decomposition(sequence):
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
    kuantisasi_bits = ''
    for rssi in rssi_block:
        if rssi >= Q1[0] and rssi <= Q1[1]:
            kuantisasi_bits += '10'
        elif rssi >= Q2[0] and rssi <= Q2[1]:
            kuantisasi_bits += '11'
        elif rssi >= Q3[0] and rssi <= Q3[1]:
            kuantisasi_bits += '01'
        elif rssi >= Q4[0] and rssi <= Q4[1]:
            kuantisasi_bits += '00'
    return kuantisasi_bits

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
    kuantisasi_bits = ''
    for rssi in rssi_block:
        if rssi >= Q1[0] and rssi <= Q1[1]:
            kuantisasi_bits += '1'
        elif rssi >= Q2[0] and rssi <= Q2[1]:
            kuantisasi_bits += '0'
    return kuantisasi_bits

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
    kuantisasi_bits = ''
    for rssi in rssi_block:
        if rssi >= Q1[0] and rssi <= Q1[1]:
            kuantisasi_bits += '000'
        elif rssi >= Q2[0] and rssi <= Q2[1]:
            kuantisasi_bits += '001'
        elif rssi >= Q3[0] and rssi <= Q3[1]:
            kuantisasi_bits += '011'
        elif rssi >= Q4[0] and rssi <= Q4[1]:
            kuantisasi_bits += '010'
        elif rssi >= Q5[0] and rssi <= Q5[1]:
            kuantisasi_bits += '110'
        elif rssi >= Q6[0] and rssi <= Q6[1]:
            kuantisasi_bits += '111'
        elif rssi >= Q7[0] and rssi <= Q7[1]:
            kuantisasi_bits += '101'
        elif rssi >= Q8[0] and rssi <= Q8[1]:
            kuantisasi_bits += '100'
    return kuantisasi_bits

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

class InformationReconciliation:
    _PRIM_POLY = 37 

    def __init__(self, n=31, k=6):
        self.n    = n
        self.k    = k
        self._bch = bchlib.BCH(5, prim_poly=self._PRIM_POLY)  # t=5

    # ── helper: bit array ↔ bytes ────────────────────────────────────── #

    @staticmethod
    def _bits_to_bytes(bits: np.ndarray) -> bytearray:
        """Pack array of 0/1 bits → bytearray, MSB first."""
        nb  = (len(bits) + 7) // 8
        out = bytearray(nb)
        for i, b in enumerate(bits):
            if b:
                out[i >> 3] |= 0x80 >> (i & 7)
        return out

    @staticmethod
    def _bytes_to_bits(ba: bytearray, n_bits: int) -> np.ndarray:
        """Unpack bytearray → array of 0/1 bits, MSB first, truncated to n_bits."""
        bits = []
        for byte in ba:
            for s in range(7, -1, -1):
                bits.append((byte >> s) & 1)
        return np.array(bits[:n_bits], dtype=np.uint8)

    def _encode_to_codeword(self, chunk_padded: np.ndarray) -> np.ndarray:
        data_b = self._bits_to_bytes(chunk_padded[:self.k])  # hanya k bit data
        ecc_b  = self._bch.encode(data_b)                    # ECC bytes dari bchlib
        ecc_bits = self._bytes_to_bits(
            bytearray(ecc_b),
            len(ecc_b) * 8
        )
        # Codeword = [k bit data | ecc bits] — sistematis, sama seperti galois systematic=True
        codeword = np.concatenate([chunk_padded[:self.k], ecc_bits])
        # Pastikan panjang = n
        if len(codeword) < self.n:
            codeword = np.concatenate([codeword, np.zeros(self.n - len(codeword), dtype=np.uint8)])
        return codeword[:self.n].astype(np.uint8)

    # ── Bob side ─────────────────────────────────────────────────────── #

    def bob_process_chunked(self, KB: np.ndarray):

        chunks_data = []
        num_chunks  = (len(KB) + self.k - 1) // self.k

        for i in range(num_chunks):
            start           = i * self.k
            chunk           = KB[start : start + self.k].copy()
            original_length = len(chunk)

            # Pad ke k bits jika chunk terakhir lebih pendek
            if len(chunk) < self.k:
                chunk = np.concatenate([
                    chunk,
                    np.zeros(self.k - len(chunk), dtype=np.uint8)
                ])

            # Pad ke n bits untuk XOR dengan codeword
            KB_padded = np.concatenate([
                chunk,
                np.zeros(self.n - self.k, dtype=np.uint8)
            ])

            # Encode → codeword (n bits), lalu XOR — sama persis dengan versi galois
            c  = self._encode_to_codeword(KB_padded)
            sk = (KB_padded ^ c).astype(np.uint8)  # sk = KB_padded XOR codeword

            crc = zlib.crc32(chunk[:original_length].tobytes())

            chunks_data.append({
                'sk'             : sk,
                'crc'            : crc,
                'original_length': original_length,
                'chunk_index'    : i,
            })

        return chunks_data

    # ── Alice side ───────────────────────────────────────────────────── #

    def alice_process_chunked(self, KA: np.ndarray, chunks_data: list, verbose=True):

        reconstructed_key = []
        failed_chunks     = []
        chunk_details     = []

        for chunk_info in chunks_data:
            chunk_idx       = chunk_info['chunk_index']
            expected_length = chunk_info['original_length']
            start           = chunk_idx * self.k

            # Ambil chunk Alice
            if start >= len(KA):
                chunk_a      = np.zeros(expected_length, dtype=np.uint8)
                actual_length = 0
            else:
                end_idx      = min(start + expected_length, len(KA))
                chunk_a      = KA[start : end_idx].copy()
                actual_length = len(chunk_a)
                if len(chunk_a) < expected_length:
                    chunk_a = np.concatenate([
                        chunk_a,
                        np.zeros(expected_length - len(chunk_a), dtype=np.uint8)
                    ])

            # Pad ke k jika perlu
            if len(chunk_a) < self.k:
                chunk_a = np.concatenate([
                    chunk_a,
                    np.zeros(self.k - len(chunk_a), dtype=np.uint8)
                ])

            # Pad ke n untuk XOR
            KA_padded = np.concatenate([
                chunk_a,
                np.zeros(self.n - self.k, dtype=np.uint8)
            ])

            # r = KA_padded XOR sk  — identik dengan versi galois
            sk_arr = np.array(chunk_info['sk'], dtype=np.uint8)
            r      = (KA_padded ^ sk_arr).astype(np.uint8)

            chunk_status = {'chunk_idx': chunk_idx, 'status': 'UNKNOWN', 'reason': ''}

            try:
                # Pisahkan r menjadi data_bits (k bit) dan ecc_bits (n-k bit)
                data_bits = r[:self.k]
                ecc_bits  = r[self.k:self.n]

                data_b = bytearray(self._bits_to_bytes(data_bits))
                ecc_b  = bytearray(self._bits_to_bytes(ecc_bits))

                # Decode BCH: cari dan hitung jumlah error
                nerr = self._bch.decode(data_b, ecc_b)

                if nerr < 0:
                    # Di luar kemampuan koreksi BCH
                    if verbose:
                        print(f"    Chunk {chunk_idx}: BCH decode failed (nerr={nerr}, uncorrectable)")
                    chunk_status.update(status='FAIL', reason=f'BCH uncorrectable (nerr={nerr})')
                    failed_chunks.append(chunk_idx)
                    chunk_details.append(chunk_status)
                    continue

                # Koreksi error in-place
                self._bch.correct(data_b, ecc_b)

                # Ambil message (k bit) dari data yang sudah dikoreksi
                KA_prime_chunk = self._bytes_to_bits(data_b, self.k)
                KA_prime_valid = KA_prime_chunk[:expected_length]

                # Length check
                if actual_length != expected_length:
                    if verbose:
                        print(f"    Chunk {chunk_idx}: Length mismatch (Alice:{actual_length}, Bob:{expected_length})")
                    chunk_status.update(status='FAIL', reason='Length mismatch')
                    failed_chunks.append(chunk_idx)
                    chunk_details.append(chunk_status)
                    continue

                # CRC check
                vA_k = zlib.crc32(KA_prime_valid.tobytes())
                vB_k = chunk_info['crc']
                if vA_k != vB_k:
                    if verbose:
                        print(f"    Chunk {chunk_idx}: CRC mismatch (Alice:{hex(vA_k)}, Bob:{hex(vB_k)})")
                    chunk_status.update(status='FAIL', reason='CRC mismatch')
                    failed_chunks.append(chunk_idx)
                    chunk_details.append(chunk_status)
                    continue

                if verbose:
                    # print(f"    Chunk {chunk_idx}: ✓ SUCCESS (CRC:{hex(vA_k)}, errors corrected:{nerr})")
                    pass

                chunk_status.update(status='SUCCESS', reason='CRC match')
                reconstructed_key.extend(KA_prime_valid.tolist())
                chunk_details.append(chunk_status)

            except Exception as e:
                if verbose:
                    print(f"    Chunk {chunk_idx}: Exception — {e}")
                chunk_status.update(status='ERROR', reason=str(e))
                failed_chunks.append(chunk_idx)
                chunk_details.append(chunk_status)

        total_chunks   = len(chunks_data)
        success_chunks = total_chunks - len(failed_chunks)

        if verbose and total_chunks > 1:
            print(f"    Chunk Summary: {success_chunks}/{total_chunks} chunks succeeded")
            if failed_chunks:
                print(f"    Failed chunks: {failed_chunks}")

        if not failed_chunks:
            return np.array(reconstructed_key, dtype=np.uint8), 1
        else:
            return None, 0


def reconciliation_per_block_with_calibration(rssi_A_full, rssi_B_full, delta_alpha=0.2, NC=3, block_size=BLOCK_SIZE, verbose=True):
    ir = InformationReconciliation()
    final_key = ""
    success_indices = []
    blocks_data = []
    keys_A_all = []
    keys_B_all = []

    total_blocks = len(rssi_A_full) // block_size
    total_error_before = 0
    total_error_after = 0
    dropped_blocks = 0
    calibration_counts = []

    total_time_complexity = 0.0
    total_time_quantization = 0.0
    total_time_reconciliation = 0.0

    complexity_values = []
    m_values = []
    alpha_values = []

    print(f"\n{'='*70}")
    print(f"RECONCILIATION WITH PER-BLOCK COMPLEXITY ESTIMATION")
    print(f"{'='*70}")
    print(f"Total blocks: {total_blocks} | Block size: {block_size} RSSI")
    print(f"Δα={delta_alpha} | NC={NC}")
    print(f"{'='*70}\n")

    for block_idx in range(total_blocks):
        start_idx = block_idx * block_size
        end_idx = (block_idx + 1) * block_size
        rssi_A_block = rssi_A_full[start_idx:end_idx]
        rssi_B_block = rssi_B_full[start_idx:end_idx]

        print(f"{'='*70}")
        print(f"BLOCK {block_idx} (RSSI {start_idx}-{end_idx-1})")
        print(f"{'='*70}")

        _t0 = time.time()
        complexity_block = calculate_complexity_for_block(rssi_A_block)
        m_block = determine_m_from_complexity(complexity_block)
        alpha_init_block = guard_band(complexity_block, m_block)
        total_time_complexity += time.time() - _t0

        print(f"Complexity: {complexity_block:.4f} | m: {m_block} | Initial α: {alpha_init_block:.4f}")
        complexity_values.append(complexity_block)
        m_values.append(m_block)
        alpha_values.append(alpha_init_block)

        success = False
        alpha_k = alpha_init_block
        kdr_block = 0
        block_info = {
            'block_id': block_idx,
            'complexity': f"{complexity_block:.4f}",
            'm': m_block,
            'alpha_init': f"{alpha_init_block:.4f}"
        }

        for nc in range(NC + 1):
            print(f"\nCalibration {nc}: α={alpha_k:.4f}")
            _t1 = time.time()
            key_A_block = kuantisasi_per_block(rssi_A_block, alpha_k, m_block)
            key_B_block = kuantisasi_per_block(rssi_B_block, alpha_k, m_block)
            total_time_quantization += time.time() - _t1

            # print(f"  Alice bits: {key_A_block[:20]}... ({len(key_A_block)} bits)")
            # print(f"  Bob bits  : {key_B_block[:20]}... ({len(key_B_block)} bits)")

            if len(key_A_block) != len(key_B_block):
                min_len = min(len(key_A_block), len(key_B_block))
                key_A_block = key_A_block[:min_len]
                key_B_block = key_B_block[:min_len]

            KA_k = to_bits(key_A_block)
            KB_k = to_bits(key_B_block)
            err_before = count_bit_errors(KA_k, KB_k)
            kdr_block = err_before / len(KB_k) if len(KB_k) > 0 else 0

            num_chunks = (len(KB_k) + ir.k - 1) // ir.k
            if num_chunks > 1:
                print(f"  Total chunks: {num_chunks}")
                for chunk_idx in range(min(num_chunks, 5)):
                    start = chunk_idx * ir.k
                    end = min(start + ir.k, len(KB_k))
                    chunk_errors = count_bit_errors(KA_k[start:end], KB_k[start:end])
                    chunk_kdr = chunk_errors / (end - start) if (end - start) > 0 else 0
                    # print(f"    Pre-BCH Chunk {chunk_idx}: {chunk_errors} errors / {end-start} bits (KDR: {chunk_kdr:.2%})")
                if num_chunks > 5:
                    # print(f"    ... and {num_chunks - 5} more chunks")
                    pass

            print(f"  Bits: {len(key_A_block)} | Errors: {err_before} | KDR: {kdr_block:.4f} ({kdr_block:.2%})")

            if nc == 0:
                total_error_before += err_before
                keys_A_all.append(key_A_block)
                keys_B_all.append(key_B_block)
                block_info['key_before'] = key_A_block
                block_info['key_length'] = len(key_A_block)
                block_info['errors_before'] = err_before
                block_info['kdr_before'] = f"{kdr_block:.4f}"

            print(f"  → Trying BCH reconciliation...")
            _t2 = time.time()
            try:
                if len(KB_k) < ir.k:
                    print(f"  ✗ Block too short ({len(KB_k)} < {ir.k}), dropping")
                    total_time_reconciliation += time.time() - _t2
                    break
                chunks_data = ir.bob_process_chunked(KB_k)
                KA_prime, IRCk = ir.alice_process_chunked(KA_k, chunks_data, verbose=True)
                if IRCk == 1:
                    err_after = count_bit_errors(KA_prime, KB_k)
                    total_error_after += err_after
                    final_key += "".join(KA_prime.astype(str))
                    success_indices.append(block_idx)
                    calibration_counts.append(nc)
                    success = True
                    block_info['status'] = 'SUCCESS'
                    block_info['key_after'] = "".join(KA_prime.astype(str))
                    block_info['errors_after'] = err_after
                    block_info['calibrations'] = nc
                    block_info['alpha_final'] = f"{alpha_k:.4f}"
                    block_info['kdr_final'] = f"{kdr_block:.4f}"
                    block_info['IRCk'] = 1
                    print(f"  ✓ SUCCESS - IRC=1 | KDR: {kdr_block:.2%} | Calibrations: {nc}")
                    total_time_reconciliation += time.time() - _t2
                    break
                else:
                    print(f"  ✗ IRC=0")
                    total_time_reconciliation += time.time() - _t2
                    if nc < NC:
                        alpha_k += delta_alpha
                        print(f"  → Will try calibration {nc+1} with α={alpha_k:.4f}")
                    else:
                        print(f"  ✗ No more calibrations, DROPPING block")
                        break
            except Exception as e:
                print(f"  ✗ BCH Error: {str(e)}")
                total_time_reconciliation += time.time() - _t2
                if nc < NC:
                    alpha_k += delta_alpha
                else:
                    break

        if not success:
            dropped_blocks += 1
            block_info['status'] = 'DROPPED'
            block_info['IRCk'] = 0
            block_info['key_after'] = ''
            block_info['kdr_final'] = f"{kdr_block:.4f}"
            print(f"\n✗ Result: DROPPED (IRC=0 after {NC+1} attempts)")

        print()
        blocks_data.append(block_info)

    total_bits_quantization = sum(len(k) for k in keys_A_all)
    total_errors_quantization = sum(
        count_bit_errors(to_bits(keys_A_all[i]), to_bits(keys_B_all[i]))
        for i in range(len(keys_A_all))
    )
    overall_kdr = total_errors_quantization / total_bits_quantization if total_bits_quantization > 0 else 0

    print(f"\n{'='*70}")
    print(f"QUANTIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total blocks  : {total_blocks}")
    print(f"Total bits    : {total_bits_quantization} bits")
    print(f"Total errors  : {total_errors_quantization}")
    print(f"Overall KDR   : {overall_kdr:.4f} ({overall_kdr:.2%})")

    print(f"\n{'='*70}")
    print(f"RECONCILIATION SUMMARY")
    print(f"{'='*70}")
    print(f"Success : {len(success_indices)}/{total_blocks} blocks ({len(success_indices)/total_blocks:.1%})")
    print(f"Dropped : {dropped_blocks}/{total_blocks} blocks")
    print(f"Final key (before PA): {len(final_key)} bits")

    return final_key, success_indices, {
        'total_blocks': total_blocks,
        'successful_blocks': len(success_indices),
        'dropped_blocks': dropped_blocks,
        'drop_rate': dropped_blocks / total_blocks,
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
    }

#=================== PRIVACY AMPLIFICATION ===============

NIST_APEN_M   = 2
NIST_ENTROPY_THRESHOLD = 0.01

def hex_to_bit_array(hex_str):
    print(repr(hex_str))
    print(len(hex_str))
    bit_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)
    return np.fromiter(bit_str, dtype=np.int8)

def nist_frequency_test(bits):
    n = len(bits)
    if n < 16:
        return 0.0
    s = sum(1 if b == 1 else -1 for b in bits)
    s_obs = abs(s) / math.sqrt(n)
    return float(erfc(s_obs / math.sqrt(2.0)))

def nist_block_frequency_test(bits, m=128):
    n = len(bits)
    if m <= 0 or n < m:
        return 0.0
    N = n // m
    if N == 0:
        return 0.0
    cs = 0.0
    for i in range(N):
        block = bits[i*m:(i+1)*m]
        pi = sum(block) / m
        cs += 4.0 * m * (pi - 0.5)**2
    return float(gammaincc(N/2.0, cs/2.0))

def nist_runs_test(bits):
    n = len(bits)
    if n < 100:
        return 0.0
    pi = sum(bits) / n
    tau = 2.0 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return 0.0
    v_obs = 1 + sum(1 for i in range(1, n) if bits[i] != bits[i-1])
    num = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    if den == 0:
        return 0.0
    return float(erfc(num / den))

def nist_longest_runs_test(bits):
    n = len(bits)
    if n < 128:
        return 0.0
    if n < 6272:
        M = 8; v_values = [1,2,3,4]; pi_vals = [0.21484375,0.3671875,0.23046875,0.1875]
    elif n < 750000:
        M = 128; v_values = [4,5,6,7,8,9]
        pi_vals = [0.1174035788,0.242955959,0.249363483,0.17517706,0.102701071,0.112398847]
    else:
        M = 10000; v_values = [10,11,12,13,14,15,16]
        pi_vals = [0.0882,0.2092,0.2483,0.1933,0.1208,0.0675,0.0727]
    N = n // M
    if N == 0:
        return 0.0
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

def nist_cumulative_sums_test(bits, mode='forward'):
    n = len(bits)
    if n < 100:
        return 0.0
    x = np.array([1 if b == 1 else -1 for b in bits], dtype=int)
    if mode == 'reverse':
        x = x[::-1]
    s = np.cumsum(x)
    z = int(np.max(np.abs(s)))
    if z == 0:
        return 1.0
    sum1 = 0.0; sum2 = 0.0
    for k in range(int(np.floor((-n/z + 1)/4)), int(np.floor((n/z - 1)/4)) + 1):
        sum1 += math.exp(-((4*k+1)**2) * z*z / (2.0*n))
    for k in range(int(np.floor((-n/z - 3)/4)), int(np.floor((n/z - 1)/4)) + 1):
        sum2 += math.exp(-((4*k+3)**2) * z*z / (2.0*n))
    p = 1.0 - sum1 + sum2
    return float(max(0.0, min(1.0, p)))

def nist_fft_test(bits):
    n = len(bits)
    if n < 1000:
        return 0.0
    x = np.array([2*b - 1 for b in bits], dtype=float)
    s = np.abs(fft(x))[:n//2]
    T = math.sqrt(math.log(1.0/0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = np.sum(s < T)
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return float(erfc(abs(d) / math.sqrt(2.0)))

def nist_approximate_entropy_test(bits, m=NIST_APEN_M):
    n = len(bits)
    if n < max(16, 2*m + 1):
        return 0.0
    def _phi(mm):
        counts = {}
        for i in range(n):
            pattern = tuple(bits[(i+j) % n] for j in range(mm))
            counts[pattern] = counts.get(pattern, 0) + 1
        total = 0.0
        for c in counts.values():
            p = c / n
            total += p * math.log(p)
        return total
    phi_m  = _phi(m)
    phi_m1 = _phi(m + 1)
    ap_en  = phi_m - phi_m1
    chi_sq = 2.0 * n * (math.log(2.0) - ap_en)
    return float(gammaincc(2**(m-1), chi_sq / 2.0))

def nist_serial_test(bits, m=2):
    n = len(bits)
    if n < max(16, 2*m + 1):
        return (0.0, 0.0)
    def psi(mm):
        counts = {}
        for i in range(n):
            patt = tuple(bits[(i+j) % n] for j in range(mm))
            counts[patt] = counts.get(patt, 0) + 1
        s = sum(v*v for v in counts.values())
        return (s * (2.0**mm) / n) - n
    psi_m   = psi(m)
    psi_m1  = psi(m + 1)
    psi_m_1 = psi(m - 1) if m > 1 else 0.0
    d1 = max(psi_m - psi_m1, 0.0)
    d2 = max(psi_m - psi_m_1, 0.0)
    p1 = gammaincc(2**(m-1)/2.0, d1/2.0)
    p2 = gammaincc(2**(m-2)/2.0, d2/2.0) if m > 1 else 0.0
    return (float(p1), float(p2))

def privacy_amplification_multi_key(reconciled_key_str):

    if not reconciled_key_str:
        print("Error: Bit hasil rekonsiliasi kosong!")
        return [], None, None, None

    BASE_CHUNK_SIZE = 10
    MAX_CHUNK_SIZE  = 45
    CHUNK_STEP      = 2

    n_total = len(reconciled_key_str)

    all_keys = []
    all_key_bits = []
    seen_keys = set()

    duplicate_count = 0
    pos = 0
    chunk_idx = 0
    stop_flag = False

    time_abbc = 0
    time_sha = 0

    # ==========================================================
    # ABS + SHA-256 PER CHUNK
    # ==========================================================

    while pos < n_total and not stop_flag:

        chunk_size = BASE_CHUNK_SIZE
        chunk_bits_str = None

        start_time_abbc = time.time()

        # ── ADAPTIVE CHUNK SIZE ──────────────────────────────
        while True:

            remaining = n_total - pos

            if remaining <= 0:
                break

            if remaining < chunk_size:
                candidate = reconciled_key_str[pos:]
            else:
                candidate = reconciled_key_str[
                    pos : pos + chunk_size
                ]

            count_1 = candidate.count('1')
            count_0 = candidate.count('0')

            balance = (
                abs(count_1 - count_0) / len(candidate)
            )

            if balance < 0.3:

                chunk_bits_str = candidate
                break

            else:

                next_chunk_size = chunk_size + CHUNK_STEP

                if next_chunk_size > MAX_CHUNK_SIZE:

                    # Check remaining bits as residual candidate
                    candidate = reconciled_key_str[pos:]

                    count_1 = candidate.count('1')
                    count_0 = candidate.count('0')

                    balance = (
                        abs(count_1 - count_0)
                        / len(candidate)
                    )

                    if balance <= 0.5:
                        chunk_bits_str = candidate
                    else:
                        stop_flag = True

                    break

                else:
                    chunk_size = next_chunk_size

        # ── NO VALID CHUNK ──────────────────────────────────
        if not chunk_bits_str:
            break

        pos += len(chunk_bits_str)

        time_abbc += time.time() - start_time_abbc

        # ======================================================
        # SHA-256
        # ======================================================

        start_time_sha = time.time()

        hash_obj = hashlib.sha256(
            chunk_bits_str.encode()
        )

        key_hex = hash_obj.hexdigest()

        # ── DUPLICATE CHECK ─────────────────────────────────
        if key_hex in seen_keys:
            duplicate_count += 1
            chunk_idx += 1
            continue

        seen_keys.add(key_hex)

        # Convert SHA-256 output to bits
        key_bits = hex_to_bit_array(key_hex)

        # Store key information
        entry = {
            'chunk_idx': chunk_idx,
            'chunk_size': len(chunk_bits_str),
            'input_bits': chunk_bits_str,
            'key_hex': key_hex,
            'bits': key_bits
        }

        all_keys.append(entry)

        # Store bits for later NIST evaluation
        all_key_bits.extend(key_bits)

        chunk_idx += 1

        time_sha += time.time() - start_time_sha

    # ==========================================================
    # CONCATENATE ALL SHA-256 OUTPUTS
    # ==========================================================

    concatenated_key_bits = all_key_bits

    print("\n" + "=" * 70)
    print("HASIL ABS + SHA-256")
    print("=" * 70)

    print(f"Total chunk diproses   : {chunk_idx}")
    print(f"Chunk duplikat dibuang : {duplicate_count}")
    print(f"Key yang dihasilkan    : {len(all_keys)}")
    print(f"Total bit untuk NIST   : {len(concatenated_key_bits)}")

    if len(all_keys) == 0:
        print("Tidak ada key yang dihasilkan.")
        return [], None, time_abbc, time_sha

    # ==========================================================
    # NIST TEST PADA HASIL CONCATENATED SHA-256
    # ==========================================================

    print("\n" + "=" * 70)
    print("NIST RANDOMNESS TEST")
    print("=" * 70)

    nist_results = {}

    nist_results['Frequency'] = (
        nist_frequency_test(concatenated_key_bits)
    )

    nist_results['Block Frequency'] = (
        nist_block_frequency_test(
            concatenated_key_bits,
            m=128
        )
    )

    nist_results['Runs'] = (
        nist_runs_test(concatenated_key_bits)
    )

    nist_results['Longest Run'] = (
        nist_longest_runs_test(concatenated_key_bits)
    )

    nist_results['Cumulative Sums Forward'] = (
        nist_cumulative_sums_test(
            concatenated_key_bits,
            mode='forward'
        )
    )

    nist_results['Cumulative Sums Reverse'] = (
        nist_cumulative_sums_test(
            concatenated_key_bits,
            mode='reverse'
        )
    )

    nist_results['FFT'] = (
        nist_fft_test(concatenated_key_bits)
    )

    nist_results['Approximate Entropy'] = (
        nist_approximate_entropy_test(
            concatenated_key_bits,
            m=NIST_APEN_M
        )
    )

    serial_p1, serial_p2 = nist_serial_test(
        concatenated_key_bits,
        m=2
    )

    nist_results['Serial 1'] = serial_p1
    nist_results['Serial 2'] = serial_p2

    # ==========================================================
    # DISPLAY RESULTS
    # ==========================================================

    for test_name, p_value in nist_results.items():

        status = (
            "✓ PASS"
            if p_value > NIST_ENTROPY_THRESHOLD
            else "✗ FAIL"
        )

        print(
            f"{test_name:<30} : "
            f"{p_value:.6f}  {status}"
        )

    return (
        all_keys,
        nist_results,
        time_abbc,
        time_sha
    )


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
            key_after = block.get('key_after', '')
            if key_after:
                for bit in key_after:
                    writer.writerow([bit, bit])
    print(f"✓ Reconciliation results saved to: {filename}")

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_cp = len(rssi_A_raw) * 2 # Fixed channel probing time
    start_wall = time.time()
    start_cpu = time.process_time()

    # KALMAN FILTER PREPROCESSING

    KF_PARAMS = {
        'a'               : 1,
        'h'               : 1,
        'R'               : 1,
        'Q'               : 0.01,
        'xaposteriori_0'  : -5,
        'paposteriori_0'  : 1
    }

    rssi_A = rssi_A_raw
    rssi_B = rssi_B_raw

    # PER-BLOCK PROCESSING + RECONCILIATION

    kunci_ir_alice, blok_sukses, stats = reconciliation_per_block_with_calibration(
        rssi_A_full=rssi_A,
        rssi_B_full=rssi_B,
        delta_alpha=0.2,
        NC=3,
        block_size=BLOCK_SIZE,
        verbose=True
    )

    time_complexity             = stats['time_complexity']
    time_complexity_and_quant   = stats['time_quantization']
    time_reconciliation         = stats['time_reconciliation']

    total_bits_after_reconciliation = len(kunci_ir_alice)

    # PRIVACY AMPLIFICATION — PER-CHUNK SHA-256 + NIST FILTER

    if len(kunci_ir_alice) > 0:
        all_keys, nist_results, time_abbc, time_sha = privacy_amplification_multi_key(
            reconciled_key_str=kunci_ir_alice,
        )

        # KGR CALCULATION

        bits_after_pa = len(all_keys) * 256 if all_keys else 0

        total_time_after_quant = time_cp + time_complexity_and_quant + time_complexity + time_csp
        total_time_after_recon = total_time_after_quant + time_reconciliation
        total_time_after_pa    = total_time_after_recon + time_abbc + time_sha 
        bits_after_quantization = stats['total_bits_quantization']
        kgr_quantization   = bits_after_quantization         / total_time_after_quant
        kgr_reconciliation = total_bits_after_reconciliation / total_time_after_recon
        kgr_pa             = bits_after_pa                   / total_time_after_pa if bits_after_pa > 0 else 0.0

        print(f"\nTime Breakdown:")
        print(f"  Time CP (Channel Probing)  : {time_cp:.6f} s (FIXED)")
        print(f"  Time Complexity            : {time_complexity:.6f} s")
        print(f"  Time CSP          : {time_csp:.6f} s")
        print(f"  Time Quantization          : {time_complexity_and_quant:.6f} s")
        print(f"  Time Reconciliation        : {time_reconciliation:.6f} s")
        print(f"  Time ABBC        : {time_abbc:.6f} s")
        print(f"  Time SHA-256               : {time_sha:.6f} s")

        print(f"\nCumulative Times:")
        print(f"  After Quantization         : {total_time_after_quant:.6f} s")
        print(f"  After Reconciliation       : {total_time_after_recon:.6f} s")
        print(f"  After PA                   : {total_time_after_pa:.6f} s")

        print(f"\nBit Generation:")
        print(f"  Bits after Quantization    : {bits_after_quantization} bits")
        print(f"  Bits after Reconciliation  : {total_bits_after_reconciliation} bits")
        print(f"  Keys generated by ABS     : {len(all_keys)} key(s) × 256 bit = {bits_after_pa} bits")

        print(f"\n{'='*70}")
        print(f"  KGR After Quantization     : {kgr_quantization:.4f} bits/second")
        print(f"  KGR After Reconciliation   : {kgr_reconciliation:.4f} bits/second")
        print(f"  KGR After PA               : {kgr_pa:.4f} bits/second")
        print(f"{'='*70}")

        time_data = {
            'cp'               : time_cp,
            # 'kalman'           : time_kf,
            'complexity'       : time_complexity_and_quant,
            'quantization'     : 0,
            'reconciliation'   : time_reconciliation,
            # 'pa'               : time_pa,
            'total_after_quant': total_time_after_quant,
            'total_after_recon': total_time_after_recon,
            'total_after_pa'   : total_time_after_pa,
        }
        kgr_data = {
            'kgr_quant'   : kgr_quantization,
            'kgr_recon'   : kgr_reconciliation,
            'kgr_pa'      : kgr_pa,
            'bits_quant'  : bits_after_quantization,
            'bits_recon'  : total_bits_after_reconciliation,
            'bits_pa'     : bits_after_pa,
            # 'keys_total'  : total_bits_after_reconciliation // PA_CHUNK_SIZE,
            'keys_passed' : len(all_keys),
        }

        # AES ENCRYPTION — pakai key dengan entropy tertinggi

        if all_keys:
            selected_key = all_keys[0]
            kunci_aes_256 = selected_key['key_hex']

            print(f"Menggunakan key (Chunk {selected_key['chunk_idx']})")
            print(f"Key (hex): {kunci_aes_256.upper()}")

            pesan_awal = "Halo Bob! Ini pesan rahasia kita berdua."
            print(f"\nPlaintext : '{pesan_awal}'")

            iv, pesan_terenkripsi = alice_encrypt(
                pesan_awal,
                kunci_aes_256
            )

            print(f"Ciphertext: {pesan_terenkripsi}")
            print(f"IV        : {iv}")

            pesan_akhir = bob_decrypt(
                iv,
                pesan_terenkripsi,
                kunci_aes_256
            )

            print(f"Decrypted : '{pesan_akhir}'")

            if pesan_awal == pesan_akhir:
                print("✓ Status: SUCCESS")
            else:
                print("✗ Status: FAILED")

        else:
            print("⚠ Tidak ada key yang dihasilkan oleh ABS.")

        # NIST RANDOMNESS TESTS

        if all_keys:
            print("\n" + "=" * 70)
            print("NIST RANDOMNESS TEST - CONCATENATED SHA-256 OUTPUT")
            print("=" * 70)

            print(f"  Total generated keys : {len(all_keys)}")
            print(f"  Total bits tested    : {len(all_keys) * 256}")

            print(f"\n  Frequency Test       : {nist_results['Frequency']:.6f}")
            print(f"  Block Frequency Test : {nist_results['Block Frequency']:.6f}")
            print(f"  Runs Test            : {nist_results['Runs']:.6f}")
            print(f"  Longest Run Test     : {nist_results['Longest Run']:.6f}")
            print(f"  FFT Test             : {nist_results['FFT']:.6f}")
            print(f"  Cumulative Sums Fwd  : {nist_results['Cumulative Sums Forward']:.6f}")
            print(f"  Cumulative Sums Rev  : {nist_results['Cumulative Sums Reverse']:.6f}")
            print(f"  Approximate Entropy  : {nist_results['Approximate Entropy']:.6f}")
            print(f"  Serial Test          : "
                f"p1={nist_results['Serial 1']:.6f}, "
                f"p2={nist_results['Serial 2']:.6f}")

        else:
            print("  Tidak ada key yang dihasilkan ABS.")

        print(f"\n{'='*70}")
        print(f"All results exported successfully!")
        print(f"Files created with timestamp: {timestamp}")
        print(f"{'='*70}")

    else:
        print("\n✗ ERROR: No key generated from reconciliation!")

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