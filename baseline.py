import math
import pandas as pd
import numpy as np
import zlib
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import binascii
import time
import csv
from datetime import datetime
from scipy.special import erfc, gammaincc
from numpy.fft import fft
import bchlib

BLOCK_SIZE = 40

# baca file csv
df = pd.read_csv("combined_V2V_searah30.csv")
# df = pd.read_csv("combined_v2i lurus 30.csv")
#df = pd.read_csv("darmahusada/combined_V2Ibelok20.csv")
#df = pd.read_csv("darmahusada/combined_V2Ilurus30.csv")
# df = pd.read_csv("pocan/combined_v2vlawan30.csv")
# df = pd.read_csv("pocan/combined_v2vsearah30.csv")
# df = pd.read_csv("pocan/combined_v2ilurus30km.csv")
# df = pd.read_csv("combined_v2i lurus 30.csv")
rssi_A = df["rssi_alice"].to_list()
rssi_B = df["rssi_bob"].to_list()
# print(len(rssi_A))
# print(len(rssi_B))

#=============================================================
# =================== TES KOMPLEKSITAS =======================
#=============================================================

def lz76_decomposition(sequence):
    """LZ76 compression untuk menghitung kompleksitas"""
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
    
    # Normalisasi: jumlah komponen / panjang sequence
    complexity = len(components) / n if n > 0 else 0
    return complexity, components

def calculate_complexity_for_block(rssi_block):
    """Hitung kompleksitas untuk satu diversity block"""
    complexity, components = lz76_decomposition(rssi_block)
    return complexity

def determine_m_from_complexity(complexity):
    """Tentukan nilai m berdasarkan kompleksitas"""
    if complexity < 0.3:
        return 2
    elif complexity < 0.675:
        return 4
    else:
        return 8

def guard_band(complexity, m):
    """Hitung guard band parameter alpha berdasarkan kompleksitas dan m"""
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
#============== KUANTISASI =====================
#===============================================

def kuantisasi_4_block(rssi_block, a):
    """Kuantisasi untuk 1 diversity block (40 RSSI measurements) dengan m=4"""
    q0 = min(rssi_block)
    q4 = max(rssi_block)
    q2 = sum(rssi_block)/len(rssi_block)

    q1_arr = [rssi for rssi in rssi_block if rssi >= q0 and rssi <= q2]
    q3_arr = [rssi for rssi in rssi_block if rssi > q2 and rssi <= q4]
    
    q1 = sum(q1_arr)/len(q1_arr) if len(q1_arr) > 0 else q0
    q3 = sum(q3_arr)/len(q3_arr) if len(q3_arr) > 0 else q4

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
    """Kuantisasi untuk 1 diversity block dengan m=2"""
    q0 = min(rssi_block)
    q2 = max(rssi_block)
    q1 = sum(rssi_block)/len(rssi_block)

    temp_s1_Arr = [rssi for rssi in rssi_block if rssi > q0 and rssi <= q2]
    mean_s1 = sum(temp_s1_Arr)/len(temp_s1_Arr) if len(temp_s1_Arr) > 0 else 0
    s1 = math.sqrt(sum((rssi - mean_s1)**2 for rssi in temp_s1_Arr) / len(temp_s1_Arr)) if len(temp_s1_Arr) > 0 else 0

    Q1 = [q0, q1 - (a * s1)]
    Q2 = [q1 + (a * s1), q2]

    kuantisasi_bits = ''
    for rssi in rssi_block:
        if rssi >= Q1[0] and rssi <= Q1[1]:
            kuantisasi_bits += '0'
        elif rssi >= Q2[0] and rssi <= Q2[1]:
            kuantisasi_bits += '1'

    return kuantisasi_bits

def kuantisasi_8_block(rssi_block, a):
    """Kuantisasi untuk 1 diversity block dengan m=8"""
    q0 = min(rssi_block)
    q8 = max(rssi_block)
    q4 = sum(rssi_block)/len(rssi_block)

    q2_arr = [rssi for rssi in rssi_block if rssi >= q0 and rssi <= q4]
    q2 = sum(q2_arr)/len(q2_arr) if len(q2_arr) > 0 else q0

    q6_arr = [rssi for rssi in rssi_block if rssi > q4 and rssi <= q8]
    q6 = sum(q6_arr)/len(q6_arr) if len(q6_arr) > 0 else q8

    q1_arr = [rssi for rssi in rssi_block if rssi >= q0 and rssi <= q2]
    q1 = sum(q1_arr)/len(q1_arr) if len(q1_arr) > 0 else q0

    q3_arr = [rssi for rssi in rssi_block if rssi > q2 and rssi <= q4]
    q3 = sum(q3_arr)/len(q3_arr) if len(q3_arr) > 0 else q2

    q5_arr = [rssi for rssi in rssi_block if rssi > q4 and rssi <= q6]
    q5 = sum(q5_arr)/len(q5_arr) if len(q5_arr) > 0 else q4

    q7_arr = [rssi for rssi in rssi_block if rssi > q6 and rssi <= q8]
    q7 = sum(q7_arr)/len(q7_arr) if len(q7_arr) > 0 else q6

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
    """Wrapper function untuk kuantisasi per diversity block"""
    if m == 2:
        return kuantisasi_2_block(rssi_block, a)
    elif m == 4:
        return kuantisasi_4_block(rssi_block, a)
    elif m == 8:
        return kuantisasi_8_block(rssi_block, a)
    else:
        raise ValueError(f"Invalid m value: {m}")

#=========================================
#=========== REKONSILIASI ================
#=========================================

def to_bits(bit_string):
    """Convert string bit ke numpy array"""
    return np.array([int(b) for b in bit_string], dtype=np.uint8)

def count_bit_errors(bits_a, bits_b):
    """Hitung jumlah bit yang berbeda antara dua sequence"""
    bits_a = np.array(bits_a, dtype=int)
    bits_b = np.array(bits_b, dtype=int)
    min_len = min(len(bits_a), len(bits_b))
    return int(np.sum(bits_a[:min_len] != bits_b[:min_len]))

def count_chunk_errors(key_str_a, key_str_b, chunk_idx, chunk_size):
    """Hitung errors untuk chunk tertentu"""
    start = chunk_idx * chunk_size
    end = start + chunk_size
    
    chunk_a = key_str_a[start:end]
    chunk_b = key_str_b[start:end]
    
    bits_a = to_bits(chunk_a)
    bits_b = to_bits(chunk_b)
    
    return count_bit_errors(bits_a, bits_b)


class InformationReconciliation:
    """
    BCH-based Information Reconciliation menggunakan bchlib.
    Logic rekonsiliasi identik dengan versi galois (XOR-based),
    hanya mengganti backend library dari galois ke bchlib.

    Parameter BCH: t=5, prim_poly=37 (x^5+x^2+1), n=31, k=6.
    """
    _PRIM_POLY = 37  # primitive polynomial x^5 + x^2 + 1 untuk GF(2^5)

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
        """
        Encode chunk (k bits, padded ke n) → codeword (n bits) via bchlib.
        Replikasi logika galois: c = bch.encode(chunk_gf) di mana c adalah
        codeword n-bit = [data | ecc].
        """
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
        """
        Bob: encode setiap chunk BCH lalu XOR codeword dengan padded key,
        kirim sk = KB_padded XOR codeword  (identik dengan versi galois).
        Return list of dict.
        """
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
        """
        Alice: terima chunks_data dari Bob, koreksi error tiap chunk dengan BCH.
        Logic identik versi galois: r = KA_padded XOR sk → decode BCH → ambil message.
        Return: (reconstructed_key np.uint8 | None, IRC_flag 1 | 0)
        """
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
                    print(f"    Chunk {chunk_idx}: ✓ SUCCESS (CRC:{hex(vA_k)}, errors corrected:{nerr})")
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
    """
    Proses rekonsiliasi dengan PER-BLOCK complexity estimation
    
    LOGIKA BARU (sesuai paper):
    1. Untuk SETIAP diversity block:
       a. Hitung kompleksitas block
       b. Tentukan m berdasarkan kompleksitas
       c. Hitung alpha berdasarkan kompleksitas dan m
       d. Kuantisasi dengan (m, alpha) tersebut
       e. Coba BCH
       f. Jika IRC=1 → LOLOS
       g. Jika IRC=0 DAN nc < NC → increase alpha, ulangi dari step d
       h. Jika IRC=0 DAN nc == NC → DROP block
    """
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
    print(f"Strategy: Each block has its own complexity → m → alpha")
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
        
# ===== STEP 1: COMPLEXITY ESTIMATION PER BLOCK =====
        _t0 = time.time()
        complexity_block = calculate_complexity_for_block(rssi_A_block)
        m_block = determine_m_from_complexity(complexity_block)
        alpha_init_block = guard_band(complexity_block, m_block)
        total_time_complexity += time.time() - _t0
        
        print(f"Complexity: {complexity_block:.4f}")
        print(f"Selected m: {m_block}")
        print(f"Initial α: {alpha_init_block:.4f}")
        
        # Simpan nilai-nilai per block
        complexity_values.append(complexity_block)
        m_values.append(m_block)
        alpha_values.append(alpha_init_block)
        
        success = False
        alpha_k = alpha_init_block
        block_info = {
            'block_id': block_idx,
            'complexity': f"{complexity_block:.4f}",
            'm': m_block,
            'alpha_init': f"{alpha_init_block:.4f}"
        }
        
        # ===== CALIBRATION LOOP (BCH-FIRST) =====
        for nc in range(NC + 1):
            print(f"\nCalibration {nc}: α={alpha_k:.4f}")
            
            # STEP 4: KUANTISASI dengan (m, alpha) block ini
            _t1 = time.time()
            key_A_block = kuantisasi_per_block(rssi_A_block, alpha_k, m_block)
            key_B_block = kuantisasi_per_block(rssi_B_block, alpha_k, m_block)
            total_time_quantization += time.time() - _t1
            
            print(f"  Alice bits (original): {key_A_block[:20]}... ({len(key_A_block)} bits)")
            print(f"  Bob bits (original)  : {key_B_block[:20]}... ({len(key_B_block)} bits)")
            
            # Truncate ke panjang yang sama
            if len(key_A_block) != len(key_B_block):
                min_len = min(len(key_A_block), len(key_B_block))
                # print(f"  → Length mismatch! Truncating to {min_len} bits")
                key_A_block = key_A_block[:min_len]
                key_B_block = key_B_block[:min_len]
            
            KA_k = to_bits(key_A_block)
            KB_k = to_bits(key_B_block)
            
            # Hitung KDR
            err_before = count_bit_errors(KA_k, KB_k)
            kdr_block = err_before / len(KB_k) if len(KB_k) > 0 else 0
            
            # ✅ Hitung error per chunk untuk analisis
            num_chunks = (len(KB_k) + ir.k - 1) // ir.k
            if num_chunks > 1:
                # print(f"  Total chunks: {num_chunks}")
                for chunk_idx in range(min(num_chunks, 5)):  # Max 5 chunks untuk display
                    start = chunk_idx * ir.k
                    end = min(start + ir.k, len(KB_k))
                    chunk_errors = count_bit_errors(KA_k[start:end], KB_k[start:end])
                    chunk_kdr = chunk_errors / (end - start) if (end - start) > 0 else 0
                    print(f"    Pre-BCH Chunk {chunk_idx}: {chunk_errors} errors / {end-start} bits (KDR: {chunk_kdr:.2%})")
                if num_chunks > 5:
                    print(f"    ... and {num_chunks - 5} more chunks")
                    # pass
            
            print(f"  Bits: {len(key_A_block)} | Errors: {err_before} | KDR: {kdr_block:.4f} ({kdr_block:.2%})")
            
            # Simpan untuk calibration pertama
            if nc == 0:
                total_error_before += err_before
                keys_A_all.append(key_A_block)
                keys_B_all.append(key_B_block)
                block_info['key_before'] = key_A_block
                block_info['key_length'] = len(key_A_block)
                block_info['errors_before'] = err_before
                block_info['kdr_before'] = f"{kdr_block:.4f}"
            
            # STEP 5: COBA BCH
            print(f"  → Trying BCH reconciliation...")
            
            _t2 = time.time()
            try:
                if len(KB_k) < ir.k:
                    print(f"  ✗ Block too short ({len(KB_k)} < {ir.k}), dropping")
                    total_time_reconciliation += time.time() - _t2
                    break
                
                chunks_data = ir.bob_process_chunked(KB_k)
                KA_prime, IRCk = ir.alice_process_chunked(KA_k, chunks_data, verbose=True)
                
                # STEP 6: CEK IRC
                if IRCk == 1:
                    # ✅ SUCCESS
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
                    
                    print(f"  ✓ SUCCESS - IRC=1")
                    print(f"    Complexity={complexity_block:.4f}, m={m_block}, α_final={alpha_k:.4f}")
                    print(f"    KDR: {kdr_block:.2%} | Calibrations: {nc}")
                    total_time_reconciliation += time.time() - _t2
                    break
                    
                else:
                    # ❌ IRC=0
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
                    print(f"  → Will try calibration {nc+1} with α={alpha_k:.4f}")
                else:
                    break
        
        if not success:
            dropped_blocks += 1
            block_info['status'] = 'DROPPED'
            block_info['IRCk'] = 0
            block_info['key_after'] = ''
            block_info['kdr_final'] = f"{kdr_block:.4f}"
            print(f"\n✗ Result: DROPPED (IRC=0 after {NC+1} attempts)")
        
        # print()
        blocks_data.append(block_info)
    
    # Statistik
    total_bits_quantization = sum(len(k) for k in keys_A_all)
    total_errors_quantization = sum(
        count_bit_errors(to_bits(keys_A_all[i]), to_bits(keys_B_all[i])) 
        for i in range(len(keys_A_all))
    )
    overall_kdr = total_errors_quantization / total_bits_quantization if total_bits_quantization > 0 else 0
    print(f"Overall KDR: {overall_kdr:.4f} ({overall_kdr:.2%})")
    
    print(f"\n{'='*70}")
    print(f"COMPLEXITY DISTRIBUTION")
    print(f"{'='*70}")
    if len(complexity_values) > 0:
        print(f"Min complexity: {min(complexity_values):.4f}")
        print(f"Max complexity: {max(complexity_values):.4f}")
        print(f"Avg complexity: {sum(complexity_values)/len(complexity_values):.4f}")
        
        m2_count = m_values.count(2)
        m4_count = m_values.count(4)
        m8_count = m_values.count(8)
        # print(f"\nm distribution:")
        print(f"  m=2: {m2_count} blocks ({m2_count/len(m_values)*100:.1f}%)")
        print(f"  m=4: {m4_count} blocks ({m4_count/len(m_values)*100:.1f}%)")
        print(f"  m=8: {m8_count} blocks ({m8_count/len(m_values)*100:.1f}%)")
    print(f"{'='*70}\n")
    
    print(f"{'='*70}")
    print(f"QUANTIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total blocks: {total_blocks}")
    print(f"NKEY (Total bits): {total_bits_quantization} bits")
    print(f"Total errors: {total_errors_quantization}")
    print(f"Overall KDR: {overall_kdr:.4f} ({overall_kdr:.2%})")
    print(f"{'='*70}\n")
    
    print(f"{'='*70}")
    print(f"RECONCILIATION SUMMARY")
    print(f"{'='*70}")
    print(f"Success: {len(success_indices)}/{total_blocks} blocks ({len(success_indices)/total_blocks:.1%})")
    print(f"Dropped: {dropped_blocks}/{total_blocks} blocks ({dropped_blocks/total_blocks:.1%})")
    print(f"Final key: {len(final_key)} bits")
    
    if len(calibration_counts) > 0:
        avg_cal = sum(calibration_counts) / len(calibration_counts)
        print(f"Average calibrations per successful block: {avg_cal:.2f}")
    
    print(f"{'='*70}\n")
    
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

#=========================================================
#=================== PRIVACY AMPLIFICATION ===============
#=========================================================
def finalize_to_aes256(reconciled_key_str):
    """Privacy Amplification menggunakan SHA-256"""
    if not reconciled_key_str:
        print("Error: Kunci hasil IR kosong!")
        return None

    hash_object = hashlib.sha256(reconciled_key_str.encode())
    final_key_hex = hash_object.hexdigest()

    print(f"Input IR Bits: {len(reconciled_key_str)} bits")
    print(f"Final Key (Hex): {final_key_hex.upper()}")
    print(f"Key Strength: 256 bits")

    return final_key_hex

#==========================================================
#==================== AES =================================
#==========================================================
def alice_encrypt(pesan_asli, key_hex):
    """Alice melakukan enkripsi menggunakan AES-256-CBC"""
    key_bytes = binascii.unhexlify(key_hex)
    cipher = AES.new(key_bytes, AES.MODE_CBC)
    iv = cipher.iv

    pesan_padded = pad(pesan_asli.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(pesan_padded)

    return binascii.hexlify(iv).decode(), binascii.hexlify(ciphertext).decode()

def bob_decrypt(iv_hex, ciphertext_hex, key_hex):
    """Bob melakukan dekripsi menggunakan AES-256-CBC"""
    key_bytes = binascii.unhexlify(key_hex)
    iv_bytes = binascii.unhexlify(iv_hex)
    ciphertext_bytes = binascii.unhexlify(ciphertext_hex)

    cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
    pesan_decrypted = unpad(cipher.decrypt(ciphertext_bytes), AES.block_size)
    
    return pesan_decrypted.decode('utf-8')

#==========================================================
#==================== SAVE FUNCTIONS ======================
#==========================================================

def save_quantization_results(rssi_A, rssi_B, keys_A, keys_B, filename="quantization_results.csv"):
    """Save hasil kuantisasi ke CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Bit_Alice', 'Bit_Bob'])
        
        for key_a, key_b in zip(keys_A, keys_B):
            min_len = min(len(key_a), len(key_b))
            for bit_idx in range(min_len):
                writer.writerow([key_a[bit_idx], key_b[bit_idx]])
    
    # print(f"✓ Quantization results saved to: {filename}")

def save_reconciliation_results(blocks_data, filename="reconciliation_results.csv"):
    """Save hasil rekonsiliasi ke CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Bit_Alice', 'Bit_Bob'])
        
        for block in blocks_data:
            key_after = block.get('key_after', '')
            if key_after:
                for bit in key_after:
                    writer.writerow([bit, bit])
    
    # print(f"✓ Reconciliation results saved to: {filename}")

def save_final_key(final_key_bits, final_key_hex, filename="final_key.csv"):
    """Save kunci final ke CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Bit'])
        for bit in final_key_bits:
            writer.writerow([bit])
    
    # print(f"✓ Final key saved to: {filename}")

def save_timing_and_kgr(time_data, kgr_data, filename="timing_kgr_results.csv"):
    """Save timing dan KGR ke CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Unit'])
        
        writer.writerow(['Time_CP', f"{time_data['cp']:.6f}", 'seconds'])
        writer.writerow(['Time_Complexity', f"{time_data['complexity']:.6f}", 'seconds'])
        writer.writerow(['Time_Quantization', f"{time_data['quantization']:.6f}", 'seconds'])
        writer.writerow(['Time_Reconciliation', f"{time_data['reconciliation']:.6f}", 'seconds'])
        writer.writerow(['Time_Privacy_Amplification', f"{time_data['pa']:.6f}", 'seconds'])
        writer.writerow(['Total_Time_After_Quant', f"{time_data['total_after_quant']:.6f}", 'seconds'])
        writer.writerow(['Total_Time_After_Recon', f"{time_data['total_after_recon']:.6f}", 'seconds'])
        writer.writerow(['Total_Time_After_PA', f"{time_data['total_after_pa']:.6f}", 'seconds'])
        
        writer.writerow(['KGR_After_Quantization', f"{kgr_data['kgr_quant']:.4f}", 'bits/second'])
        writer.writerow(['KGR_After_Reconciliation', f"{kgr_data['kgr_recon']:.4f}", 'bits/second'])
        writer.writerow(['KGR_After_PA', f"{kgr_data['kgr_pa']:.4f}", 'bits/second'])
        
        writer.writerow(['Bits_After_Quantization', kgr_data['bits_quant'], 'bits'])
        writer.writerow(['Bits_After_Reconciliation', kgr_data['bits_recon'], 'bits'])
        writer.writerow(['Bits_After_PA', kgr_data['bits_pa'], 'bits'])
    
    # print(f"✓ Timing and KGR results saved to: {filename}")

def save_statistics(stats, filename="statistics_summary.csv"):
    """Save statistik keseluruhan ke CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        
        writer.writerow(['Total_Blocks', stats['total_blocks']])
        writer.writerow(['Successful_Blocks', stats['successful_blocks']])
        writer.writerow(['Dropped_Blocks', stats['dropped_blocks']])
        writer.writerow(['Success_Rate', f"{stats['successful_blocks']/stats['total_blocks']:.4f}"])
        writer.writerow(['Drop_Rate', f"{stats['drop_rate']:.4f}"])
        writer.writerow(['Total_Errors_Before_IR', stats['total_error_before']])
        writer.writerow(['Total_Errors_After_IR', stats['total_error_after']])
        
        if len(stats['calibration_counts']) > 0:
            avg_cal = sum(stats['calibration_counts']) / len(stats['calibration_counts'])
            writer.writerow(['Avg_Calibrations_Per_Block', f"{avg_cal:.2f}"])
    
    # print(f"✓ Statistics saved to: {filename}")

def save_per_block_complexity(stats, filename="per_block_complexity.csv"):
    """Save complexity, m, alpha untuk setiap block"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Block_ID', 'Complexity', 'm', 'Alpha_Init', 'Alpha_Final', 'Status', 'Calibrations', 'KDR_Before', 'KDR_Final'])
        
        for block in stats['blocks_data']:
            writer.writerow([
                block['block_id'],
                block.get('complexity', 'N/A'),
                block.get('m', 'N/A'),
                block.get('alpha_init', 'N/A'),
                block.get('alpha_final', 'N/A'),
                block.get('status', 'N/A'),
                block.get('calibrations', 'N/A'),
                block.get('kdr_before', 'N/A'),
                block.get('kdr_final', 'N/A')
            ])
    
    print(f"✓ Per-block complexity data saved to: {filename}")

#====================================================
#=========== NIST RANDOMNESS TESTS ==================
#====================================================

BLOCK_M  = 128
APEN_M   = 2
SERIAL_M = 2

def nist_frequency_test(bits):
    n = len(bits)
    if n < 16:
        return 0.0
    s = sum(1 if b == 1 else -1 for b in bits)
    s_obs = abs(s) / math.sqrt(n)
    return float(erfc(s_obs / math.sqrt(2.0)))

def nist_block_frequency_test(bits, m=BLOCK_M):
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
        M = 8
        v_values = [1, 2, 3, 4]
        pi_vals  = [0.21484375, 0.3671875, 0.23046875, 0.1875]
    elif n < 750000:
        M = 128
        v_values = [4, 5, 6, 7, 8, 9]
        pi_vals  = [
            0.1174035788, 0.242955959, 0.249363483,
            0.17517706,   0.102701071, 0.112398847
        ]
    else:
        M = 10000
        v_values = [10, 11, 12, 13, 14, 15, 16]
        pi_vals  = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    N = n // M
    if N == 0:
        return 0.0

    v_counts = [0] * len(v_values)

    for i in range(N):
        block = bits[i*M:(i+1)*M]
        longest = 0
        run = 0
        for b in block:
            if b == 1:
                run += 1
                longest = max(longest, run)
            else:
                run = 0

        if longest <= v_values[0]:
            v_counts[0] += 1
        elif longest >= v_values[-1]:
            v_counts[-1] += 1
        else:
            for j in range(1, len(v_values)):
                if longest == v_values[j]:
                    v_counts[j] += 1
                    break

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

    sum1 = 0.0
    sum2 = 0.0

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

def nist_approximate_entropy_test(bits, m=APEN_M):
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

def nist_serial_test(bits, m=SERIAL_M):
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

def hex_to_bit_array(hex_str):
    bit_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)
    return np.fromiter(bit_str, dtype=np.int8)

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
    print("\n" + "="*70)
    print("PHYSICAL LAYER KEY GENERATION")
    print("PER-BLOCK COMPLEXITY ESTIMATION")
    print("="*70)
    print("\nIMPROVEMENT: Each diversity block has its own:")
    print("  - Complexity estimation")
    print("  - m value selection")
    print("  - Alpha calculation")
    print("  - Quantization parameters")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_cp = 160  # Fixed channel probing time
    start_wall = time.time()
    start_cpu = time.process_time()
    # for i in range (0, 1000):        
    #============== STEP 1-2-3: PER-BLOCK PROCESSING =========================
    print("STEP 1-3: Per-Block Complexity → m → Alpha → Quantization → Reconciliation")
    print("-" * 70)
    
    # Semua proses complexity, quantization, dan reconciliation ada di dalam fungsi ini
    kunci_ir_alice, blok_sukses, stats = reconciliation_per_block_with_calibration(
        rssi_A_full=rssi_A,
        rssi_B_full=rssi_B,
        delta_alpha=0.2,
        NC=3,
        block_size=BLOCK_SIZE,
        verbose=True
    )
    
    # Pecah waktu processing menjadi komponen
    # Estimasi: complexity + quantization = 20% dari total, reconciliation = 80%
    time_complexity        = stats['time_complexity']
    time_complexity_and_quant = stats['time_quantization']   # dipakai di KGR
    time_reconciliation    = stats['time_reconciliation']
    
    print(f"  Estimated complexity+quantization: {time_complexity_and_quant:.6f} seconds")
    print(f"  Estimated reconciliation: {time_reconciliation:.6f} seconds")
    
    # Save results
    save_quantization_results(
        rssi_A, rssi_B,
        stats['keys_A_all'], stats['keys_B_all'],
        filename=f"quantization_results_perblock_{timestamp}.csv"
    )
    
    save_reconciliation_results(
        stats['blocks_data'],
        filename=f"reconciliation_results_perblock_{timestamp}.csv"
    )
    
    save_per_block_complexity(
        stats,
        filename=f"per_block_complexity_{timestamp}.csv"
    )
    
    total_bits_after_reconciliation = len(kunci_ir_alice)
    
    #=================== STEP 4: PRIVACY AMPLIFICATION ========================
    if len(kunci_ir_alice) > 0:
        print(f"\n{'='*70}")
        print(f"STEP 4: PRIVACY AMPLIFICATION")
        print(f"{'='*70}")
        
        start_pa = time.time()
        kunci_aes_256 = finalize_to_aes256(kunci_ir_alice)
        time_pa = time.time() - start_pa
        
        print(f"Time Privacy Amplification: {time_pa:.6f} seconds")
        
        save_final_key(
            kunci_ir_alice, kunci_aes_256,
            filename=f"final_key_perblock_{timestamp}.csv"
        )
        
        total_bits_after_pa = 256
        
        #=================== KGR CALCULATION ========================
        print(f"\n{'='*70}")
        print(f"{'KEY GENERATION RATE (KGR) ANALYSIS':^70}")
        print(f"{'='*70}")
        
        print(f"\nTime Breakdown:")
        print(f"  Time CP (Channel Probing)         : {time_cp:.6f} seconds (FIXED)")
        print(f"  Time complexity    : {time_complexity:.6f} seconds")
        print(f"  Time Quantization    : {time_complexity_and_quant:.6f} seconds")
        print(f"  Time Reconciliation               : {time_reconciliation:.6f} seconds")
        print(f"  Time Privacy Amplification        : {time_pa:.6f} seconds")
        
        # Hitung waktu kumulatif
        total_time_after_quant = time_cp + time_complexity_and_quant + time_complexity
        total_time_after_recon = time_cp + time_complexity_and_quant + time_reconciliation + time_complexity
        total_time_after_pa = time_cp + time_complexity_and_quant + time_reconciliation + time_pa + time_complexity
        
        print(f"\nCumulative Times:")
        print(f"  After Quantization                : {total_time_after_quant:.6f} seconds")
        print(f"  After Reconciliation              : {total_time_after_recon:.6f} seconds")
        print(f"  After Privacy Amplification       : {total_time_after_pa:.6f} seconds")
        
        # Hitung KGR
        bits_after_quantization = stats['total_bits_quantization']
        kgr_quantization = bits_after_quantization / total_time_after_quant
        kgr_reconciliation = total_bits_after_reconciliation / total_time_after_recon
        kgr_pa = total_bits_after_pa / total_time_after_pa
        
        print(f"\n{'='*70}")
        print(f"  KGR After Quantization            : {kgr_quantization:.4f} bits/second")
        print(f"  KGR After Reconciliation          : {kgr_reconciliation:.4f} bits/second")
        print(f"  KGR After Privacy Amplification   : {kgr_pa:.4f} bits/second")
        print(f"{'='*70}")
        
        print(f"\nBit Generation:")
        print(f"  Bits after Quantization           : {bits_after_quantization} bits")
        print(f"  Bits after Reconciliation         : {total_bits_after_reconciliation} bits")
        print(f"  Bits after Privacy Amplification  : {total_bits_after_pa} bits")
        
        time_data = {
            'cp': time_cp,
            'complexity': time_complexity_and_quant,
            'quantization': 0,  # Already included in complexity time
            'reconciliation': time_reconciliation,
            'pa': time_pa,
            'total_after_quant': total_time_after_quant,
            'total_after_recon': total_time_after_recon,
            'total_after_pa': total_time_after_pa
        }
        
        kgr_data = {
            'kgr_quant': kgr_quantization,
            'kgr_recon': kgr_reconciliation,
            'kgr_pa': kgr_pa,
            'bits_quant': bits_after_quantization,
            'bits_recon': total_bits_after_reconciliation,
            'bits_pa': total_bits_after_pa
        }
        
        save_timing_and_kgr(
            time_data, kgr_data,
            filename=f"timing_kgr_results_perblock_{timestamp}.csv"
        )
        
        save_statistics(
            stats,
            filename=f"statistics_summary_perblock_{timestamp}.csv"
        )
        
        # =================== STEP 5: AES ENCRYPTION/DECRYPTION ======================
        print(f"\n{'='*70}")
        print(f"STEP 5: AES-256 ENCRYPTION & DECRYPTION")
        print(f"{'='*70}")
        
        pesan_awal = "Halo Bob! Ini pesan rahasia kita berdua."
        print(f"Plaintext: '{pesan_awal}'")
        
        iv, pesan_terenkripsi = alice_encrypt(pesan_awal, kunci_aes_256)
        print(f"Ciphertext: {pesan_terenkripsi}")
        print(f"IV: {iv}")
        
        pesan_akhir = bob_decrypt(iv, pesan_terenkripsi, kunci_aes_256)
        print(f"Decrypted: '{pesan_akhir}'")
        
        if pesan_awal == pesan_akhir:
            # print("✓ Status: SUCCESS")
            print("✓ Status: SUCCESS")
        else:
            # print("✗ Status: FAILED")
            print("✗ Status: FAILED")

        # =================== STEP 6: NIST RANDOMNESS TESTS ======================
        print(f"\n{'='*70}")
        print(f"STEP 6: NIST RANDOMNESS TESTS")
        print(f"{'='*70}")
        
        bits_arr = hex_to_bit_array(kunci_aes_256)
        
        print(f"Frequency Test:              {nist_frequency_test(bits_arr):.6f}")
        print(f"Runs Test:                   {nist_runs_test(bits_arr):.6f}")
        print(f"FFT Test:                    {nist_fft_test(bits_arr):.6f}")
        print(f"Cumulative Sums (Forward):   {nist_cumulative_sums_test(bits_arr, 'forward'):.6f}")
        print(f"Approximate Entropy:         {nist_approximate_entropy_test(bits_arr, 2):.6f}")
        serial_result = nist_serial_test(bits_arr, 2)
        print(f"Serial Test:                 p1={serial_result[0]:.6f}, p2={serial_result[1]:.6f}")
        
        print(f"\n{'='*70}")
        print(f"All results exported successfully!")
        print(f"Files created with timestamp: {timestamp}")
        print(f"Suffix: _perblock")
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

    print("power avg:", power)
    print("cpu avg:", cpu_usage)