/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! 4-Bit Non-Cryptographic Hash Weight & Matmul Engine (`h4x2` + `f16` Block Scales).
//!
//! Implements and simulates:
//! 1. **Non-Cryptographic Multiplicative Hash Primitives** (3-op integer multiply-shift-XOR
//!    for zero-cost coordinate-indexed stochastic rounding, Rademacher sign hashing, and
//!    salted block-Cuckoo bucket assignment).
//! 2. **4-Bit Hash + `f16` (16-bit) Block-Scaled Matmul on the Tensor Core Crossbar**,
//!    eliminating standard NVFP4's `f8e4m3fn` (3-bit mantissa) scale bottleneck and
//!    systematic Round-to-Nearest (RTN) bias.
//! 3. **Full 16-State Bijective Hash Codebook** (reclaiming `f4e2m1fn`'s wasted `-0.0` code
//!    point so all $2^4 = 16$ bit patterns represent distinct values).
//! 4. **Collision-Managed Hash-Folded Crossbar Matmul ($K \to K / c$)** using **Sparsity +
//!    Per-Block Hash Salt + Stochastic Rounding** to execute matrix multiplications in
//!    $1/c$ Tensor Core crossbar cycles while keeping collision error strictly unbiased.

#[cfg(test_standalone)]
#[path = "dtype.rs"]
pub mod dtype;

#[cfg(not(test_standalone))]
use crate::dtype::f4e2m1fnx2;
#[cfg(test_standalone)]
use dtype::f4e2m1fnx2;
use half::f16;

/// Hardware decode table for IEEE/NVIDIA `f4e2m1fn` nibble `0x0..=0xF`.
/// Note that nibble `0x8` is `-0.0`, which duplicates `0x0` (`+0.0`).
pub const E2M1_DECODE_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // 0x0..=0x7 (positive)
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // 0x8..=0xF (negative)
];

/// Full 16-state bijective codebook (`2^4 = 16` distinct values).
/// Reclaims state `0x8` (`-0.0` in E2M1) as `+2.5` (filling the largest interior quantization
/// jump between `2.0` and `3.0` or `0.0..0.5`), reserving `0x0 = 0.0` for exact sparsity.
pub const BIJECTIVE16_DECODE_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 6.0, // 0..=8 (zero + 8 positive levels)
    -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // 9..=15 (7 negative levels)
];

/// Positive magnitude sorted codebook for `f4e2m1fn` (`0..=7`).
const E2M1_POS_MAGS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Packed pair of 4-bit non-cryptographic hash nibbles in one byte.
///
/// Layout is bit-compatible with [`f4e2m1fnx2`] so `h4x2` buffers can be passed
/// directly into hardware `mmaf_scaled` Tensor Core crossbar tiles without re-packing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct h4x2(pub u8);

impl h4x2 {
    #[inline]
    pub const fn from_nibbles(low: u8, high: u8) -> Self {
        Self((low & 0x0F) | ((high & 0x0F) << 4))
    }

    #[inline]
    pub const fn low(self) -> u8 {
        self.0 & 0x0F
    }

    #[inline]
    pub const fn high(self) -> u8 {
        (self.0 >> 4) & 0x0F
    }

    #[inline]
    pub const fn as_f4e2m1fnx2(self) -> f4e2m1fnx2 {
        f4e2m1fnx2::from_bits(self.0)
    }
}

// ---------------------------------------------------------------------------
// 1. Fast Non-Cryptographic Multiplicative Hash Primitives
// ---------------------------------------------------------------------------

/// Ultra-fast 32-bit non-cryptographic multiplicative mixer (Murmur3/SplitMix32 style).
///
/// Takes 2 multiplies and 3 XOR-shifts on GPU integer ALUs (executes in parallel
/// with Tensor Core MMA instructions).
#[inline]
pub const fn hash_mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    x
}

/// Generates a deterministic, uniform `[0.0, 1.0)` float from a `(row, col, salt)` coordinate
/// using a non-cryptographic hash for zero-state stochastic rounding.
#[inline]
pub fn hash_stochastic_unit(row: usize, col: usize, salt: u32) -> f32 {
    let key = (row as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((col as u32).wrapping_mul(0x6C8E_9CF5))
        ^ salt;
    let h = hash_mix32(key);
    // Use upper 24 bits for uniform [0, 1) mantissa
    (h >> 8) as f32 * (1.0 / ((1u32 << 24) as f32))
}

/// Maps a coordinate `k` within a block of size `block_size` into a compressed bucket
/// `0..compressed_size` and a Rademacher sign `{-1.0, +1.0}` using a salted multiplicative hash.
#[inline]
pub const fn hash_bucket_and_sign(
    k_in_block: usize,
    compressed_size: usize,
    salt: u8,
) -> (usize, f32) {
    let key = ((k_in_block as u32).wrapping_mul(0x9E37_79B9)) ^ ((salt as u32).wrapping_mul(0x85EB_CA6B));
    let h = hash_mix32(key);
    let bucket = (h as usize) % compressed_size;
    let sign = if (h >> 31) == 0 { 1.0f32 } else { -1.0f32 };
    (bucket, sign)
}

// ---------------------------------------------------------------------------
// 2. FP8 E4M3FN Software Simulation (for exact NVFP4 baseline comparison)
// ---------------------------------------------------------------------------

/// Quantizes an `f32` scale to `f8e4m3fn` and returns its decoded `f32` value,
/// matching the hardware behavior of `convert_tile::<f8e4m3fn>` in `nvfp4_moe.rs`.
#[inline]
pub fn quantize_decode_f8e4m3fn(val: f32) -> f32 {
    if val == 0.0 || val.is_nan() {
        return 0.0;
    }
    let sign = if val < 0.0 { -1.0 } else { 1.0 };
    let abs_v = val.abs().min(448.0);
    let exp = abs_v.log2().floor() as i32;
    if exp < -9 {
        return 0.0;
    }
    let clamped_exp = exp.clamp(-6, 8);
    let base = 2.0f32.powi(clamped_exp);
    let mant = ((abs_v / base - 1.0) * 8.0).round().clamp(0.0, 7.0) / 8.0;
    sign * base * (1.0 + mant)
}

/// Quantizes a normalized value `v_norm` (in `[-6.0, 6.0]`) to a 4-bit `f4e2m1fn` nibble
/// using standard deterministic Round-To-Nearest-Even (RTN).
#[inline]
pub fn quantize_e2m1_rtn(v_norm: f32) -> u8 {
    if v_norm == 0.0 || v_norm.is_nan() {
        return 0;
    }
    let sign_bit = if v_norm < 0.0 { 0x8u8 } else { 0x0u8 };
    let a = v_norm.abs().min(6.0);
    let mag_nibble = if a < 0.25 {
        0
    } else if a <= 0.75 {
        1
    } else if a < 1.25 {
        2
    } else if a <= 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    if mag_nibble == 0 {
        0
    } else {
        sign_bit | mag_nibble
    }
}

/// Quantizes a normalized value `v_norm` (in `[-6.0, 6.0]`) to a 4-bit `f4e2m1fn` nibble
/// using a **non-cryptographic hash for stochastic rounding** with controllable dither width
/// `dither_width in [0.0, 1.0]` (`0.0` = low-variance RTN, `1.0` = full unbiased SR).
#[inline]
pub fn quantize_e2m1_hash(
    v_norm: f32,
    row: usize,
    col: usize,
    salt: u32,
    dither_width: f32,
) -> u8 {
    if v_norm == 0.0 || v_norm.is_nan() {
        return 0;
    }
    let sign_bit = if v_norm < 0.0 { 0x8u8 } else { 0x0u8 };
    let a = v_norm.abs().min(6.0);

    let idx = if a < 0.5 {
        0
    } else if a < 1.0 {
        1
    } else if a < 1.5 {
        2
    } else if a < 2.0 {
        3
    } else if a < 3.0 {
        4
    } else if a < 4.0 {
        5
    } else if a < 6.0 {
        6
    } else {
        return sign_bit | 7;
    };

    let low = E2M1_POS_MAGS[idx];
    let high = E2M1_POS_MAGS[idx + 1];
    let frac = (a - low) / (high - low);
    // Hash-generated uniform dither in [-0.5 * dither_width, +0.5 * dither_width]
    let u = hash_stochastic_unit(row, col, salt) - 0.5;
    let mag_nibble = if frac + u * dither_width >= 0.5 {
        (idx + 1) as u8
    } else {
        idx as u8
    };

    if mag_nibble == 0 {
        0
    } else {
        sign_bit | mag_nibble
    }
}

/// Quantizes a normalized value `v_norm` (in `[-6.0, 6.0]`) into the **16-state bijective
/// codebook** (`BIJECTIVE16_DECODE_TABLE`) using non-cryptographic hash rounding.
#[inline]
pub fn quantize_bijective16_hash(
    v_norm: f32,
    row: usize,
    col: usize,
    salt: u32,
    dither_width: f32,
) -> u8 {
    if v_norm == 0.0 || v_norm.is_nan() {
        return 0;
    }
    let u = (hash_stochastic_unit(row, col, salt) - 0.5) * dither_width;
    if v_norm > 0.0 {
        const POS: [f32; 9] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 6.0];
        let a = v_norm.min(6.0);
        let mut idx = 0usize;
        while idx < 8 && a >= POS[idx + 1] {
            idx += 1;
        }
        if idx >= 8 {
            return 8;
        }
        let frac = (a - POS[idx]) / (POS[idx + 1] - POS[idx]);
        if frac + u >= 0.5 {
            (idx + 1) as u8
        } else {
            idx as u8
        }
    } else {
        const NEG_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let a = v_norm.abs().min(6.0);
        let mut idx = 0usize;
        while idx < 7 && a >= NEG_MAG[idx + 1] {
            idx += 1;
        }
        let chosen = if idx >= 7 {
            7
        } else {
            let frac = (a - NEG_MAG[idx]) / (NEG_MAG[idx + 1] - NEG_MAG[idx]);
            if frac + u >= 0.5 { idx + 1 } else { idx }
        };
        if chosen == 0 {
            0
        } else {
            (8 + chosen) as u8
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Block-Quantized Representations (NVFP4 vs. Hash4 + f16 Block Scales)
// ---------------------------------------------------------------------------

/// Standard NVFP4 block-quantized matrix (`f4e2m1fnx2` nibbles + `f8e4m3fn` scales per 16 elements).
pub struct NvFp4Matrix {
    pub rows: usize,
    pub cols: usize,
    pub packed: Vec<f4e2m1fnx2>, // [rows, cols / 2]
    pub scales_f8: Vec<f32>,     // [rows, cols / 16], quantized to f8e4m3fn
}

impl NvFp4Matrix {
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(cols % 16, 0, "cols must be a multiple of 16");
        let mut packed = Vec::with_capacity(rows * (cols / 2));
        let mut scales_f8 = Vec::with_capacity(rows * (cols / 16));

        for r in 0..rows {
            for b in 0..(cols / 16) {
                let start = r * cols + b * 16;
                let block = &data[start..start + 16];
                let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let scale_f8 = quantize_decode_f8e4m3fn(amax / 6.0);
                let inv_scale = if scale_f8 > 0.0 { 1.0 / scale_f8 } else { 0.0 };
                scales_f8.push(scale_f8);

                for pair in 0..8 {
                    let v0 = block[pair * 2] * inv_scale;
                    let v1 = block[pair * 2 + 1] * inv_scale;
                    let n0 = quantize_e2m1_rtn(v0);
                    let n1 = quantize_e2m1_rtn(v1);
                    packed.push(f4e2m1fnx2::from_nibbles(n0, n1));
                }
            }
        }
        Self {
            rows,
            cols,
            packed,
            scales_f8,
        }
    }

    /// Simulates the hardware `mmaf_scaled` Tensor Core crossbar matmul `C = A @ B.T`.
    pub fn matmul_crossbar(&self, rhs: &NvFp4Matrix) -> Vec<f32> {
        assert_eq!(self.cols, rhs.cols);
        let m = self.rows;
        let n = rhs.rows;
        let k_blocks = self.cols / 16;
        let mut out = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for b in 0..k_blocks {
                    let sa = self.scales_f8[i * k_blocks + b];
                    let sb = rhs.scales_f8[j * k_blocks + b];
                    let mut block_dot = 0.0f32;
                    let base_a = i * (self.cols / 2) + b * 8;
                    let base_b = j * (rhs.cols / 2) + b * 8;
                    for p in 0..8 {
                        let pa = self.packed[base_a + p];
                        let pb = rhs.packed[base_b + p];
                        let a0 = E2M1_DECODE_TABLE[pa.low().0 as usize];
                        let a1 = E2M1_DECODE_TABLE[pa.high().0 as usize];
                        let b0 = E2M1_DECODE_TABLE[pb.low().0 as usize];
                        let b1 = E2M1_DECODE_TABLE[pb.high().0 as usize];
                        block_dot += a0 * b0 + a1 * b1;
                    }
                    acc += block_dot * (sa * sb);
                }
                out[i * n + j] = acc;
            }
        }
        out
    }
}

/// 4-Bit Non-Cryptographic Hash Matrix with **16-bit (`f16`) Block Scales**.
///
/// Supports both:
/// - **Hardware Crossbar E2M1 Mode** (`bijective_16 = false`): Packed bytes are 100% compatible
///   with `mmaf_scaled` Tensor Core instructions while using `f16` block scales and
///   hash-based stochastic rounding.
/// - **Full 16-State Bijective Mode** (`bijective_16 = true`): Uses all $2^4 = 16$ bit patterns
///   (`BIJECTIVE16_DECODE_TABLE`) with `f16` block scales.
pub struct Hash4F16Matrix {
    pub rows: usize,
    pub cols: usize,
    pub packed: Vec<h4x2>,    // [rows, cols / 2]
    pub scales_f16: Vec<f16>, // [rows, cols / 16] — true 16-bit block scales!
    pub bijective_16: bool,
}

impl Hash4F16Matrix {
    pub fn from_f32_with_hash(
        data: &[f32],
        rows: usize,
        cols: usize,
        salt: u32,
        dither_width: f32,
        sparsity_rel_threshold: f32,
        bijective_16: bool,
    ) -> Self {
        assert_eq!(cols % 16, 0, "cols must be a multiple of 16");
        let mut packed = Vec::with_capacity(rows * (cols / 2));
        let mut scales_f16 = Vec::with_capacity(rows * (cols / 16));

        for r in 0..rows {
            for b in 0..(cols / 16) {
                let start = r * cols + b * 16;
                let block = &data[start..start + 16];
                let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let scale = f16::from_f32(amax / 6.0);
                let scale_f32 = scale.to_f32();
                let inv_scale = if scale_f32 > 0.0 { 1.0 / scale_f32 } else { 0.0 };
                let prune_cutoff = amax * sparsity_rel_threshold;
                scales_f16.push(scale);

                for pair in 0..8 {
                    let c0 = b * 16 + pair * 2;
                    let c1 = c0 + 1;
                    let raw0 = if block[pair * 2].abs() < prune_cutoff {
                        0.0
                    } else {
                        block[pair * 2] * inv_scale
                    };
                    let raw1 = if block[pair * 2 + 1].abs() < prune_cutoff {
                        0.0
                    } else {
                        block[pair * 2 + 1] * inv_scale
                    };

                    let n0 = if bijective_16 {
                        quantize_bijective16_hash(raw0, r, c0, salt, dither_width)
                    } else {
                        quantize_e2m1_hash(raw0, r, c0, salt, dither_width)
                    };
                    let n1 = if bijective_16 {
                        quantize_bijective16_hash(raw1, r, c1, salt, dither_width)
                    } else {
                        quantize_e2m1_hash(raw1, r, c1, salt, dither_width)
                    };
                    packed.push(h4x2::from_nibbles(n0, n1));
                }
            }
        }

        Self {
            rows,
            cols,
            packed,
            scales_f16,
            bijective_16,
        }
    }

    /// Executes the 4-bit crossbar inner product with 16-bit (`f16`) block scales.
    pub fn matmul_crossbar(&self, rhs: &Hash4F16Matrix) -> Vec<f32> {
        assert_eq!(self.cols, rhs.cols);
        assert_eq!(self.bijective_16, rhs.bijective_16);
        let table = if self.bijective_16 {
            &BIJECTIVE16_DECODE_TABLE
        } else {
            &E2M1_DECODE_TABLE
        };

        let m = self.rows;
        let n = rhs.rows;
        let k_blocks = self.cols / 16;
        let mut out = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for b in 0..k_blocks {
                    let sa = self.scales_f16[i * k_blocks + b].to_f32();
                    let sb = rhs.scales_f16[j * k_blocks + b].to_f32();
                    let mut block_dot = 0.0f32;
                    let base_a = i * (self.cols / 2) + b * 8;
                    let base_b = j * (rhs.cols / 2) + b * 8;
                    for p in 0..8 {
                        let pa = self.packed[base_a + p];
                        let pb = rhs.packed[base_b + p];
                        let a0 = table[pa.low() as usize];
                        let a1 = table[pa.high() as usize];
                        let b0 = table[pb.low() as usize];
                        let b1 = table[pb.high() as usize];
                        block_dot += a0 * b0 + a1 * b1;
                    }
                    acc += block_dot * (sa * sb);
                }
                out[i * n + j] = acc;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 4. Collision-Managed Hash-Folded Crossbar Matmul (K -> K / c)
// ---------------------------------------------------------------------------

/// Finds the optimal 4-bit block salt (`0..16`) that minimizes hash bucket collisions
/// for the non-zero support of a sparse block ("Block Cuckoo / Choice-of-Salt Hashing").
pub fn find_min_collision_salt(block: &[f32], compressed_size: usize) -> (u8, usize) {
    let mut best_salt = 0u8;
    let mut best_collisions = usize::MAX;

    for salt in 0u8..16u8 {
        let mut counts = vec![0u8; compressed_size];
        let mut collisions = 0usize;
        for (k, &v) in block.iter().enumerate() {
            if v.abs() > 1e-7 {
                let (bucket, _) = hash_bucket_and_sign(k, compressed_size, salt);
                if counts[bucket] > 0 {
                    collisions += 1;
                }
                counts[bucket] += 1;
            }
        }
        if collisions < best_collisions {
            best_collisions = collisions;
            best_salt = salt;
            if collisions == 0 {
                break;
            }
        }
    }
    (best_salt, best_collisions)
}

/// Folds a sparse matrix `[rows, cols]` along `cols` into `[rows, cols / fold_factor]`
/// using a **non-cryptographic multiplicative hash + stochastic rounding + `f16` block scales**,
/// so the 4-bit Tensor Core crossbar executes in `1 / fold_factor` of the cycles!
///
/// Collision control uses the mechanisms from our thesis:
/// 1. **Sparsity**: Only salient non-zero elements occupy hash buckets ($P(\text{collision}) \propto p^2$).
/// 2. **Salted Block-Cuckoo Hashing + Rademacher Sign Hash + Stochastic Rounding**:
///    Minimizes bucket collisions per block and makes any residual collision strictly unbiased
///    ($\mathbb{E}[s_a s_b x_a w_b] = 0$ for $a \ne b$).
pub fn hash_fold_to_crossbar_f16(
    data: &[f32],
    rows: usize,
    cols: usize,
    fold_factor: usize,
    block_salts: &[u8],
    sr_salt: u32,
    dither_width: f32,
) -> (Hash4F16Matrix, usize, usize) {
    assert!(fold_factor >= 1 && (cols / fold_factor) % 16 == 0);
    let compressed_cols = cols / fold_factor;
    let src_block_len = 16 * fold_factor;
    let num_blocks = cols / src_block_len;
    assert_eq!(block_salts.len(), num_blocks);

    let mut folded_f32 = vec![0.0f32; rows * compressed_cols];
    let mut total_nonzero = 0usize;
    let mut total_collisions = 0usize;

    for r in 0..rows {
        for b in 0..num_blocks {
            let salt = block_salts[b];
            let src_start = r * cols + b * src_block_len;
            let dst_start = r * compressed_cols + b * 16;
            let mut bucket_occupancy = [0u8; 16];

            for k_in_block in 0..src_block_len {
                let v = data[src_start + k_in_block];
                if v.abs() > 1e-7 {
                    total_nonzero += 1;
                    let (bucket, sign) = hash_bucket_and_sign(k_in_block, 16, salt);
                    if bucket_occupancy[bucket] > 0 {
                        total_collisions += 1;
                    }
                    bucket_occupancy[bucket] = bucket_occupancy[bucket].saturating_add(1);
                    folded_f32[dst_start + bucket] += sign * v;
                }
            }
        }
    }

    let mat = Hash4F16Matrix::from_f32_with_hash(
        &folded_f32,
        rows,
        compressed_cols,
        sr_salt,
        dither_width,
        0.0,
        false,
    );
    (mat, total_nonzero, total_collisions)
}

// ============================================================================
// 5. TurboQuant 2-Bit SWAR Transport + Pre-Sent Error Ledger -> E2M1 Crossbar
// ============================================================================

/// Unpacks 8 weights (`half_sel` = `0` for low 8 weights, `1` for high 8 weights) from a
/// 32-bit SWAR-interleaved 2-bit TurboQuant word (`tq_word`) into 8 packed 4-bit `E2M1`
/// nibbles (`u32`), and applies up to 2 Pre-Sent Error Ledger patches per 8-weight half.
///
/// ### 2-Instruction SWAR Bulk Unpack
/// The 4 inner Gaussian `E2M1` centroids `{-1.0, -0.5, +0.5, +1.0}` have 4-bit nibble encodings:
/// - `+0.5 = 0x1 (0001b)` (`sign=0, mag=0, b0=1`)
/// - `+1.0 = 0x2 (0010b)` (`sign=0, mag=1, b0=0`)
/// - `-0.5 = 0x9 (1001b)` (`sign=1, mag=0, b0=1`)
/// - `-1.0 = 0xA (1010b)` (`sign=1, mag=1, b0=0`)
///
/// Every nibble satisfies `b3 = sign, b2 = 0, b1 = mag, b0 = !mag`, so 8 2-bit codes expand
/// to 8 `E2M1` nibbles in two bitwise operations:
/// ```text
/// sm = (tq_word << half_sel) & 0xAAAA_AAAA;
/// w  = sm | (((!sm) >> 1) & 0x1111_1111);
/// ```
#[inline]
pub const fn unpack_tq2_half_with_ledger(
    tq_word: u32,
    half_sel: u32,
    leg_top2: u16,
    leg_top4: u16,
    num_patches: usize,
) -> u32 {
    let sm = (tq_word << half_sel) & 0xAAAA_AAAAu32;
    let mut w = sm | (((!sm) >> 1) & 0x1111_1111u32);

    if num_patches >= 2 {
        let p0 = ((leg_top2 >> (half_sel * 8)) & 0xFF) as u32;
        let sh0 = (p0 & 7) * 4;
        let nib0 = p0 >> 4;
        w = (w & !(0xFu32 << sh0)) | (nib0 << sh0);
    }
    if num_patches >= 4 {
        let p1 = ((leg_top4 >> (half_sel * 8)) & 0xFF) as u32;
        let sh1 = (p1 & 7) * 4;
        let nib1 = p1 >> 4;
        w = (w & !(0xFu32 << sh1)) | (nib1 << sh1);
    }
    w
}

/// Transport-compressed matrix representation combining:
/// 1. `tq2_words`: `2.0 bits/weight` SWAR-interleaved 2-bit TurboQuant bulk stream (`[rows, cols / 16]`)
/// 2. `scales_f8`: `0.5 bits/weight` `f8e4m3fn` block scales (`[rows, cols / 16]`)
/// 3. `ledger_top2`: `1.0 bit/weight` Pre-Sent Top-2 Error Ledger (`[rows, cols / 16]`, 1 byte per 8 weights)
/// 4. `ledger_top4`: Optional extra `1.0 bit/weight` Top-4 Error Ledger (`[rows, cols / 16]`)
#[derive(Clone, Debug)]
pub struct TurboQuantLedgerMatrix {
    pub rows: usize,
    pub cols: usize,
    pub tq2_words: Vec<u32>,
    pub scales_f8: Vec<f32>,
    pub ledger_top2: Vec<u16>,
    pub ledger_top4: Vec<u16>,
}

impl TurboQuantLedgerMatrix {
    /// Encodes an `f32` matrix `[rows, cols]` (`cols % 16 == 0`) into a 2-bit SWAR TurboQuant stream
    /// plus a Pre-Sent Error Ledger (`ledger_top2` and `ledger_top4`).
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        assert_eq!(cols % 16, 0);

        let num_blocks = rows * (cols / 16);
        let mut tq2_words = Vec::with_capacity(num_blocks);
        let mut scales_f8 = Vec::with_capacity(num_blocks);
        let mut ledger_top2 = Vec::with_capacity(num_blocks);
        let mut ledger_top4 = Vec::with_capacity(num_blocks);

        for blk in 0..num_blocks {
            let slice = &data[blk * 16..(blk + 1) * 16];
            let amax = slice.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

            let mut best_scale = 0.0f32;
            let mut best_err = f32::INFINITY;
            let mut best_tq_word = 0u32;
            let mut best_leg2 = 0u16;
            let mut best_leg4 = 0u16;

            for step in 0..9 {
                let mult = 0.70 + 0.075 * (step as f32);
                let s_dec = quantize_decode_f8e4m3fn(((amax / 6.0) * mult).min(448.0));
                let inv_s = if amax > 0.0 && s_dec > 0.0 {
                    1.0 / s_dec
                } else {
                    0.0
                };

                let mut tq_word = 0u32;
                let mut leg2 = 0u16;
                let mut leg4 = 0u16;
                let mut total_sq_err = 0.0f32;

                for h in 0..2usize {
                    let mut err_gain = [0.0f32; 8];
                    let mut true_nib = [0u8; 8];
                    let mut bulk_recon = [0.0f32; 8];

                    for i in 0..8usize {
                        let raw_w = slice[h * 8 + i];
                        let v = raw_w * inv_s;
                        let nib = quantize_e2m1_rtn(v);
                        true_nib[i] = nib;

                        let sign_bit = if v < 0.0 { 1u32 } else { 0u32 };
                        let mag_bit = if v.abs() >= 0.75 { 1u32 } else { 0u32 };

                        let bit_s = i * 4 + (3 - h);
                        let bit_m = i * 4 + (1 - h);
                        tq_word |= (sign_bit << bit_s) | (mag_bit << bit_m);

                        let b_nib = ((sign_bit << 3) | (mag_bit << 1) | (mag_bit ^ 1)) as u8;
                        bulk_recon[i] = E2M1_DECODE_TABLE[b_nib as usize] * s_dec;
                        let patch_recon = E2M1_DECODE_TABLE[nib as usize] * s_dec;
                        let e_bulk = raw_w - bulk_recon[i];
                        let e_patch = raw_w - patch_recon;
                        err_gain[i] = (e_bulk * e_bulk) - (e_patch * e_patch);
                    }

                    let mut best0 = 0usize;
                    let mut max_g0 = -1.0f32;
                    for i in 0..8 {
                        if err_gain[i] > max_g0 {
                            max_g0 = err_gain[i];
                            best0 = i;
                        }
                    }
                    err_gain[best0] = -1.0;

                    let mut best1 = 0usize;
                    let mut max_g1 = -1.0f32;
                    for i in 0..8 {
                        if err_gain[i] > max_g1 {
                            max_g1 = err_gain[i];
                            best1 = i;
                        }
                    }

                    let byte_p0 = ((best0 as u8) & 0x7) | ((true_nib[best0] & 0xF) << 4);
                    let byte_p1 = ((best1 as u8) & 0x7) | ((true_nib[best1] & 0xF) << 4);
                    leg2 |= (byte_p0 as u16) << (h * 8);
                    leg4 |= (byte_p1 as u16) << (h * 8);

                    for i in 0..8 {
                        let rec = if i == best0 {
                            E2M1_DECODE_TABLE[true_nib[i] as usize] * s_dec
                        } else {
                            bulk_recon[i]
                        };
                        let d = slice[h * 8 + i] - rec;
                        total_sq_err += d * d;
                    }
                }

                if total_sq_err < best_err {
                    best_err = total_sq_err;
                    best_scale = s_dec;
                    best_tq_word = tq_word;
                    best_leg2 = leg2;
                    best_leg4 = leg4;
                }
            }

            tq2_words.push(best_tq_word);
            scales_f8.push(best_scale);
            ledger_top2.push(best_leg2);
            ledger_top4.push(best_leg4);
        }

        Self {
            rows,
            cols,
            tq2_words,
            scales_f8,
            ledger_top2,
            ledger_top4,
        }
    }

    /// Unpacks the 2-bit SWAR stream + Error Ledger (`num_patches` = `0`, `2`, or `4`)
    /// directly into hardware-native `NvFp4Matrix` (`E2M1` nibbles + `f8e4m3fn` scales).
    pub fn unpack_to_nvfp4(&self, num_patches: usize) -> NvFp4Matrix {
        let mut packed = Vec::with_capacity(self.rows * (self.cols / 2));
        for blk in 0..self.tq2_words.len() {
            let tq_word = self.tq2_words[blk];
            let leg2 = self.ledger_top2[blk];
            let leg4 = self.ledger_top4[blk];

            let w0 = unpack_tq2_half_with_ledger(tq_word, 0, leg2, leg4, num_patches);
            let w1 = unpack_tq2_half_with_ledger(tq_word, 1, leg2, leg4, num_patches);

            for b in 0..4 {
                packed.push(f4e2m1fnx2::from_bits(((w0 >> (b * 8)) & 0xFF) as u8));
            }
            for b in 0..4 {
                packed.push(f4e2m1fnx2::from_bits(((w1 >> (b * 8)) & 0xFF) as u8));
            }
        }
        NvFp4Matrix {
            rows: self.rows,
            cols: self.cols,
            packed,
            scales_f8: self.scales_f8.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_matmul_nt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for c in 0..k {
                    sum += a[i * k + c] * b[j * k + c];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    fn compute_metrics(ref_out: &[f32], test_out: &[f32]) -> (f32, f32, f32) {
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        let mut dot = 0.0f64;
        let mut sum_sq_test = 0.0f64;
        let mut sum_signed_err = 0.0f64;

        for (&r, &t) in ref_out.iter().zip(test_out.iter()) {
            let err = (t - r) as f64;
            sum_sq_err += err * err;
            sum_sq_ref += (r as f64) * (r as f64);
            sum_sq_test += (t as f64) * (t as f64);
            dot += (r as f64) * (t as f64);
            sum_signed_err += err;
        }
        let rel_l2 = (sum_sq_err.sqrt() / sum_sq_ref.sqrt().max(1e-12)) as f32;
        let cos_sim = (dot / (sum_sq_ref.sqrt() * sum_sq_test.sqrt()).max(1e-12)) as f32;
        let mean_bias = (sum_signed_err / (ref_out.len() as f64)) as f32;
        (rel_l2, cos_sim, mean_bias)
    }

    #[test]
    fn hash4_f16_beats_nvfp4_in_both_single_pass_inference_and_rl() {
        let m = 16usize;
        let n = 16usize;
        let k = 512usize;

        // Generate realistic neural activations and weights with varied block scales
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for i in 0..m {
            for c in 0..k {
                let u = hash_stochastic_unit(i, c, 0x1234_5678) * 2.0 - 1.0;
                // Wide dynamic range across blocks stresses f8e4m3fn's 3-bit mantissa
                let block_scale = 0.03 + 3.7 * hash_stochastic_unit(i, c / 16, 0xAAAA);
                a[i * k + c] = u * block_scale;
            }
        }
        for j in 0..n {
            for c in 0..k {
                let u = hash_stochastic_unit(j + 100, c, 0x8765_4321) * 2.0 - 1.0;
                let block_scale = 0.02 + 2.9 * hash_stochastic_unit(j + 100, c / 16, 0xBBBB);
                b[j * k + c] = u * block_scale;
            }
        }

        let ref_out = reference_matmul_nt(&a, &b, m, n, k);

        // 1. Standard NVFP4 (f4e2m1fnx2 + f8e4m3fn 8-bit scales + RTN)
        let nvfp4_a = NvFp4Matrix::from_f32(&a, m, k);
        let nvfp4_b = NvFp4Matrix::from_f32(&b, n, k);
        let nvfp4_out = nvfp4_a.matmul_crossbar(&nvfp4_b);
        let (nvfp4_rel_l2, nvfp4_cos, _) = compute_metrics(&ref_out, &nvfp4_out);

        // 2. Hash4-E2M1 + f16 (16-bit) block scales (Single-Pass Inference, dither=0.0)
        let hash4_inf_a = Hash4F16Matrix::from_f32_with_hash(&a, m, k, 0x1111, 0.0, 0.0, false);
        let hash4_inf_b = Hash4F16Matrix::from_f32_with_hash(&b, n, k, 0x2222, 0.0, 0.0, false);
        let hash4_inf_out = hash4_inf_a.matmul_crossbar(&hash4_inf_b);
        let (hash4_inf_rel_l2, hash4_inf_cos, _) = compute_metrics(&ref_out, &hash4_inf_out);

        // 3. Hash4-Bijective16 + f16 (16-bit) block scales (Single-Pass Inference, reclaiming -0.0)
        let bij_inf_a = Hash4F16Matrix::from_f32_with_hash(&a, m, k, 0x1111, 0.0, 0.0, true);
        let bij_inf_b = Hash4F16Matrix::from_f32_with_hash(&b, n, k, 0x2222, 0.0, 0.0, true);
        let bij_inf_out = bij_inf_a.matmul_crossbar(&bij_inf_b);
        let (bij_inf_rel_l2, bij_inf_cos, _) = compute_metrics(&ref_out, &bij_inf_out);

        // 4. Hash4-Bijective16 + f16 scales + Full Stochastic Hash Rounding (8-step RL expectation)
        let mut avg_hash4_out = vec![0.0f32; m * n];
        let trials = 8usize;
        for t in 0..trials {
            let ha = Hash4F16Matrix::from_f32_with_hash(
                &a,
                m,
                k,
                0x1000 + (t as u32) * 37,
                1.0,
                0.0,
                true,
            );
            let hb = Hash4F16Matrix::from_f32_with_hash(
                &b,
                n,
                k,
                0x2000 + (t as u32) * 53,
                1.0,
                0.0,
                true,
            );
            let out_t = ha.matmul_crossbar(&hb);
            for idx in 0..(m * n) {
                avg_hash4_out[idx] += out_t[idx] / (trials as f32);
            }
        }
        let (sr_avg_rel_l2, sr_avg_cos, _) = compute_metrics(&ref_out, &avg_hash4_out);

        println!(
            "\n=== Matmul [16x512] @ [512x16] Accuracy Benchmark ===\n  \
             1. Standard NVFP4 (f8e4m3 scale, 15-state E2M1):   rel_l2 = {nvfp4_rel_l2:.5}, cos = {nvfp4_cos:.6}\n  \
             2. Hash4-Crossbar (f16 scale, 15-state, 1-pass):   rel_l2 = {hash4_inf_rel_l2:.5}, cos = {hash4_inf_cos:.6}\n  \
             3. Hash4-Bij16    (f16 scale, 16-state, 1-pass):   rel_l2 = {bij_inf_rel_l2:.5}, cos = {bij_inf_cos:.6}\n  \
             4. Hash4-Bij16-SR (f16 scale, 8-step RL rollout):  rel_l2 = {sr_avg_rel_l2:.5}, cos = {sr_avg_cos:.6}"
        );

        // Both single-pass f16 block scale AND 16-state bijective beat standard NVFP4!
        assert!(
            hash4_inf_rel_l2 < nvfp4_rel_l2,
            "f16 block scales ({hash4_inf_rel_l2}) must beat f8e4m3fn scales ({nvfp4_rel_l2})"
        );
        assert!(
            bij_inf_rel_l2 < hash4_inf_rel_l2,
            "16-state bijective ({bij_inf_rel_l2}) must beat 15-state E2M1 ({hash4_inf_rel_l2})"
        );
        assert!(
            sr_avg_rel_l2 < bij_inf_rel_l2,
            "Stochastic Hash Rounding ({sr_avg_rel_l2}) must further reduce error over single-pass ({bij_inf_rel_l2})"
        );
    }

    #[test]
    fn sparsity_and_salted_hashing_control_collisions_in_2x_folded_crossbar() {
        let m = 16usize;
        let n = 16usize;
        let k = 512usize;
        let fold_factor = 2usize;
        let src_block_len = 16 * fold_factor; // 32 -> folded to 16
        let num_blocks = k / src_block_len;

        // Simulate MoE FC2 after ReLU-squared + structured pruning (~87.5% sparse: 4 active per 32-block)
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for blk in 0..num_blocks {
            for slot in 0..4 {
                let c = blk * src_block_len
                    + ((hash_mix32((blk * 17 + slot * 31) as u32) as usize) % src_block_len);
                for i in 0..m {
                    a[i * k + c] = (hash_stochastic_unit(i, c, 0x7777) * 2.0 - 1.0) * 1.5;
                }
                for j in 0..n {
                    b[j * k + c] = (hash_stochastic_unit(j + 32, c, 0x8888) * 2.0 - 1.0) * 1.5;
                }
            }
        }

        let ref_out = reference_matmul_nt(&a, &b, m, n, k);

        // 1. Unsalted hash folding (fixed salt = 0 for all blocks)
        let unsalted = vec![0u8; num_blocks];
        let (fa_unsalted, nonzero_a, col_unsalted) =
            hash_fold_to_crossbar_f16(&a, m, k, fold_factor, &unsalted, 0xAAAA, 0.0);
        let (fb_unsalted, _, _) =
            hash_fold_to_crossbar_f16(&b, n, k, fold_factor, &unsalted, 0xBBBB, 0.0);
        let out_unsalted = fa_unsalted.matmul_crossbar(&fb_unsalted);
        let (rel_unsalted, cos_unsalted, _) = compute_metrics(&ref_out, &out_unsalted);

        // 2. Salted Block-Cuckoo Hash Folding (picks 4-bit salt per block that minimizes collisions)
        let mut optimal_salts = Vec::with_capacity(num_blocks);
        for blk in 0..num_blocks {
            let block_slice = &a[blk * src_block_len..(blk + 1) * src_block_len];
            let (best_salt, _) = find_min_collision_salt(block_slice, 16);
            optimal_salts.push(best_salt);
        }
        let (fa_salted, _, col_salted) =
            hash_fold_to_crossbar_f16(&a, m, k, fold_factor, &optimal_salts, 0xAAAA, 0.0);
        let (fb_salted, _, _) =
            hash_fold_to_crossbar_f16(&b, n, k, fold_factor, &optimal_salts, 0xBBBB, 0.0);
        let out_salted = fa_salted.matmul_crossbar(&fb_salted);
        let (rel_salted, cos_salted, _) = compute_metrics(&ref_out, &out_salted);

        println!(
            "\n=== 2x Hash-Folded Crossbar (K=512 -> K_hash=256, 2.0x Crossbar OP/s) ===\n  \
             Active nonzeros: {nonzero_a}\n  \
             Unsalted Hash Folding: collisions = {col_unsalted:3}, rel_l2 = {rel_unsalted:.5}, cos = {cos_unsalted:.6}\n  \
             Salted Cuckoo Folding: collisions = {col_salted:3}, rel_l2 = {rel_salted:.5}, cos = {cos_salted:.6}"
        );

        assert_eq!(
            col_salted, 0,
            "4-bit per-block salt should achieve 0 collisions on 4-of-32 sparse blocks!"
        );
        assert!(rel_salted < rel_unsalted);
        assert!(cos_salted > 0.99);
    }

    #[test]
    fn turboquant_swar_with_error_ledger_recovers_nvfp4_accuracy() {
        let m = 16usize;
        let n = 16usize;
        let k = 256usize;

        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for i in 0..m {
            for c in 0..k {
                let u = hash_stochastic_unit(i, c, 0x1357) * 2.0 - 1.0;
                a[i * k + c] = u * 1.8;
            }
        }
        for j in 0..n {
            for c in 0..k {
                let u1 = hash_stochastic_unit(j, c, 0x2468).max(1e-7);
                let u2 = hash_stochastic_unit(j, c, 0x9BDF);
                let z = (-2.0 * u1.ln()).sqrt() * (6.2831853 * u2).cos();
                // 87.5% Gaussian bulk + 12.5% heavy outliers
                b[j * k + c] = if c % 8 == 0 {
                    z.signum() * (4.2 + 1.5 * u2)
                } else {
                    z * 0.65
                };
            }
        }

        let ref_out = reference_matmul_nt(&a, &b, m, n, k);
        let nvfp4_a = NvFp4Matrix::from_f32(&a, m, k);
        let tq_b = TurboQuantLedgerMatrix::from_f32(&b, n, k);

        let b_top0 = tq_b.unpack_to_nvfp4(0);
        let b_top2 = tq_b.unpack_to_nvfp4(2);
        let b_top4 = tq_b.unpack_to_nvfp4(4);

        let (rel0, cos0, _) = compute_metrics(&ref_out, &nvfp4_a.matmul_crossbar(&b_top0));
        let (rel2, cos2, _) = compute_metrics(&ref_out, &nvfp4_a.matmul_crossbar(&b_top2));
        let (rel4, cos4, _) = compute_metrics(&ref_out, &nvfp4_a.matmul_crossbar(&b_top4));

        println!(
            "\n=== TurboQuant SWAR + Error Ledger [16x256] @ [256x16] ===\n  \
             Top-0 (2.50 bits/w): rel_l2 = {rel0:.5}, cos = {cos0:.6}\n  \
             Top-2 (3.50 bits/w): rel_l2 = {rel2:.5}, cos = {cos2:.6}\n  \
             Top-4 (4.50 bits/w): rel_l2 = {rel4:.5}, cos = {cos4:.6}"
        );

        assert!(rel2 < rel0 * 0.35, "Top-2 Error Ledger must slash Rel L2 error ({rel2} vs {rel0})");
        assert!(rel4 < rel2, "Top-4 Error Ledger must further reduce error ({rel4} vs {rel2})");
        assert!(cos2 > 0.98, "Top-2 Cosine Similarity must exceed 0.98 (got {cos2})");
    }
}
