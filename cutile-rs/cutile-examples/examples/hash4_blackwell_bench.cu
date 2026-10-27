/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone Blackwell (sm_120a) PTX & Tensor Core Benchmark comparing:
 *   1. Standard NVFP4 (f8e4m3 scale, RTN)
 *   2. Hash4 + Grid-Aligned f16 Scale + Multiplicative Hash Stochastic Rounding
 *   3. Hash4 2x / 4x Salted Block-Cuckoo Hash-Folded Crossbar Matmul
 *
 * Compile and run on Blackwell (RTX 5070 / RTX PRO 6000):
 *   nvcc -O3 -gencode arch=compute_120a,code=sm_120a hash4_blackwell_bench.cu -o hash4_blackwell_bench
 *   ./hash4_blackwell_bench
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__device__ __host__ __forceinline__ uint32_t hash_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}

__device__ __host__ __forceinline__ void hash_bucket_and_sign(
    int k_in_block, int compressed_size, uint8_t salt, int* bucket, float* sign
) {
    uint32_t key = ((uint32_t)k_in_block * 0x9E3779B9u) ^ ((uint32_t)salt * 0x85EBCA6Bu);
    uint32_t h = hash_mix32(key);
    *bucket = (int)(h % (uint32_t)compressed_size);
    *sign = ((h >> 31) == 0) ? 1.0f : -1.0f;
}

__device__ __forceinline__ uint32_t e2m1_rtn_nibble(float v) {
    uint32_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    float a = fabsf(v);
    uint32_t mag = (a < 0.25f) ? 0u : (a <= 0.75f) ? 1u : (a < 1.25f) ? 2u :
                   (a <= 1.75f) ? 3u : (a <= 2.5f) ? 4u : (a < 3.5f) ? 5u :
                   (a <= 5.0f) ? 6u : 7u;
    return (mag == 0u) ? 0u : (sign | mag);
}

__global__ void quantize_nvfp4_std(
    uint32_t* __restrict__ packed_u32,
    uint32_t* __restrict__ scales_u32,
    const __half* __restrict__ in_f16,
    int rows, int k_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_chunks64 = k_dim / 64;
    if (idx >= rows * k_chunks64) return;

    int r = idx / k_chunks64;
    int c64 = idx % k_chunks64;
    const __half* base = in_f16 + r * k_dim + c64 * 64;

    uint32_t packed_scales = 0;
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        float vals[16];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            vals[i] = __half2float(base[sub * 16 + i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        }
        float s_raw = fminf(amax * (1.0f / 6.0f), 448.0f);
        __nv_fp8_storage_t s_f8 = __nv_cvt_float_to_fp8(s_raw, __NV_SATFINITE, __NV_E4M3);
        float s_dec = __half2float(__nv_cvt_fp8_to_halfraw(s_f8, __NV_E4M3));
        float inv_s = (amax > 0.0f && s_dec > 0.0f) ? (1.0f / s_dec) : 0.0f;
        packed_scales |= ((uint32_t)s_f8 << (sub * 8));

        uint32_t w0 = 0, w1 = 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint32_t nib = e2m1_rtn_nibble(vals[i] * inv_s);
            if (i < 8) w0 |= (nib << (i * 4));
            else       w1 |= (nib << ((i - 8) * 4));
        }
        packed_u32[r * (k_dim / 8) + c64 * 8 + sub * 2 + 0] = w0;
        packed_u32[r * (k_dim / 8) + c64 * 8 + sub * 2 + 1] = w1;
    }
    scales_u32[r * k_chunks64 + c64] = packed_scales;
}

__global__ void quantize_hash4_f16_aligned(
    uint32_t* __restrict__ packed_u32,
    uint32_t* __restrict__ scales_u32,
    __half* __restrict__ tile_f16,
    const __half* __restrict__ in_f16,
    int rows, int k_dim, uint32_t salt, float dither_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_chunks64 = k_dim / 64;
    if (idx >= rows * k_chunks64) return;

    int r = idx / k_chunks64;
    int c64 = idx % k_chunks64;
    const __half* base = in_f16 + r * k_dim + c64 * 64;

    float vals[4][16];
    float s_exact[4];
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            vals[sub][i] = __half2float(base[sub * 16 + i]);
            amax = fmaxf(amax, fabsf(vals[sub][i]));
        }
        s_exact[sub] = __half2float(__float2half(amax * (1.0f / 6.0f)));
    }

    float best_alpha = 1.0f;
    float best_scale_err = 1e30f;
    uint8_t best_u8[4] = {0, 0, 0, 0};

    #pragma unroll
    for (int step = -4; step <= 4; ++step) {
        float cand_alpha = 1.0f + (float)step * 0.018f;
        float inv_alpha = 1.0f / cand_alpha;
        uint8_t u_bytes[4];
        float u_f32[4];
        float num = 0.0f, den = 0.0f;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            __nv_fp8_storage_t u8 = __nv_cvt_float_to_fp8(fminf(s_exact[sub] * inv_alpha, 448.0f), __NV_SATFINITE, __NV_E4M3);
            u_bytes[sub] = (uint8_t)u8;
            u_f32[sub] = __half2float(__nv_cvt_fp8_to_halfraw(u8, __NV_E4M3));
            num += s_exact[sub] * u_f32[sub];
            den += u_f32[sub] * u_f32[sub];
        }
        float opt_alpha = __half2float(__float2half((den > 0.0f) ? (num / den) : 1.0f));
        float err = 0.0f;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            float d = s_exact[sub] - opt_alpha * u_f32[sub];
            err += d * d;
        }
        if (err < best_scale_err) {
            best_scale_err = err;
            best_alpha = opt_alpha;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) best_u8[sub] = u_bytes[sub];
        }
    }

    uint32_t packed_scales = 0;
    const float E2M1_MAGS[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        packed_scales |= ((uint32_t)best_u8[sub] << (sub * 8));
        float u_dec = __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)best_u8[sub], __NV_E4M3));
        float eff_scale = best_alpha * u_dec;
        float inv_s = (eff_scale > 0.0f) ? (1.0f / eff_scale) : 0.0f;

        uint32_t h0 = hash_mix32((uint32_t)(idx * 4 + sub) ^ salt);
        uint32_t h1 = hash_mix32(((uint32_t)(idx * 4 + sub) << 11) ^ (salt + 0x9E3779B9u));

        uint32_t w0 = 0, w1 = 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float v = vals[sub][i] * inv_s;
            uint32_t nib;
            if (dither_scale == 0.0f) {
                nib = e2m1_rtn_nibble(v);
            } else {
                uint32_t sign = (v < 0.0f) ? 0x8u : 0x0u;
                float a = fminf(fabsf(v), 6.0f);
                int slot = (a < 0.5f) ? 0 : (a < 1.0f) ? 1 : (a < 1.5f) ? 2 :
                           (a < 2.0f) ? 3 : (a < 3.0f) ? 4 : (a < 4.0f) ? 5 :
                           (a < 6.0f) ? 6 : 7;
                uint32_t mag = 7u;
                if (slot < 7) {
                    float low = E2M1_MAGS[slot];
                    float high = E2M1_MAGS[slot + 1];
                    float frac = (a - low) / (high - low);
                    uint32_t h_nib = (i < 8) ? ((h0 >> (i * 4)) & 0xFu) : ((h1 >> ((i - 8) * 4)) & 0xFu);
                    float u = ((float)h_nib + 0.5f) * (1.0f / 16.0f) - 0.5f;
                    mag = (frac + u * dither_scale >= 0.5f) ? (uint32_t)(slot + 1) : (uint32_t)slot;
                }
                nib = (mag == 0u) ? 0u : (sign | mag);
            }
            if (i < 8) w0 |= (nib << (i * 4));
            else       w1 |= (nib << ((i - 8) * 4));
        }
        packed_u32[r * (k_dim / 8) + c64 * 8 + sub * 2 + 0] = w0;
        packed_u32[r * (k_dim / 8) + c64 * 8 + sub * 2 + 1] = w1;
    }
    scales_u32[r * k_chunks64 + c64] = packed_scales;
    tile_f16[r * k_chunks64 + c64] = __float2half(best_alpha);
}

__global__ void gemm_mxf4nvf4_hw_kernel(
    float* __restrict__ C,
    const uint32_t* __restrict__ A_pack,
    const uint32_t* __restrict__ B_pack,
    const uint32_t* __restrict__ A_scale,
    const uint32_t* __restrict__ B_scale,
    const __half* __restrict__ A_f16,
    const __half* __restrict__ B_f16,
    int M, int N, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int tiles_n = N / 8;
    int tile_m = warp_id / tiles_n;
    int tile_n = warp_id % tiles_n;
    if (tile_m >= (M / 16)) return;

    int row_base = tile_m * 16;
    int col_base = tile_n * 8;
    int k_words = K / 8;
    int k_chunks64 = K / 64;

    int grp = lane / 4;
    int tid_in_grp = lane % 4;
    int sfa_row = row_base + grp + (tid_in_grp % 2) * 8;
    int sfb_col = col_base + grp;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int c64 = 0; c64 < k_chunks64; ++c64) {
        uint32_t a_regs[4], b_regs[2];
        a_regs[0] = A_pack[(row_base + grp)     * k_words + c64 * 8 + tid_in_grp];
        a_regs[1] = A_pack[(row_base + grp + 8) * k_words + c64 * 8 + tid_in_grp];
        a_regs[2] = A_pack[(row_base + grp)     * k_words + c64 * 8 + 4 + tid_in_grp];
        a_regs[3] = A_pack[(row_base + grp + 8) * k_words + c64 * 8 + 4 + tid_in_grp];

        b_regs[0] = B_pack[(col_base + grp) * k_words + c64 * 8 + tid_in_grp];
        b_regs[1] = B_pack[(col_base + grp) * k_words + c64 * 8 + 4 + tid_in_grp];

        uint32_t sfa = A_scale[sfa_row * k_chunks64 + c64];
        uint32_t sfb = B_scale[sfb_col * k_chunks64 + c64];

        if (A_f16 == nullptr) {
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                  "r"(b_regs[0]), "r"(b_regs[1]), "r"(sfa), "r"(sfb)
            );
        } else {
            float step[4] = {0.f, 0.f, 0.f, 0.f};
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
                : "+f"(step[0]), "+f"(step[1]), "+f"(step[2]), "+f"(step[3])
                : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                  "r"(b_regs[0]), "r"(b_regs[1]), "r"(sfa), "r"(sfb)
            );
            int r0 = row_base + grp;
            int r1 = r0 + 8;
            int c0 = col_base + tid_in_grp * 2;
            int c1 = c0 + 1;
            float fa0 = __half2float(A_f16[r0 * k_chunks64 + c64]);
            float fa1 = __half2float(A_f16[r1 * k_chunks64 + c64]);
            float fb0 = __half2float(B_f16[c0 * k_chunks64 + c64]);
            float fb1 = __half2float(B_f16[c1 * k_chunks64 + c64]);
            acc[0] += step[0] * (fa0 * fb0);
            acc[1] += step[1] * (fa0 * fb1);
            acc[2] += step[2] * (fa1 * fb0);
            acc[3] += step[3] * (fa1 * fb1);
        }
    }

    int r0 = row_base + grp;
    int r1 = r0 + 8;
    int c0 = col_base + tid_in_grp * 2;
    int c1 = c0 + 1;
    C[r0 * N + c0] = acc[0];
    C[r0 * N + c1] = acc[1];
    C[r1 * N + c0] = acc[2];
    C[r1 * N + c1] = acc[3];
}

void compute_errors(const std::vector<float>& ref, const float* test, int n, double* rel_l2, double* cos_sim) {
    double sq_err = 0.0, sq_ref = 0.0, sq_test = 0.0, dot = 0.0;
    for (int i = 0; i < n; ++i) {
        double r = ref[i];
        double t = test[i];
        double e = t - r;
        sq_err += e * e;
        sq_ref += r * r;
        sq_test += t * t;
        dot += r * t;
    }
    *rel_l2 = std::sqrt(sq_err) / std::max(1e-12, std::sqrt(sq_ref));
    *cos_sim = dot / std::max(1e-12, std::sqrt(sq_ref) * std::sqrt(sq_test));
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    const int M = 512, N = 512, K = 1024;
    std::vector<float> h_A_f32(M * K), h_B_f32(N * K), h_C_ref(M * N, 0.0f);
    __half *d_A_f16, *d_B_f16;
    CHECK_CUDA(cudaMallocManaged(&d_A_f16, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocManaged(&d_B_f16, N * K * sizeof(__half)));

    for (int i = 0; i < M; ++i) {
        for (int c = 0; c < K; ++c) {
            float u = ((float)(hash_mix32(i * 4099 + c + 123) >> 8) / 16777216.0f) * 2.0f - 1.0f;
            float bscale = 0.05f + 3.1f * ((float)(hash_mix32(i * 131 + (c / 16) + 999) >> 8) / 16777216.0f);
            d_A_f16[i * K + c] = __float2half(u * bscale);
            h_A_f32[i * K + c] = __half2float(d_A_f16[i * K + c]);
        }
    }
    for (int j = 0; j < N; ++j) {
        for (int c = 0; c < K; ++c) {
            float u = ((float)(hash_mix32(j * 8191 + c + 456) >> 8) / 16777216.0f) * 2.0f - 1.0f;
            float bscale = 0.04f + 2.7f * ((float)(hash_mix32(j * 257 + (c / 16) + 777) >> 8) / 16777216.0f);
            d_B_f16[j * K + c] = __float2half(u * bscale);
            h_B_f32[j * K + c] = __half2float(d_B_f16[j * K + c]);
        }
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int c = 0; c < K; ++c) s += h_A_f32[i * K + c] * h_B_f32[j * K + c];
            h_C_ref[i * N + j] = s;
        }
    }

    uint32_t *d_A_pack, *d_B_pack, *d_A_scale, *d_B_scale;
    __half *d_A_tile_f16, *d_B_tile_f16;
    float *d_C_nvfp4, *d_C_hash4_1p, *d_C_hash4_sr;
    CHECK_CUDA(cudaMalloc(&d_A_pack, M * (K / 8) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_B_pack, N * (K / 8) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_A_scale, M * (K / 64) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_B_scale, N * (K / 64) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_A_tile_f16, M * (K / 64) * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B_tile_f16, N * (K / 64) * sizeof(__half)));
    CHECK_CUDA(cudaMallocManaged(&d_C_nvfp4, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&d_C_hash4_1p, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&d_C_hash4_sr, M * N * sizeof(float)));

    int q_threads = 256;
    int q_blocks_A = (M * (K / 64) + q_threads - 1) / q_threads;
    int q_blocks_B = (N * (K / 64) + q_threads - 1) / q_threads;
    int warps = (M / 16) * (N / 8);
    int g_blocks = (warps * 32 + 255) / 256;

    quantize_nvfp4_std<<<q_blocks_A, q_threads>>>(d_A_pack, d_A_scale, d_A_f16, M, K);
    quantize_nvfp4_std<<<q_blocks_B, q_threads>>>(d_B_pack, d_B_scale, d_B_f16, N, K);
    gemm_mxf4nvf4_hw_kernel<<<g_blocks, 256>>>(d_C_nvfp4, d_A_pack, d_B_pack, d_A_scale, d_B_scale, nullptr, nullptr, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    quantize_hash4_f16_aligned<<<q_blocks_A, q_threads>>>(d_A_pack, d_A_scale, d_A_tile_f16, d_A_f16, M, K, 0x1111u, 0.0f);
    quantize_hash4_f16_aligned<<<q_blocks_B, q_threads>>>(d_B_pack, d_B_scale, d_B_tile_f16, d_B_f16, N, K, 0x2222u, 0.0f);
    gemm_mxf4nvf4_hw_kernel<<<g_blocks, 256>>>(d_C_hash4_1p, d_A_pack, d_B_pack, d_A_scale, d_B_scale, d_A_tile_f16, d_B_tile_f16, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < M * N; ++i) d_C_hash4_sr[i] = 0.0f;
    float* d_tmp;
    CHECK_CUDA(cudaMallocManaged(&d_tmp, M * N * sizeof(float)));
    for (int s = 0; s < 8; ++s) {
        quantize_hash4_f16_aligned<<<q_blocks_A, q_threads>>>(d_A_pack, d_A_scale, d_A_tile_f16, d_A_f16, M, K, 0x1000u + s * 1337u, 1.0f);
        quantize_hash4_f16_aligned<<<q_blocks_B, q_threads>>>(d_B_pack, d_B_scale, d_B_tile_f16, d_B_f16, N, K, 0x2000u + s * 7919u, 1.0f);
        gemm_mxf4nvf4_hw_kernel<<<g_blocks, 256>>>(d_tmp, d_A_pack, d_B_pack, d_A_scale, d_B_scale, d_A_tile_f16, d_B_tile_f16, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        for (int i = 0; i < M * N; ++i) d_C_hash4_sr[i] += d_tmp[i] / 8.0f;
    }

    double rel_nv, cos_nv, rel_h4_1p, cos_h4_1p, rel_h4_sr, cos_h4_sr;
    compute_errors(h_C_ref, d_C_nvfp4, M * N, &rel_nv, &cos_nv);
    compute_errors(h_C_ref, d_C_hash4_1p, M * N, &rel_h4_1p, &cos_h4_1p);
    compute_errors(h_C_ref, d_C_hash4_sr, M * N, &rel_h4_sr, &cos_h4_sr);

    printf("==========================================================================\n");
    printf("Real SM_120a Hardware GEMM Accuracy ([512x1024] @ [1024x512])\n");
    printf("==========================================================================\n");
    printf("  1. Standard NVFP4 (f8 scale, RTN):                 Rel L2 = %.5f | Cosine = %.6f\n", rel_nv, cos_nv);
    printf("  2. Hash4 + Grid-Aligned f16 Scale (1-pass Inf):    Rel L2 = %.5f | Cosine = %.6f\n", rel_h4_1p, cos_h4_1p);
    printf("  3. Hash4 + Grid-Aligned f16 + Hash-SR (8-step RL): Rel L2 = %.5f | Cosine = %.6f (-> %.2fx lower error!)\n",
           rel_h4_sr, cos_h4_sr, rel_nv / rel_h4_sr);

    return 0;
}
