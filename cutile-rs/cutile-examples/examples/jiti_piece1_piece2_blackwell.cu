// ============================================================================
// JITI Piece 1 & Piece 2 (Tile-Coalesced) Stress Test on Blackwell SM_120a
//
// Compile & Run:
//   nvcc -O3 -gencode arch=compute_120a,code=sm_120a \
//        jiti_piece1_piece2_blackwell.cu -o jiti_piece1_piece2_blackwell
//   ./jiti_piece1_piece2_blackwell
// ============================================================================

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

__device__ __host__ __forceinline__ float decode_e2m1(uint32_t nib) {
    const float T[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return T[nib & 0xFu];
}

__device__ __host__ __forceinline__ uint32_t e2m1_rtn_nibble(float v) {
    uint32_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    float a = fabsf(v);
    uint32_t mag = (a < 0.25f) ? 0u : (a <= 0.75f) ? 1u : (a < 1.25f) ? 2u :
                   (a <= 1.75f) ? 3u : (a <= 2.5f) ? 4u : (a < 3.5f) ? 5u :
                   (a <= 5.0f) ? 6u : 7u;
    return (mag == 0u) ? 0u : (sign | mag);
}

__global__ void quantize_nvfp4_std_kernel(
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
    for (int sub = 0; sub < 4; ++sub) {
        float vals[16], amax = 0.0f;
        for (int i = 0; i < 16; ++i) {
            vals[i] = __half2float(base[sub * 16 + i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        }
        __nv_fp8_storage_t s_f8 = __nv_cvt_float_to_fp8(fminf(amax / 6.0f, 448.0f), __NV_SATFINITE, __NV_E4M3);
        float s_dec = __half2float(__nv_cvt_fp8_to_halfraw(s_f8, __NV_E4M3));
        float inv_s = (amax > 0.0f && s_dec > 0.0f) ? (1.0f / s_dec) : 0.0f;
        packed_scales |= ((uint32_t)s_f8 << (sub * 8));

        uint32_t w0 = 0, w1 = 0;
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

__global__ void pack_jiti_coalesced_tiles_kernel(
    uint32_t* __restrict__ W_tq_msb2_coal,   // [(N/16), (K/64), 2, 32]
    uint32_t* __restrict__ W_leg_lsb2_coal,  // [(N/16), (K/64), 2, 32]
    uint16_t* __restrict__ W_leg_top2_coal,  // [(N/16), (K/64), 2, 32]
    const uint32_t* __restrict__ W_pack_std, // [N, K / 8]
    int N, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int k_chunks64 = K / 64;
    int total_warps = (N / 16) * k_chunks64;
    if (warp_id >= total_warps) return;

    int tile_w = warp_id / k_chunks64;
    int c64    = warp_id % k_chunks64;
    int w_row_base = tile_w * 16;
    int k_words = K / 8;

    int grp = lane / 4;
    int tid_in_grp = lane % 4;

    for (int half_k = 0; half_k < 2; ++half_k) {
        int kw = c64 * 8 + half_k * 4 + tid_in_grp;
        uint32_t w_r0 = W_pack_std[(w_row_base + grp)     * k_words + kw];
        uint32_t w_r1 = W_pack_std[(w_row_base + grp + 8) * k_words + kw];

        uint32_t tq_msb2  = (w_r0 & 0xCCCCCCCCu) | ((w_r1 & 0xCCCCCCCCu) >> 2);
        uint32_t leg_lsb2 = (w_r0 & 0x33333333u) | ((w_r1 & 0x33333333u) << 2);

        uint16_t leg_top2 = 0;
        for (int h = 0; h < 2; ++h) {
            uint32_t w_tru = (h == 0) ? w_r0 : w_r1;
            uint32_t msb_h = w_tru & 0xCCCCCCCCu;
            uint32_t e1    = (msb_h >> 2) & 0x11111111u;
            uint32_t w_app = msb_h | ((~e1 & 0x11111111u) << 1) | e1;
            int best_i = 0;
            float max_err = -1.0f;
            for (int i = 0; i < 8; ++i) {
                uint32_t tn = (w_tru >> (i * 4)) & 0xFu;
                uint32_t an = (w_app >> (i * 4)) & 0xFu;
                float err = fabsf(decode_e2m1(tn) - decode_e2m1(an));
                if (err > max_err) {
                    max_err = err;
                    best_i = i;
                }
            }
            uint32_t best_nib = (w_tru >> (best_i * 4)) & 0xFu;
            uint8_t byte_p = (uint8_t)((best_i & 7) | (best_nib << 4));
            leg_top2 |= ((uint16_t)byte_p << (h * 8));
        }

        size_t out_idx = ((size_t)(tile_w * k_chunks64 + c64) * 2 + half_k) * 32 + lane;
        W_tq_msb2_coal[out_idx]  = tq_msb2;
        W_leg_lsb2_coal[out_idx] = leg_lsb2;
        W_leg_top2_coal[out_idx] = leg_top2;
    }
}

__device__ __forceinline__ void unpack_jiti_pair_from_single_u32(
    uint32_t tq_msb2,
    uint32_t leg_lsb2,
    uint16_t leg_top2,
    int mode,
    uint32_t* out_r0,
    uint32_t* out_r1
) {
    if (mode == 2) {
        *out_r0 = (tq_msb2 & 0xCCCCCCCCu) | (leg_lsb2 & 0x33333333u);
        *out_r1 = ((tq_msb2 << 2u) & 0xCCCCCCCCu) | ((leg_lsb2 >> 2u) & 0x33333333u);
        return;
    }
    uint32_t msb0 = tq_msb2 & 0xCCCCCCCCu;
    uint32_t e1_0 = (msb0 >> 2u) & 0x11111111u;
    uint32_t w0   = msb0 | ((~e1_0 & 0x11111111u) << 1u) | e1_0;

    uint32_t msb1 = (tq_msb2 << 2u) & 0xCCCCCCCCu;
    uint32_t e1_1 = (msb1 >> 2u) & 0x11111111u;
    uint32_t w1   = msb1 | ((~e1_1 & 0x11111111u) << 1u) | e1_1;

    if (mode == 1) {
        uint32_t p0 = leg_top2 & 0xFFu;
        uint32_t sh0 = (p0 & 7u) * 4u;
        w0 = (w0 & ~(0xFu << sh0)) | ((p0 >> 4u) << sh0);

        uint32_t p1 = (leg_top2 >> 8u) & 0xFFu;
        uint32_t sh1 = (p1 & 7u) * 4u;
        w1 = (w1 & ~(0xFu << sh1)) | ((p1 >> 4u) << sh1);
    }
    *out_r0 = w0;
    *out_r1 = w1;
}

__global__ void gemm_standard_splitk_kernel(
    float* __restrict__ C,
    const uint32_t* __restrict__ X_pack,
    const uint32_t* __restrict__ X_scale,
    const uint32_t* __restrict__ W_pack,
    const uint32_t* __restrict__ W_scale,
    int M_tokens, int N, int K, int split_k
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int tiles_n = N / 8;
    int tile_n = warp_id % tiles_n;
    int k_slice = warp_id / tiles_n;
    if (k_slice >= split_k) return;

    int col_base = tile_n * 8;
    int k_words = K / 8;
    int k_chunks64 = K / 64;
    int chunks_per_slice = k_chunks64 / split_k;
    int c64_start = k_slice * chunks_per_slice;
    int c64_end   = c64_start + chunks_per_slice;

    int grp = lane / 4;
    int tid_in_grp = lane % 4;
    int sfa_row = grp + (tid_in_grp % 2) * 8;
    int sfb_col = col_base + grp;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int c64 = c64_start; c64 < c64_end; ++c64) {
        uint32_t a_regs[4], b_regs[2];
        a_regs[0] = X_pack[grp       * k_words + c64 * 8 + tid_in_grp];
        a_regs[1] = X_pack[(grp + 8) * k_words + c64 * 8 + tid_in_grp];
        a_regs[2] = X_pack[grp       * k_words + c64 * 8 + 4 + tid_in_grp];
        a_regs[3] = X_pack[(grp + 8) * k_words + c64 * 8 + 4 + tid_in_grp];

        b_regs[0] = W_pack[(col_base + grp) * k_words + c64 * 8 + tid_in_grp];
        b_regs[1] = W_pack[(col_base + grp) * k_words + c64 * 8 + 4 + tid_in_grp];

        uint32_t sfa = X_scale[sfa_row * k_chunks64 + c64];
        uint32_t sfb = W_scale[sfb_col * k_chunks64 + c64];

        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
              "r"(b_regs[0]), "r"(b_regs[1]), "r"(sfa), "r"(sfb)
        );
    }

    int r0 = grp, r1 = grp + 8;
    int c0 = col_base + tid_in_grp * 2, c1 = c0 + 1;
    if (r0 < M_tokens) {
        atomicAdd(&C[r0 * N + c0], acc[0]);
        atomicAdd(&C[r0 * N + c1], acc[1]);
    }
    if (r1 < M_tokens) {
        atomicAdd(&C[r1 * N + c0], acc[2]);
        atomicAdd(&C[r1 * N + c1], acc[3]);
    }
}

__global__ void gemm_jiti_coalesced_splitk_kernel(
    float* __restrict__ C,
    const uint32_t* __restrict__ W_tq_msb2_coal,
    const uint32_t* __restrict__ W_leg_lsb2_coal,
    const uint16_t* __restrict__ W_leg_top2_coal,
    const uint32_t* __restrict__ W_scale,
    const uint32_t* __restrict__ X_pack,
    const uint32_t* __restrict__ X_scale,
    int M_tokens, int N, int K, int split_k, int mode
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int tiles_w = N / 16;
    int tile_w = warp_id % tiles_w;
    int k_slice = warp_id / tiles_w;
    if (k_slice >= split_k) return;

    int w_row_base = tile_w * 16;
    int k_words = K / 8;
    int k_chunks64 = K / 64;
    int chunks_per_slice = k_chunks64 / split_k;
    int c64_start = k_slice * chunks_per_slice;
    int c64_end   = c64_start + chunks_per_slice;

    int grp = lane / 4;
    int tid_in_grp = lane % 4;
    int sfa_w_row = w_row_base + grp + (tid_in_grp % 2) * 8;
    int sfb_x_col = grp;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int c64 = c64_start; c64 < c64_end; ++c64) {
        size_t base_idx0 = ((size_t)(tile_w * k_chunks64 + c64) * 2 + 0) * 32 + lane;
        size_t base_idx1 = ((size_t)(tile_w * k_chunks64 + c64) * 2 + 1) * 32 + lane;

        uint32_t w_regs[4];
        unpack_jiti_pair_from_single_u32(
            W_tq_msb2_coal[base_idx0],
            (mode == 2) ? W_leg_lsb2_coal[base_idx0] : 0,
            (mode == 1) ? W_leg_top2_coal[base_idx0] : 0,
            mode, &w_regs[0], &w_regs[1]
        );
        unpack_jiti_pair_from_single_u32(
            W_tq_msb2_coal[base_idx1],
            (mode == 2) ? W_leg_lsb2_coal[base_idx1] : 0,
            (mode == 1) ? W_leg_top2_coal[base_idx1] : 0,
            mode, &w_regs[2], &w_regs[3]
        );

        uint32_t x_regs[2];
        x_regs[0] = X_pack[grp * k_words + c64 * 8 + tid_in_grp];
        x_regs[1] = X_pack[grp * k_words + c64 * 8 + 4 + tid_in_grp];

        uint32_t sfa = W_scale[sfa_w_row * k_chunks64 + c64];
        uint32_t sfb = X_scale[sfb_x_col * k_chunks64 + c64];

        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(w_regs[0]), "r"(w_regs[1]), "r"(w_regs[2]), "r"(w_regs[3]),
              "r"(x_regs[0]), "r"(x_regs[1]), "r"(sfa), "r"(sfb)
        );
    }

    int n0 = w_row_base + grp, n1 = n0 + 8;
    int m0 = tid_in_grp * 2,   m1 = m0 + 1;
    if (m0 < M_tokens) {
        atomicAdd(&C[m0 * N + n0], acc[0]);
        atomicAdd(&C[m0 * N + n1], acc[2]);
    }
    if (m1 < M_tokens) {
        atomicAdd(&C[m1 * N + n0], acc[1]);
        atomicAdd(&C[m1 * N + n1], acc[3]);
    }
}

void compute_errors(const float* ref, const float* test, int n, double* rel_l2, double* cos_sim, float* max_diff) {
    double sq_err = 0.0, sq_ref = 0.0, sq_test = 0.0, dot = 0.0;
    float md = 0.0f;
    for (int i = 0; i < n; ++i) {
        double r = ref[i], t = test[i], e = t - r;
        sq_err += e * e; sq_ref += r * r; sq_test += t * t; dot += r * t;
        md = std::max(md, std::fabs(ref[i] - test[i]));
    }
    *rel_l2 = std::sqrt(sq_err) / std::max(1e-12, std::sqrt(sq_ref));
    *cos_sim = dot / std::max(1e-12, std::sqrt(sq_ref) * std::sqrt(sq_test));
    *max_diff = md;
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("==================================================================================\n");
    printf("JITI Piece 1 & Piece 2 (Tile-Coalesced) Stress Test on %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("==================================================================================\n\n");

    const int N = 8192, K = 4096;
    __half *d_X_f16, *d_W_f16;
    CHECK_CUDA(cudaMallocManaged(&d_X_f16, 16 * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocManaged(&d_W_f16, (size_t)N * K * sizeof(__half)));

    for (int i = 0; i < 16 * K; ++i) {
        float u = ((float)(hash_mix32(i + 101) >> 8) / 16777216.0f) * 2.0f - 1.0f;
        d_X_f16[i] = __float2half(u * 1.5f);
    }
    for (size_t i = 0; i < (size_t)N * K; ++i) {
        uint32_t h1 = hash_mix32(i * 17 + 456);
        uint32_t h2 = hash_mix32(i * 31 + 789);
        float u1 = std::max(1e-7f, (float)(h1 >> 8) / 16777216.0f);
        float u2 = (float)(h2 >> 8) / 16777216.0f;
        float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
        d_W_f16[i] = __float2half(z * 1.1f);
    }

    uint32_t *d_X_pack, *d_X_scale, *d_W_pack, *d_W_scale;
    uint32_t *d_W_tq_msb2_coal, *d_W_leg_lsb2_coal;
    uint16_t *d_W_leg_top2_coal;
    float *d_C_std, *d_C_jiti;

    CHECK_CUDA(cudaMalloc(&d_X_pack, 16 * (K / 8) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_X_scale, 16 * (K / 64) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_W_pack, (size_t)N * (K / 8) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_W_scale, (size_t)N * (K / 64) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_W_tq_msb2_coal, (size_t)N * (K / 16) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_W_leg_lsb2_coal, (size_t)N * (K / 16) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_W_leg_top2_coal, (size_t)N * (K / 16) * sizeof(uint16_t)));
    CHECK_CUDA(cudaMallocManaged(&d_C_std, 16 * N * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&d_C_jiti, 16 * N * sizeof(float)));

    quantize_nvfp4_std_kernel<<<(16 * (K / 64) + 255) / 256, 256>>>(d_X_pack, d_X_scale, d_X_f16, 16, K);
    quantize_nvfp4_std_kernel<<<(N * (K / 64) + 255) / 256, 256>>>(d_W_pack, d_W_scale, d_W_f16, N, K);

    int pack_warps = (N / 16) * (K / 64);
    pack_jiti_coalesced_tiles_kernel<<<(pack_warps * 32 + 255) / 256, 256>>>(
        d_W_tq_msb2_coal, d_W_leg_lsb2_coal, d_W_leg_top2_coal, d_W_pack, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int split_k = 4;
    const int M_tok = 1;
    int warps_std  = (N / 8) * split_k;
    int warps_jiti = (N / 16) * split_k;
    int blocks_std  = (warps_std * 32 + 255) / 256;
    int blocks_jiti = (warps_jiti * 32 + 255) / 256;

    for (int r = 0; r < 10; ++r) {
        gemm_standard_splitk_kernel<<<blocks_std, 256>>>(d_C_std, d_X_pack, d_X_scale, d_W_pack, d_W_scale, M_tok, N, K, split_k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    const int runs = 500;
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; ++r) {
        gemm_standard_splitk_kernel<<<blocks_std, 256>>>(d_C_std, d_X_pack, d_X_scale, d_W_pack, d_W_scale, M_tok, N, K, split_k);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_std = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_std, start, stop));
    float us_std = (ms_std * 1000.0f) / runs;

    CHECK_CUDA(cudaMemset(d_C_std, 0, 16 * N * sizeof(float)));
    gemm_standard_splitk_kernel<<<blocks_std, 256>>>(d_C_std, d_X_pack, d_X_scale, d_W_pack, d_W_scale, M_tok, N, K, split_k);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("--- Single-Token (M=1) MoE Gate+Up Projection [N=8192, K=4096] (33.55M weights) ---\n");
    printf("  [Baseline] Standard NVFP4 (X in m16, W in n8)          : %5.2f us\n\n", us_std);

    const char* mode_names[3] = {
        "Mode 0: JITI Coalesced + 2-Bit MSB Bulk Only (2.50 bits/w)",
        "Mode 1: JITI Coalesced + 2-Bit MSB + Top-2 Patch (3.50 bits/w)",
        "Mode 2: JITI Coalesced + 2-Bit MSB + Exact LSB2 (100% Bit-Exact NVFP4)"
    };

    for (int mode = 0; mode < 3; ++mode) {
        CHECK_CUDA(cudaMemset(d_C_jiti, 0, 16 * N * sizeof(float)));
        gemm_jiti_coalesced_splitk_kernel<<<blocks_jiti, 256>>>(
            d_C_jiti, d_W_tq_msb2_coal, d_W_leg_lsb2_coal, d_W_leg_top2_coal, d_W_scale,
            d_X_pack, d_X_scale, M_tok, N, K, split_k, mode);
        CHECK_CUDA(cudaDeviceSynchronize());

        double rel_l2, cos_sim;
        float max_diff;
        compute_errors(d_C_std, d_C_jiti, M_tok * N, &rel_l2, &cos_sim, &max_diff);

        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < runs; ++r) {
            gemm_jiti_coalesced_splitk_kernel<<<blocks_jiti, 256>>>(
                d_C_jiti, d_W_tq_msb2_coal, d_W_leg_lsb2_coal, d_W_leg_top2_coal, d_W_scale,
                d_X_pack, d_X_scale, M_tok, N, K, split_k, mode);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_jiti = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_jiti, start, stop));
        float us_jiti = (ms_jiti * 1000.0f) / runs;

        printf("  %s:\n", mode_names[mode]);
        printf("    Latency = %5.2f us (%4.2fx speedup vs Std NVFP4) | vs NVFP4: Rel L2 = %.6f | Cosine = %.6f | Max Diff = %.6f\n",
               us_jiti, us_std / us_jiti, rel_l2, cos_sim, max_diff);
    }

    return 0;
}
