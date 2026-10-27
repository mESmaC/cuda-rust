// ============================================================================
// TurboQuant 2-Bit Transport + Pre-Sent Error Ledger -> Blackwell SM_120a MMA
//
// Compile & Run on Blackwell (RTX 50-series / RTX PRO 6000 Server Edition):
//   nvcc -O3 -gencode arch=compute_120a,code=sm_120a \
//        turboquant_error_ledger_blackwell.cu -o turboquant_error_ledger_blackwell
//   ./turboquant_error_ledger_blackwell
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

// Hardware E2M1 decode table for 4-bit nibble (0..15)
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

// ============================================================================
// 2-Instruction Bit-Interleaved SWAR TurboQuant 2-Bit Codebook!
//
// Observe the 4 inner E2M1 Gaussian bulk centroids {-1.0, -0.5, +0.5, +1.0}:
//   +0.5 = 0x1 (0001b) -> sign=0, mag=0, b0 = ~mag = 1
//   +1.0 = 0x2 (0010b) -> sign=0, mag=1, b0 = ~mag = 0
//   -0.5 = 0x9 (1001b) -> sign=1, mag=0, b0 = ~mag = 1
//   -1.0 = 0xA (1010b) -> sign=1, mag=1, b0 = ~mag = 0
// Every nibble has b3 = sign, b2 = 0, b1 = mag, b0 = ~mag!
// So if we pack 16 weights (2 halves of 8 weights) into a 32-bit `tq_word`:
//   - Half 0 (weights 0..7): sign_i at bit (4*i + 3), mag_i at bit (4*i + 1)  [mask 0xAAAAAAAA]
//   - Half 1 (weights 8..15): sign_i at bit (4*i + 2), mag_i at bit (4*i + 0) [mask 0x55555555]
// Then unpacking 8 weights to 8 E2M1 nibbles on the GPU is TWO SWAR instructions:
//   uint32_t sm = (tq_word << half_sel) & 0xAAAAAAAAu;
//   uint32_t w  = sm | (((~sm) >> 1) & 0x11111111u);
// And the Pre-Sent Error Ledger patches exact 0.0 and outliers up to +/-6.0!
// ============================================================================

__global__ void encode_turboquant_with_ledger_kernel(
    uint32_t* __restrict__ tq2_u32,          // [ROWS, K / 16] (2.0 bits/elem SWAR-interleaved bulk)
    uint32_t* __restrict__ scales_u32,       // [ROWS, K / 64] (0.5 bits/elem ue4m3 block scales)
    uint16_t* __restrict__ ledger_top2_u16,  // [ROWS, K / 16] (1.0 bit/elem Top-2 Error Ledger: 1 patch per 8 weights)
    uint16_t* __restrict__ ledger_top4_u16,  // [ROWS, K / 16] (extra 1.0 bit/elem Top-4 Error Ledger: 2nd patch per 8 weights)
    const __half* __restrict__ in_f16,       // [ROWS, K]
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
        float vals[16];
        float amax = 0.0f;
        for (int i = 0; i < 16; ++i) {
            vals[i] = __half2float(base[sub * 16 + i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        }

        __nv_fp8_storage_t best_s_f8 = 0;
        float best_err = 1e30f;
        uint32_t best_tq_word = 0;
        uint16_t best_leg2 = 0;
        uint16_t best_leg4 = 0;

        for (int step = 0; step < 9; ++step) {
            float mult = 0.70f + 0.075f * (float)step; // 0.70 .. 1.30 of (amax / 6.0)
            float s_raw = fminf(amax * (1.0f / 6.0f) * mult, 448.0f);
            __nv_fp8_storage_t s_f8 = __nv_cvt_float_to_fp8(s_raw, __NV_SATFINITE, __NV_E4M3);
            float s_dec = __half2float(__nv_cvt_fp8_to_halfraw(s_f8, __NV_E4M3));
            float inv_s = (amax > 0.0f && s_dec > 0.0f) ? (1.0f / s_dec) : 0.0f;

            uint32_t tq_word = 0;
            uint16_t leg2 = 0, leg4 = 0;
            float total_sq_err = 0.0f;

            for (int h = 0; h < 2; ++h) {
                float err_mag[8];
                uint32_t true_nib[8];
                float bulk_recon[8];

                for (int i = 0; i < 8; ++i) {
                    float raw_w = vals[h * 8 + i];
                    float v = raw_w * inv_s;
                    true_nib[i] = e2m1_rtn_nibble(v);

                    // 2-bit bulk code in {-1.0, -0.5, +0.5, +1.0}
                    uint32_t sign_bit = (v < 0.0f) ? 1u : 0u;
                    uint32_t mag_bit  = (fabsf(v) >= 0.75f) ? 1u : 0u;

                    int bit_s = i * 4 + (3 - h);
                    int bit_m = i * 4 + (1 - h);
                    tq_word |= (sign_bit << bit_s) | (mag_bit << bit_m);

                    uint32_t b_nib = (sign_bit << 3) | (mag_bit << 1) | (mag_bit ^ 1u);
                    bulk_recon[i] = decode_e2m1(b_nib) * s_dec;
                    float patch_recon = decode_e2m1(true_nib[i]) * s_dec;
                    float e_bulk = raw_w - bulk_recon[i];
                    float e_patch = raw_w - patch_recon;
                    err_mag[i] = (e_bulk * e_bulk) - (e_patch * e_patch);
                }

                int best0 = 0, best1 = 0;
                float max_gain0 = -1.0f;
                for (int i = 0; i < 8; ++i) {
                    if (err_mag[i] > max_gain0) {
                        max_gain0 = err_mag[i];
                        best0 = i;
                    }
                }
                err_mag[best0] = -1.0f;
                float max_gain1 = -1.0f;
                for (int i = 0; i < 8; ++i) {
                    if (err_mag[i] > max_gain1) {
                        max_gain1 = err_mag[i];
                        best1 = i;
                    }
                }

                uint8_t byte_p0 = (uint8_t)((best0 & 0x7) | ((true_nib[best0] & 0xF) << 4));
                uint8_t byte_p1 = (uint8_t)((best1 & 0x7) | ((true_nib[best1] & 0xF) << 4));
                leg2 |= ((uint16_t)byte_p0 << (h * 8));
                leg4 |= ((uint16_t)byte_p1 << (h * 8));

                for (int i = 0; i < 8; ++i) {
                    float rec = (i == best0) ? (decode_e2m1(true_nib[i]) * s_dec) : bulk_recon[i];
                    float d = vals[h * 8 + i] - rec;
                    total_sq_err += d * d;
                }
            }

            if (total_sq_err < best_err) {
                best_err = total_sq_err;
                best_s_f8 = s_f8;
                best_tq_word = tq_word;
                best_leg2 = leg2;
                best_leg4 = leg4;
            }
        }

        packed_scales |= ((uint32_t)best_s_f8 << (sub * 8));
        int blk16_idx = r * (k_dim / 16) + c64 * 4 + sub;
        tq2_u32[blk16_idx] = best_tq_word;
        ledger_top2_u16[blk16_idx] = best_leg2;
        ledger_top4_u16[blk16_idx] = best_leg4;
    }
    scales_u32[r * k_chunks64 + c64] = packed_scales;
}

__device__ __forceinline__ uint32_t unpack_tq2_half_with_ledger(
    uint32_t tq_word,
    uint32_t half_sel, // 0 or 1
    uint16_t leg_top2,
    uint16_t leg_top4,
    int num_patches
) {
    // 1. SWAR 2-instruction bulk unpack: 8 x 2-bit codes -> 8 x 4-bit E2M1 nibbles {-1.0, -0.5, +0.5, +1.0}
    uint32_t sm = (tq_word << half_sel) & 0xAAAAAAAAu;
    uint32_t w  = sm | (((~sm) >> 1) & 0x11111111u);

    // 2. Apply Pre-Sent Error Ledger patch for this 8-weight half
    if (num_patches >= 2) {
        uint32_t p0 = (leg_top2 >> (half_sel * 8u)) & 0xFFu;
        uint32_t sh0 = (p0 & 7u) * 4u;
        uint32_t nib0 = p0 >> 4u;
        w = (w & ~(0xFu << sh0)) | (nib0 << sh0);
    }
    if (num_patches >= 4) {
        uint32_t p1 = (leg_top4 >> (half_sel * 8u)) & 0xFFu;
        uint32_t sh1 = (p1 & 7u) * 4u;
        uint32_t nib1 = p1 >> 4u;
        w = (w & ~(0xFu << sh1)) | (nib1 << sh1);
    }
    return w;
}

__global__ void gemm_fused_tq_ledger_mxf4_kernel(
    float* __restrict__ C,
    const uint32_t* __restrict__ A_pack_nvfp4,
    const uint32_t* __restrict__ A_scale,
    const uint32_t* __restrict__ B_tq2,
    const uint32_t* __restrict__ B_scale,
    const uint16_t* __restrict__ B_ledger_top2,
    const uint16_t* __restrict__ B_ledger_top4,
    int M, int N, int K, int num_patches
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
    int k_blk16 = K / 16;
    int k_chunks64 = K / 64;

    int grp = lane / 4;
    int tid_in_grp = lane % 4;
    int sfa_row = row_base + grp + (tid_in_grp % 2) * 8;
    int sfb_col = col_base + grp;

    uint32_t half_sel = (uint32_t)(tid_in_grp % 2);
    int sub0 = tid_in_grp / 2;
    int sub1 = 2 + (tid_in_grp / 2);

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int c64 = 0; c64 < k_chunks64; ++c64) {
        uint32_t a_regs[4];
        a_regs[0] = A_pack_nvfp4[(row_base + grp)     * k_words + c64 * 8 + tid_in_grp];
        a_regs[1] = A_pack_nvfp4[(row_base + grp + 8) * k_words + c64 * 8 + tid_in_grp];
        a_regs[2] = A_pack_nvfp4[(row_base + grp)     * k_words + c64 * 8 + 4 + tid_in_grp];
        a_regs[3] = A_pack_nvfp4[(row_base + grp + 8) * k_words + c64 * 8 + 4 + tid_in_grp];

        int b_idx0 = (col_base + grp) * k_blk16 + c64 * 4 + sub0;
        int b_idx1 = (col_base + grp) * k_blk16 + c64 * 4 + sub1;

        uint32_t b_regs[2];
        b_regs[0] = unpack_tq2_half_with_ledger(
            B_tq2[b_idx0], half_sel,
            (num_patches >= 2) ? B_ledger_top2[b_idx0] : 0,
            (num_patches >= 4) ? B_ledger_top4[b_idx0] : 0,
            num_patches
        );
        b_regs[1] = unpack_tq2_half_with_ledger(
            B_tq2[b_idx1], half_sel,
            (num_patches >= 2) ? B_ledger_top2[b_idx1] : 0,
            (num_patches >= 4) ? B_ledger_top4[b_idx1] : 0,
            num_patches
        );

        uint32_t sfa = A_scale[sfa_row * k_chunks64 + c64];
        uint32_t sfb = B_scale[sfb_col * k_chunks64 + c64];

        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
              "r"(b_regs[0]), "r"(b_regs[1]), "r"(sfa), "r"(sfb)
        );
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

__global__ void gemm_nvfp4_std_kernel(
    float* __restrict__ C,
    const uint32_t* __restrict__ A_pack,
    const uint32_t* __restrict__ A_scale,
    const uint32_t* __restrict__ B_pack,
    const uint32_t* __restrict__ B_scale,
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

        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
              "r"(b_regs[0]), "r"(b_regs[1]), "r"(sfa), "r"(sfb)
        );
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
        double r = ref[i], t = test[i], e = t - r;
        sq_err += e * e; sq_ref += r * r; sq_test += t * t; dot += r * t;
    }
    *rel_l2 = std::sqrt(sq_err) / std::max(1e-12, std::sqrt(sq_ref));
    *cos_sim = dot / std::max(1e-12, std::sqrt(sq_ref) * std::sqrt(sq_test));
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("==================================================================================\n");
    printf("TurboQuant + Pre-Sent Error Ledger Benchmark on %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("==================================================================================\n\n");

    const int M = 512, N = 512, K = 1024;
    std::vector<float> h_A_f32(M * K), h_B_f32(N * K), h_C_ref(M * N, 0.0f);
    __half *d_A_f16, *d_B_f16;
    CHECK_CUDA(cudaMallocManaged(&d_A_f16, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMallocManaged(&d_B_f16, N * K * sizeof(__half)));

    // Realistic LLM/MoE weight distribution: Gaussian bulk (sigma=0.65) + outlier channels (up to +/-5.8)
    for (int i = 0; i < M; ++i) {
        for (int c = 0; c < K; ++c) {
            float u = ((float)(hash_mix32(i * 4099 + c + 123) >> 8) / 16777216.0f) * 2.0f - 1.0f;
            d_A_f16[i * K + c] = __float2half(u * 1.8f);
            h_A_f32[i * K + c] = __half2float(d_A_f16[i * K + c]);
        }
    }
    for (int j = 0; j < N; ++j) {
        for (int c = 0; c < K; ++c) {
            uint32_t h1 = hash_mix32(j * 8191 + c + 456);
            uint32_t h2 = hash_mix32(j * 131071 + c * 17 + 789);
            float u1 = std::max(1e-7f, (float)(h1 >> 8) / 16777216.0f);
            float u2 = (float)(h2 >> 8) / 16777216.0f;
            float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
            // 87.5% Gaussian bulk (sigma = 0.65, clustered around +/-0.5..1.0), 12.5% heavy outliers (~4.5..6.0)
            float val = ((c % 8) == 0) ? ((z >= 0.0f ? 1.0f : -1.0f) * (4.2f + 1.5f * u2)) : (z * 0.65f);
            d_B_f16[j * K + c] = __float2half(val);
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

    uint32_t *d_A_pack, *d_A_scale, *d_B_pack_nvfp4, *d_B_scale_nvfp4;
    uint32_t *d_B_tq2, *d_B_scale_tq;
    uint16_t *d_B_leg2, *d_B_leg4;
    float *d_C_out;

    CHECK_CUDA(cudaMalloc(&d_A_pack, M * (K / 8) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_A_scale, M * (K / 64) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_B_pack_nvfp4, N * (K / 8) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_B_scale_nvfp4, N * (K / 64) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_B_tq2, N * (K / 16) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_B_scale_tq, N * (K / 64) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_B_leg2, N * (K / 16) * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_B_leg4, N * (K / 16) * sizeof(uint16_t)));
    CHECK_CUDA(cudaMallocManaged(&d_C_out, M * N * sizeof(float)));

    int q_threads = 256;
    int q_blocks_A = (M * (K / 64) + q_threads - 1) / q_threads;
    int q_blocks_B = (N * (K / 64) + q_threads - 1) / q_threads;
    int warps = (M / 16) * (N / 8);
    int g_blocks = (warps * 32 + 255) / 256;

    quantize_nvfp4_std<<<q_blocks_A, q_threads>>>(d_A_pack, d_A_scale, d_A_f16, M, K);
    quantize_nvfp4_std<<<q_blocks_B, q_threads>>>(d_B_pack_nvfp4, d_B_scale_nvfp4, d_B_f16, N, K);
    encode_turboquant_with_ledger_kernel<<<q_blocks_B, q_threads>>>(
        d_B_tq2, d_B_scale_tq, d_B_leg2, d_B_leg4, d_B_f16, N, K
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("--- 1. Real SM_120a Hardware GEMM Accuracy & Speed ([512x1024] @ [1024x512]) ---\n");
    for (int patches : {0, 2, 4}) {
        for (int r = 0; r < 10; ++r) {
            gemm_fused_tq_ledger_mxf4_kernel<<<g_blocks, 256>>>(
                d_C_out, d_A_pack, d_A_scale, d_B_tq2, d_B_scale_tq, d_B_leg2, d_B_leg4, M, N, K, patches
            );
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        const int runs = 200;
        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < runs; ++r) {
            gemm_fused_tq_ledger_mxf4_kernel<<<g_blocks, 256>>>(
                d_C_out, d_A_pack, d_A_scale, d_B_tq2, d_B_scale_tq, d_B_leg2, d_B_leg4, M, N, K, patches
            );
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        double rel_l2, cos_sim;
        compute_errors(h_C_ref, d_C_out, M * N, &rel_l2, &cos_sim);
        float bits_per_w = 2.50f + (float)patches * 0.50f;
        float p4608_eff_gbs = 2.0f * (16.0f / bits_per_w);
        printf("  TurboQuant-2bit + Top-%d Error Ledger (%4.2f bits/w | %4.1f GB/s eff on P4608): %5.2f us | Rel L2 = %.5f | Cosine = %.6f\n",
               patches, bits_per_w, p4608_eff_gbs, (ms * 1000.0f) / runs, rel_l2, cos_sim);
    }

    // Standard NVFP4 baseline
    {
        for (int r = 0; r < 10; ++r) {
            gemm_nvfp4_std_kernel<<<g_blocks, 256>>>(d_C_out, d_A_pack, d_A_scale, d_B_pack_nvfp4, d_B_scale_nvfp4, M, N, K);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        const int runs = 200;
        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < runs; ++r) {
            gemm_nvfp4_std_kernel<<<g_blocks, 256>>>(d_C_out, d_A_pack, d_A_scale, d_B_pack_nvfp4, d_B_scale_nvfp4, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        double rel_l2, cos_sim;
        compute_errors(h_C_ref, d_C_out, M * N, &rel_l2, &cos_sim);
        printf("  Standard NVFP4 Baseline (no transport) (%4.2f bits/w | %4.1f GB/s eff on P4608): %5.2f us | Rel L2 = %.5f | Cosine = %.6f\n\n",
               4.50f, 2.0f * (16.0f / 4.50f), (ms * 1000.0f) / runs, rel_l2, cos_sim);
    }

    // ============================================================================
    // 2. Live PCIe Host-to-GPU Streaming + Fused Unpack & GEMM Benchmark
    // ============================================================================
    printf("--- 2. Live PCIe H2D Expert Streaming + Fused SM_120a GEMM (MoE Expert [4096x4096], M=128) ---\n");
    const int SM = 128, SN = 4096, SK = 4096;
    const size_t num_w = (size_t)SN * SK;

    size_t bytes_fp16     = num_w * sizeof(__half);          // 33.55 MB (16.0 bits/w)
    size_t bytes_nvfp4_w  = (num_w / 8) * sizeof(uint32_t);  //  8.39 MB ( 4.0 bits/w)
    size_t bytes_scales   = (num_w / 64) * sizeof(uint32_t); //  1.05 MB ( 0.5 bits/w)
    size_t bytes_tq2_bulk = (num_w / 16) * sizeof(uint32_t); //  4.19 MB ( 2.0 bits/w)
    size_t bytes_leg2     = (num_w / 16) * sizeof(uint16_t); //  2.10 MB ( 1.0 bits/w)

    void *h_pin_fp16, *h_pin_nvfp4, *h_pin_tq2, *h_pin_leg2, *h_pin_scales;
    CHECK_CUDA(cudaMallocHost(&h_pin_fp16, bytes_fp16));
    CHECK_CUDA(cudaMallocHost(&h_pin_nvfp4, bytes_nvfp4_w + bytes_scales));
    CHECK_CUDA(cudaMallocHost(&h_pin_tq2, bytes_tq2_bulk));
    CHECK_CUDA(cudaMallocHost(&h_pin_leg2, bytes_leg2));
    CHECK_CUDA(cudaMallocHost(&h_pin_scales, bytes_scales));

    uint32_t *d_SA_pack, *d_SA_scale, *d_SB_nvfp4, *d_SB_scale, *d_SB_tq2;
    uint16_t *d_SB_leg2;
    __half *d_SB_fp16;
    float *d_SC;
    CHECK_CUDA(cudaMalloc(&d_SA_pack, SM * (SK / 8) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_SA_scale, SM * (SK / 64) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_SB_fp16, bytes_fp16));
    CHECK_CUDA(cudaMalloc(&d_SB_nvfp4, bytes_nvfp4_w + bytes_scales));
    CHECK_CUDA(cudaMalloc(&d_SB_tq2, bytes_tq2_bulk));
    CHECK_CUDA(cudaMalloc(&d_SB_leg2, bytes_leg2));
    CHECK_CUDA(cudaMalloc(&d_SB_scale, bytes_scales));
    CHECK_CUDA(cudaMalloc(&d_SC, SM * SN * sizeof(float)));

    int s_warps = (SM / 16) * (SN / 8);
    int s_blocks = (s_warps * 32 + 255) / 256;
    const int s_runs = 50;

    for (int r = 0; r < 5; ++r) {
        CHECK_CUDA(cudaMemcpyAsync(d_SB_fp16, h_pin_fp16, bytes_fp16, cudaMemcpyHostToDevice));
        quantize_nvfp4_std<<<(SN*(SK/64)+255)/256, 256>>>(d_SB_nvfp4, d_SB_scale, d_SB_fp16, SN, SK);
        gemm_nvfp4_std_kernel<<<s_blocks, 256>>>(d_SC, d_SA_pack, d_SA_scale, d_SB_nvfp4, d_SB_scale, SM, SN, SK);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < s_runs; ++r) {
        CHECK_CUDA(cudaMemcpyAsync(d_SB_fp16, h_pin_fp16, bytes_fp16, cudaMemcpyHostToDevice));
        quantize_nvfp4_std<<<(SN*(SK/64)+256-1)/256, 256>>>(d_SB_nvfp4, d_SB_scale, d_SB_fp16, SN, SK);
        gemm_nvfp4_std_kernel<<<s_blocks, 256>>>(d_SC, d_SA_pack, d_SA_scale, d_SB_nvfp4, d_SB_scale, SM, SN, SK);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_fp16 = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp16, start, stop));
    ms_fp16 /= s_runs;

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < s_runs; ++r) {
        CHECK_CUDA(cudaMemcpyAsync(d_SB_nvfp4, h_pin_nvfp4, bytes_nvfp4_w + bytes_scales, cudaMemcpyHostToDevice));
        gemm_nvfp4_std_kernel<<<s_blocks, 256>>>(d_SC, d_SA_pack, d_SA_scale, d_SB_nvfp4, d_SB_scale, SM, SN, SK);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_nvfp4 = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_nvfp4, start, stop));
    ms_nvfp4 /= s_runs;

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < s_runs; ++r) {
        CHECK_CUDA(cudaMemcpyAsync(d_SB_scale, h_pin_scales, bytes_scales, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyAsync(d_SB_leg2, h_pin_leg2, bytes_leg2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyAsync(d_SB_tq2, h_pin_tq2, bytes_tq2_bulk, cudaMemcpyHostToDevice));
        gemm_fused_tq_ledger_mxf4_kernel<<<s_blocks, 256>>>(
            d_SC, d_SA_pack, d_SA_scale, d_SB_tq2, d_SB_scale, d_SB_leg2, nullptr, SM, SN, SK, 2
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_tq_full = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tq_full, start, stop));
    ms_tq_full /= s_runs;

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < s_runs; ++r) {
        CHECK_CUDA(cudaMemcpyAsync(d_SB_tq2, h_pin_tq2, bytes_tq2_bulk, cudaMemcpyHostToDevice));
        gemm_fused_tq_ledger_mxf4_kernel<<<s_blocks, 256>>>(
            d_SC, d_SA_pack, d_SA_scale, d_SB_tq2, d_SB_scale, d_SB_leg2, nullptr, SM, SN, SK, 2
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_tq_cached_ledger = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tq_cached_ledger, start, stop));
    ms_tq_cached_ledger /= s_runs;

    printf("  [A] Stream FP16 Weights (33.55 MB) + Quant + GEMM            : %6.3f ms (%5.2fx speedup)\n",
           ms_fp16, ms_fp16 / ms_fp16);
    printf("  [B] Stream Standard NVFP4 (9.44 MB) + GEMM                   : %6.3f ms (%5.2fx speedup vs FP16)\n",
           ms_nvfp4, ms_fp16 / ms_nvfp4);
    printf("  [C] Stream Error Ledger (3.15 MB) -> TQ-2bit (4.19 MB) + GEMM: %6.3f ms (%5.2fx speedup vs FP16 | %4.2fx vs NVFP4)\n",
           ms_tq_full, ms_fp16 / ms_tq_full, ms_nvfp4 / ms_tq_full);
    printf("  [D] VRAM-Resident Ledger + Stream TQ-2bit Only (4.19 MB)     : %6.3f ms (%5.2fx speedup vs FP16 | %4.2fx vs NVFP4)\n",
           ms_tq_cached_ledger, ms_fp16 / ms_tq_cached_ledger, ms_nvfp4 / ms_tq_cached_ledger);

    return 0;
}
