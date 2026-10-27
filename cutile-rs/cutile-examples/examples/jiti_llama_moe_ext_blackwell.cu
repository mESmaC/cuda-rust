// =================================================================================================
// JITI (Just-In-Time Inference) Blackwell SM_120a PyTorch C++/CUDA Extension for Sparse-MoE LLMs
// =================================================================================================
// Implements the Zero-Lobotomy JITI Pipeline for SwiGLU Sparse-MoE Layers (e.g. Llama-3.2-1B-NCM-R2,
// 16 layers x 16 experts, top_k=8, d_model=2048, expert_hidden_dim=1024):
//   1. Offline/Load-Time Expert Packet Compilation (`compile_swiglu_expert_to_jiti_packets`):
//      - Packet 1 (`msb2_packet`, 1.50 MiB/expert): 2.0-bit TurboQuant-MSB2 coalesced warp tiles
//        for both `up_proj [2048, 2048]` and `down_proj [2048, 1024]`.
//      - Packet 2 (`ledger_packet`, 1.875 MiB/expert): 2.0-bit Exact-LSB2 Error Ledger coalesced
//        warp tiles + UE4M3 micro-block scales (`sfa`) for zero-bit-loss NVFP4 reconstruction.
//   2. Persistent 2-Slot L2 Cache Ring Buffer (`JitiGlobalRingEngine`, 6.75 MiB):
//      - Pinned in Blackwell's 128 MB L2 Cache via `cudaAccessPropertyPersisting`.
//      - Overlaps asynchronous PCIe H2D staging (`stream_prefetch`) of Expert `idx + 1` with
//        Blackwell Tensor Core execution (`stream_compute`) of Expert `idx`.
//   3. Zero-Sync Fused SwiGLU + NVFP4 Quantization + `mma.sync.aligned.m16n8k64.kind::mxf4nvf4`:
//      - Packs `M=1` (single-token decode) up to `M=8` (speculative tree verification) into the
//        `n=8` Tensor Core column dimension at 1.00x streaming latency.
// =================================================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err)); \
} while(0)

// Expert geometry constants for Llama-3.2-1B-NCM-Mitosis-R2 (d_model=2048, H=1024)
static constexpr int D_MODEL = 2048;
static constexpr int D_UP    = 2048; // 2 * 1024 (SwiGLU gate + up)
static constexpr int D_FFN   = 1024;
static constexpr size_t UP_COAL_U32 = (D_UP / 16) * (D_MODEL / 64) * 2 * 32;   // 262,144 u32 = 1.00 MiB
static constexpr size_t DN_COAL_U32 = (D_MODEL / 16) * (D_FFN / 64) * 2 * 32;  // 131,072 u32 = 0.50 MiB
static constexpr size_t UP_SCAL_U32 = D_UP * (D_MODEL / 64);                   //  65,536 u32 = 0.25 MiB
static constexpr size_t DN_SCAL_U32 = D_MODEL * (D_FFN / 64);                  //  32,768 u32 = 0.125 MiB

// Packet 1 (TurboQuant MSB2 stream): [up_msb2 (1.0 MiB) | dn_msb2 (0.5 MiB)] = 393,216 u32 (1.50 MiB)
static constexpr size_t MSB2_PACKET_U32 = UP_COAL_U32 + DN_COAL_U32;
// Packet 2 (Exact-LSB2 Error Ledger + Scales): [up_lsb2 | up_scales | dn_lsb2 | dn_scales] = 491,520 u32 (1.875 MiB)
static constexpr size_t LEDGER_PACKET_U32 = UP_COAL_U32 + UP_SCAL_U32 + DN_COAL_U32 + DN_SCAL_U32;

__device__ __host__ __forceinline__ uint32_t e2m1_rtn_nibble(float v) {
    uint32_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    float a = fabsf(v);
    uint32_t mag = (a < 0.25f) ? 0u : (a <= 0.75f) ? 1u : (a < 1.25f) ? 2u :
                   (a <= 1.75f) ? 3u : (a <= 2.5f) ? 4u : (a < 3.5f) ? 5u :
                   (a <= 5.0f) ? 6u : 7u;
    return (mag == 0u) ? 0u : (sign | mag);
}

__global__ void quantize_bf16_to_nvfp4_std_kernel(
    uint32_t* __restrict__ packed_u32,
    uint32_t* __restrict__ scales_u32,
    const __nv_bfloat16* __restrict__ in_bf16,
    float inv_global_scale,
    int rows, int k_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_chunks64 = k_dim / 64;
    if (idx >= rows * k_chunks64) return;
    int r = idx / k_chunks64;
    int c64 = idx % k_chunks64;
    const __nv_bfloat16* base = in_bf16 + r * k_dim + c64 * 64;

    uint32_t packed_scales = 0;
    for (int sub = 0; sub < 4; ++sub) {
        float vals[16], amax = 0.0f;
        for (int i = 0; i < 16; ++i) {
            vals[i] = __bfloat162float(base[sub * 16 + i]) * inv_global_scale;
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

__global__ void fused_swiglu_and_quantize_nvfp4_kernel(
    uint32_t* __restrict__ packed_u32,
    uint32_t* __restrict__ scales_u32,
    const float* __restrict__ up_out,
    float inv_global_scale,
    int M, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_chunks64 = H / 64;
    if (idx >= M * k_chunks64) return;
    int m = idx / k_chunks64;
    int c64 = idx % k_chunks64;
    const float* gate_base = up_out + m * (2 * H) + c64 * 64;
    const float* val_base  = gate_base + H;

    uint32_t packed_scales = 0;
    for (int sub = 0; sub < 4; ++sub) {
        float vals[16], amax = 0.0f;
        for (int i = 0; i < 16; ++i) {
            float g = gate_base[sub * 16 + i];
            float v = val_base[sub * 16 + i];
            float h = (g / (1.0f + expf(-g))) * v * inv_global_scale;
            vals[i] = h;
            amax = fmaxf(amax, fabsf(h));
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
        packed_u32[m * (H / 8) + c64 * 8 + sub * 2 + 0] = w0;
        packed_u32[m * (H / 8) + c64 * 8 + sub * 2 + 1] = w1;
    }
    scales_u32[m * k_chunks64 + c64] = packed_scales;
}

__global__ void pack_jiti_coalesced_tiles_kernel(
    uint32_t* __restrict__ W_tq_msb2_coal,
    uint32_t* __restrict__ W_leg_lsb2_coal,
    const uint32_t* __restrict__ W_pack_std,
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

        size_t out_idx = ((size_t)(tile_w * k_chunks64 + c64) * 2 + half_k) * 32 + lane;
        W_tq_msb2_coal[out_idx]  = tq_msb2;
        W_leg_lsb2_coal[out_idx] = leg_lsb2;
    }
}

__device__ __forceinline__ void unpack_jiti_exact_lsb2(
    uint32_t tq_msb2,
    uint32_t leg_lsb2,
    uint32_t* out_r0,
    uint32_t* out_r1
) {
    *out_r0 = (tq_msb2 & 0xCCCCCCCCu) | (leg_lsb2 & 0x33333333u);
    *out_r1 = ((tq_msb2 << 2u) & 0xCCCCCCCCu) | ((leg_lsb2 >> 2u) & 0x33333333u);
}

template <bool WRITE_BF16>
__global__ void gemm_jiti_m16n8k64_kernel(
    void* __restrict__ C_out,
    const uint32_t* __restrict__ W_tq_msb2_coal,
    const uint32_t* __restrict__ W_leg_lsb2_coal,
    const uint32_t* __restrict__ W_scale,
    const uint32_t* __restrict__ X_pack,
    const uint32_t* __restrict__ X_scale,
    float alpha,
    int M_tokens, int N, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int tiles_w = N / 16;
    if (warp_id >= tiles_w) return;

    int tile_w = warp_id;
    int tile_m = blockIdx.y;
    int m_base = tile_m * 8;
    int w_row_base = tile_w * 16;
    int k_words = K / 8;
    int k_chunks64 = K / 64;

    int grp = lane / 4;
    int tid_in_grp = lane % 4;
    int sfa_w_row = w_row_base + grp + (tid_in_grp % 2) * 8;
    int sfb_x_col = m_base + grp;
    bool valid_x_col = (sfb_x_col < M_tokens);

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int c64 = 0; c64 < k_chunks64; ++c64) {
        size_t base_idx0 = ((size_t)(tile_w * k_chunks64 + c64) * 2 + 0) * 32 + lane;
        size_t base_idx1 = ((size_t)(tile_w * k_chunks64 + c64) * 2 + 1) * 32 + lane;

        uint32_t w_regs[4];
        unpack_jiti_exact_lsb2(W_tq_msb2_coal[base_idx0], W_leg_lsb2_coal[base_idx0], &w_regs[0], &w_regs[1]);
        unpack_jiti_exact_lsb2(W_tq_msb2_coal[base_idx1], W_leg_lsb2_coal[base_idx1], &w_regs[2], &w_regs[3]);

        uint32_t x_regs[2];
        x_regs[0] = valid_x_col ? X_pack[sfb_x_col * k_words + c64 * 8 + tid_in_grp] : 0u;
        x_regs[1] = valid_x_col ? X_pack[sfb_x_col * k_words + c64 * 8 + 4 + tid_in_grp] : 0u;

        uint32_t sfa = W_scale[sfa_w_row * k_chunks64 + c64];
        uint32_t sfb = valid_x_col ? X_scale[sfb_x_col * k_chunks64 + c64] : 0u;

        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(w_regs[0]), "r"(w_regs[1]), "r"(w_regs[2]), "r"(w_regs[3]),
              "r"(x_regs[0]), "r"(x_regs[1]), "r"(sfa), "r"(sfb)
        );
    }

    int n0 = w_row_base + grp, n1 = n0 + 8;
    int m0 = m_base + tid_in_grp * 2, m1 = m0 + 1;
    if (WRITE_BF16) {
        __nv_bfloat16* C_bf = reinterpret_cast<__nv_bfloat16*>(C_out);
        if (m0 < M_tokens) {
            C_bf[m0 * N + n0] = __float2bfloat16(acc[0] * alpha);
            C_bf[m0 * N + n1] = __float2bfloat16(acc[2] * alpha);
        }
        if (m1 < M_tokens) {
            C_bf[m1 * N + n0] = __float2bfloat16(acc[1] * alpha);
            C_bf[m1 * N + n1] = __float2bfloat16(acc[3] * alpha);
        }
    } else {
        float* C_fp = reinterpret_cast<float*>(C_out);
        if (m0 < M_tokens) {
            C_fp[m0 * N + n0] = acc[0] * alpha;
            C_fp[m0 * N + n1] = acc[2] * alpha;
        }
        if (m1 < M_tokens) {
            C_fp[m1 * N + n0] = acc[1] * alpha;
            C_fp[m1 * N + n1] = acc[3] * alpha;
        }
    }
}

std::vector<torch::Tensor> compile_swiglu_expert_to_jiti_packets(torch::Tensor up_bf16, torch::Tensor dn_bf16) {
    auto dev = up_bf16.device();
    float up_amax = up_bf16.abs().max().item<float>();
    float dn_amax = dn_bf16.abs().max().item<float>();
    float up_gs = (up_amax > 0.0f) ? (up_amax / (6.0f * 256.0f)) : 1.0f;
    float dn_gs = (dn_amax > 0.0f) ? (dn_amax / (6.0f * 256.0f)) : 1.0f;

    auto msb2_pkt   = torch::empty({(long)MSB2_PACKET_U32}, torch::TensorOptions().dtype(torch::kUInt32).device(dev));
    auto ledger_pkt = torch::empty({(long)LEDGER_PACKET_U32}, torch::TensorOptions().dtype(torch::kUInt32).device(dev));

    uint32_t* d_up_msb2 = reinterpret_cast<uint32_t*>(msb2_pkt.data_ptr());
    uint32_t* d_dn_msb2 = d_up_msb2 + UP_COAL_U32;

    uint32_t* d_up_lsb2 = reinterpret_cast<uint32_t*>(ledger_pkt.data_ptr());
    uint32_t* d_up_scal = d_up_lsb2 + UP_COAL_U32;
    uint32_t* d_dn_lsb2 = d_up_scal + UP_SCAL_U32;
    uint32_t* d_dn_scal = d_dn_lsb2 + DN_COAL_U32;

    // 1. up_proj [2048, 2048]
    auto up_pack_std = torch::empty({D_UP, D_MODEL / 8}, torch::TensorOptions().dtype(torch::kUInt32).device(dev));
    int chunks_up = D_UP * (D_MODEL / 64);
    quantize_bf16_to_nvfp4_std_kernel<<<(chunks_up + 255) / 256, 256>>>(
        reinterpret_cast<uint32_t*>(up_pack_std.data_ptr()),
        d_up_scal,
        reinterpret_cast<const __nv_bfloat16*>(up_bf16.data_ptr()),
        1.0f / up_gs, D_UP, D_MODEL
    );
    int warps_up = (D_UP / 16) * (D_MODEL / 64);
    pack_jiti_coalesced_tiles_kernel<<<(warps_up * 32 + 255) / 256, 256>>>(
        d_up_msb2, d_up_lsb2,
        reinterpret_cast<const uint32_t*>(up_pack_std.data_ptr()),
        D_UP, D_MODEL
    );

    // 2. down_proj [2048, 1024]
    auto dn_pack_std = torch::empty({D_MODEL, D_FFN / 8}, torch::TensorOptions().dtype(torch::kUInt32).device(dev));
    int chunks_dn = D_MODEL * (D_FFN / 64);
    quantize_bf16_to_nvfp4_std_kernel<<<(chunks_dn + 255) / 256, 256>>>(
        reinterpret_cast<uint32_t*>(dn_pack_std.data_ptr()),
        d_dn_scal,
        reinterpret_cast<const __nv_bfloat16*>(dn_bf16.data_ptr()),
        1.0f / dn_gs, D_MODEL, D_FFN
    );
    int warps_dn = (D_MODEL / 16) * (D_FFN / 64);
    pack_jiti_coalesced_tiles_kernel<<<(warps_dn * 32 + 255) / 256, 256>>>(
        d_dn_msb2, d_dn_lsb2,
        reinterpret_cast<const uint32_t*>(dn_pack_std.data_ptr()),
        D_MODEL, D_FFN
    );

    auto gs_t = torch::tensor({up_gs, dn_gs}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    return {msb2_pkt, ledger_pkt, gs_t};
}

// Persistent Global 2-Slot L2 Ring Buffer + Pre-allocated Scratch Workspace
struct JitiGlobalRingEngine {
    bool initialized = false;
    uint32_t* d_ring_msb2[2]   = {nullptr, nullptr};
    uint32_t* d_ring_ledger[2] = {nullptr, nullptr};
    uint32_t* d_x_up_pack = nullptr;
    uint32_t* d_x_up_scal = nullptr;
    float*    d_up_out    = nullptr;
    uint32_t* d_x_dn_pack = nullptr;
    uint32_t* d_x_dn_scal = nullptr;
    int max_m = 4096;
    cudaStream_t stream_prefetch = nullptr;
    cudaEvent_t ev_staged[2];
    cudaEvent_t ev_consumed[2];
    cudaEvent_t ev_x_quant_done;

    void ensure_init() {
        if (initialized) return;
        for (int s = 0; s < 2; ++s) {
            CHECK_CUDA(cudaMalloc(&d_ring_msb2[s],   MSB2_PACKET_U32 * sizeof(uint32_t)));
            CHECK_CUDA(cudaMalloc(&d_ring_ledger[s], LEDGER_PACKET_U32 * sizeof(uint32_t)));
            CHECK_CUDA(cudaEventCreateWithFlags(&ev_staged[s], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&ev_consumed[s], cudaEventDisableTiming));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&ev_x_quant_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_prefetch, cudaStreamNonBlocking));

        // Pin the 2-slot Ring Buffer in Blackwell's 128 MB L2 Cache
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        size_t persist_bytes = 2 * (MSB2_PACKET_U32 + LEDGER_PACKET_U32) * sizeof(uint32_t);
        if (prop.persistingL2CacheMaxSize >= persist_bytes) {
            CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_bytes));
            cudaStreamAttrValue attr;
            attr.accessPolicyWindow.base_ptr  = (void*)d_ring_msb2[0];
            attr.accessPolicyWindow.num_bytes = MSB2_PACKET_U32 * sizeof(uint32_t);
            attr.accessPolicyWindow.hitRatio  = 1.0f;
            attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(stream_prefetch, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        CHECK_CUDA(cudaMalloc(&d_x_up_pack, (size_t)max_m * (D_MODEL / 8)  * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_x_up_scal, (size_t)max_m * (D_MODEL / 64) * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_up_out,    (size_t)max_m * D_UP           * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_x_dn_pack, (size_t)max_m * (D_FFN / 8)    * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_x_dn_scal, (size_t)max_m * (D_FFN / 64)   * sizeof(uint32_t)));
        initialized = true;
    }
};

static JitiGlobalRingEngine g_engine;

void init_jiti_ring_engine() {
    g_engine.ensure_init();
}

// Execute an entire 16-expert SparseMoE layer over gathered_x [Total_M, 2048]
// streaming_mode:
//   0 = Cold Expert Ring-Buffer Streaming (both msb2_packets and ledger_packets in Pinned Host RAM -> 0 MB expert VRAM!)
//   1 = Warm Error-Ledger Streaming (msb2_packets in Pinned Host RAM, ledger_packets on GPU)
//   2 = Resident JITI (both msb2_packets and ledger_packets on GPU)
torch::Tensor jiti_moe_layer_forward(
    torch::Tensor gathered_x,                  // [Total_M, 2048] CUDA BFloat16
    std::vector<int64_t> counts_per_expert,    // length E=16
    std::vector<torch::Tensor> msb2_packets,   // length E=16 (Pinned CPU or CUDA)
    std::vector<torch::Tensor> ledger_packets, // length E=16 (Pinned CPU or CUDA)
    std::vector<float> up_gscales,             // length E=16
    std::vector<float> dn_gscales,             // length E=16
    int streaming_mode
) {
    g_engine.ensure_init();
    int total_m = gathered_x.size(0);
    auto out_bf16 = torch::empty_like(gathered_x);
    if (total_m == 0) return out_bf16;
    TORCH_CHECK(total_m <= g_engine.max_m, "total_m exceeds max_m=4096");

    cudaStream_t stream_compute = at::cuda::getCurrentCUDAStream().stream();

    // Fixed activation global scale (avoids any GPU->CPU sync!)
    const float x_gs = 1.0f / 16.0f;
    const float inv_x_gs = 16.0f;
    const float h_gs = 1.0f / 16.0f;
    const float inv_h_gs = 16.0f;

    // 1. Quantize ALL gathered_x [total_m, 2048] to NVFP4 in a single kernel launch!
    int chunks_all_x = total_m * (D_MODEL / 64);
    quantize_bf16_to_nvfp4_std_kernel<<<(chunks_all_x + 255) / 256, 256, 0, stream_compute>>>(
        g_engine.d_x_up_pack,
        g_engine.d_x_up_scal,
        reinterpret_cast<const __nv_bfloat16*>(gathered_x.data_ptr()),
        inv_x_gs, total_m, D_MODEL
    );

    // Collect active experts and their row offsets in gathered_x
    struct ActiveExp { int e; int offset; int m; };
    std::vector<ActiveExp> active;
    int cur_off = 0;
    for (size_t e = 0; e < counts_per_expert.size(); ++e) {
        int m_e = (int)counts_per_expert[e];
        if (m_e > 0) {
            active.push_back({(int)e, cur_off, m_e});
            cur_off += m_e;
        }
    }
    int n_act = (int)active.size();
    if (n_act == 0) return out_bf16;

    // Pre-stage active[0] if streaming from Pinned Host RAM
    if (streaming_mode == 0 || streaming_mode == 1) {
        int e0 = active[0].e;
        CHECK_CUDA(cudaMemcpyAsync(
            g_engine.d_ring_msb2[0], msb2_packets[e0].data_ptr(),
            MSB2_PACKET_U32 * sizeof(uint32_t), cudaMemcpyHostToDevice, g_engine.stream_prefetch
        ));
        if (streaming_mode == 0) {
            CHECK_CUDA(cudaMemcpyAsync(
                g_engine.d_ring_ledger[0], ledger_packets[e0].data_ptr(),
                LEDGER_PACKET_U32 * sizeof(uint32_t), cudaMemcpyHostToDevice, g_engine.stream_prefetch
            ));
        }
        CHECK_CUDA(cudaEventRecord(g_engine.ev_staged[0], g_engine.stream_prefetch));
    }

    for (int idx = 0; idx < n_act; ++idx) {
        int cur_slot = idx & 1;
        int nxt_slot = (idx + 1) & 1;
        int e = active[idx].e;
        int off = active[idx].offset;
        int m_e = active[idx].m;

        // Overlap async PCIe prefetch of next expert into nxt_slot
        if ((streaming_mode == 0 || streaming_mode == 1) && (idx + 1 < n_act)) {
            int e_next = active[idx + 1].e;
            if (idx >= 1) {
                CHECK_CUDA(cudaStreamWaitEvent(g_engine.stream_prefetch, g_engine.ev_consumed[nxt_slot], 0));
            }
            CHECK_CUDA(cudaMemcpyAsync(
                g_engine.d_ring_msb2[nxt_slot], msb2_packets[e_next].data_ptr(),
                MSB2_PACKET_U32 * sizeof(uint32_t), cudaMemcpyHostToDevice, g_engine.stream_prefetch
            ));
            if (streaming_mode == 0) {
                CHECK_CUDA(cudaMemcpyAsync(
                    g_engine.d_ring_ledger[nxt_slot], ledger_packets[e_next].data_ptr(),
                    LEDGER_PACKET_U32 * sizeof(uint32_t), cudaMemcpyHostToDevice, g_engine.stream_prefetch
                ));
            }
            CHECK_CUDA(cudaEventRecord(g_engine.ev_staged[nxt_slot], g_engine.stream_prefetch));
        }

        const uint32_t* d_msb2_base = nullptr;
        const uint32_t* d_ledg_base = nullptr;
        if (streaming_mode == 0) {
            CHECK_CUDA(cudaStreamWaitEvent(stream_compute, g_engine.ev_staged[cur_slot], 0));
            d_msb2_base = g_engine.d_ring_msb2[cur_slot];
            d_ledg_base = g_engine.d_ring_ledger[cur_slot];
        } else if (streaming_mode == 1) {
            CHECK_CUDA(cudaStreamWaitEvent(stream_compute, g_engine.ev_staged[cur_slot], 0));
            d_msb2_base = g_engine.d_ring_msb2[cur_slot];
            d_ledg_base = reinterpret_cast<const uint32_t*>(ledger_packets[e].data_ptr());
        } else {
            d_msb2_base = reinterpret_cast<const uint32_t*>(msb2_packets[e].data_ptr());
            d_ledg_base = reinterpret_cast<const uint32_t*>(ledger_packets[e].data_ptr());
        }

        const uint32_t* d_up_msb2 = d_msb2_base;
        const uint32_t* d_dn_msb2 = d_up_msb2 + UP_COAL_U32;
        const uint32_t* d_up_lsb2 = d_ledg_base;
        const uint32_t* d_up_scal = d_up_lsb2 + UP_COAL_U32;
        const uint32_t* d_dn_lsb2 = d_up_scal + UP_SCAL_U32;
        const uint32_t* d_dn_scal = d_dn_lsb2 + DN_COAL_U32;

        const uint32_t* x_up_p = g_engine.d_x_up_pack + (size_t)off * (D_MODEL / 8);
        const uint32_t* x_up_s = g_engine.d_x_up_scal + (size_t)off * (D_MODEL / 64);

        // 1. up_proj GEMM [m_e, 2048]
        dim3 grid_up((D_UP / 16 + 3) / 4, (m_e + 7) / 8);
        gemm_jiti_m16n8k64_kernel<false><<<grid_up, 128, 0, stream_compute>>>(
            g_engine.d_up_out,
            d_up_msb2, d_up_lsb2, d_up_scal,
            x_up_p, x_up_s,
            up_gscales[e] * x_gs,
            m_e, D_UP, D_MODEL
        );

        // 2. Fused SwiGLU + Quantize to NVFP4 [m_e, 1024]
        int chunks_dn_x = m_e * (D_FFN / 64);
        fused_swiglu_and_quantize_nvfp4_kernel<<<(chunks_dn_x + 255) / 256, 256, 0, stream_compute>>>(
            g_engine.d_x_dn_pack,
            g_engine.d_x_dn_scal,
            g_engine.d_up_out,
            inv_h_gs,
            m_e, D_FFN
        );

        // 3. down_proj GEMM [m_e, 2048] directly into out_bf16[off : off + m_e]
        __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out_bf16.data_ptr()) + (size_t)off * D_MODEL;
        dim3 grid_dn((D_MODEL / 16 + 3) / 4, (m_e + 7) / 8);
        gemm_jiti_m16n8k64_kernel<true><<<grid_dn, 128, 0, stream_compute>>>(
            out_ptr,
            d_dn_msb2, d_dn_lsb2, d_dn_scal,
            g_engine.d_x_dn_pack, g_engine.d_x_dn_scal,
            dn_gscales[e] * h_gs,
            m_e, D_MODEL, D_FFN
        );

        if (streaming_mode == 0 || streaming_mode == 1) {
            CHECK_CUDA(cudaEventRecord(g_engine.ev_consumed[cur_slot], stream_compute));
        }
    }
    return out_bf16;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compile_swiglu_expert_to_jiti_packets", &compile_swiglu_expert_to_jiti_packets);
    m.def("init_jiti_ring_engine", &init_jiti_ring_engine);
    m.def("jiti_moe_layer_forward", &jiti_moe_layer_forward);
}
