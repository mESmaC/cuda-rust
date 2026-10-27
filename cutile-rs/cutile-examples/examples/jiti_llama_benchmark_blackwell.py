#!/usr/bin/env python3
"""
JITI (Just-In-Time Inference) Blackwell SM_120a End-to-End Sparse-MoE LLM Benchmark Suite
=========================================================================================
Benchmarks Pre-CUDA-Kernel PyTorch BF16 Sparse-MoE vs. Post-CUDA-Kernel JITI Blackwell
(`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4`) on `mesmac/MCL4-COE-v9-Llama-3.2-1B-NCM-Mitosis-R2`
(2.431B parameters, 1.611B routed expert parameters across 16 layers x 16 experts, top_k=8).

Modes Evaluated:
  - Baseline: Standard PyTorch BF16 SparseMoE (all 256 experts resident in GPU VRAM)
  - JITI Mode 0: Cold Ring-Buffer Streaming (0 MB permanent expert VRAM; both 2.0-bit
    TurboQuant-MSB2 and 2.0-bit Exact-LSB2 Error Ledger streamed from Pinned Host RAM
    into the 6.75 MiB 2-Slot L2 Cache Ring Buffer)
  - JITI Mode 1: Warm Error-Ledger + 2.0-bit TurboQuant-MSB2 PCIe Streaming (480 MB warm
    Error Ledger + Scales in VRAM; 1.50 MiB/expert TurboQuant-MSB2 streamed over PCIe)
  - JITI Mode 2: All-Resident JITI NVFP4 (864 MB compressed JITI packets in VRAM)
"""

from __future__ import annotations

import gc
import os
import pathlib
import time
import types
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


def load_jiti_blackwell_extension(cu_path: str | pathlib.Path | None = None):
    if cu_path is None:
        cu_path = pathlib.Path(__file__).with_name("jiti_llama_moe_ext_blackwell.cu")
    cu_src = pathlib.Path(cu_path).read_text(encoding="utf-8")
    # Strip PYBIND11_MODULE from cuda_sources so load_inline can bind cleanly
    split_marker = "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)"
    cuda_body = cu_src.split(split_marker)[0]
    cpp_src = """
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> compile_swiglu_expert_to_jiti_packets(torch::Tensor up_bf16, torch::Tensor dn_bf16);
void init_jiti_ring_engine();
torch::Tensor jiti_moe_layer_forward(
    torch::Tensor gathered_x,
    std::vector<int64_t> counts_per_expert,
    std::vector<torch::Tensor> msb2_packets,
    std::vector<torch::Tensor> ledger_packets,
    std::vector<float> up_gscales,
    std::vector<float> dn_gscales,
    int streaming_mode
);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compile_swiglu_expert_to_jiti_packets", &compile_swiglu_expert_to_jiti_packets);
    m.def("init_jiti_ring_engine", &init_jiti_ring_engine);
    m.def("jiti_moe_layer_forward", &jiti_moe_layer_forward);
}
"""
    ext = load_inline(
        name="jiti_llama_ext_v2",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_body],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_120a,code=sm_120a", "--use_fast_math"],
        verbose=False,
    )
    ext.init_jiti_ring_engine()
    return ext


def set_jiti_mode(model, mode: int) -> None:
    """
    Configure JITI expert residency mode across all 16 Sparse-MoE layers:
      - mode 0: Cold Expert Ring-Buffer Streaming (0 MB permanent expert VRAM)
      - mode 1: Warm Error-Ledger Streaming (480 MB Ledger+Scales in VRAM; 2.0-bit MSB2 in Pinned Host RAM)
      - mode 2: All-Resident JITI NVFP4 (864 MB MSB2 + Ledger+Scales in VRAM)
    """
    for blk in list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers):
        moe = blk.moe
        moe._jiti_streaming_mode = mode
        if mode == 0:
            moe._jiti_msb2_active = moe._jiti_msb2_cpu
            moe._jiti_ledger_active = moe._jiti_ledger_cpu
        elif mode == 1:
            moe._jiti_msb2_active = moe._jiti_msb2_cpu
            moe._jiti_ledger_active = [t.to("cuda", non_blocking=True) for t in moe._jiti_ledger_cpu]
        elif mode == 2:
            moe._jiti_msb2_active = [t.to("cuda", non_blocking=True) for t in moe._jiti_msb2_cpu]
            moe._jiti_ledger_active = [t.to("cuda", non_blocking=True) for t in moe._jiti_ledger_cpu]
        else:
            raise ValueError(f"Unsupported JITI mode: {mode}")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def patch_model_with_jiti(model, jiti_ext, initial_mode: int = 0) -> dict[str, float]:
    """
    Compiles all 256 routed SwiGLU experts (1.611B parameters) into JITI packets
    (2.0-bit TurboQuant-MSB2 + 2.0-bit Exact-LSB2 Error Ledger + UE4M3 scales),
    unloads all 256 BF16 experts from GPU VRAM, and patches `SparseMoE.forward`.
    """
    moe_layers = list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers)
    t0 = time.perf_counter()
    total_msb2_bytes = 0
    total_ledger_bytes = 0

    for blk in moe_layers:
        moe = blk.moe
        moe.mitotic_ortho_coef = 0.0
        moe._grouped_weight_cache = None

        msb2_cpu_pinned = []
        ledger_cpu_pinned = []
        up_gs_list = []
        dn_gs_list = []

        for exp in moe.experts:
            msb2_gpu, ledger_gpu, gs_cpu = jiti_ext.compile_swiglu_expert_to_jiti_packets(
                exp.up_proj.weight.detach(),
                exp.down_proj.weight.detach(),
            )
            msb2_pin = msb2_gpu.cpu().pin_memory()
            ledg_pin = ledger_gpu.cpu().pin_memory()
            msb2_cpu_pinned.append(msb2_pin)
            ledger_cpu_pinned.append(ledg_pin)
            up_gs_list.append(float(gs_cpu[0].item()))
            dn_gs_list.append(float(gs_cpu[1].item()))
            total_msb2_bytes += msb2_pin.numel() * 4
            total_ledger_bytes += ledg_pin.numel() * 4

            del msb2_gpu, ledger_gpu
            exp.up_proj.weight = torch.nn.Parameter(
                torch.empty(0, device="cuda", dtype=torch.bfloat16), requires_grad=False
            )
            exp.down_proj.weight = torch.nn.Parameter(
                torch.empty(0, device="cuda", dtype=torch.bfloat16), requires_grad=False
            )

        moe._jiti_msb2_cpu = msb2_cpu_pinned
        moe._jiti_ledger_cpu = ledger_cpu_pinned
        moe._jiti_up_gs = up_gs_list
        moe._jiti_dn_gs = dn_gs_list
        moe._jiti_streaming_mode = 0
        moe._jiti_msb2_active = msb2_cpu_pinned
        moe._jiti_ledger_active = ledger_cpu_pinned

    def jiti_sparse_moe_forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor | None = None,
        *,
        prior_expert_mask: torch.Tensor | None = None,
        dense_expert_indices=None,
        dense_subset_k=None,
    ):
        flat = x.reshape(-1, x.shape[-1])
        flat_valid = (
            torch.ones(flat.shape[0], dtype=torch.bool, device=x.device)
            if valid is None
            else valid.reshape(-1)
        )
        active_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        if active_indices.numel() == 0:
            zero = x.new_zeros((), dtype=torch.float32)
            loads = x.new_zeros(self.num_experts, dtype=torch.float32)
            self.last_selected_mask = x.new_zeros((*x.shape[:-1], self.num_experts), dtype=torch.float32)
            return torch.zeros_like(x), zero, zero, loads

        active_x = flat.index_select(0, active_indices)
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = F.linear(active_x.float(), self.router.weight.float())
            pair_tensors = self._get_mitotic_pair_tensors(x.device)
            probabilities = F.softmax(router_logits, dim=-1)
            routed_probs = probabilities
            if pair_tensors is not None:
                p_t, c_t = pair_tensors
                routed_probs = probabilities.clone()
                p_wins = router_logits.index_select(1, p_t) >= router_logits.index_select(1, c_t)
                routed_probs[:, c_t] = torch.where(
                    p_wins, torch.zeros_like(p_wins, dtype=routed_probs.dtype), probabilities.index_select(1, c_t)
                )
                routed_probs[:, p_t] = torch.where(
                    p_wins, probabilities.index_select(1, p_t), torch.zeros_like(p_wins, dtype=routed_probs.dtype)
                )
            if prior_expert_mask is not None and self.recurrent_coverage_penalty > 0.0:
                flat_prior = (
                    prior_expert_mask.reshape(-1, self.num_experts)
                    .index_select(0, active_indices)
                    .to(routed_probs.dtype)
                )
                selection_probs = routed_probs * (1.0 - self.recurrent_coverage_penalty * flat_prior).clamp_min(0.05)
                _, selected = selection_probs.topk(self.top_k, dim=-1)
                weights = routed_probs.gather(-1, selected)
            else:
                weights, selected = routed_probs.topk(self.top_k, dim=-1)
            if self.norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            if self.routed_scaling_factor != 1.0:
                weights = weights * self.routed_scaling_factor

            one_hot_sel = F.one_hot(selected, num_classes=self.num_experts)
            full_sel_mask = x.new_zeros((flat.shape[0], self.num_experts), dtype=torch.float32)
            full_sel_mask.index_copy_(0, active_indices, one_hot_sel.sum(dim=1).detach().to(torch.float32))
            self.last_selected_mask = full_sel_mask.view(*x.shape[:-1], self.num_experts)
            counts_int = one_hot_sel.sum(dim=(0, 1))
            counts = counts_int.float() / max(1, selected.numel())
            aux_loss = x.new_zeros((), dtype=torch.float32)
            router_z_loss = x.new_zeros((), dtype=torch.float32)

        k_eff = selected.shape[1]
        flat_selected = selected.reshape(-1)
        perm = flat_selected.argsort(stable=True)
        sorted_tokens = torch.div(perm, k_eff, rounding_mode="floor")
        sorted_weights = weights.reshape(-1).index_select(0, perm).unsqueeze(-1)
        gathered_x = active_x.index_select(0, sorted_tokens).to(torch.bfloat16).contiguous()

        all_expert_out = jiti_ext.jiti_moe_layer_forward(
            gathered_x,
            counts_int.detach().cpu().tolist(),
            self._jiti_msb2_active,
            self._jiti_ledger_active,
            self._jiti_up_gs,
            self._jiti_dn_gs,
            self._jiti_streaming_mode,
        )

        weighted = (all_expert_out * sorted_weights.to(all_expert_out.dtype)).to(active_x.dtype)
        unpermed = torch.empty_like(weighted)
        unpermed.index_copy_(0, perm, weighted)
        active_out = unpermed.view(active_x.shape[0], k_eff, active_x.shape[-1]).sum(dim=1).to(active_x.dtype)

        output = torch.zeros_like(flat)
        output.index_copy_(0, active_indices, active_out)
        return output.view_as(x), aux_loss, router_z_loss, counts.detach()

    for blk in moe_layers:
        blk.moe.forward = types.MethodType(jiti_sparse_moe_forward, blk.moe)

    set_jiti_mode(model, initial_mode)
    torch.cuda.synchronize()
    return {
        "compile_s": time.perf_counter() - t0,
        "pinned_msb2_mb": total_msb2_bytes / (1024 ** 2),
        "pinned_ledger_mb": total_ledger_bytes / (1024 ** 2),
        "vram_after_unload_mb": torch.cuda.memory_allocated() / (1024 ** 2),
    }
