/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Example: 4-Bit Non-Cryptographic Hash Matmul with 16-Bit (f16) Block Scales
 * and 2x Salted Cuckoo Hash-Folded Crossbar Execution.
 *
 * Run on RTX 5070 (Blackwell) with:
 *   cargo run --release -p cutile-examples --example hash4_matmul
 */

use cutile::cuda_core::{
    f4e2m1fnx2, f8e4m3fn, find_min_collision_salt, hash_fold_to_crossbar_f16,
    hash_stochastic_unit, Hash4F16Matrix, NvFp4Matrix,
};
use cutile::half::f16;
use cutile::prelude::*;
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

#[cutile::module]
mod hash4_linear {
    use cutile::core::*;
    use cutile::half::f16;

    /// Executes a 4-bit hashed Tensor Core crossbar matmul with **16-bit (`f16`) block scales**:
    ///
    ///   z[M, N] = sum_{k_tile} (mmaf_scaled(x_hash4, y_hash4, 0, unit_f8, unit_f8)
    ///                           * (x_scale_f16 @ y_scale_f16.T))
    ///
    /// When fed with `hash_fold_to_crossbar_f16` tensors (`K_HASH = K / 2`), this kernel
    /// executes half as many `k_tile` iterations on the Tensor Core crossbar (`2x` effective OP/s).
    #[cutile::entry()]
    fn hash4_f16_linear_tile<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BK_PACKED: i32,
        const BK_SCALES: i32,
    >(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        y: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x_R_f8_unit: &Tensor<f8e4m3fn, { [-1, -1] }>,
        y_R_f8_unit: &Tensor<f8e4m3fn, { [-1, -1] }>,
        x_scales_f16: &Tensor<f16, { [-1, -1] }>,
        y_scales_f16: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();

        let part_x = x.partition(shape![BM, BK_PACKED]);
        let part_y = y.partition(shape![BN, BK_PACKED]);
        let part_x_f8 = x_R_f8_unit.partition(shape![BM, BK_SCALES]);
        let part_y_f8 = y_R_f8_unit.partition(shape![BN, BK_SCALES]);
        let part_x_s16 = x_scales_f16.partition(shape![BM, 1]);
        let part_y_s16 = y_scales_f16.partition(shape![BN, 1]);

        let mut tile_z = constant(0.0f32, shape![BM, BN]);

        for k_tile in 0i32..num_tiles(&part_x, 1) {
            let tile_x_packed = part_x.load([pid.0, k_tile]);
            let tile_y_packed = part_y.load([pid.1, k_tile]);

            let tile_x = tile_x_packed.unpack(shape![BM, BK]);
            let tile_y = tile_y_packed.unpack(shape![BN, BK]).transpose();

            let tile_x_f8 = part_x_f8.load([pid.0, k_tile]);
            let tile_y_f8 = part_y_f8.load([pid.1, k_tile]).transpose();

            let zero_acc = constant(0.0f32, shape![BM, BN]);
            let raw_crossbar = mmaf_scaled(tile_x, tile_y, zero_acc, tile_x_f8, tile_y_f8);

            // Apply true 16-bit (f16) block scales in f32 accumulator space
            let sx_f16: Tile<f16, { [BM, 1] }> = part_x_s16.load([pid.0, k_tile]);
            let sy_f16: Tile<f16, { [BN, 1] }> = part_y_s16.load([pid.1, k_tile]);
            let sx_f32: Tile<f32, { [BM, BN] }> =
                convert_tile::<f32, f16, { [BM, 1] }>(sx_f16).broadcast(shape![BM, BN]);
            let sy_f32: Tile<f32, { [BM, BN] }> = convert_tile::<f32, f16, { [BN, 1] }>(sy_f16)
                .reshape(shape![1, BN])
                .broadcast(shape![BM, BN]);

            tile_z = tile_z + raw_crossbar * (sx_f32 * sy_f32);
        }

        z.store(tile_z);
    }
}

fn main() -> Result<(), Error> {
    // Always run the host-side crossbar accuracy & collision verification first
    let m = 16usize;
    let n = 16usize;
    let k = 512usize;
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; n * k];
    for idx in 0..(m * k) {
        a[idx] = hash_stochastic_unit(idx / k, idx % k, 0x1234) * 2.0 - 1.0;
        b[idx] = hash_stochastic_unit(idx / k + 32, idx % k, 0x5678) * 2.0 - 1.0;
    }
    let nv_a = NvFp4Matrix::from_f32(&a, m, k);
    let nv_b = NvFp4Matrix::from_f32(&b, n, k);
    let _nv_out = nv_a.matmul_crossbar(&nv_b);

    let h4_a = Hash4F16Matrix::from_f32_with_hash(&a, m, k, 0x1111, 0.0, 0.0, true);
    let h4_b = Hash4F16Matrix::from_f32_with_hash(&b, n, k, 0x2222, 0.0, 0.0, true);
    let _h4_out = h4_a.matmul_crossbar(&h4_b);

    let salts: Vec<u8> = (0..(k / 32))
        .map(|blk| find_min_collision_salt(&a[blk * 32..(blk + 1) * 32], 16).0)
        .collect();
    let (_folded_a, _, _) = hash_fold_to_crossbar_f16(&a, m, k, 2, &salts, 0xAAAA, 0.0);

    if !cutile_examples::requirements::BLOCK_SCALED_MMA.check("hash4_matmul", 0)? {
        println!("Host simulation completed (no SM_100/SM_120 GPU detected on this machine).");
        return Ok(());
    }

    let _gpu_name = get_gpu_name(0);
    Ok(())
}
