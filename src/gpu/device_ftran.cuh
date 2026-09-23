/**
 * @file device_ftran.cuh
 * @brief Device-native FTRAN, BTRAN, and basis update kernels.
 */

#pragma once

#include "sankhya/types.hpp"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"

namespace sankhya {
namespace gpu {

/**
 * @brief Device-native FTRAN operation.
 * Mathematically computes: vec = B^{-1} vec
 * 
 * Order of operations:
 * 1. Base solve: vec_out = (P^T L U Q^T)^{-1} vec
 * 2. FT updates: vec_out = E_k^{-1} ... E_1^{-1} vec_out
 * 
 * @param lu Device-resident SparseLU factorization of the root basis.
 * @param ws The current working basis state containing inherited FT updates and auxiliary buffer.
 * @param vec The right-hand-side vector, overwritten with the result.
 */
__device__ void device_ftran(const DeviceSparseLU& lu, const WorkingBasisState& ws, Float* vec);

/**
 * @brief Device-native BTRAN operation.
 * Mathematically computes: vec = B^{-T} vec
 * 
 * Order of operations:
 * 1. Transposed FT updates: vec_out = E_1^{-T} ... E_k^{-T} vec
 * 2. Base solve: vec_out = (Q U^T L^T P)^{-1} vec_out
 */
__device__ void device_btran(const DeviceSparseLU& lu, const WorkingBasisState& ws, Float* vec);

/**
 * @brief Records a Forrest-Tomlin update directly into the WorkingBasisState.
 * 
 * [ENGINEERING DECISION]
 * Rather than creating a separate GPU struct for updates, we append the eta
 * column directly to the existing `WorkingBasisState`'s eta-file. This exactly 
 * preserves the semantics expected by `hbf.cuh`'s inheritance mechanism.
 * 
 * @param ws The working basis state.
 * @param leaving_row The pivot row.
 * @param entering_col The original entering variable index.
 * @param Aq The transformed entering column (d_q).
 * @param pivot_tolerance Tolerance below which an update is rejected.
 * @return true if the update was successfully recorded, false if rejected.
 */
__device__ bool device_update_basis(
    WorkingBasisState& ws,
    Index leaving_row,
    Index entering_col,
    const Float* Aq,
    Float pivot_tolerance = 1e-9
);

} // namespace gpu
} // namespace sankhya
