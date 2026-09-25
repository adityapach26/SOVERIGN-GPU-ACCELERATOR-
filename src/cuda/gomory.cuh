#pragma once

#include "sankhya/types.hpp"
#include "gpu/device_model.cuh"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"
#include "gpu/vram_arena.cuh"
#include "cuts/cut_pool.hpp"

namespace sankhya {
namespace gpu {

/**
 * @brief Generates a warp-level Gomory Mixed-Integer (GMI) cut from the resident GPU basis.
 * 
 * Computes the tableau row using the resident LU factors via BTRAN entirely on the GPU,
 * preventing expensive host-device roundtrips of the basis inverse. The fractional
 * coefficients are computed natively in a CUDA kernel and handed off to the CPU's
 * global CutPool manager as an inequality (a^T x <= rhs).
 * 
 * @param model The device-resident problem model
 * @param lu The device-resident root basis factorization
 * @param ws The working basis state (used for BTRAN and scratchpad)
 * @param basic_row The simplex row index of the fractional basic variable
 * @param basic_value The fractional value (b_bar_i)
 * @param arena The VRAMArena for transient allocations
 * @return cuts::Cut The mathematically equivalent cut represented for the global pool
 */
cuts::Cut generate_gomory_cut(
    const DeviceModel& model,
    const DeviceSparseLU& lu,
    WorkingBasisState& ws,
    Index basic_row,
    Float basic_value,
    VRAMArena& arena
);

} // namespace gpu
} // namespace sankhya

