#pragma once

#include "sankhya/types.hpp"
#include "gpu/device_model.cuh"
#include "gpu/vram_arena.cuh"
#include <vector>

namespace sankhya {
namespace gpu {

/**
 * @brief Computes the continuous projection direction for the Feasibility Pump.
 * 
 * Computes d = - A^T (A x_tilde - b) using mixed-precision Tensor Core operations (WMMA).
 * This provides a steepest-descent direction toward the equality constraint manifold.
 * 
 * @param model The device model containing the constraint matrix A and rhs b.
 * @param x_tilde The rounded/integer target point.
 * @param arena The VRAMArena for temporary device allocations.
 * @return std::vector<Float> The projection direction d (on host).
 */
std::vector<Float> compute_fp_projection_direction(
    const DeviceModel& model,
    const std::vector<Float>& x_tilde,
    VRAMArena& arena
);

} // namespace gpu
} // namespace sankhya
