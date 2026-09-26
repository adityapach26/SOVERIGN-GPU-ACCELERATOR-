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

/**
 * @brief Strictly audits an integer candidate solution using FP64 CUDA-core arithmetic.
 *
 * After the low-precision (FP16 WMMA) Feasibility Pump heuristic proposes a candidate,
 * this function re-evaluates every equality constraint A x = b in strict double precision
 * on CUDA cores to guard against false incumbents caused by FP16/TF32 precision loss.
 *
 * Accepts the candidate only if:
 *   max_i |( A x )_i - b_i| <= kDefaultFeasibilityTol   (1e-6)
 *
 * @param model     Device model containing CSC matrix A and rhs b (in device memory).
 * @param candidate The proposed integer candidate x (host vector, size == model.cols).
 * @return true  iff max |Ax - b| <= kDefaultFeasibilityTol (strictly FP64).
 * @return false if any constraint is violated beyond tolerance.
 */
bool verify_incumbent_fp64(
    const DeviceModel& model,
    const std::vector<Float>& candidate
);

} // namespace gpu
} // namespace sankhya
