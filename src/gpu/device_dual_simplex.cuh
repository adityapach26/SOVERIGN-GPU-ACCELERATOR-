/**
 * @file device_dual_simplex.cuh
 * @brief Pre-Phase 13.1 Single-Node GPU Dual Simplex Foundation
 */
#pragma once

#include "sankhya/types.hpp"
#include "gpu/device_model.cuh"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"

namespace sankhya {
namespace gpu {

/**
 * @brief Device-side dual simplex statuses
 */
enum class DeviceSimplexStatus {
    Optimal = 0,
    Infeasible = 1,
    IterationLimit = 2,
    NumericalFailure = 3
};

/**
 * @brief Single-node dual simplex iteration loop kernel.
 * 
 * Performs iterative dual-simplex algorithm directly on the GPU using
 * device_ftran, device_btran, and device_update_basis.
 * 
 * @param model Immutable problem state.
 * @param lu Device-resident root basis factorization.
 * @param ws HBF working basis state for this node.
 * @param obj_sign +1.0 for Minimize, -1.0 for Maximize.
 * @param max_iterations Maximum iterations before returning IterationLimit.
 * @param status_out Output buffer for the final status.
 */
__global__ void dual_simplex_kernel(
    DeviceModel model,
    DeviceSparseLU lu,
    WorkingBasisState ws,
    Float obj_sign,
    Index max_iterations,
    DeviceSimplexStatus* status_out
);

} // namespace gpu
} // namespace sankhya
