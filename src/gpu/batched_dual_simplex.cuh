/**
 * @file batched_dual_simplex.cuh
 * @brief Phase 13.1 Batched Sibling-Node Dual Simplex Launch
 */
#pragma once

#include <cstdint>
#include "sankhya/types.hpp"
#include "gpu/device_model.cuh"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"
#include "gpu/device_dual_simplex.cuh"
#include <vector>

namespace sankhya {
namespace gpu {

/**
 * @brief Dispatches K sibling dual-simplex solves in ONE kernel launch.
 * 
 * @param model Immutable problem state.
 * @param lu Device-resident root basis factorization.
 * @param d_ws_array Device pointer to an array of K WorkingBasisState structs.
 * @param obj_sign +1.0 for Minimize, -1.0 for Maximize.
 * @param max_iterations Maximum iterations per node.
 * @param d_status_array Device pointer to array of length K for output statuses.
 * @param d_iter_count_array Device pointer to array of length K for output iteration counts.
 * @param num_nodes The number of sibling nodes to launch (K).
 */
void launch_batched_dual_simplex(
    const DeviceModel& model,
    const DeviceSparseLU& lu,
    WorkingBasisState* d_ws_array,
    Float obj_sign,
    Index max_iterations,
    DeviceSimplexStatus* d_status_array,
    Index* d_iter_count_array,
    uint32_t num_nodes
);

} // namespace gpu
} // namespace sankhya

