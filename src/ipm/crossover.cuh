/**
 * @file crossover.cuh
 * @brief Phase 16.1: In-VRAM Megiddo Crossover
 */

#pragma once

#include "sankhya/types.hpp"
#include "simplex/basis.hpp"
#include "gpu/device_model.cuh"
#include <vector>

namespace sankhya {
namespace ipm {

/**
 * @brief Performs Megiddo Basis Purification / crossover entirely in VRAM.
 * 
 * Takes an interior point solution (x_ipm) and pivots strictly interior variables
 * to their bounds while maintaining feasibility, yielding an extreme-point basic 
 * feasible solution (vertex) suitable for downstream branch-and-cut.
 * 
 * [ENGINEERING DECISION]
 * The interface strictly matches the required specification without an explicit
 * VRAMArena argument. Therefore, temporary crossover state is allocated via 
 * standard device mechanisms (cudaMalloc) and freed before returning. The resulting 
 * Basis object is returned to the host, and it is the caller's responsibility to
 * deposit it into the HBFManager's mutable arena (via set_root_basis).
 */
simplex::Basis generate_crossover_basis(
    const gpu::DeviceModel& model,
    const std::vector<Float>& x_ipm
);

} // namespace ipm
} // namespace sankhya

