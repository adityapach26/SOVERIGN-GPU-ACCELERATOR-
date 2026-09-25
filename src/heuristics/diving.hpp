#pragma once
#include <vector>
#include "core/problem.hpp"
#include "sankhya/types.hpp"
#include "milp/tree.hpp"

// Forward declarations for GPU structs to respect CPU/CUDA boundaries
namespace sankhya {
namespace gpu {
    class HBFManager;
    struct DeviceModel;
    struct DeviceSparseLU;
    class VRAMArena;
}
namespace heuristics {

/**
 * @brief Applies the Basic Diving Heuristic.
 *        Plunges depth-first by fixing the closest-to-integer fractional variable.
 *        Does NOT create persistent nodes in the NodeQueue.
 */
bool apply_diving_heuristic(
    const core::Model& host_model,
    const gpu::DeviceModel& device_model,
    const gpu::DeviceSparseLU& lu,
    gpu::HBFManager& hbf,
    gpu::VRAMArena& arena,
    uint32_t current_node_id,
    const std::vector<Float>& initial_x,
    milp::GlobalIncumbent& incumbent
);

} // namespace heuristics
} // namespace sankhya

