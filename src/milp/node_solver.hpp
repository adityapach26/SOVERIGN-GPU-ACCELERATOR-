#pragma once

#include "milp/tree.hpp"
#include "core/problem.hpp"
#include <vector>

namespace sankhya {
namespace gpu {
    class HBFManager;
    struct DeviceModel;
    struct DeviceSparseLU;
    class VRAMArena;
    
    // Explicitly redefine to avoid including CUDA headers (__global__) in CPU translation units
    enum class DeviceSimplexStatus {
        Optimal = 0,
        Infeasible = 1,
        IterationLimit = 2,
        NumericalFailure = 3
    };
}

namespace milp {

// Isolated CPU decision logic for testability
void process_node_result(
    const MILPNode& node,
    GlobalIncumbent& incumbent,
    const std::vector<Float>& host_x,
    Float host_obj,
    gpu::DeviceSimplexStatus status,
    const core::Model& host_model
);

// Full GPU dispatch function
void solve_node(
    const MILPNode& node,
    gpu::HBFManager& hbf,
    GlobalIncumbent& incumbent,
    const core::Model& host_model,
    const gpu::DeviceModel& device_model,
    const gpu::DeviceSparseLU& lu,
    gpu::VRAMArena& arena
);

} // namespace milp
} // namespace sankhya
