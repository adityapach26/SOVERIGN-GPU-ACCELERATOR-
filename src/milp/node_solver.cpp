#include "milp/node_solver.hpp"
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

#include "gpu/hbf.cuh"
#include "gpu/vram_arena.cuh"

// Forward declare to avoid including __global__ from device_dual_simplex.cuh in a .cpp file
namespace sankhya {
namespace gpu {
enum class DeviceSimplexStatus {
    Optimal = 0,
    Infeasible = 1,
    IterationLimit = 2,
    NumericalFailure = 3
};

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
}
}

namespace sankhya {
namespace milp {

void process_node_result(
    const MILPNode& /*node*/, // Unused by 18.1 rules
    GlobalIncumbent& incumbent,
    const std::vector<Float>& host_x,
    Float host_obj,
    gpu::DeviceSimplexStatus status,
    const core::Model& host_model
) {
    // 1. Non-optimal handling
    if (status == gpu::DeviceSimplexStatus::Infeasible) {
        // Node is pruned by infeasibility. Do not update incumbent.
        return;
    }
    if (status != gpu::DeviceSimplexStatus::Optimal) {
        // Unbounded, iteration limit, numerical failure -> cannot be used as incumbent
        return;
    }

    // 2. Pruning by bound logic
    if (host_obj >= incumbent.get_obj()) {
        // Pruned by bound. Do not update.
        return;
    }

    // 3. Integrality check
    bool is_integer_feasible = true;
    for (std::size_t i = 0; i < host_x.size(); ++i) {
        if (host_model.vtype[i] != VariableType::Continuous) {
            Float val = host_x[i];
            Float diff = std::abs(val - std::round(val));
            if (diff > 1e-5) {
                is_integer_feasible = false;
                break;
            }
        }
    }

    // 4. Update GlobalIncumbent if integer feasible and strictly better
    if (is_integer_feasible) {
        incumbent.update(host_x, host_obj);
    }
    // If optimal, better bound, but fractional -> remains a candidate for branching (Phase 18.2+)
}

void solve_node(
    const MILPNode& node,
    gpu::HBFManager& hbf,
    GlobalIncumbent& incumbent,
    const core::Model& host_model,
    const gpu::DeviceModel& device_model,
    const gpu::DeviceSparseLU& lu,
    gpu::VRAMArena& arena
) {
    const Index num_cols = static_cast<Index>(host_model.obj.size());
    const Index m = static_cast<Index>(host_model.rhs.size());

    // 1. Working basis state allocated
    gpu::WorkingBasisState ws = hbf.allocate_working_state(m, num_cols);

    // 2. Inherit parent basis and apply bound changes
    hbf.inherit_basis(node.hbf_id, device_model, ws);

    // 3. Allocate device output arrays for the batched dispatcher (num_nodes = 1)
    gpu::WorkingBasisState* d_ws_array = static_cast<gpu::WorkingBasisState*>(
        arena.alloc(sizeof(gpu::WorkingBasisState)));
    cudaMemcpy(d_ws_array, &ws, sizeof(gpu::WorkingBasisState), cudaMemcpyHostToDevice);

    gpu::DeviceSimplexStatus* d_status = static_cast<gpu::DeviceSimplexStatus*>(
        arena.alloc(sizeof(gpu::DeviceSimplexStatus)));
    
    Index* d_iter_count = static_cast<Index*>(
        arena.alloc(sizeof(Index)));

    Float obj_sign = (host_model.sense == OptimizationSense::Minimize) ? 1.0 : -1.0;
    Index max_iterations = 1000;

    // 4. Trigger dual simplex on GPU
    gpu::launch_batched_dual_simplex(
        device_model,
        lu,
        d_ws_array,
        obj_sign,
        max_iterations,
        d_status,
        d_iter_count,
        1
    );

    // 5. Read back results
    gpu::DeviceSimplexStatus h_status;
    cudaMemcpy(&h_status, d_status, sizeof(gpu::DeviceSimplexStatus), cudaMemcpyDeviceToHost);

    std::vector<Float> h_x(num_cols, 0.0);
    cudaMemcpy(h_x.data(), ws.x, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);

    // 6. Compute objective value
    Float h_obj = 0.0;
    for (Index i = 0; i < num_cols; ++i) {
        h_obj += host_model.obj[static_cast<std::size_t>(i)] * h_x[static_cast<std::size_t>(i)];
    }

    // 7. Cleanup device allocations
    arena.free(d_iter_count);
    arena.free(d_status);
    arena.free(d_ws_array);
    hbf.free_working_state(ws);

    // 8. Process result logic
    process_node_result(node, incumbent, h_x, h_obj, h_status, host_model);
}

} // namespace milp
} // namespace sankhya
