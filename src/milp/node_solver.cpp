#include "milp/node_solver.hpp"
#include <cmath>
#include <iostream>

#include "gpu/node_dispatch.cuh"
#include "gpu/hbf.cuh"
#include "gpu/vram_arena.cuh"

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

    Float obj_sign = (host_model.sense == OptimizationSense::Minimize) ? 1.0 : -1.0;
    Index max_iterations = 1000;

    // 3. Trigger dual simplex via C++ bridge
    gpu::NodeDispatchResult result = gpu::dispatch_single_node(
        device_model,
        lu,
        ws,
        obj_sign,
        max_iterations,
        arena,
        num_cols
    );

    // 4. Compute objective value
    Float h_obj = 0.0;
    for (Index i = 0; i < num_cols; ++i) {
        h_obj += host_model.obj[static_cast<std::size_t>(i)] * result.x[static_cast<std::size_t>(i)];
    }

    // 5. Cleanup device allocations
    hbf.free_working_state(ws);

    // 6. Process result logic
    process_node_result(node, incumbent, result.x, h_obj, result.status, host_model);
}

} // namespace milp
} // namespace sankhya
