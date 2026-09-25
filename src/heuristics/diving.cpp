#include "heuristics/diving.hpp"
#include "heuristics/rounding.hpp"
#include "gpu/node_dispatch.cuh"
#include "gpu/hbf.cuh"
#include "gpu/vram_arena.cuh"
#include "milp/node_solver.hpp"
#include <cmath>
#include <algorithm>

namespace sankhya {
namespace heuristics {

bool apply_diving_heuristic(
    const core::Model& host_model,
    const gpu::DeviceModel& device_model,
    const gpu::DeviceSparseLU& lu,
    gpu::HBFManager& hbf,
    gpu::VRAMArena& arena,
    uint32_t current_node_id,
    const std::vector<Float>& initial_x,
    milp::GlobalIncumbent& incumbent
) {
    std::vector<Float> current_x = initial_x;
    uint32_t current_parent = current_node_id;
    
    // We arbitrarily place our temporary diving node IDs far away from early B&B nodes
    // to avoid potential ID conflicts. HBFManager guarantees O(1) allocation but respects ID namespace.
    uint32_t dive_id_start = 4000; 
    int max_depth = 50; 
    bool found_incumbent = false;
    
    const Index num_cols = static_cast<Index>(host_model.obj.size());
    const Index m = static_cast<Index>(host_model.rhs.size());

    // Allocate working basis state ONCE for the dive
    gpu::WorkingBasisState ws = hbf.allocate_working_state(m, num_cols);

    // Initial inheritance purely to bootstrap the device state and read the bounds back
    hbf.inherit_basis(current_node_id, device_model, ws);
    
    // Extract actual bounds taking effect at this node via the C++-CUDA bridge
    std::vector<Float> current_lb, current_ub;
    gpu::fetch_working_bounds(ws, num_cols, current_lb, current_ub);

    Float obj_sign = (host_model.sense == OptimizationSense::Minimize) ? 1.0 : -1.0;
    Index max_iterations = 1000;

    for (int depth = 0; depth < max_depth; ++depth) {
        Index best_j = -1;
        Float min_dist = math::kInfinity;
        Float target_bound = 0.0;
        bool is_upper_bound = false;

        // 1. Identify fractional integer variables
        for (std::size_t j = 0; j < current_x.size(); ++j) {
            if (host_model.vtype[j] != VariableType::Continuous) {
                Float xj = current_x[j];
                Float fl = std::floor(xj);
                Float ce = std::ceil(xj);
                
                // Use repository numerical tolerance conventions
                Float dist_down = xj - fl;
                Float dist_up = ce - xj;
                
                if (dist_down > math::kDefaultFeasibilityTol && dist_up > math::kDefaultFeasibilityTol) {
                    Float dist = std::min(dist_down, dist_up);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_j = static_cast<Index>(j);
                        if (dist_down < dist_up) {
                            // Nearest is floor, so branch down: x <= floor
                            target_bound = fl;
                            is_upper_bound = true;
                        } else {
                            // Nearest is ceil, so branch up: x >= ceil
                            target_bound = ce;
                            is_upper_bound = false;
                        }
                    }
                }
            }
        }

        // 2. No remaining fractional variables -> feasible candidate?
        if (best_j == -1) {
            if (is_integer_feasible(host_model, current_x, &current_lb, &current_ub)) {
                Float obj = 0.0;
                for (std::size_t i = 0; i < current_x.size(); ++i) {
                    obj += host_model.obj[i] * current_x[i];
                }
                if (host_model.sense == OptimizationSense::Maximize) {
                    obj = -obj; // Ensure compatibility with existing minimization incumbent logic
                }
                incumbent.update(current_x, obj);
                found_incumbent = true;
            }
            break; // Dive ends
        }

        // 3. Fix/branch that variable toward its nearest integer direction
        gpu::BoundDelta delta;
        delta.var_idx = best_j;
        
        if (is_upper_bound) {
            delta.new_lb = current_lb[best_j];
            delta.new_ub = target_bound;
            current_ub[best_j] = target_bound;
        } else {
            delta.new_lb = target_bound;
            current_lb[best_j] = target_bound;
            delta.new_ub = current_ub[best_j];
        }

        uint32_t next_id = dive_id_start + depth;
        // The orchestrating B&B manager frees all nodes eventually. 
        hbf.create_node(next_id, current_parent, {delta});

        // Re-inherit to transparently apply the bound delta to the GPU working state
        hbf.inherit_basis(next_id, device_model, ws);

        // 4. Recompute the resulting relaxation using the existing project mechanisms
        gpu::NodeDispatchResult result = gpu::dispatch_single_node(
            device_model, lu, ws, obj_sign, max_iterations, arena, num_cols
        );

        if (result.status != gpu::DeviceSimplexStatus::Optimal) {
            break; // Dive becomes infeasible or hit iteration limit
        }
        
        current_x = result.x;
        current_parent = next_id;
    }
    
    hbf.free_working_state(ws);
    return found_incumbent;
}

} // namespace heuristics
} // namespace sankhya
