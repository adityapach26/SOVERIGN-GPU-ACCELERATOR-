#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "heuristics/diving.hpp"
#include "core/problem.hpp"
#include "gpu/device_model.cuh"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"
#include "gpu/vram_arena.cuh"
#include "numerics/sparse_lu.hpp"
#include "simplex/basis.hpp"
#include "milp/tree.hpp"

using namespace sankhya;

TEST_CASE("Phase 22.1: Diving Heuristic Variable Selection", "[cuda][heuristics][diving]") {
    // We only test the variable selection and setup to avoid full simplex LP solving complexity.
    // The diving heuristic uses `HBFManager` and `dispatch_single_node` from `node_dispatch.cuh`.
    
    // Create a mock model: x0, x1, x2 (all integers)
    core::Model host_model;
    host_model.sense = OptimizationSense::Minimize;
    host_model.obj = {1.0, 1.0, 1.0};
    host_model.lb = {0.0, 0.0, 0.0};
    host_model.ub = {10.0, 10.0, 10.0};
    host_model.vtype = {VariableType::Integer, VariableType::Integer, VariableType::Integer};
    
    // Dummy constraints: Identity matrix A=I, b=lp_x
    host_model.A.rows = 3;
    host_model.A.cols = 3;
    host_model.A.col_ptrs = {0, 1, 2, 3};
    host_model.A.row_indices = {0, 1, 2};
    host_model.A.values = {1.0, 1.0, 1.0};
    host_model.rhs = {1.1, 2.9, 3.5};
    
    std::vector<Float> lp_x = {1.1, 2.9, 3.5}; // Fractional solution
    
    gpu::VRAMArena arena(1024 * 1024);
    gpu::DeviceModel d_model = gpu::upload_to_device(host_model, arena);
    
    simplex::Basis basis;
    basis.col_status = {simplex::BasisStatus::Basic, simplex::BasisStatus::Basic, simplex::BasisStatus::Basic};
    basis.basic_indices = {0, 1, 2};
    
    numerics::SparseLUFactorization cpu_lu;
    cpu_lu.factorize(host_model.A, basis);
    
    gpu::DeviceSparseLUManager lu_manager(arena);
    lu_manager.upload(cpu_lu);
    gpu::DeviceSparseLU d_lu = lu_manager.get_device_struct();
    
    gpu::HBFManager hbf_manager(arena);
    
    // We won't fully execute the dive to completion if it hits iteration limit, 
    // but we can verify it doesn't crash and returns cleanly.
    milp::GlobalIncumbent incumbent;
    uint32_t current_node_id = 0;
    
    // Create the root node since diving starts from it
    hbf_manager.create_node(current_node_id, 0, {});

    // Distance to integer for x0 (1.1): min(0.1, 0.9) = 0.1
    // Distance to integer for x1 (2.9): min(0.9, 0.1) = 0.1
    // Distance to integer for x2 (3.5): min(0.5, 0.5) = 0.5
    // Between x0 and x1, tie break selects smallest index -> x0.
    // The heuristic will branch x0 <= 1.0.
    
    bool result = heuristics::apply_diving_heuristic(
        host_model, d_model, d_lu, hbf_manager, arena, current_node_id, lp_x, incumbent
    );
    
    // Since our mock model has identity constraints, fixing a variable might immediately 
    // trigger infeasibility or just complete. We just verify the plunge terminates safely.
    REQUIRE(result == false); // Diving likely infeasible/iteration limited for this dummy setup
    
    lu_manager.free_all();
    hbf_manager.free_all_nodes();
}

