#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include "cuda/batched_dual.cuh"
#include "gpu/hbf.cuh"
#include "gpu/device_model.cuh"
#include "gpu/vram_arena.cuh"
#include <vector>

using namespace sankhya;

TEST_CASE("Step 13.1 Batched Dual Simplex Launch", "[cuda][batched_dual]") {
    // 1. Setup persistent DeviceModel
    Index m = 10;
    Index n = 20;
    std::vector<Float> h_lb(n, 0.0);
    std::vector<Float> h_ub(n, 10.0);
    std::vector<Float> h_obj(n, 1.0); // Cost of 1.0 for all vars
    
    // For test verification: x_3 cost = 5.0
    h_obj[3] = 5.0; 
    
    gpu::DeviceModel model(m, n, h_lb, h_ub, h_obj);
    
    gpu::VRAMArena arena(1024 * 1024 * 10); // 10MB
    gpu::HBFManager hbf_manager(arena);
    
    // Setup Root Basis
    std::vector<Index> root_basis(m);
    for (Index i = 0; i < m; ++i) root_basis[i] = i;
    hbf_manager.set_root_basis(root_basis);
    
    // 2. Create Sibling Nodes with differing BoundDeltas
    // Parent node
    auto parent = hbf_manager.create_node(0, 0, {}); // Root parent
    
    // Sibling 1: var 3 has lower bound 2.0
    gpu::BoundDelta d1;
    d1.var_idx = 3;
    d1.new_lb = 2.0;
    d1.new_ub = 10.0;
    auto child1 = hbf_manager.create_node(1, 0, {d1});
    
    // Sibling 2: var 3 has lower bound 4.0
    gpu::BoundDelta d2;
    d2.var_idx = 3;
    d2.new_lb = 4.0;
    d2.new_ub = 10.0;
    auto child2 = hbf_manager.create_node(2, 0, {d2});
    
    // Capture state before launch to test immutability
    std::vector<Float> lb_before(n);
    cudaMemcpy(lb_before.data(), model.lb, n * sizeof(Float), cudaMemcpyDeviceToHost);
    
    // 3. Batched Launch
    cuda::BatchedDualSolver solver(hbf_manager, model);
    solver.batched_dual_simplex({1, 2});
    
    // 4. Verify results
    auto objs = solver.get_objectives();
    auto stats = solver.get_statuses();
    
    REQUIRE(objs.size() == 2);
    REQUIRE(stats.size() == 2);
    
    // Status should be -99 (Prerequisite limitation documented)
    REQUIRE(stats[0] == -99);
    REQUIRE(stats[1] == -99);
    
    // Expected objective values:
    // Base objective (all lb=0 except var 3) = 0.0
    // Node 1: var 3 lb = 2.0, cost = 5.0 -> obj = 10.0
    // Node 2: var 3 lb = 4.0, cost = 5.0 -> obj = 20.0
    REQUIRE(objs[0] == 10.0);
    REQUIRE(objs[1] == 20.0);
    
    // Verification that Node 1 didn't overwrite Node 2
    REQUIRE(objs[0] != objs[1]);
    
    // 5. Verify DeviceModel immutability
    std::vector<Float> lb_after(n);
    cudaMemcpy(lb_after.data(), model.lb, n * sizeof(Float), cudaMemcpyDeviceToHost);
    
    for (Index i = 0; i < n; ++i) {
        REQUIRE(lb_before[i] == lb_after[i]);
    }
}
