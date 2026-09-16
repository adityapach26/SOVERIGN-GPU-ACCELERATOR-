#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "gpu/batched_dual_simplex.cuh"
#include "gpu/device_ftran.cuh"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"
#include "numerics/sparse_lu.hpp"
#include "core/problem.hpp"
#include <vector>

using namespace sankhya;

TEST_CASE("Phase 13.1 Batched Sibling-Node Dual Simplex", "[cuda][dual][batching]") {
    core::Model host_model;
    host_model.sense = OptimizationSense::Minimize;
    host_model.rows = 2;
    host_model.cols = 4;
    host_model.nnz = 7;
    
    // Nontrivial Mathematically Certified LP:
    // Min z = 4x1 + 5x2
    // -3x1 - 3x2 + s1 + s2 = -6
    // -2x1 - 1x2 + 0  + s2 = -3
    
    host_model.A.rows = 2;
    host_model.A.cols = 4;
    host_model.A.col_ptrs = {0, 2, 4, 5, 7};
    host_model.A.row_indices = {0, 1, 0, 1, 0, 0, 1};
    host_model.A.values = {-3.0, -2.0, -3.0, -1.0, 1.0, 1.0, 1.0};
    
    host_model.obj = {4.0, 5.0, 0.0, 0.0};
    host_model.lb = {0.0, 0.0, 0.0, 0.0};
    host_model.ub = {1e30, 1e30, 1e30, 1e30};
    host_model.rhs = {-6.0, -3.0};
    
    simplex::Basis host_basis;
    host_basis.col_status = {simplex::BasisStatus::AtLower, simplex::BasisStatus::AtLower,
                             simplex::BasisStatus::Basic, simplex::BasisStatus::Basic};
    
    // Initial basis B0 = {s1, s2} -> cols 2 and 3
    host_basis.basic_indices = {2, 3};

    // ==========================================
    // GPU EXECUTION PREPARATION
    // ==========================================
    gpu::VRAMArena arena(1024 * 1024);
    
    numerics::SparseLUFactorization root_lu;
    root_lu.factorize(host_model.A, host_basis);

    gpu::DeviceModel d_model = gpu::upload_to_device(host_model, arena);
    gpu::DeviceSparseLUManager manager(arena);
    manager.upload(root_lu);
    gpu::DeviceSparseLU d_lu = manager.get_device_struct();

    gpu::HBFManager hbf_manager(arena);

    // Create Root Node (ID = 100)
    std::vector<gpu::BoundDelta> root_deltas;
    hbf_manager.create_node(100, 100, root_deltas);

    // Create Sibling 1 (ID = 101) - BoundDelta on x1
    std::vector<gpu::BoundDelta> child1_deltas = {{0, 5.0, 10.0}};
    hbf_manager.create_node(101, 100, child1_deltas);

    // Create Sibling 2 (ID = 102) - BoundDelta on x2
    std::vector<gpu::BoundDelta> child2_deltas = {{1, 2.0, 8.0}};
    hbf_manager.create_node(102, 100, child2_deltas);

    uint32_t num_nodes = 2;
    std::vector<gpu::WorkingBasisState> h_ws_array;
    std::vector<uint32_t> child_ids = {101, 102};

    // Sibling 1 initial x (Standard certified scale)
    std::vector<Float> host_x_1 = {0.0, 0.0, -3.0, -3.0};
    // Sibling 2 initial x (Scaled by 0.5 to prove independent solver state trace)
    std::vector<Float> host_x_2 = {0.0, 0.0, -1.5, -1.5};

    for (int i = 0; i < num_nodes; ++i) {
        gpu::WorkingBasisState ws = hbf_manager.allocate_working_state(2, 4, 100, 10);
        
        // Inherit basis applies the BoundDeltas to ws.lb and ws.ub
        hbf_manager.inherit_basis(child_ids[i], d_model, ws);

        // Upload independent primal solution
        const std::vector<Float>& x_init = (i == 0) ? host_x_1 : host_x_2;
        cudaMemcpy(ws.x, x_init.data(), 4 * sizeof(Float), cudaMemcpyHostToDevice);

        // Both nodes share the same initial basis structure for this test
        std::vector<bool> host_is_basic = {false, false, true, true};
        cudaMemcpy(ws.is_basic, host_is_basic.data(), 4 * sizeof(bool), cudaMemcpyHostToDevice);

        h_ws_array.push_back(ws);
    }

    gpu::WorkingBasisState* d_ws_array = static_cast<gpu::WorkingBasisState*>(
        arena.allocate(num_nodes * sizeof(gpu::WorkingBasisState))
    );
    cudaMemcpy(d_ws_array, h_ws_array.data(), num_nodes * sizeof(gpu::WorkingBasisState), cudaMemcpyHostToDevice);

    gpu::DeviceSimplexStatus* d_status_array = static_cast<gpu::DeviceSimplexStatus*>(
        arena.allocate(num_nodes * sizeof(gpu::DeviceSimplexStatus))
    );
    Index* d_iter_count_array = static_cast<Index*>(
        arena.allocate(num_nodes * sizeof(Index))
    );

    // ==========================================
    // BATCHED LAUNCH (PHASE 13.1)
    // ==========================================
    
    // SINGLE KERNEL LAUNCH executing multiple nodes concurrently
    gpu::launch_batched_dual_simplex(
        d_model,
        d_lu,
        d_ws_array,
        1.0, // obj_sign
        100, // max_iter
        d_status_array,
        d_iter_count_array,
        num_nodes
    );

    cudaDeviceSynchronize();

    // ==========================================
    // INDEPENDENT RESULT VERIFICATION
    // ==========================================

    std::vector<gpu::DeviceSimplexStatus> h_status(num_nodes);
    cudaMemcpy(h_status.data(), d_status_array, num_nodes * sizeof(gpu::DeviceSimplexStatus), cudaMemcpyDeviceToHost);

    std::vector<Index> h_iter(num_nodes);
    cudaMemcpy(h_iter.data(), d_iter_count_array, num_nodes * sizeof(Index), cudaMemcpyDeviceToHost);

    // Verify Sibling 1
    REQUIRE(h_status[0] == gpu::DeviceSimplexStatus::Optimal);
    REQUIRE(h_iter[0] >= 2);

    std::vector<Float> gpu_x_1(4);
    cudaMemcpy(gpu_x_1.data(), h_ws_array[0].x, 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    REQUIRE(gpu_x_1[0] == Catch::Approx(1.0).margin(1e-7));
    REQUIRE(gpu_x_1[1] == Catch::Approx(1.0).margin(1e-7));

    Float obj_1 = 4.0 * gpu_x_1[0] + 5.0 * gpu_x_1[1];
    REQUIRE(obj_1 == Catch::Approx(9.0).margin(1e-7));

    std::vector<Float> gpu_lb_1(4);
    cudaMemcpy(gpu_lb_1.data(), h_ws_array[0].lb, 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    REQUIRE(gpu_lb_1[0] == Catch::Approx(5.0)); // BoundDelta applied

    // Verify Sibling 2 (Independent Solve)
    REQUIRE(h_status[1] == gpu::DeviceSimplexStatus::Optimal);
    REQUIRE(h_iter[1] >= 2);

    std::vector<Float> gpu_x_2(4);
    cudaMemcpy(gpu_x_2.data(), h_ws_array[1].x, 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    REQUIRE(gpu_x_2[0] == Catch::Approx(0.5).margin(1e-7));
    REQUIRE(gpu_x_2[1] == Catch::Approx(0.5).margin(1e-7));

    Float obj_2 = 4.0 * gpu_x_2[0] + 5.0 * gpu_x_2[1];
    REQUIRE(obj_2 == Catch::Approx(4.5).margin(1e-7));

    std::vector<Float> gpu_lb_2(4);
    cudaMemcpy(gpu_lb_2.data(), h_ws_array[1].lb, 4 * sizeof(Float), cudaMemcpyDeviceToHost);
    REQUIRE(gpu_lb_2[1] == Catch::Approx(2.0)); // BoundDelta applied

    // Prove that sibling state contamination did not occur
    REQUIRE(gpu_lb_1[0] != gpu_lb_2[0]);
    REQUIRE(gpu_x_1[0] != gpu_x_2[0]);
    REQUIRE(obj_1 != obj_2);

    arena.free(d_iter_count_array);
    arena.free(d_status_array);
    arena.free(d_ws_array);
    for(int i = 0; i < num_nodes; i++) hbf_manager.free_working_state(h_ws_array[i]);
    manager.free_all();
}
