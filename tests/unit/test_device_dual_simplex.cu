#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "gpu/device_dual_simplex.cuh"
#include "gpu/device_ftran.cuh"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"
#include "numerics/sparse_lu.hpp"
#include "simplex/dual.hpp"
#include "core/problem.hpp"
#include <vector>

using namespace sankhya;

__global__ void run_gpu_dual_simplex(
    gpu::DeviceModel model,
    gpu::DeviceSparseLU lu,
    gpu::WorkingBasisState ws,
    Float obj_sign,
    Index max_iter,
    gpu::DeviceSimplexStatus* status_out,
    Index* iter_out
) {
    gpu::dual_simplex_kernel(model, lu, ws, obj_sign, max_iter, status_out, iter_out);
}

TEST_CASE("Pre-Phase 13.1 GPU Dual Simplex - Exact CPU Parity & Multi-Iter", "[cuda][dual]") {
    core::Model host_model;
    host_model.sense = OptimizationSense::Minimize;
    host_model.rows = 2;
    host_model.cols = 5;
    host_model.nnz = 8;
    
    // Nontrivial Mathematically Certified Multi-Iteration LP
    // Min z = 3x1 + 4x2 + 5x3
    //  1x1 + 0x2 - 2x3 + 1s1 - 1s2 =  1
    // -2x1 - 2x2 - 1x3 + 0s1 + 1s2 = -6
    
    host_model.A.rows = 2;
    host_model.A.cols = 5;
    host_model.A.col_ptrs = {0, 2, 3, 5, 6, 8};
    host_model.A.row_indices = {0, 1, 1, 0, 1, 0, 0, 1};
    host_model.A.values = {1.0, -2.0, -2.0, -2.0, -1.0, 1.0, -1.0, 1.0};
    
    host_model.obj = {3.0, 4.0, 5.0, 0.0, 0.0};
    host_model.lb = {0.0, 0.0, 0.0, 0.0, 0.0};
    host_model.ub = {1e30, 1e30, 1e30, 1e30, 1e30};
    host_model.rhs = {1.0, -6.0};
    
    simplex::Basis host_basis;
    host_basis.col_status = {simplex::BasisStatus::AtLower, simplex::BasisStatus::AtLower, simplex::BasisStatus::AtLower,
                             simplex::BasisStatus::Basic, simplex::BasisStatus::Basic};
    
    // Initial basis B0 = {s1, s2} -> cols 3 and 4
    // B0 = [[1, -1], [0, 1]] (NOT Identity)
    host_basis.basic_indices = {3, 4};
    
    // EXACT Ax=b verification:
    // x = {0, 0, 0, -5, -6}
    // Eq 1: 1(0) - 2(0) + 1(-5) - 1(-6) = 1 (Matches RHS)
    // Eq 2: -2(0) - 2(0) - 1(0) + 0(-5) + 1(-6) = -6 (Matches RHS)
    std::vector<Float> host_x = {0.0, 0.0, 0.0, -5.0, -6.0};
    
    // CPU dual simplex to get reference result
    numerics::SparseLUFactorization cpu_lu;
    // Make copies since dual_simplex_phase2 modifies them
    simplex::Basis cpu_basis = host_basis;
    std::vector<Float> cpu_x = host_x;
    
    simplex::SimplexStatus cpu_status = simplex::dual_simplex_phase2(
        host_model, cpu_basis, cpu_x, cpu_lu
    );
    
    REQUIRE(cpu_status == simplex::SimplexStatus::Optimal);
    // Expected mathematically certified optimal solution: x1 = 1, x2 = 2
    REQUIRE(cpu_x[0] == Catch::Approx(1.0).margin(1e-7));
    REQUIRE(cpu_x[1] == Catch::Approx(2.0).margin(1e-7));
    REQUIRE(cpu_x[2] == Catch::Approx(0.0).margin(1e-7));
    REQUIRE(cpu_x[3] == Catch::Approx(0.0).margin(1e-7));
    REQUIRE(cpu_x[4] == Catch::Approx(0.0).margin(1e-7));

    // ==========================================
    // GPU EXECUTION
    // ==========================================
    gpu::VRAMArena arena(1024 * 1024);
    
    // Factorize the INITIAL basis for GPU root
    numerics::SparseLUFactorization root_lu;
    root_lu.factorize(host_model.A, host_basis);

    gpu::DeviceModel d_model = gpu::upload_to_device(host_model, arena);
    gpu::DeviceSparseLUManager manager(arena);
    manager.upload(root_lu);
    gpu::DeviceSparseLU d_lu = manager.get_device_struct();

    gpu::HBFManager hbf_manager(arena);
    gpu::WorkingBasisState ws = hbf_manager.allocate_working_state(2, 5, 100, 10);

    // Initialize the working state identically to the start of the CPU phase
    int zero_int = 0;
    Index zero_idx = 0;
    cudaMemcpy(ws.error_code, &zero_int, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.num_eta_cols, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.eta_nnz, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.eta_col_starts, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);
    
    cudaMemcpy(ws.basis_indices, host_basis.basic_indices.data(), 2 * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.x, host_x.data(), 5 * sizeof(Float), cudaMemcpyHostToDevice);
    
    std::vector<bool> host_is_basic(5, false);
    for (Index b : host_basis.basic_indices) host_is_basic[b] = true;
    cudaMemcpy(ws.is_basic, host_is_basic.data(), 5 * sizeof(bool), cudaMemcpyHostToDevice);

    gpu::DeviceSimplexStatus* d_status = static_cast<gpu::DeviceSimplexStatus*>(arena.allocate(sizeof(gpu::DeviceSimplexStatus)));
    Index* d_iter = static_cast<Index*>(arena.allocate(sizeof(Index)));

    // Execute exactly one node
    run_gpu_dual_simplex<<<1, 32>>>(d_model, d_lu, ws, 1.0, 100, d_status, d_iter);
    cudaDeviceSynchronize();

    gpu::DeviceSimplexStatus h_status = gpu::DeviceSimplexStatus::IterationLimit;
    cudaMemcpy(&h_status, d_status, sizeof(gpu::DeviceSimplexStatus), cudaMemcpyDeviceToHost);
    
    Index h_iter = 0;
    cudaMemcpy(&h_iter, d_iter, sizeof(Index), cudaMemcpyDeviceToHost);

    // PROOF: Verify multiple iterations occurred natively on GPU
    // Iteration 1 leaves s2, enters x1 (solution remains primal infeasible at s1=-2)
    // Iteration 2 leaves s1, enters x2 (optimal)
    REQUIRE(h_iter >= 2);

    REQUIRE(h_status == gpu::DeviceSimplexStatus::Optimal);

    std::vector<Float> gpu_x(5);
    cudaMemcpy(gpu_x.data(), ws.x, 5 * sizeof(Float), cudaMemcpyDeviceToHost);

    // Compare GPU solution to CPU solution EXACTLY
    for (Index i = 0; i < 5; ++i) {
        REQUIRE(gpu_x[i] == Catch::Approx(cpu_x[i]).margin(1e-7));
    }

    // Verify basis alignment (basic indices)
    std::vector<Index> gpu_basis_indices(2);
    cudaMemcpy(gpu_basis_indices.data(), ws.basis_indices, 2 * sizeof(Index), cudaMemcpyDeviceToHost);

    std::vector<bool> gpu_is_basic(5);
    cudaMemcpy(gpu_is_basic.data(), ws.is_basic, 5 * sizeof(bool), cudaMemcpyDeviceToHost);

    // Check CPU basic indices are present and match exactly the state boolean array
    for (Index i = 0; i < 2; ++i) {
        Index b_idx = cpu_basis.basic_indices[i];
        REQUIRE(gpu_is_basic[b_idx] == true);
        REQUIRE(gpu_basis_indices[i] == b_idx);
    }

    arena.free(d_iter);
    arena.free(d_status);
    hbf_manager.free_working_state(ws);
    manager.free_all();
}
