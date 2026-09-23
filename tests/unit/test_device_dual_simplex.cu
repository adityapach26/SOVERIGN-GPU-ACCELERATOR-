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


TEST_CASE("Pre-Phase 13.1 GPU Dual Simplex - Mathematically Certified Pivot Certificate", "[cuda][dual]") {
    core::Model host_model;
    host_model.sense = OptimizationSense::Minimize;
    host_model.add_variable(4.0, 0.0, 1e30);
    host_model.add_variable(5.0, 0.0, 1e30);
    host_model.add_variable(0.0, 0.0, 1e30);
    host_model.add_variable(0.0, 0.0, 1e30);
    
    host_model.add_constraint({0, 1, 2, 3}, {-3.0, -3.0, 1.0, 1.0}, -6.0);
    host_model.add_constraint({0, 1, 3}, {-2.0, -1.0, 1.0}, -3.0);
    
    host_model.finalize();
    
    simplex::Basis host_basis;
    host_basis.col_status = {simplex::BasisStatus::AtLower, simplex::BasisStatus::AtLower,
                             simplex::BasisStatus::Basic, simplex::BasisStatus::Basic};
    
    // Initial basis B0 = {s1, s2} -> cols 2 and 3
    // B0 = [[1, 1], [0, 1]] (NOT Identity)
    host_basis.basic_indices = {2, 3};
    
    // EXACT Ax=b verification:
    // x = {0, 0, -3, -3}
    // Eq 1: -3(0) - 3(0) + 1(-3) + 1(-3) = -6 (Matches RHS)
    // Eq 2: -2(0) - 1(0) + 0(-3) + 1(-3) = -3 (Matches RHS)
    std::vector<Float> host_x = {0.0, 0.0, -3.0, -3.0};
    
    // CPU dual simplex to get reference result
    numerics::SparseLUFactorization cpu_lu;
    simplex::Basis cpu_basis = host_basis;
    std::vector<Float> cpu_x = host_x;
    
    simplex::SimplexStatus cpu_status = simplex::dual_simplex_phase2(
        host_model, cpu_basis, cpu_x, cpu_lu
    );
    
    // Test Requirement 1 & 2: Assert CPU behavior against mathematical certificate
    REQUIRE(cpu_status == simplex::SimplexStatus::Optimal);
    REQUIRE(cpu_x[0] == Catch::Approx(1.0).margin(1e-7));
    REQUIRE(cpu_x[1] == Catch::Approx(1.0).margin(1e-7));
    REQUIRE(cpu_x[2] == Catch::Approx(0.0).margin(1e-7));
    REQUIRE(cpu_x[3] == Catch::Approx(0.0).margin(1e-7));

    // ==========================================
    // GPU EXECUTION
    // ==========================================
    gpu::VRAMArena arena(1024 * 1024);
    
    numerics::SparseLUFactorization root_lu;
    root_lu.factorize(host_model.A, host_basis);

    gpu::DeviceModel d_model = gpu::upload_to_device(host_model, arena);
    gpu::DeviceSparseLUManager manager(arena);
    manager.upload(root_lu);
    gpu::DeviceSparseLU d_lu = manager.get_device_struct();

    gpu::HBFManager hbf_manager(arena);
    gpu::WorkingBasisState ws = hbf_manager.allocate_working_state(2, 4, 100, 10);

    int zero_int = 0;
    Index zero_idx = 0;
    cudaMemcpy(ws.error_code, &zero_int, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.num_eta_cols, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.eta_nnz, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.eta_col_starts, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);
    
    // Test Requirement 3: Initialize GPU with exact same valid initial state
    cudaMemcpy(ws.basis_indices, host_basis.basic_indices.data(), 2 * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.x, host_x.data(), 4 * sizeof(Float), cudaMemcpyHostToDevice);
    
    std::vector<uint8_t> host_is_basic(4, 0);
    for (Index b : host_basis.basic_indices) host_is_basic[b] = 1;
    cudaMemcpy(ws.is_basic, host_is_basic.data(), 4 * sizeof(bool), cudaMemcpyHostToDevice);

    gpu::DeviceSimplexStatus* d_status = static_cast<gpu::DeviceSimplexStatus*>(arena.allocate(sizeof(gpu::DeviceSimplexStatus)));
    Index* d_iter = static_cast<Index*>(arena.allocate(sizeof(Index)));

    gpu::dual_simplex_kernel<<<1, 32>>>(d_model, d_lu, ws, 1.0, 100, d_status, d_iter);
    cudaDeviceSynchronize();

    gpu::DeviceSimplexStatus h_status = gpu::DeviceSimplexStatus::IterationLimit;
    cudaMemcpy(&h_status, d_status, sizeof(gpu::DeviceSimplexStatus), cudaMemcpyDeviceToHost);
    
    Index h_iter = 0;
    cudaMemcpy(&h_iter, d_iter, sizeof(Index), cudaMemcpyDeviceToHost);

    // Test Requirement 4: Assert GPU iteration count >= 2
    // Certificate: Iteration 1 leaves s1, enters x2. Iteration 2 leaves s2, enters x1.
    REQUIRE(h_iter >= 2);

    // Test Requirement 5: Assert GPU status == Optimal
    REQUIRE(h_status == gpu::DeviceSimplexStatus::Optimal);

    std::vector<Float> gpu_x(4);
    cudaMemcpy(gpu_x.data(), ws.x, 4 * sizeof(Float), cudaMemcpyDeviceToHost);

    // Test Requirement 6: Compare complete GPU x against CPU x
    for (Index i = 0; i < 4; ++i) {
        REQUIRE(gpu_x[i] == Catch::Approx(cpu_x[i]).margin(1e-7));
    }
    
    // Test Requirement 8: Verify final feasibility numerically where practical
    REQUIRE(gpu_x[0] >= -1e-7);
    REQUIRE(gpu_x[1] >= -1e-7);
    REQUIRE(gpu_x[2] >= -1e-7);
    REQUIRE(gpu_x[3] >= -1e-7);

    // Test Requirement 9: Compare objective values
    Float cpu_obj = 4.0 * cpu_x[0] + 5.0 * cpu_x[1];
    Float gpu_obj = 4.0 * gpu_x[0] + 5.0 * gpu_x[1];
    REQUIRE(gpu_obj == Catch::Approx(9.0).margin(1e-7));
    REQUIRE(gpu_obj == Catch::Approx(cpu_obj).margin(1e-7));

    // Test Requirement 7: Compare GPU basis against CPU basis
    std::vector<Index> gpu_basis_indices(2);
    cudaMemcpy(gpu_basis_indices.data(), ws.basis_indices, 2 * sizeof(Index), cudaMemcpyDeviceToHost);

    std::vector<uint8_t> gpu_is_basic(4);
    cudaMemcpy(gpu_is_basic.data(), ws.is_basic, 4 * sizeof(bool), cudaMemcpyDeviceToHost);

    for (Index i = 0; i < 2; ++i) {
        Index b_idx = cpu_basis.basic_indices[i];
        REQUIRE(gpu_is_basic[b_idx] == 1);
        REQUIRE(gpu_basis_indices[i] == b_idx);
    }

    arena.free(d_iter);
    arena.free(d_status);
    hbf_manager.free_working_state(ws);
    manager.free_all();
}
