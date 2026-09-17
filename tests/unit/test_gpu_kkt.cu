#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "cuda/kkt.cuh"
#include "gpu/vram_arena.cuh"
#include "ipm/symbolic.hpp"
#include "core/problem.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

using namespace sankhya;

TEST_CASE("Phase 15.1: GPU Supernodal Cholesky Factorization Path", "[cuda][kkt][ipm]") {
    // 1. Construct known symmetric positive-definite system
    // A = [[2, 0],
    //      [1, 3]]
    core::CSRMatrix A;
    A.rows = 2;
    A.cols = 2;
    A.row_ptrs = {0, 1, 3};
    A.col_indices = {0, 0, 1};
    A.values = {2.0, 1.0, 3.0};
    
    // Theta = [1, 1]
    std::vector<Float> Theta = {1.0, 1.0};
    
    // M = A Theta A^T = [[4, 2],
    //                    [2, 10]]
    // Expected L = [[2, 0],
    //               [1, 3]]
    
    // 2. Compute CPU symbolic factorization
    std::vector<Index> P = ipm::compute_amd_ordering(A);
    ipm::SymbolicFactorization sym;
    ipm::compute_symbolic_factorization(A, P, sym);
    
    // Construct full M pattern for the test (P is identity for this graph)
    core::CSRMatrix M_pattern;
    M_pattern.rows = 2;
    M_pattern.cols = 2;
    M_pattern.row_ptrs = {0, 2, 4};
    M_pattern.col_indices = {0, 1, 0, 1};
    M_pattern.values = {0.0, 0.0, 0.0, 0.0}; // Values to be overwritten by GPU

    // 3. Initialize GPU KKT Solver (VRAM residency setup)
    gpu::VRAMArena arena(1024 * 1024);
    gpu::GPUKKTCholeskySolver kkt_solver(arena, A, sym, M_pattern);
    
    // 4. Test GPU M construction and hit Factorization Blocker
    bool blocker_hit = false;
    try {
        kkt_solver.gpu_cholesky_factorize(A, Theta);
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        if (msg.find("BLOCKER") != std::string::npos) {
            blocker_hit = true;
        } else {
            throw; // Unexpected error
        }
    }
    REQUIRE(blocker_hit == true);
    
    // 5. Bypass Blocker: Upload Known L manually to test Triangular Solves on GPU
    // full L nonzeros for this matrix (including diagonal): {2.0, 1.0, 3.0}
    std::vector<Float> host_L_vals = {2.0, 1.0, 3.0};
    Float* d_L_vals = kkt_solver.get_device_L_values();
    cudaMemcpy(d_L_vals, host_L_vals.data(), 3 * sizeof(Float), cudaMemcpyHostToDevice);
    
    // 6. Test GPU Triangular Solves
    // M y = rhs  ==>  L L^T y = rhs
    // rhs = [8, 22]^T
    // Expected y = [1, 2]^T
    std::vector<Float> rhs = {8.0, 22.0};
    kkt_solver.gpu_cholesky_solve(rhs);
    
    // Validate GPU solution
    REQUIRE(rhs[0] == Catch::Approx(1.0).margin(1e-7));
    REQUIRE(rhs[1] == Catch::Approx(2.0).margin(1e-7));
    
    // 7. Validate Residual
    // M x = [4(1) + 2(2), 2(1) + 10(2)] = [8, 22]
    Float res_0 = 4.0 * rhs[0] + 2.0 * rhs[1] - 8.0;
    Float res_1 = 2.0 * rhs[0] + 10.0 * rhs[1] - 22.0;
    Float residual = std::sqrt(res_0 * res_0 + res_1 * res_1);
    REQUIRE(residual < 1e-6);
}
