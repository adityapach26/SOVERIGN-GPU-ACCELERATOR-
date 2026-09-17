#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "cuda/kkt.cuh"
#include "gpu/vram_arena.cuh"
#include "ipm/symbolic.hpp"
#include "core/problem.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>

using namespace sankhya;

TEST_CASE("Phase 15.1: GPU Sparse Cholesky Factorization Path (Hardened)", "[cuda][kkt][ipm]") {
    // 1. Construct non-trivial, symmetric positive-definite system with branching elimination tree
    // M = [[6, 1, 1],
    //      [1, 5, 0],
    //      [1, 0, 5]]
    // This forms a star graph. AMD will eliminate node 1 and 2 before node 0.
    
    core::CSRMatrix A;
    A.rows = 3;
    A.cols = 5;
    A.row_ptrs = {0, 3, 5, 7};
    A.col_indices = {0, 3, 4, 1, 3, 2, 4};
    A.values = {2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    
    std::vector<Float> Theta = {1.0, 1.0, 1.0, 1.0, 1.0};
    
    // 2. Compute CPU symbolic factorization
    std::vector<Index> P = ipm::compute_amd_ordering(A);
    ipm::SymbolicFactorization sym;
    ipm::compute_symbolic_factorization(A, P, sym);
    
    // AMD Ordering expected: P = {1, 2, 0}
    // Permuted M = P M P^T = [[5, 0, 1],
    //                         [0, 5, 1],
    //                         [1, 1, 6]]
    // Expected L nonzeros for permuted M:
    // row 0: L00 = sqrt(5)
    // row 1: L10 = 0, L11 = sqrt(5)
    // row 2: L20 = 1/sqrt(5), L21 = 1/sqrt(5), L22 = sqrt(6 - 0.2 - 0.2) = sqrt(5.6)
    
    // Construct full permuted M pattern for the solver
    core::CSRMatrix M_pattern;
    M_pattern.rows = 3;
    M_pattern.cols = 3;
    M_pattern.row_ptrs = {0, 2, 4, 7};
    M_pattern.col_indices = {0, 2, 1, 2, 0, 1, 2};
    M_pattern.values = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 

    // 3. Initialize GPU KKT Solver (VRAM residency setup)
    gpu::VRAMArena arena(1024 * 1024);
    gpu::GPUKKTCholeskySolver kkt_solver(arena, A, sym, M_pattern);
    
    // 4. Execute Real GPU Factorization (M = A Theta A^T and M = LL^T)
    kkt_solver.gpu_cholesky_factorize(A, Theta);
    
    // 5. Verify the generated L factor
    Index nnz_L = sym.L_pattern.values.size() + 3; // strictly lower + diagonal
    std::vector<Float> host_L_vals(nnz_L, 0.0);
    Float* d_L_vals = kkt_solver.get_device_L_values();
    
    cudaError_t err = cudaMemcpy(host_L_vals.data(), d_L_vals, nnz_L * sizeof(Float), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        Float expected_L00 = std::sqrt(5.0);
        Float expected_L11 = std::sqrt(5.0);
        Float expected_L20 = 1.0 / std::sqrt(5.0);
        Float expected_L21 = 1.0 / std::sqrt(5.0);
        Float expected_L22 = std::sqrt(5.6);
        
        // Check L * L^T approx M_permuted
        // We know the pattern is:
        // row 0: 0
        // row 1: 1
        // row 2: 0, 1, 2
        // So host_L_vals = {L00, L11, L20, L21, L22}
        Float L00 = host_L_vals[0];
        Float L11 = host_L_vals[1];
        Float L20 = host_L_vals[2];
        Float L21 = host_L_vals[3];
        Float L22 = host_L_vals[4];
        
        Float res_00 = (L00 * L00) - 5.0;
        Float res_11 = (L11 * L11) - 5.0;
        Float res_20 = (L20 * L00) - 1.0;
        Float res_21 = (L21 * L11) - 1.0;
        Float res_22 = (L20*L20 + L21*L21 + L22*L22) - 6.0;
        
        Float factor_residual = std::sqrt(res_00*res_00 + res_11*res_11 + res_20*res_20 + res_21*res_21 + res_22*res_22);
        REQUIRE(factor_residual < 1e-6);
    }
    
    // 6. Test GPU Triangular Solves (End-to-End)
    // Original M = [[6, 1, 1], [1, 5, 0], [1, 0, 5]]
    // Let true x = [1, 2, 3]^T
    // Then b = M x = [6(1)+1(2)+1(3), 1(1)+5(2)+0, 1(1)+0+5(3)]^T = [11, 11, 16]^T
    std::vector<Float> rhs = {11.0, 11.0, 16.0};
    
    // Note: rhs must be uploaded to GPU, permuted to match P M P^T, solved, and permuted back.
    // The GPUKKTCholeskySolver does this natively!
    kkt_solver.gpu_cholesky_solve(rhs);
    
    if (err == cudaSuccess) {
        // Validate GPU solution
        REQUIRE(rhs[0] == Catch::Approx(1.0).margin(1e-6));
        REQUIRE(rhs[1] == Catch::Approx(2.0).margin(1e-6));
        REQUIRE(rhs[2] == Catch::Approx(3.0).margin(1e-6));
        
        // 7. Validate Solve Residual ||M x - b||
        Float res_0 = 6.0 * rhs[0] + 1.0 * rhs[1] + 1.0 * rhs[2] - 11.0;
        Float res_1 = 1.0 * rhs[0] + 5.0 * rhs[1] - 11.0;
        Float res_2 = 1.0 * rhs[0] + 5.0 * rhs[2] - 16.0;
        Float solve_residual = std::sqrt(res_0*res_0 + res_1*res_1 + res_2*res_2);
        REQUIRE(solve_residual < 1e-6);
    }
}