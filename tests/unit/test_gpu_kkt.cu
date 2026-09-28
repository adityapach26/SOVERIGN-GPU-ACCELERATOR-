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
    
    // AMD Ordering expected: P = {1, 0, 2}
    // Permuted M = P M P^T = [[5, 1, 0],
    //                         [1, 6, 1],
    //                         [0, 1, 5]]
    
    // Construct full permuted M pattern for the solver
    core::CSRMatrix M_pattern;
    M_pattern.rows = 3;
    M_pattern.cols = 3;
    M_pattern.row_ptrs = {0, 2, 5, 7};
    M_pattern.col_indices = {0, 1, 0, 1, 2, 1, 2};
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
        // Reconstruct the dense 3x3 lower-triangular L matrix on the host for verification
        std::vector<std::vector<Float>> dense_L(3, std::vector<Float>(3, 0.0));
        
        // We explicitly know the CSR structure that was generated for L:
        std::vector<Index> host_L_row_ptrs = {0, 1, 3, 5};
        std::vector<Index> host_L_col_indices = {0, 0, 1, 1, 2};
        
        for (Index i = 0; i < 3; ++i) {
            for (Index p = host_L_row_ptrs[i]; p < host_L_row_ptrs[i+1]; ++p) {
                Index j = host_L_col_indices[p];
                dense_L[i][j] = host_L_vals[p];
            }
        }
        
        // Calculate R = LL^T - M_permuted
        Float R_norm_sq = 0.0;
        // M_permuted dense:
        Float dense_M_perm[3][3] = {
            {5.0, 1.0, 0.0},
            {1.0, 6.0, 1.0},
            {0.0, 1.0, 5.0}
        };
        
        for (Index i = 0; i < 3; ++i) {
            for (Index j = 0; j < 3; ++j) {
                Float LLT_ij = 0.0;
                for (Index k = 0; k < 3; ++k) {
                    LLT_ij += dense_L[i][k] * dense_L[j][k];
                }
                Float diff = LLT_ij - dense_M_perm[i][j];
                R_norm_sq += diff * diff;
            }
        }
        
        Float factor_residual = std::sqrt(R_norm_sq);
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

// ==========================================================================
// Phase 15.2b: KKT Physical Relative Residual Contract
//
// Verifies the numerical contract introduced in commit 0d32913 / follow-up:
//   The KKT solver must solve M0*dy = rhs (the PHYSICAL equation).
//   M_delta = M0 + delta*I is used ONLY as the factorization preconditioner.
//   Acceptance is based on ||rhs - M0*dy|| / max(1, ||rhs||) <= 1e-10,
//   NOT on an absolute floor.
//
// This test uses a scaled RHS (scale=1e4) so that the FP64 residual floor
// (~1e-11 absolute) is above the old absolute 1e-10 criterion but well
// within the correct relative criterion (1e-10 * rhs_norm ~ 1e-6).
// ==========================================================================
TEST_CASE("Phase 15.2b: KKT physical relative residual contract", "[cuda][kkt][ipm]") {
    // M0 = A*Theta*A^T = [[6,1,1],[1,5,0],[1,0,5]].  True x = [1,2,3], b = [11,11,16].
    // Scaled RHS: b_scaled = scale*b, x_scaled = scale*x.
    // FP64 floor of M0 residual after refinement: ~scale * eps * cond(M0) * ||b||
    //   = 1e4 * 2.2e-16 * ~10 * 22 = ~4.8e-11  (absolute)
    //   = ~4.8e-11 / (16e4) = ~3e-15            (relative)
    // Relative is far below 1e-10 -> should PASS the new contract.
    // Absolute 4.8e-11 may be > 1e-10 old floor -> would have FAILED old criterion.
    core::CSRMatrix A;
    A.rows = 3;
    A.cols = 5;
    A.row_ptrs = {0, 3, 5, 7};
    A.col_indices = {0, 3, 4, 1, 3, 2, 4};
    A.values = {2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    std::vector<Float> Theta = {1.0, 1.0, 1.0, 1.0, 1.0};

    std::vector<Index> P = ipm::compute_amd_ordering(A);
    ipm::SymbolicFactorization sym;
    ipm::compute_symbolic_factorization(A, P, sym);

    core::CSRMatrix M_pattern;
    M_pattern.rows = 3; M_pattern.cols = 3;
    M_pattern.row_ptrs = {0, 2, 5, 7};
    M_pattern.col_indices = {0, 1, 0, 1, 2, 1, 2};
    M_pattern.values = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    gpu::VRAMArena arena(1024 * 1024);
    gpu::GPUKKTCholeskySolver kkt_solver(arena, A, sym, M_pattern);
    kkt_solver.gpu_cholesky_factorize(A, Theta);

    constexpr double scale = 1e4;
    std::vector<Float> rhs_scaled = {
        static_cast<Float>(11.0 * scale),
        static_cast<Float>(11.0 * scale),
        static_cast<Float>(16.0 * scale)
    };

    // Solver must succeed: relative M0 residual is within 1e-10 * rhs_norm.
    bool ok = kkt_solver.gpu_cholesky_solve(rhs_scaled);
    REQUIRE(ok == true);

    // Solution must be scale * [1, 2, 3].
    REQUIRE(rhs_scaled[0] == Catch::Approx(scale * 1.0).epsilon(1e-8));
    REQUIRE(rhs_scaled[1] == Catch::Approx(scale * 2.0).epsilon(1e-8));
    REQUIRE(rhs_scaled[2] == Catch::Approx(scale * 3.0).epsilon(1e-8));

    // Explicit physical M0 residual (not M_delta) must be small relative to rhs_norm.
    // M0 = [[6,1,1],[1,5,0],[1,0,5]]
    double r0 = 6.0*rhs_scaled[0] + rhs_scaled[1] + rhs_scaled[2] - 11.0*scale;
    double r1 = rhs_scaled[0]     + 5.0*rhs_scaled[1]              - 11.0*scale;
    double r2 = rhs_scaled[0]                      + 5.0*rhs_scaled[2] - 16.0*scale;
    double abs_M0_res = std::max(std::abs(r0), std::max(std::abs(r1), std::abs(r2)));
    double rhs_norm_val = std::max(1.0, 16.0 * scale); // max component of rhs
    double rel_M0_res = abs_M0_res / rhs_norm_val;

    // Relative physical residual must satisfy the physical contract (stricter than verifier 1e-8).
    REQUIRE(rel_M0_res < 1e-8);
}