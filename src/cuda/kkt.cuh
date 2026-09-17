/**
 * @file kkt.cuh
 * @brief Phase 15 Step 15.1: GPU Supernodal Cholesky Factorization
 */
#pragma once

#include "core/sparse_matrix.hpp"
#include "gpu/vram_arena.cuh"
#include "sankhya/types.hpp"
#include "ipm/symbolic.hpp"
#include <vector>
#include <cusparse.h>

namespace sankhya {
namespace gpu {

/**
 * @brief GPU-resident solver for IPM KKT normal equations M = A Theta A^T.
 * 
 * Architecture:
 * 1. CPU orchestration initializes structural data.
 * 2. M is constructed numerically directly on GPU using a custom exact dot-product kernel.
 * 3. Cholesky factorization is conceptually intended for GPU natively.
 *    [B] Engineering Decision: We implement a custom, exact, up-looking sparse Cholesky factorization
 *    executed entirely on the GPU utilizing elimination-tree level parallelism.
 * 4. Forward and backward triangular solves execute natively via cuSPARSE.
 */
class GPUKKTCholeskySolver {
public:
    /**
     * @brief Initializes persistent GPU solver structures.
     * 
     * @param arena Memory arena for persistent VRAM allocations.
     * @param A The immutable model matrix (CSR).
     * @param sym Computed CPU symbolic factorization structure (from Phase 14.1).
     * @param M_pattern The CSR structure for the permuted normal matrix M.
     */
    GPUKKTCholeskySolver(
        VRAMArena& arena, 
        const core::CSRMatrix& A, 
        const ipm::SymbolicFactorization& sym,
        const core::CSRMatrix& M_pattern
    );

    ~GPUKKTCholeskySolver();

    /**
     * @brief Constructs M = A Theta A^T on the GPU and factorizes M = LL^T.
     * 
     * @param A The CSRMatrix model
     * @param Theta The scale vector Theta.
     */
    void gpu_cholesky_factorize(const core::CSRMatrix& A, const std::vector<Float>& Theta);

    /**
     * @brief Solves LL^T \Delta y = rhs on the GPU.
     * 
     * @param rhs RHS vector. Will be overwritten with solution.
     */
    void gpu_cholesky_solve(std::vector<Float>& rhs);

    /**
     * [B] Engineering Decision: Expose device-resident versions for the Mehrotra IPM loop
     * to avoid full host-device round-trips for every IPM iteration.
     */
    void gpu_cholesky_factorize_device(const Float* d_Theta);
    void gpu_cholesky_solve_device(Float* d_rhs);

    // Expose device pointers for testing and verification
    Float* get_device_L_values() const { return d_L_vals_; }

private:
    VRAMArena& arena_;
    
    // Matrix dimensions
    Index m_;
    Index n_;

    // Device pointers for A
    Index* d_A_row_ptrs_;
    Index* d_A_col_indices_;
    Float* d_A_vals_;

    // Device pointers for permuted M
    Index* d_M_row_ptrs_;
    Index* d_M_col_indices_;
    Float* d_M_vals_;

    // Device pointers for L
    Index* d_L_row_ptrs_;
    Index* d_L_col_indices_;
    Float* d_L_vals_;

    // Device pointer for L^T (cuSPARSE requires explicit transpose setup or handles it via descriptor)
    // Actually cuSPARSE handles transpose internally for triangular solve.

    // Device pointers for permutations
    Index* d_P_;

    Index num_levels_;
    std::vector<Index> level_ptrs_;
    Index* d_level_nodes_;
    int* d_factorization_error_;

    // cuSPARSE Context and Descriptors
    cusparseHandle_t handle_;
    cusparseSpMatDescr_t descr_L_;
    
    // Workspaces
    Float* d_z_; // intermediate solution for forward solve
    
    // cuSPARSE SpSV Data
    cusparseSpSVDescr_t spsv_descr_L_;
    cusparseSpSVDescr_t spsv_descr_LT_;
    cusparseDnVecDescr_t vec_rhs_;
    cusparseDnVecDescr_t vec_z_;
    void* d_spsv_buffer_;

    void initialize_cusparse();
};

} // namespace gpu
} // namespace sankhya
