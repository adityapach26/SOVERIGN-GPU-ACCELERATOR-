/**
 * @file symbolic.hpp
 * @brief Phase 14 Step 14.1: CPU Symbolic Phase and AMD Ordering
 */
#pragma once

#include "core/sparse_matrix.hpp"
#include <vector>

namespace sankhya {
namespace ipm {

/**
 * @brief Symbolic factorization structure for AA^T.
 * 
 * Engineering Decision [B]: We represent the symbolic Cholesky structure of AA^T 
 * storing the permutation, inverse permutation, elimination tree, and symbolic 
 * non-zero pattern of the lower triangular Cholesky factor L.
 */
struct SymbolicFactorization {
    // Permutation vector P. P[i] = original_index
    // This is the elimination ordering.
    std::vector<Index> P;
    
    // Inverse permutation P_inv. P_inv[original_index] = i
    std::vector<Index> P_inv;
    
    // Elimination tree parent array for the permuted matrix.
    // parent[j] is the first row index i > j such that L_{i,j} != 0.
    // parent[j] = -1 if j is a root.
    std::vector<Index> parent;
    
    // Symbolic structure of lower triangular factor L for the permuted matrix.
    // L_pattern has L_pattern.rows = A.rows, L_pattern.cols = A.rows.
    // It is strictly lower triangular (diagonal is implicit).
    core::CSRMatrix L_pattern;
};

/**
 * @brief Computes a basic Minimum Degree ordering on the AA^T structural graph.
 * 
 * Engineering Decision [B]: Implement a basic minimum-degree heuristic 
 * rather than introducing an external AMD dependency. This computes the 
 * elimination ordering permutation P. Tie-breaking is deterministic 
 * (lower original vertex index wins).
 * 
 * @param A The original CSR matrix.
 * @return std::vector<Index> The elimination ordering permutation P.
 */
std::vector<Index> compute_amd_ordering(const core::CSRMatrix& A);

/**
 * @brief Computes the symbolic factorization (elimination tree and fill pattern) for AA^T.
 * 
 * @param A The original CSR matrix.
 * @param P The permutation vector from compute_amd_ordering.
 * @param out_factorization Output structure populated with the symbolic information.
 */
void compute_symbolic_factorization(
    const core::CSRMatrix& A,
    const std::vector<Index>& P,
    SymbolicFactorization& out_factorization
);

} // namespace ipm
} // namespace sankhya
