/**
 * @file sparse_lu.hpp
 * @brief Sparse LU factorization of the basis matrix
 *
 * Concrete implementation of BasisFactorization using sparse LU
 * decomposition with Markowitz-threshold pivoting.
 *
 *   PBQ = LU
 *
 * where P and Q are row and column permutation matrices chosen for
 * sparsity-aware pivoting (Markowitz criterion with a threshold
 * tolerance).
 *
 * Storage layout:
 *   - L is unit lower triangular (diagonal elements are implicitly 1
 *     and are NOT stored).  Off-diagonal entries are stored column-wise
 *     in compressed sparse column (CSC-like) format.
 *   - U (including its diagonal) is stored row-wise in compressed
 *     sparse row (CSR-like) format.
 *   - Row and column permutations (P, Q) are stored as index vectors
 *     together with their inverses so that both forward and inverse
 *     look-ups are O(1).
 */

#pragma once

#include <vector>
#include <stdexcept>

#include "numerics/factorization.hpp"

namespace sankhya {
namespace numerics {

class SparseLUFactorization : public BasisFactorization {
public:
    /**
     * @brief Establish the LU factorization from the constraint matrix
     *        and current basis
     * @param A     the constraint matrix in CSC form
     * @param basis the current basis (basic_indices selects columns of A)
     */
    void factorize(
        const core::CSCMatrix& A,
        const simplex::Basis& basis
    ) override;

    /**
     * @brief Forward transformation: solve B x = rhs in place
     * @param rhs on entry the right-hand side; on exit the solution x
     */
    void ftran(
        std::vector<Float>& rhs
    ) override;

    /**
     * @brief Backward transformation: solve B^T x = rhs in place
     * @param rhs on entry the right-hand side; on exit the solution x
     */
    void btran(
        std::vector<Float>& rhs
    ) override;

    /**
     * @brief Update the basis factorization after a basis change
     *
     * Step 6.1 placeholder — performs full refactorization.
     * Forrest-Tomlin updates are deferred to a later step.
     *
     * @param leaving_row  the row index of the leaving variable
     * @param entering_col the column index of the entering variable
     * @param Aq           the column of A corresponding to the entering variable
     */
    void update(
        Index leaving_row,
        Index entering_col,
        const std::vector<Float>& Aq
    ) override;

    bool needs_refactorization(
        const std::vector<Float>& x_B,
        const std::vector<Float>& b_B
    ) const override;

private:
    Index m_ = 0;  ///< basis dimension

    // L factor — compressed sparse column (CSC-like), unit lower triangular
    // (diagonal = 1 not stored)
    std::vector<Float> L_vals_;      ///< L factor values
    std::vector<Index> L_rows_;      ///< L factor row indices
    std::vector<Index> L_col_ptrs_;  ///< L factor column pointers (size m_+1)

    // U factor — compressed sparse row (CSR-like), including diagonal
    std::vector<Float> U_vals_;      ///< U factor values
    std::vector<Index> U_cols_;      ///< U factor column indices
    std::vector<Index> U_row_ptrs_;  ///< U factor row pointers (size m_+1)

    // Permutation vectors for sparsity-aware pivoting
    std::vector<Index> perm_row_;      ///< row permutation: perm_row_[permuted] = original
    std::vector<Index> perm_col_;      ///< column permutation: perm_col_[permuted] = original
    std::vector<Index> inv_perm_row_;  ///< inverse row perm: inv_perm_row_[original] = permuted
    std::vector<Index> inv_perm_col_;  ///< inverse col perm: inv_perm_col_[original] = permuted

    bool factorized_ = false;  ///< whether a valid factorization is held

    // References for refactorization
    const core::CSCMatrix* A_ptr_ = nullptr;
    simplex::Basis basis_copy_;

    // Forrest-Tomlin update tracking
    struct FTUpdate {
        Index k; // The column index in U that was replaced
        std::vector<Float> w; // The multipliers w_i for i = k ... m-2
    };
    std::vector<FTUpdate> ft_updates_; // Sequence of eta-like row transformations
};

} // namespace numerics
} // namespace sankhya
