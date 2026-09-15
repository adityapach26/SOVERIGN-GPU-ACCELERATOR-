/**
 * @file factorization.hpp
 * @brief Abstract basis factorization interface for SANKHYA / VAJRA-OPT
 *
 * Defines the pure virtual CPU-side interface for basis factorization,
 * forward transformation (FTRAN), backward transformation (BTRAN),
 * and basis update operations.
 *
 * Step 4.1: Interface boundary only — no production numerical
 * factorization algorithm is implemented here.
 */

#pragma once

#include <vector>

#include "sankhya/types.hpp"
#include "core/sparse_matrix.hpp"
#include "simplex/basis.hpp"

namespace sankhya {
namespace numerics {

/**
 * @brief Abstract interface for basis factorization
 *
 * A BasisFactorization maintains a representation of the current basis
 * matrix B (the m columns of A selected by the basis) and supports:
 *
 *   - factorize(A, basis): establish the factorization from scratch
 *   - ftran(rhs):          solve B x = rhs in place
 *   - btran(rhs):          solve B^T x = rhs in place
 *   - update(...):         rank-1 update when a basis column changes
 *
 * Concrete implementations may use dense or sparse representations,
 * LU decomposition, or other factorization strategies.
 */
class BasisFactorization {
public:
    virtual ~BasisFactorization() = default;

    /**
     * @brief Establish the basis factorization from the constraint matrix
     *        and current basis
     * @param A     the constraint matrix in CSC form
     * @param basis the current basis (basic_indices selects columns of A)
     */
    virtual void factorize(
        const core::CSCMatrix& A,
        const simplex::Basis& basis
    ) = 0;

    /**
     * @brief Forward transformation: solve B x = rhs in place
     * @param rhs on entry the right-hand side; on exit the solution x
     */
    virtual void ftran(
        std::vector<Float>& rhs
    ) = 0;

    /**
     * @brief Backward transformation: solve B^T x = rhs in place
     * @param rhs on entry the right-hand side; on exit the solution x
     */
    virtual void btran(
        std::vector<Float>& rhs
    ) = 0;

    /**
     * @brief Update the basis factorization after a basis change
     *
     * Represents replacing the column associated with leaving_row
     * in the basis with the entering column Aq.
     *
     * @param leaving_row  the row index of the leaving variable
     * @param entering_col the column index of the entering variable
     * @param Aq           the column of A corresponding to the entering variable
     */
    virtual void update(
        Index leaving_row,
        Index entering_col,
        const std::vector<Float>& Aq
    ) = 0;

    /**
     * @brief Determine if the factorization requires numerical refresh
     *
     * Monitors residual || B x_B - b_B ||_infty against a configured tolerance.
     *
     * @param x_B current basis solution
     * @param b_B corresponding right-hand side for the basis
     * @return true if refactorization is needed
     */
    virtual bool needs_refactorization(
        const std::vector<Float>& x_B,
        const std::vector<Float>& b_B
    ) const {
        (void)x_B;
        (void)b_B;
        return false;
    }
};

} // namespace numerics
} // namespace sankhya

