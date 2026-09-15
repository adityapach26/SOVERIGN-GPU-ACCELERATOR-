/**
 * @file primal.hpp
 * @brief L0 CPU Primal Simplex (Phase II) for SANKHYA / VAJRA-OPT
 *
 * Provides a correctness-first dense-backed Phase II revised primal simplex
 * with Dantzig pricing and standard ratio test.
 *
 * Step 4.2: No Phase I, Big-M, sparse LU, GPU, or advanced pricing/pivoting.
 */

#pragma once

#include <vector>

#include "sankhya/types.hpp"
#include "core/problem.hpp"
#include "simplex/basis.hpp"
#include "numerics/factorization.hpp"

namespace sankhya {
namespace simplex {

/**
 * @brief Termination status of the primal simplex algorithm
 */
enum class SimplexStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit
};

/**
 * @brief Phase II revised primal simplex with Dantzig pricing
 *
 * Solves an LP in equality form (A x = b, lb <= x <= ub) starting from
 * a known Phase-II feasible basis.
 *
 * @param model      the LP data model (finalized)
 * @param basis      the current basis; updated in place on return
 * @param x          the primal variable vector; updated in place on return
 * @param factorizer a BasisFactorization backend for BTRAN/FTRAN/update
 * @return the simplex termination status
 *
 * @note This function does NOT implement Phase I. The caller must supply
 *       a feasible basis. If the initial basic solution is not Phase-II
 *       feasible, SimplexStatus::Infeasible is returned as an engineering
 *       interpretation of the Phase-II-only scope.
 */
SimplexStatus primal_simplex_phase2(
    const core::Model& model,
    Basis& basis,
    std::vector<Float>& x,
    numerics::BasisFactorization& factorizer
);

} // namespace simplex
} // namespace sankhya

