/**
 * @file dual.hpp
 * @brief CPU Dual Simplex (Phase II) for SANKHYA / VAJRA-OPT
 *
 * Step 7.1: Restores primal feasibility while preserving dual feasibility.
 */

#pragma once

#include <vector>

#include "sankhya/types.hpp"
#include "core/problem.hpp"
#include "simplex/basis.hpp"
#include "numerics/factorization.hpp"
#include "simplex/primal.hpp" // For SimplexStatus

namespace sankhya {
namespace simplex {

/**
 * @brief Phase II CPU Dual Simplex
 *
 * Solves an LP starting from a dual-feasible but primal-infeasible basis.
 *
 * @param model      the LP data model (finalized)
 * @param basis      the current basis; updated in place on return
 * @param x          the primal variable vector; updated in place on return
 * @param factorizer a BasisFactorization backend for BTRAN/FTRAN/update
 * @return the simplex termination status (Optimal, Infeasible for dual unbounded, IterationLimit)
 */
SimplexStatus dual_simplex_phase2(
    const core::Model& model,
    Basis& basis,
    std::vector<Float>& x,
    numerics::BasisFactorization& factorizer
);

} // namespace simplex
} // namespace sankhya

