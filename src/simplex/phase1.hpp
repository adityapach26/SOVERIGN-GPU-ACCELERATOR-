/**
 * @file phase1.hpp
 * @brief Native CPU Phase-I LP solver for SANKHYA / VAJRA-OPT
 *
 * Implements a standard artificial-variable Phase-I method to construct
 * a primal-feasible basis for an arbitrary LP in equality form:
 *
 *   min   c^T x
 *   s.t.  A x = b,   lb <= x <= ub
 *
 * After Phase I, the existing primal_simplex_phase2() is called to
 * optimize the original objective.
 *
 * This is an ADDITIVE RECOVERY PATH only.  It does NOT replace the
 * primary GPU Mehrotra IPM.  It is invoked only when Mehrotra reports
 * a non-Optimal status.
 *
 * Architectural notes
 * -------------------
 * - The temporary Phase-I model (with artificial variables) is built
 *   entirely inside this module and never exposed to the caller.
 * - The caller's original Model is never mutated.
 * - All artificial variables are removed from the basis before Phase II.
 * - Only original-model variable indices appear in the returned Basis.
 * - Determinism: tie-breaking always uses the smallest variable / row index.
 * - Lower-bound handling: Phase-I shifts each variable xj → xj' = xj - lb_j
 *   so that all shifted variables are non-negative, matching the assumption
 *   that primal_simplex_phase2 sets nonbasic variables to 0 (= lower bound).
 */

#pragma once

#include <vector>

#include "sankhya/types.hpp"
#include "core/problem.hpp"
#include "simplex/basis.hpp"
#include "simplex/primal.hpp"
#include "numerics/factorization.hpp"

namespace sankhya {
namespace simplex {

/**
 * @brief Run Phase I to find a feasible basis, then Phase II to optimality.
 *
 * Constructs the Phase-I auxiliary LP, drives artificial variables to zero,
 * then calls primal_simplex_phase2() on the original problem.
 *
 * @param model      The LP in canonical equality form (A x = b, lb <= x <= ub).
 *                   Must already be finalized (model.A is valid CSCMatrix).
 * @param basis      Out: the final basis (after Phase I and Phase II).
 * @param x          Out: the final primal solution vector (n entries).
 * @param factorizer A BasisFactorization backend (SparseLUFactorization).
 * @return SimplexStatus::Optimal       if Phase I and II both succeed.
 *         SimplexStatus::Infeasible    if Phase I cannot reach zero.
 *         SimplexStatus::Unbounded     if Phase II detects unboundedness.
 *         SimplexStatus::IterationLimit if any phase exceeds its iteration limit.
 */
SimplexStatus solve_with_phase1(
    const core::Model& model,
    Basis& basis,
    std::vector<Float>& x,
    numerics::BasisFactorization& factorizer
);

} // namespace simplex
} // namespace sankhya
