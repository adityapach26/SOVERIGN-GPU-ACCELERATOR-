#pragma once

#include <vector>
#include "core/problem.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace verifier {

/**
 * @brief Independently verifies an optimality certificate (x, pi) for a given LP model.
 *
 * Checks:
 * 1. Dimensional consistency.
 * 2. Primal feasibility: ||Ax - b|| < 1e-6 and lb <= x <= ub.
 * 3. Dual feasibility: r = c - A^T pi, with correct signs for variable bounds.
 * 4. Complementary slackness: r_i * (x_i - bound) ≈ 0.
 *
 * Uses strict double-precision math independent of solver execution state.
 *
 * @param model The LP model (in canonical equality form).
 * @param x The primal solution vector.
 * @param pi The dual solution vector (shadow prices).
 * @return true if the certificate is mathematically valid, false otherwise.
 */
bool verify_optimal(
    const core::Model& model,
    const std::vector<Float>& x,
    const std::vector<Float>& pi
);

} // namespace verifier
} // namespace sankhya
