/**
 * @file perturbation.hpp
 * @brief Symplectic Phi perturbation — Proposed Tier-2 anti-stall heuristic.
 *
 * CLASSIFICATION: [E] PROPOSED MECHANISM
 *
 * This is a deterministic golden-ratio anti-stall perturbation proposed by VAJRA-OPT.
 * It is NOT a verified theorem and MUST NOT replace Bland's rule or Harris ratio testing.
 * Expose it as an independent callable mechanism only.
 *
 * Mathematical definition:
 *
 *   b_tilde[i] = b[i] + epsilon_0 * phi^(i mod P)
 *
 * where:
 *   phi     = (sqrt(5) - 1) / 2          (golden ratio conjugate)
 *   epsilon_0 = 2^(-48)                  (exact binary fraction)
 *   P >= m, P prime                       (period, caller-specified)
 */

#pragma once

#include "sankhya/types.hpp"
#include <vector>

namespace sankhya {
namespace numerics {

/**
 * @brief Apply symplectic phi perturbation to a bounds/RHS vector.
 *
 * Perturbs each element in-place according to:
 *
 *   bounds[i] += epsilon_0 * phi^(i % P)
 *
 * where phi = (sqrt(5)-1)/2 and epsilon_0 = 2^(-48).
 *
 * PROPOSED MECHANISM: This is a deterministic Tier-2 anti-stall heuristic.
 * It is NOT proven to guarantee cycle avoidance. Do NOT invoke automatically
 * from Simplex, B&B, or other solver paths. Call explicitly when needed.
 *
 * @param bounds Vector to perturb in-place (size unchanged).
 * @param P      Prime period satisfying P >= bounds.size().
 *               Must be a prime number and >= bounds.size().
 *               Passing P = 0 or P < bounds.size() throws std::invalid_argument.
 *
 * @throws std::invalid_argument if P == 0 or P < bounds.size().
 */
void apply_symplectic_phi(std::vector<Float>& bounds, Index P);

} // namespace numerics
} // namespace sankhya
