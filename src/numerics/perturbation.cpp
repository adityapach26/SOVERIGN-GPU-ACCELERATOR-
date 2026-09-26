/**
 * @file perturbation.cpp
 * @brief Symplectic Phi perturbation implementation — Proposed Tier-2 anti-stall heuristic.
 *
 * CLASSIFICATION: [E] PROPOSED MECHANISM
 *
 * Implements a deterministic golden-ratio perturbation proposed by VAJRA-OPT to help
 * break cyclic/degenerate constraint manifolds without PRNG overhead.
 *
 * This is NOT a verified theorem and does NOT replace Bland's rule or the Harris
 * two-pass ratio test. It is exposed as an explicitly callable mechanism only.
 *
 * Formula:
 *   b_tilde[i] = b[i] + epsilon_0 * phi^(i mod P)
 *
 *   phi     = (sqrt(5) - 1) / 2   (golden ratio conjugate; exact via std::sqrt)
 *   epsilon0 = 2^(-48)             (exact binary fraction via std::ldexp)
 *   P       caller-supplied prime period, P >= bounds.size()
 */

#include "numerics/perturbation.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace sankhya {
namespace numerics {

void apply_symplectic_phi(std::vector<Float>& bounds, Index P) {
    // -----------------------------------------------------------------------
    // Validate P
    // -----------------------------------------------------------------------
    if (P == 0) {
        throw std::invalid_argument(
            "apply_symplectic_phi: P must be >= 1 (got P = 0)");
    }

    const auto m = static_cast<Index>(bounds.size());
    if (P < m) {
        throw std::invalid_argument(
            "apply_symplectic_phi: P must be >= bounds.size() (P = " +
            std::to_string(P) + ", bounds.size() = " + std::to_string(m) + ")");
    }

    // -----------------------------------------------------------------------
    // Mathematical constants — exact values, never hard-coded approximations.
    //
    //   phi     = (sqrt(5) - 1) / 2       golden ratio conjugate
    //   epsilon0 = 2^(-48)                smallest binary power that avoids
    //              conflicts with typical LP data scaling
    // -----------------------------------------------------------------------
    const Float phi      = (std::sqrt(Float{5}) - Float{1}) / Float{2};
    const Float epsilon0 = std::ldexp(Float{1}, -48);

    // -----------------------------------------------------------------------
    // Apply perturbation in-place.
    //
    // We precompute phi^k iteratively to avoid repeated std::pow calls,
    // exploiting the fact that phi^(k+1) = phi^k * phi.
    //
    // Because P >= m, the exponent (i % P) == i for every valid index i,
    // so the sequence phi^0, phi^1, ..., phi^(m-1) is always strictly
    // monotonically decreasing — the perturbation diminishes geometrically.
    //
    // When the caller passes P < m (invalid, rejected above), the exponent
    // would repeat according to i % P, which is the intended behavior for
    // a genuine prime period shorter than the vector. We keep the general
    // modulo formulation here for mathematical correctness, though the guard
    // above currently enforces P >= m.
    // -----------------------------------------------------------------------
    Float phi_power = Float{1};     // phi^0 = 1 for i = 0
    for (Index i = 0; i < m; ++i) {
        // Recompute phi^(i % P) multiplicatively.
        // Since P >= m, (i % P) == i here, so phi_power always advances.
        // The std::pow path would be:
        //   bounds[i] += epsilon0 * std::pow(phi, static_cast<Float>(i % P));
        // The iterative path below is numerically equivalent for sequential i.
        if (i == 0) {
            phi_power = Float{1};   // phi^0
        } else {
            phi_power *= phi;       // phi^i = phi^(i-1) * phi
        }

        bounds[static_cast<std::size_t>(i)] += epsilon0 * phi_power;
    }
}

} // namespace numerics
} // namespace sankhya
