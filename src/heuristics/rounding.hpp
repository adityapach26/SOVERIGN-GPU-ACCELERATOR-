#pragma once
#include <vector>
#include "core/problem.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace heuristics {

/**
 * @brief Checks if a candidate vector satisfies variable bounds, integrality,
 *        and model equality constraints.
 */
bool is_integer_feasible(
    const core::Model& model,
    const std::vector<Float>& x,
    const std::vector<Float>* custom_lb = nullptr,
    const std::vector<Float>* custom_ub = nullptr,
    Float tol = math::kDefaultFeasibilityTol
);

/**
 * @brief Applies the nearest-integer rounding heuristic to find an incumbent early.
 *        Does not silently call another optimization solver.
 */
bool apply_rounding_heuristic(
    const core::Model& model,
    const std::vector<Float>& lp_x,
    std::vector<Float>& out_incumbent
);

} // namespace heuristics
} // namespace sankhya
