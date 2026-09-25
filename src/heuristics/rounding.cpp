#include "heuristics/rounding.hpp"
#include <cmath>

namespace sankhya {
namespace heuristics {

bool is_integer_feasible(
    const core::Model& model,
    const std::vector<Float>& x,
    const std::vector<Float>* custom_lb,
    const std::vector<Float>* custom_ub,
    Float tol
) {
    if (x.size() != model.vtype.size()) return false;

    const std::vector<Float>& lb = custom_lb ? *custom_lb : model.lb;
    const std::vector<Float>& ub = custom_ub ? *custom_ub : model.ub;

    // 1. Check bounds and integrality
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] < lb[i] - tol || x[i] > ub[i] + tol) {
            return false;
        }
        if (model.vtype[i] != VariableType::Continuous) {
            Float rounded = std::round(x[i]);
            if (std::abs(x[i] - rounded) > tol) {
                return false;
            }
        }
    }

    // 2. Check linear constraints (A x = b)
    std::vector<Float> lhs(model.rhs.size(), 0.0);
    for (Index j = 0; j < model.A.cols; ++j) {
        Index start = model.A.col_ptrs[j];
        Index end = model.A.col_ptrs[j+1];
        Float xj = x[j];
        for (Index p = start; p < end; ++p) {
            lhs[model.A.row_indices[p]] += model.A.values[p] * xj;
        }
    }
    
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (std::abs(lhs[i] - model.rhs[i]) > tol) {
            return false;
        }
    }

    return true;
}

bool apply_rounding_heuristic(
    const core::Model& model,
    const std::vector<Float>& lp_x,
    std::vector<Float>& out_incumbent
) {
    if (lp_x.size() != model.vtype.size()) return false;

    std::vector<Float> candidate = lp_x;

    // Nearest-integer rounding for integer-constrained variables
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        if (model.vtype[i] != VariableType::Continuous) {
            Float rounded = std::round(candidate[i]);
            
            // Respect bounds
            if (rounded < model.lb[i]) rounded = model.lb[i];
            if (rounded > model.ub[i]) rounded = model.ub[i];
            
            candidate[i] = rounded;
        }
    }

    // Evaluate candidate
    if (is_integer_feasible(model, candidate)) {
        out_incumbent = candidate;
        return true; // Strictly feasible integer point found
    }

    // Failed rounding. Intentional return with no solver call.
    return false;
}

} // namespace heuristics
} // namespace sankhya

