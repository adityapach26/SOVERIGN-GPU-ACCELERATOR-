#include "verifier/certificate.hpp"
#include <cmath>

namespace sankhya {
namespace verifier {

bool verify_optimal(
    const core::Model& model,
    const std::vector<Float>& x,
    const std::vector<Float>& pi
) {
    // 1. Dimensional validation
    if (x.size() != static_cast<std::size_t>(model.A.cols)) return false;
    if (pi.size() != static_cast<std::size_t>(model.A.rows)) return false;
    if (model.obj.size() != x.size()) return false;
    if (model.lb.size() != x.size()) return false;
    if (model.ub.size() != x.size()) return false;
    if (model.rhs.size() != pi.size()) return false;
    if (model.A.col_ptrs.size() != x.size() + 1) return false;
    if (model.A.col_ptrs.back() < 0) return false;
    if (model.A.row_indices.size() < static_cast<std::size_t>(model.A.col_ptrs.back())) return false;
    if (model.A.values.size() < static_cast<std::size_t>(model.A.col_ptrs.back())) return false;

    // Constants
    const double epsilon_feas = 1e-6;
    const double obj_sign = (model.sense == OptimizationSense::Maximize) ? -1.0 : 1.0;

    const std::size_t n = x.size();
    const std::size_t m = pi.size();

    // 2. Primal Feasibility
    // Check bounds: l_i - eps <= x_i <= u_i + eps
    for (std::size_t i = 0; i < n; ++i) {
        double xi = static_cast<double>(x[i]);
        double lbi = static_cast<double>(model.lb[i]);
        double ubi = static_cast<double>(model.ub[i]);

        if (lbi > -static_cast<double>(math::kInfinity) && xi <= lbi - epsilon_feas) {
            return false; // Violated lower bound
        }
        if (ubi < static_cast<double>(math::kInfinity) && xi >= ubi + epsilon_feas) {
            return false; // Violated upper bound
        }
    }

    // Check Ax = b: ||Ax - b||_inf < eps
    std::vector<double> Ax(m, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        double xj = static_cast<double>(x[j]);
        Index start = model.A.col_ptrs[j];
        Index end = model.A.col_ptrs[j + 1];
        for (Index k = start; k < end; ++k) {
            Index i = model.A.row_indices[static_cast<std::size_t>(k)];
            if (i < 0 || i >= static_cast<Index>(m)) return false;
            double val = static_cast<double>(model.A.values[static_cast<std::size_t>(k)]);
            Ax[static_cast<std::size_t>(i)] += val * xj;
        }
    }

    for (std::size_t i = 0; i < m; ++i) {
        double res = std::abs(Ax[i] - static_cast<double>(model.rhs[i]));
        if (res >= epsilon_feas) {
            return false; // Violated Ax = b
        }
    }

    // 3 & 4. Dual Feasibility and Complementary Slackness
    // r = c - A^T pi
    for (std::size_t j = 0; j < n; ++j) {
        double cj = obj_sign * static_cast<double>(model.obj[j]);
        double AT_pi_j = 0.0;
        
        Index start = model.A.col_ptrs[j];
        Index end = model.A.col_ptrs[j + 1];
        for (Index k = start; k < end; ++k) {
            Index i = model.A.row_indices[static_cast<std::size_t>(k)];
            if (i < 0 || i >= static_cast<Index>(m)) return false;
            double val = static_cast<double>(model.A.values[static_cast<std::size_t>(k)]);
            AT_pi_j += val * static_cast<double>(pi[static_cast<std::size_t>(i)]);
        }
        
        double rj = cj - AT_pi_j;
        double xj = static_cast<double>(x[j]);
        double lbj = static_cast<double>(model.lb[j]);
        double ubj = static_cast<double>(model.ub[j]);

        // Dual feasibility semantics (Minimization):
        // If free variable, rj == 0
        // If bounded, sign of rj dictates which bound must be active.
        // We check complementary slackness strictly.

        if (rj >= epsilon_feas) {
            // Must be at lower bound
            if (lbj <= -static_cast<double>(math::kInfinity)) {
                return false; // Unbounded below, cannot have positive reduced cost
            }
            if (xj >= lbj + epsilon_feas) {
                return false; // Positive reduced cost but not at lower bound (Complementary slackness failed)
            }
            // Explicitly verify r_j * (x_j - lb_j) ≈ 0 for the lower-bound case
            if (std::abs(rj * (xj - lbj)) >= epsilon_feas) {
                return false;
            }
        } else if (rj <= -epsilon_feas) {
            // Must be at upper bound
            if (ubj >= static_cast<double>(math::kInfinity)) {
                return false; // Unbounded above, cannot have negative reduced cost
            }
            if (xj <= ubj - epsilon_feas) {
                return false; // Negative reduced cost but not at upper bound
            }
            if (std::abs(rj * (xj - ubj)) >= epsilon_feas) {
                return false;
            }
        } else {
            // rj ≈ 0, basic or strictly interior variable
            // No complementary slackness violation by definition of rj ≈ 0.
        }
    }

    return true;
}

} // namespace verifier
} // namespace sankhya
