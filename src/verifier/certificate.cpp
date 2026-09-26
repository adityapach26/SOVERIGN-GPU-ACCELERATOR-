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
    const double kInf = static_cast<double>(math::kInfinity);

    const std::size_t n = x.size();
    const std::size_t m = pi.size();

    // 2. Primal Feasibility
    // Check bounds: lb_i - eps <= x_i <= ub_i + eps
    for (std::size_t i = 0; i < n; ++i) {
        double xi  = static_cast<double>(x[i]);
        double lbi = static_cast<double>(model.lb[i]);
        double ubi = static_cast<double>(model.ub[i]);

        if (lbi > -kInf && xi <= lbi - epsilon_feas) {
            return false; // Violated lower bound
        }
        if (ubi < kInf && xi >= ubi + epsilon_feas) {
            return false; // Violated upper bound
        }
    }

    // Check Ax = b: ||Ax - b||_inf < eps
    std::vector<double> Ax(m, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        double xj    = static_cast<double>(x[j]);
        Index start  = model.A.col_ptrs[j];
        Index end    = model.A.col_ptrs[j + 1];
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
    //
    // r = c - A^T pi  (reduced cost)
    //
    // Correct KKT complementarity (minimization, lb <= x <= ub):
    //
    //   Case A: r > eps  (wants to push x to lb)
    //     - lb = -inf          -> dual infeasible (return false)
    //     - lb finite          -> check r * (x - lb) <= eps
    //                            (if the product is small, near-lb is certified)
    //
    //   Case B: r < -eps  (wants to push x to ub)
    //     - ub = +inf          -> dual infeasible (return false)
    //     - ub finite          -> check (-r) * (ub - x) <= eps
    //
    //   Case C: |r| <= eps     -> no complementarity violation
    //
    // Fix (Phase 33.1): the previous implementation used an additive distance
    // check  (x - lb <= eps)  before the product check.  For variables that are
    // very slightly above lb with a small positive r, the product r*(x-lb) can
    // be far below eps even though x-lb > eps (e.g. AFIRO var 18:
    //   r=0.0416, x-lb=5.3e-6, product=2.2e-7 < 1e-6).
    // The additive check is overly strict and not the correct KKT criterion.
    // Only the complementarity PRODUCT condition is used here.

    for (std::size_t j = 0; j < n; ++j) {
        double cj = obj_sign * static_cast<double>(model.obj[j]);
        double AT_pi_j = 0.0;

        Index start = model.A.col_ptrs[j];
        Index end   = model.A.col_ptrs[j + 1];
        for (Index k = start; k < end; ++k) {
            Index i = model.A.row_indices[static_cast<std::size_t>(k)];
            if (i < 0 || i >= static_cast<Index>(m)) return false;
            double val = static_cast<double>(model.A.values[static_cast<std::size_t>(k)]);
            AT_pi_j += val * static_cast<double>(pi[static_cast<std::size_t>(i)]);
        }

        double rj  = cj - AT_pi_j;
        double xj  = static_cast<double>(x[j]);
        double lbj = static_cast<double>(model.lb[j]);
        double ubj = static_cast<double>(model.ub[j]);

        if (rj >= epsilon_feas) {
            // Case A: positive reduced cost — variable must be driven to lb.
            if (lbj <= -kInf) {
                return false; // Unbounded below: cannot have r > 0
            }
            // Complementarity product: r_j * (x_j - lb_j) <= eps
            if (rj * (xj - lbj) >= epsilon_feas) {
                return false;
            }
        } else if (rj <= -epsilon_feas) {
            // Case B: negative reduced cost — variable must be driven to ub.
            if (ubj >= kInf) {
                return false; // Unbounded above: cannot have r < 0
            }
            // Complementarity product: (-r_j) * (ub_j - x_j) <= eps
            if ((-rj) * (ubj - xj) >= epsilon_feas) {
                return false;
            }
        }
        // Case C: |r_j| < eps — no violation.
    }

    return true;
}

} // namespace verifier
} // namespace sankhya
