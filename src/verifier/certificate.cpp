#include "verifier/certificate.hpp"
#include <cmath>
#include <iostream>

namespace sankhya {
namespace verifier {

bool verify_optimal(
    const core::Model& model,
    const std::vector<Float>& x,
    const std::vector<Float>& pi
) {
    // 1. Dimensional validation
    if (x.size() != static_cast<std::size_t>(model.A.cols)) {
        std::cout << "Certificate failure: component: dimensions, variable: N/A, expected x.size()=" 
                  << model.A.cols << ", got " << x.size() << "\n";
        return false;
    }
    if (pi.size() != static_cast<std::size_t>(model.A.rows)) {
        std::cout << "Certificate failure: component: dimensions, variable: N/A, expected pi.size()=" 
                  << model.A.rows << ", got " << pi.size() << "\n";
        return false;
    }
    if (model.obj.size() != x.size() || model.lb.size() != x.size() || model.ub.size() != x.size() ||
        model.A.col_ptrs.size() != x.size() + 1 || model.A.col_ptrs.back() < 0 ||
        model.A.row_indices.size() < static_cast<std::size_t>(model.A.col_ptrs.back()) ||
        model.A.values.size() < static_cast<std::size_t>(model.A.col_ptrs.back())) {
        std::cout << "Certificate failure: component: dimensions, variable: N/A, model consistency failed\n";
        return false;
    }
    if (model.rhs.size() != pi.size()) {
        std::cout << "Certificate failure: component: dimensions, variable: N/A, rhs.size() mismatch\n";
        return false;
    }

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
            std::cout << "Certificate failure:\ncomponent: primal bound\nvariable: " << i 
                      << "\nx: " << xi << "\nlb: " << lbi << "\nub: " << ubi 
                      << "\nviolation: x < lb - eps\n";
            return false;
        }
        if (ubi < kInf && xi >= ubi + epsilon_feas) {
            std::cout << "Certificate failure:\ncomponent: primal bound\nvariable: " << i 
                      << "\nx: " << xi << "\nlb: " << lbi << "\nub: " << ubi 
                      << "\nviolation: x > ub + eps\n";
            return false;
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
            std::cout << "Certificate failure:\ncomponent: primal Ax=b\nrow: " << i 
                      << "\nAx: " << Ax[i] << "\nb: " << model.rhs[i] 
                      << "\nresidual: " << res << "\n";
            return false;
        }
    }

    // 3 & 4. Dual Feasibility and Complementary Slackness
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
                std::cout << "Certificate failure:\ncomponent: dual feasibility\nvariable: " << j 
                          << "\nx: " << xj << "\nlb: " << lbj << "\nub: " << ubj 
                          << "\nc: " << cj << "\nAt_pi: " << AT_pi_j << "\nr: " << rj 
                          << "\ncomplementarity: " << rj * (xj - lbj) 
                          << "\nviolation: r > 0 but lb is -inf\n";
                return false;
            }
            double comp = rj * (xj - lbj);
            if (comp >= epsilon_feas) {
                std::cout << "Certificate failure:\ncomponent: complementary slackness (lower)\nvariable: " << j 
                          << "\nx: " << xj << "\nlb: " << lbj << "\nub: " << ubj 
                          << "\nc: " << cj << "\nAt_pi: " << AT_pi_j << "\nr: " << rj 
                          << "\ncomplementarity: " << comp << "\n";
                return false;
            }
        } else if (rj <= -epsilon_feas) {
            // Case B: negative reduced cost — variable must be driven to ub.
            if (ubj >= kInf) {
                std::cout << "Certificate failure:\ncomponent: dual feasibility\nvariable: " << j 
                          << "\nx: " << xj << "\nlb: " << lbj << "\nub: " << ubj 
                          << "\nc: " << cj << "\nAt_pi: " << AT_pi_j << "\nr: " << rj 
                          << "\ncomplementarity: " << (-rj) * (ubj - xj) 
                          << "\nviolation: r < 0 but ub is +inf\n";
                return false;
            }
            double comp = (-rj) * (ubj - xj);
            if (comp >= epsilon_feas) {
                std::cout << "Certificate failure:\ncomponent: complementary slackness (upper)\nvariable: " << j 
                          << "\nx: " << xj << "\nlb: " << lbj << "\nub: " << ubj 
                          << "\nc: " << cj << "\nAt_pi: " << AT_pi_j << "\nr: " << rj 
                          << "\ncomplementarity: " << comp << "\n";
                return false;
            }
        }
    }

    return true;
}

} // namespace verifier
} // namespace sankhya
