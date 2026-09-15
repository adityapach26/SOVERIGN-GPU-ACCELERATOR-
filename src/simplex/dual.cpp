/**
 * @file dual.cpp
 * @brief CPU Dual Simplex (Phase II) implementation
 *
 * Implements Step 7.1.
 */

#include "dual.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace sankhya {
namespace simplex {

SimplexStatus dual_simplex_phase2(
    const core::Model& model,
    Basis& basis,
    std::vector<Float>& x,
    numerics::BasisFactorization& factorizer
) {
    const Index m = model.A.rows;
    const Index n = model.A.cols;

    // Objective sign: internally reduce to minimization.
    const Float obj_sign = (model.sense == OptimizationSense::Minimize) ? 1.0 : -1.0;

    // Engineering decision: iteration limit to prevent infinite loops.
    static constexpr Index kMaxIterations = 10000;

    // 0. Initialize factorization
    factorizer.factorize(model.A, basis);

    for (Index iter = 0; iter < kMaxIterations; ++iter) {
        // 1. Leaving variable selection
        // p = argmin x_B,i subject to x_B,i < -eps
        Index p = -1;
        Float min_xb = 0.0;

        for (Index i = 0; i < m; ++i) {
            const Float xb_i = x[static_cast<std::size_t>(basis.basic_indices[static_cast<std::size_t>(i)])];
            if (xb_i < -math::kDefaultFeasibilityTol) {
                if (p == -1 || xb_i < min_xb) {
                    p = i;
                    min_xb = xb_i;
                }
            }
        }

        // If no basic variable is primal infeasible, we are optimal (feasible)
        if (p == -1) {
            return SimplexStatus::Optimal;
        }

        // 2. BTRAN: solve B^T d_pi = e_p
        std::vector<Float> d_pi(static_cast<std::size_t>(m), 0.0);
        d_pi[static_cast<std::size_t>(p)] = 1.0;
        factorizer.btran(d_pi);

        // We also need the current dual variables to compute reduced costs.
        // As per source requirement to preserve convention and prevent drift,
        // compute r_j on the fly.
        std::vector<Float> pi(static_cast<std::size_t>(m));
        for (Index i = 0; i < m; ++i) {
            const Index bi = basis.basic_indices[static_cast<std::size_t>(i)];
            pi[static_cast<std::size_t>(i)] = obj_sign * model.obj[static_cast<std::size_t>(bi)];
        }
        factorizer.btran(pi);

        // 3. Dual ratio pricing
        Index q = -1;
        Float min_theta = std::numeric_limits<Float>::infinity();

        for (Index j = 0; j < n; ++j) {
            if (basis.col_status[static_cast<std::size_t>(j)] == BasisStatus::Basic) {
                continue;
            }

            // d_{pi, j} = d_pi^T A_j
            Float d_pi_j = 0.0;
            const Index col_start = model.A.col_ptrs[static_cast<std::size_t>(j)];
            const Index col_end = model.A.col_ptrs[static_cast<std::size_t>(j) + 1];
            for (Index k = col_start; k < col_end; ++k) {
                const auto kk = static_cast<std::size_t>(k);
                d_pi_j += d_pi[static_cast<std::size_t>(model.A.row_indices[kk])] * model.A.values[kk];
            }

            // Only consider d_{pi, j} < 0
            if (d_pi_j < -math::kDefaultPivotTol) {
                // Compute r_j = c_j - pi^T A_j
                Float r_j = obj_sign * model.obj[static_cast<std::size_t>(j)];
                for (Index k = col_start; k < col_end; ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    r_j -= pi[static_cast<std::size_t>(model.A.row_indices[kk])] * model.A.values[kk];
                }

                // Numerical safety: r_j should be >= 0 in dual feasible state.
                if (r_j < 0.0) {
                    r_j = 0.0;
                }

                const Float theta_j = r_j / std::abs(d_pi_j);
                if (theta_j < min_theta) {
                    min_theta = theta_j;
                    q = j;
                }
            }
        }

        if (q == -1) {
            // Dual unbounded -> Primal infeasible
            return SimplexStatus::Infeasible;
        }

        // 4. FTRAN: solve B d = A_q
        std::vector<Float> Aq(static_cast<std::size_t>(m), 0.0);
        {
            const Index col_start = model.A.col_ptrs[static_cast<std::size_t>(q)];
            const Index col_end = model.A.col_ptrs[static_cast<std::size_t>(q) + 1];
            for (Index k = col_start; k < col_end; ++k) {
                const auto kk = static_cast<std::size_t>(k);
                Aq[static_cast<std::size_t>(model.A.row_indices[kk])] = model.A.values[kk];
            }
        }
        std::vector<Float> d = Aq;
        factorizer.ftran(d);

        // 5. Update primal variables: x_B <- x_B - theta d
        // The primal step size theta_P is x_{B,p} / d_p. 
        // Note: mathematical derivation sets d_p = d_{pi, q} < 0.
        const Float d_p = d[static_cast<std::size_t>(p)];
        const Float theta_primal = x[static_cast<std::size_t>(basis.basic_indices[static_cast<std::size_t>(p)])] / d_p;

        for (Index i = 0; i < m; ++i) {
            x[static_cast<std::size_t>(basis.basic_indices[static_cast<std::size_t>(i)])] -= theta_primal * d[static_cast<std::size_t>(i)];
        }

        // 6. Basis update
        const Index leaving_var = basis.basic_indices[static_cast<std::size_t>(p)];
        
        x[static_cast<std::size_t>(leaving_var)] = 0.0;
        basis.col_status[static_cast<std::size_t>(leaving_var)] = BasisStatus::AtLower;

        basis.basic_indices[static_cast<std::size_t>(p)] = q;
        basis.col_status[static_cast<std::size_t>(q)] = BasisStatus::Basic;
        x[static_cast<std::size_t>(q)] = theta_primal;

        // 7. Factorization update (Forrest-Tomlin)
        factorizer.update(p, q, Aq);

        // 8. Residual monitoring (Step 6.2)
        if ((iter + 1) % 50 == 0) {
            std::vector<Float> x_B(static_cast<std::size_t>(m));
            for (Index i = 0; i < m; ++i) {
                x_B[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(basis.basic_indices[static_cast<std::size_t>(i)])];
            }
            if (factorizer.needs_refactorization(x_B, model.rhs)) {
                factorizer.factorize(model.A, basis);
            }
        }
    }

    return SimplexStatus::IterationLimit;
}

} // namespace simplex
} // namespace sankhya

