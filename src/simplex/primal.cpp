/**
 * @file primal.cpp
 * @brief L0/L1/L2 CPU Primal Simplex (Phase II) implementation
 *
 * Revised primal simplex with layered pricing and ratio test:
 *
 *   Step 4.2 (L0): correctness-first Phase II, no Phase I.
 *   Step 5.1 (L1): Bland fallback after >50 consecutive degenerate pivots.
 *   Step 5.2 (L2): Devex pricing (s_j = r_j² / gamma_j) and
 *                  Harris two-pass ratio test.
 *
 * Selection hierarchy (normal → fallback):
 *   Entering:  Devex (L2)  →  Bland (L1, on stall)
 *   Leaving:   Harris two-pass (L2)  →  Bland tie-break (L1, on stall)
 */

#include "primal.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

namespace sankhya {
namespace simplex {

SimplexStatus primal_simplex_phase2(
    const core::Model& model,
    Basis& basis,
    std::vector<Float>& x,
    numerics::BasisFactorization& factorizer
) {
    const Index m = model.A.rows;
    const Index n = model.A.cols;
    const auto nn = static_cast<std::size_t>(n);

    // Objective sign: internally reduce to minimization.
    // Minimize c^T x  ⟹  obj_sign = +1
    // Maximize c^T x  ⟹  minimize (-c)^T x  ⟹  obj_sign = -1
    const Float obj_sign =
        (model.sense == OptimizationSense::Minimize) ? 1.0 : -1.0;

    // Engineering safety: explicit iteration limit to prevent unbounded loops.
    static constexpr Index kMaxIterations = 10000;

    // Step 5.1: Bland fallback on detected degenerate stalling.
    // Track consecutive degenerate pivots (theta < kDefaultFeasibilityTol).
    // After more than 50 consecutive degenerate pivots, switch to Bland's
    // rule for entering-variable selection and leaving-variable tie-breaking.
    static constexpr Index kBlandActivationThreshold = 50;
    Index degenerate_pivots = 0;
    bool use_bland = false;

    // -----------------------------------------------------------------------
    // Step 5.2: Devex reference framework (gamma_j for j = 0..n-1)
    // -----------------------------------------------------------------------
    // Engineering decision: initialize all Devex reference weights to 1.0.
    // This is the standard "reset" convention for an approximate Devex
    // framework (see Devex pricing literature: Harris, 1973). With
    // gamma_j = 1 the initial Devex score s_j = r_j² / 1.0 = r_j², which
    // degenerates to squared-reduced-cost pricing — a reasonable warm-start
    // that differentiates from plain Dantzig while the weights converge.
    //
    // This is an engineering decision; the source specifies the score
    // formula s_j = r_j² / gamma_j and "maintain reference framework γ"
    // but does not mandate a specific initial value.
    std::vector<Float> gamma(nn, 1.0);

    // -----------------------------------------------------------------------
    // Validate initial Phase-II feasibility
    // -----------------------------------------------------------------------
    for (Index i = 0; i < m; ++i) {
        const Float xb = x[static_cast<std::size_t>(
            basis.basic_indices[static_cast<std::size_t>(i)])];
        if (xb < -math::kDefaultFeasibilityTol) {
            return SimplexStatus::Infeasible;
        }
    }

    // -----------------------------------------------------------------------
    // Step 0 — Initialize factorization from supplied model and basis
    // -----------------------------------------------------------------------
    factorizer.factorize(model.A, basis);

    // -----------------------------------------------------------------------
    // Main simplex loop
    // -----------------------------------------------------------------------
    for (Index iter = 0; iter < kMaxIterations; ++iter) {

        // -------------------------------------------------------------------
        // Step 1 — BTRAN: form c_B and solve B^T π = c_B
        // -------------------------------------------------------------------
        std::vector<Float> pi(static_cast<std::size_t>(m));
        for (Index i = 0; i < m; ++i) {
            const auto bi = static_cast<std::size_t>(
                basis.basic_indices[static_cast<std::size_t>(i)]);
            pi[static_cast<std::size_t>(i)] = obj_sign * model.obj[bi];
        }
        factorizer.btran(pi);

        // -------------------------------------------------------------------
        // Step 2 — Reduced costs for nonbasic variables
        // Step 3 — Pricing: Devex, Bland, depending on use_bland
        // -------------------------------------------------------------------
        Index q = -1;

        if (use_bland) {
            // ----- L1 Bland entering rule (Step 5.1) -----
            // Select the smallest global variable index j among all nonbasic
            // variables with r_j < -kDefaultFeasibilityTol.
            for (Index j = 0; j < n; ++j) {
                if (basis.col_status[static_cast<std::size_t>(j)] ==
                    BasisStatus::Basic) {
                    continue;
                }
                Float rc = obj_sign * model.obj[static_cast<std::size_t>(j)];
                const Index col_start =
                    model.A.col_ptrs[static_cast<std::size_t>(j)];
                const Index col_end =
                    model.A.col_ptrs[static_cast<std::size_t>(j) + 1];
                for (Index k = col_start; k < col_end; ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    rc -= model.A.values[kk] *
                          pi[static_cast<std::size_t>(
                              model.A.row_indices[kk])];
                }
                if (rc < -math::kDefaultFeasibilityTol) {
                    q = j;
                    break;  // smallest index first
                }
            }
        } else {
            // ----- L2 Devex pricing (Step 5.2) -----
            // Score: s_j = r_j² / gamma_j
            // Select q = argmax s_j among nonbasic j with r_j < -eps.
            Float best_score = 0.0;

            for (Index j = 0; j < n; ++j) {
                if (basis.col_status[static_cast<std::size_t>(j)] ==
                    BasisStatus::Basic) {
                    continue;
                }
                Float rc = obj_sign * model.obj[static_cast<std::size_t>(j)];
                const Index col_start =
                    model.A.col_ptrs[static_cast<std::size_t>(j)];
                const Index col_end =
                    model.A.col_ptrs[static_cast<std::size_t>(j) + 1];
                for (Index k = col_start; k < col_end; ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    rc -= model.A.values[kk] *
                          pi[static_cast<std::size_t>(
                              model.A.row_indices[kk])];
                }
                if (rc < -math::kDefaultFeasibilityTol) {
                    // Numerical safety: ensure gamma_j is positive.
                    // Engineering decision: if gamma_j has become
                    // non-positive due to floating-point drift, clamp
                    // it to 1.0 (the initial reference weight).
                    Float gj = gamma[static_cast<std::size_t>(j)];
                    if (gj <= 0.0) {
                        gj = 1.0;
                        gamma[static_cast<std::size_t>(j)] = 1.0;
                    }
                    const Float score = (rc * rc) / gj;
                    if (score > best_score) {
                        best_score = score;
                        q = j;
                    }
                }
            }
        }

        // Optimality: no eligible entering variable
        if (q == -1) {
            return SimplexStatus::Optimal;
        }

        // -------------------------------------------------------------------
        // Step 4 — FTRAN: extract A_q and solve B d = A_q
        // -------------------------------------------------------------------
        std::vector<Float> Aq(static_cast<std::size_t>(m), 0.0);
        {
            const Index col_start =
                model.A.col_ptrs[static_cast<std::size_t>(q)];
            const Index col_end =
                model.A.col_ptrs[static_cast<std::size_t>(q) + 1];
            for (Index k = col_start; k < col_end; ++k) {
                const auto kk = static_cast<std::size_t>(k);
                Aq[static_cast<std::size_t>(model.A.row_indices[kk])] =
                    model.A.values[kk];
            }
        }
        std::vector<Float> d = Aq;
        factorizer.ftran(d);

        // -------------------------------------------------------------------
        // Step 5 — Ratio test: Harris two-pass (L2) or Bland (L1)
        // -------------------------------------------------------------------
        Index p = -1;
        Float theta = std::numeric_limits<Float>::infinity();

        if (use_bland) {
            // ----- L1 Bland leaving rule (Step 5.1) -----
            // 1. Determine the minimum eligible ratio.
            // 2. Among all rows whose ratio ties that minimum (within
            //    exact floating-point equality), select the smallest row index.
            for (Index i = 0; i < m; ++i) {
                if (d[static_cast<std::size_t>(i)] > math::kDefaultPivotTol) {
                    const Float xb = x[static_cast<std::size_t>(
                        basis.basic_indices[static_cast<std::size_t>(i)])];
                    const Float ratio =
                        xb / d[static_cast<std::size_t>(i)];
                    if (ratio < theta) {
                        theta = ratio;
                        p = i;
                    } else if (ratio == theta && (p == -1 || i < p)) {
                        // Bland tie-break: smallest row index wins
                        p = i;
                    }
                }
            }
        } else {
            // ----- L2 Harris two-pass ratio test (Step 5.2) -----
            //
            // Pass 1: Determine the maximum acceptable step Delta.
            //   Delta = min over eligible rows i of:
            //     (x_B[i] + kDefaultFeasibilityTol) / d[i]
            //   where d[i] > kDefaultPivotTol.
            //
            // This allows a slight feasibility relaxation (by eps) so
            // that numerically near-degenerate rows do not unnecessarily
            // restrict the step, while remaining within feasibility tol.

            Float Delta = std::numeric_limits<Float>::infinity();
            bool has_eligible = false;

            for (Index i = 0; i < m; ++i) {
                const auto ii = static_cast<std::size_t>(i);
                if (d[ii] > math::kDefaultPivotTol) {
                    has_eligible = true;
                    const Float xb = x[static_cast<std::size_t>(
                        basis.basic_indices[ii])];
                    const Float relaxed_ratio =
                        (xb + math::kDefaultFeasibilityTol) / d[ii];
                    if (relaxed_ratio < Delta) {
                        Delta = relaxed_ratio;
                    }
                }
            }

            if (!has_eligible) {
                // No positive d_i at all ⟹ unbounded
                // (p remains -1, handled below)
            } else {
                // Pass 2: Among rows satisfying x_B[i] / d[i] <= Delta
                // (with d[i] > kDefaultPivotTol), select p = argmax |d[i]|.
                // This picks the numerically strongest pivot element
                // within the Harris-safe region.
                Float best_abs_d = -1.0;

                for (Index i = 0; i < m; ++i) {
                    const auto ii = static_cast<std::size_t>(i);
                    if (d[ii] > math::kDefaultPivotTol) {
                        const Float xb = x[static_cast<std::size_t>(
                            basis.basic_indices[ii])];
                        const Float ratio = xb / d[ii];
                        if (ratio <= Delta) {
                            const Float abs_d = std::abs(d[ii]);
                            if (abs_d > best_abs_d) {
                                best_abs_d = abs_d;
                                p = i;
                            }
                        }
                    }
                }

                // If a leaving row was selected, compute actual theta
                // using the standard ratio (not the relaxed one).
                if (p != -1) {
                    const Float xb_p = x[static_cast<std::size_t>(
                        basis.basic_indices[static_cast<std::size_t>(p)])];
                    theta = xb_p / d[static_cast<std::size_t>(p)];
                    // Clamp theta to non-negative. A tiny negative theta
                    // can arise from numerical noise when x_B[p] ≈ 0.
                    if (theta < 0.0) {
                        theta = 0.0;
                    }
                }
            }
        }

        // Unbounded: no eligible leaving row
        if (p == -1) {
            return SimplexStatus::Unbounded;
        }

        // -------------------------------------------------------------------
        // Step 5.1 — Track consecutive degenerate pivots
        // -------------------------------------------------------------------
        if (theta < math::kDefaultFeasibilityTol) {
            ++degenerate_pivots;
            if (!use_bland && degenerate_pivots > kBlandActivationThreshold) {
                // Activate Bland fallback mode. Once activated, Bland mode
                // remains active for the rest of this solve.
                // Implementation decision: no switch-back to Devex/Harris
                // after activation, consistent with the source specification.
                use_bland = true;
            }
        } else {
            // Non-degenerate pivot breaks the consecutive stall sequence.
            degenerate_pivots = 0;
        }

        // -------------------------------------------------------------------
        // Step 6 — Primal variable update: x_B <- x_B - θ d
        // -------------------------------------------------------------------
        for (Index i = 0; i < m; ++i) {
            x[static_cast<std::size_t>(
                basis.basic_indices[static_cast<std::size_t>(i)])] -=
                theta * d[static_cast<std::size_t>(i)];
        }

        // -------------------------------------------------------------------
        // Step 7 — Basis status update
        // -------------------------------------------------------------------
        const Index leaving_var =
            basis.basic_indices[static_cast<std::size_t>(p)];

        // Leaving variable goes to its lower bound (nonbasic)
        x[static_cast<std::size_t>(leaving_var)] = 0.0;
        basis.col_status[static_cast<std::size_t>(leaving_var)] =
            BasisStatus::AtLower;

        // Entering variable occupies the leaving row's basis slot
        basis.basic_indices[static_cast<std::size_t>(p)] = q;
        basis.col_status[static_cast<std::size_t>(q)] = BasisStatus::Basic;
        x[static_cast<std::size_t>(q)] = theta;

        // -------------------------------------------------------------------
        // Step 8 — Factorization update
        // -------------------------------------------------------------------
        factorizer.update(p, q, Aq);

        // -------------------------------------------------------------------
        // Step 5.2 — Devex reference weight update
        // -------------------------------------------------------------------
        // Engineering decision: use the approximate Devex update formula.
        // After a pivot where variable q enters and variable leaving_var
        // leaves:
        //
        //   For each nonbasic variable j (j != q):
        //     gamma_j := max(kDefaultPivotTol,
        //                    gamma_j - 2 * (d_tilde_j / d_p) * alpha_j
        //                    + (d_tilde_j / d_p)^2 * gamma_q)
        //
        // where d_tilde_j = (column j of B^{-1} A)_p (the p-th component
        // of the basis representation of column j) and alpha_j is a
        // cross-term. Computing d_tilde_j for every nonbasic column is
        // expensive (requires an FTRAN per column) and is not feasible
        // with the current dense factorization interface.
        //
        // Therefore, we use the simplified Devex reference update from
        // Harris (1973): after a pivot, set the entering variable's
        // weight to 1 / ||d||² (where d is the FTRAN'd entering column),
        // and for all other nonbasic variables, scale their weights by a
        // pivot-dependent factor.
        //
        // Simplified update (engineering decision, not source-mandated):
        //   1. gamma[leaving_var] = gamma[q] (transfer weight to newly
        //      nonbasic leaving variable)
        //   2. gamma[q] is irrelevant (q is now basic)
        //   3. Periodically (every n pivots), reset all weights to 1.0
        //      to prevent drift. This is standard Devex practice.
        //
        // The weight for the newly nonbasic leaving_var inherits the
        // entering variable's reference weight, which provides a
        // reasonable approximation of its edge weight in the current
        // basis representation.
        {
            // Transfer the entering variable's Devex weight to the
            // leaving variable (which is now nonbasic).
            const auto lv = static_cast<std::size_t>(leaving_var);

            // Compute ||d||² for an improved entering-variable weight
            // estimate. This captures the steepness of the entering edge.
            Float d_sq = 0.0;
            for (Index i = 0; i < m; ++i) {
                const Float di = d[static_cast<std::size_t>(i)];
                d_sq += di * di;
            }

            // The leaving variable's new weight: use max(1, ||d||²)
            // as an approximation of its Devex reference in the new basis.
            gamma[lv] = (d_sq > 1.0) ? d_sq : 1.0;

            // Periodic Devex reset: every n iterations, reset all weights
            // to 1.0 to prevent numerical drift. This is a standard Devex
            // engineering practice (Harris, 1973).
            if ((iter + 1) % n == 0) {
                for (std::size_t j = 0; j < nn; ++j) {
                    gamma[j] = 1.0;
                }
            }
        }


        // -------------------------------------------------------------------
        // Step 6.2 — Residual monitoring and refactorization
        // -------------------------------------------------------------------
        if ((iter + 1) % 50 == 0) {
            std::vector<Float> x_B(static_cast<std::size_t>(m));
            for (Index i = 0; i < m; ++i) {
                x_B[static_cast<std::size_t>(i)] = 
                    x[static_cast<std::size_t>(basis.basic_indices[static_cast<std::size_t>(i)])];
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
