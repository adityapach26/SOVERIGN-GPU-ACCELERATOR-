#include <cstddef>
#include <limits>
#include <algorithm>
#include <utility>
/**
 * @file test_primal_simplex.cpp
 * @brief Unit tests for the L0 CPU Primal Simplex Phase II
 *
 * Step 4.2 tests: small 2-variable LP.
 * Step 5.1 tests: Beale cycling LP with Dantzig-only cycling behavior
 *                 and Bland fallback behavior.
 *
 * The test LP (in standard inequality form):
 *
 *     minimize   -3 x1 - 2 x2
 *     subject to   x1 + x2 <= 4
 *                  x1       <= 2
 *                  x1, x2   >= 0
 *
 * Equality form with slacks s1, s2:
 *
 *     minimize   c = [-3, -2, 0, 0]^T  (variables: x1, x2, s1, s2)
 *     subject to  [1 1 1 0] [x1]   [4]
 *                 [1 0 0 1] [x2] = [2]
 *                           [s1]
 *                           [s2]
 *                 x1, x2, s1, s2 >= 0
 *
 * Initial basis: {s1=col2, s2=col3}, B = I, x = [0, 0, 4, 2]
 * This basis is NOT optimal (reduced costs r_0 = -3, r_1 = -2 < 0).
 *
 * Hand-calculated pivot trace:
 *
 *   Iteration 1: enters x1 (col 0, r=-3), leaves s2 (row 1, θ=2)
 *       x = [2, 0, 2, 0], basis = {col2, col0}
 *
 *   Iteration 2: enters x2 (col 1, r=-2), leaves s1 (row 0, θ=2)
 *       x = [2, 2, 0, 0], basis = {col1, col0}
 *
 *   Iteration 3: all reduced costs >= 0 → OPTIMAL
 *
 * Optimal: x1=2, x2=2, s1=0, s2=0
 * Optimal objective: -3(2) + -2(2) = -10
 */

#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "simplex/primal.hpp"
#include "numerics/factorization.hpp"
#include "core/sparse_matrix.hpp"
#include "simplex/basis.hpp"

using sankhya::Float;
using sankhya::Index;
using sankhya::OptimizationSense;
using sankhya::VariableType;
using sankhya::core::CSCMatrix;
using sankhya::core::Model;
using sankhya::simplex::Basis;
using sankhya::simplex::BasisStatus;
using sankhya::simplex::SimplexStatus;
using sankhya::simplex::primal_simplex_phase2;

// ---------------------------------------------------------------------------
// Test-local NaiveDenseFactorization backend
// ---------------------------------------------------------------------------

namespace {

/**
 * @brief Minimal dense basis factorization — test-only backend
 *
 * Materializes the m×m basis into a dense row-major matrix and solves
 * systems via Gaussian elimination with partial pivoting.
 *
 * This class exists ONLY in this test translation unit and must NOT be
 * added to production source.
 */
class NaiveDenseFactorization : public sankhya::numerics::BasisFactorization {
public:
    void factorize(
        const CSCMatrix& A,
        const Basis& basis
    ) override {
        m_ = static_cast<Index>(basis.basic_indices.size());
        const auto mm = static_cast<std::size_t>(m_);

        // Materialize basis columns into a dense m×m row-major matrix.
        dense_.assign(mm * mm, 0.0);
        for (Index j = 0; j < m_; ++j) {
            const auto jj = static_cast<std::size_t>(j);
            const Index col = basis.basic_indices[jj];
            const Index start = A.col_ptrs[static_cast<std::size_t>(col)];
            const Index end   = A.col_ptrs[static_cast<std::size_t>(col) + 1];
            for (Index k = start; k < end; ++k) {
                const Index row = A.row_indices[static_cast<std::size_t>(k)];
                if (row < m_) {
                    dense_[static_cast<std::size_t>(row) * mm + jj] =
                        A.values[static_cast<std::size_t>(k)];
                }
            }
        }
    }

    void ftran(std::vector<Float>& rhs) override {
        solve_dense(dense_, rhs, /*transpose=*/false);
    }

    void btran(std::vector<Float>& rhs) override {
        solve_dense(dense_, rhs, /*transpose=*/true);
    }

    void update(
        Index leaving_row,
        Index /*entering_col*/,
        const std::vector<Float>& Aq
    ) override {
        // Replace the leaving_row-th column of the dense basis with Aq.
        const auto mm = static_cast<std::size_t>(m_);
        const auto lr = static_cast<std::size_t>(leaving_row);
        for (Index i = 0; i < m_; ++i) {
            dense_[static_cast<std::size_t>(i) * mm + lr] =
                Aq[static_cast<std::size_t>(i)];
        }
    }

private:
    Index m_ = 0;
    std::vector<Float> dense_;  // row-major m×m

    /**
     * @brief Solve a dense system via Gaussian elimination with partial pivoting
     *
     * If transpose is false, solves M x = rhs.
     * If transpose is true,  solves M^T x = rhs.
     * Overwrites rhs with the solution.
     */
    static void solve_dense(
        const std::vector<Float>& matrix,
        std::vector<Float>& rhs,
        bool transpose
    ) {
        const auto n = static_cast<Index>(rhs.size());
        if (n == 0) return;
        const auto nn = static_cast<std::size_t>(n);

        // Work on a copy so the stored matrix is preserved.
        std::vector<Float> M = matrix;

        // Transpose in place if needed.
        if (transpose) {
            for (Index i = 0; i < n; ++i) {
                for (Index j = i + 1; j < n; ++j) {
                    const auto ii = static_cast<std::size_t>(i);
                    const auto jj = static_cast<std::size_t>(j);
                    std::swap(M[ii * nn + jj], M[jj * nn + ii]);
                }
            }
        }

        // Forward elimination with partial pivoting.
        for (Index col = 0; col < n; ++col) {
            const auto cc = static_cast<std::size_t>(col);

            // Find pivot.
            Index pivot_row = col;
            Float max_val = std::abs(M[cc * nn + cc]);
            for (Index row = col + 1; row < n; ++row) {
                const Float v = std::abs(
                    M[static_cast<std::size_t>(row) * nn + cc]);
                if (v > max_val) {
                    max_val = v;
                    pivot_row = row;
                }
            }
            if (max_val < 1e-15) {
                throw std::runtime_error(
                    "Singular matrix in NaiveDenseFactorization");
            }

            // Swap rows.
            if (pivot_row != col) {
                for (Index j = col; j < n; ++j) {
                    const auto jj = static_cast<std::size_t>(j);
                    std::swap(
                        M[cc * nn + jj],
                        M[static_cast<std::size_t>(pivot_row) * nn + jj]);
                }
                std::swap(rhs[cc],
                          rhs[static_cast<std::size_t>(pivot_row)]);
            }

            // Eliminate below.
            const Float diag = M[cc * nn + cc];
            for (Index row = col + 1; row < n; ++row) {
                const auto rr = static_cast<std::size_t>(row);
                const Float factor = M[rr * nn + cc] / diag;
                for (Index j = col + 1; j < n; ++j) {
                    M[rr * nn + static_cast<std::size_t>(j)] -=
                        factor * M[cc * nn + static_cast<std::size_t>(j)];
                }
                M[rr * nn + cc] = 0.0;
                rhs[rr] -= factor * rhs[cc];
            }
        }

        // Back substitution.
        for (Index row = n - 1; row >= 0; --row) {
            const auto rr = static_cast<std::size_t>(row);
            Float sum = rhs[rr];
            for (Index j = row + 1; j < n; ++j) {
                sum -= M[rr * nn + static_cast<std::size_t>(j)] *
                       rhs[static_cast<std::size_t>(j)];
            }
            rhs[rr] = sum / M[rr * nn + rr];
        }
    }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test-helper: build the standard 2-variable LP in equality form
// ---------------------------------------------------------------------------

namespace {

/**
 * Build the test LP model:
 *
 *   minimize  -3 x1 - 2 x2
 *   s.t.       x1 + x2 + s1      = 4
 *              x1           + s2  = 2
 *              x1, x2, s1, s2 >= 0
 *
 * Variables: x1(0), x2(1), s1(2), s2(3)
 * A is 2×4, b = [4, 2]
 */
Model make_test_model() {
    Model m;
    m.sense = OptimizationSense::Minimize;

    // Variables: x1, x2, s1, s2 (all continuous, lb=0, ub=+inf)
    m.add_variable(-3.0);  // x1
    m.add_variable(-2.0);  // x2
    m.add_variable( 0.0);  // s1
    m.add_variable( 0.0);  // s2

    // Constraints in equality form:
    //   row 0: x1 + x2 + s1 = 4
    m.add_constraint({0, 1, 2}, {1.0, 1.0, 1.0}, 4.0);
    //   row 1: x1 + s2 = 2
    m.add_constraint({0, 3}, {1.0, 1.0}, 2.0);

    m.finalize();
    return m;
}

/**
 * Build the initial slack-variable basis: {s1=col2, s2=col3}.
 * 4 variables total, 2 basic.
 */
Basis make_slack_basis() {
    Basis b;
    b.col_status = {
        BasisStatus::AtLower,  // x1 nonbasic
        BasisStatus::AtLower,  // x2 nonbasic
        BasisStatus::Basic,    // s1 basic
        BasisStatus::Basic     // s2 basic
    };
    b.basic_indices = {2, 3};  // row 0 → s1, row 1 → s2
    return b;
}

/**
 * Build the initial feasible solution for the slack basis.
 * x1=0, x2=0, s1=4, s2=2
 */
std::vector<Float> make_initial_x() {
    return {0.0, 0.0, 4.0, 2.0};
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Step 5.1 test helpers: Beale cycling LP
// ---------------------------------------------------------------------------

namespace {

/**
 * Build the classic Beale cycling LP in equality form.
 *
 * The original Beale problem (with explicit slack variables) is:
 *
 *   minimize  -3/4 x4 + 20 x5 - 1/2 x6 + 6 x7
 *
 *   s.t.  1/4 x4 -  8 x5 -   x6 + 9 x7 + x1         = 0
 *         1/2 x4 - 12 x5 - 1/2 x6 + 12 x7     + x2   = 0
 *                                    x6             + x3 = 1
 *
 *         x1, x2, x3, x4, x5, x6, x7 >= 0
 *
 * Variables indexed 0..6:
 *   x1=col0, x2=col1, x3=col2, x4=col3, x5=col4, x6=col5, x7=col6
 *
 * Objective: c = [0, 0, 0, -3/4, 20, -1/2, 6]
 *
 * A is 3×7, b = [0, 0, 1]
 *
 * Initial basis: {x1=col0, x2=col1, x3=col2}, B = I
 * Initial solution: x = [0, 0, 1, 0, 0, 0, 0]
 *
 * This is Phase-II feasible (x_B = [0, 0, 1] >= 0).
 *
 * Under Dantzig pricing, the simplex cycles indefinitely because the
 * first two basic variables (x1, x2) have value 0, causing degenerate
 * pivots with theta = 0.
 *
 * Known optimum (reachable by Bland's rule):
 *   x6 = 1, all others = 0 → objective = -1/2
 *   Or equivalently: any feasible point where x6=1 and x3=0 gives obj = -1/2
 *   since the constraint x6 + x3 = 1 forces x6 = 1 when x3 = 0.
 *   With x4=x5=x7=0: row 0 gives x1 = x6 = 1, row 1 gives x2 = x6/2 = 1/2.
 *   So optimal: x1=1, x2=1/2, x3=0, x4=0, x5=0, x6=1, x7=0, obj = -1/2.
 */
Model make_chvatal_model() {
    Model m;
    m.sense = OptimizationSense::Minimize;

    m.add_variable(-10.0);  // x1 (col 0)
    m.add_variable( 57.0);  // x2 (col 1)
    m.add_variable(  9.0);  // x3 (col 2)
    m.add_variable( 24.0);  // x4 (col 3)
    m.add_variable(  0.0);  // x5 (col 4) - slack row 0
    m.add_variable(  0.0);  // x6 (col 5) - slack row 1
    m.add_variable(  0.0);  // x7 (col 6) - slack row 2

    // Row 0: 0.5 x1 - 5.5 x2 - 2.5 x3 + 9 x4 + x5 = 0
    m.add_constraint({0, 1, 2, 3, 4},
                     {0.5, -5.5, -2.5, 9.0, 1.0}, 0.0);

    // Row 1: 0.5 x1 - 1.5 x2 - 0.5 x3 + x4 + x6 = 0
    m.add_constraint({0, 1, 2, 3, 5},
                     {0.5, -1.5, -0.5, 1.0, 1.0}, 0.0);

    // Row 2: x1 + x7 = 1
    m.add_constraint({0, 6}, {1.0, 1.0}, 1.0);

    m.finalize();
    return m;
}

Basis make_chvatal_basis() {
    Basis b;
    b.col_status = {
        BasisStatus::AtLower,  // x1
        BasisStatus::AtLower,  // x2
        BasisStatus::AtLower,  // x3
        BasisStatus::AtLower,  // x4
        BasisStatus::Basic,    // x5
        BasisStatus::Basic,    // x6
        BasisStatus::Basic     // x7
    };
    b.basic_indices = {4, 5, 6};  // row 0 -> col 4, row 1 -> col 5, row 2 -> col 6
    return b;
}

std::vector<Float> make_chvatal_initial_x() {
    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
}

/**
 * Test-local wrapper that disables the Bland fallback by lowering the
 * iteration limit to a value where cycling is detected (Dantzig cycles
 * within a few iterations, so 10000 is more than enough for the cycle
 * to repeat many times without Bland ever activating if it weren't for
 * the counter > 50 threshold).
 *
 * Engineering decision: To demonstrate Dantzig-only cycling behavior,
 * we rely on the production code's existing iteration limit of 10000.
 * The Beale LP under Dantzig pricing cycles with a cycle length of ~6
 * degenerate pivots. After >50 consecutive degenerate pivots, Bland
 * activates and terminates.
 *
 * To test the Dantzig-only case, we need to prevent Bland from activating.
 * We do this by calling the production function (which has Bland enabled)
 * on a DIFFERENT formulation that does NOT cycle but merely stalls,
 * OR by accepting that the production code with Bland will solve it.
 *
 * Actually, since the production code includes Bland fallback with
 * threshold > 50, and the Beale cycle length is ~6, Bland activates
 * around iteration 51-57 (after 50+ consecutive degenerate pivots).
 * Therefore, the production code WILL solve it (Bland kicks in).
 *
 * For the Dantzig-only test (demonstrating cycling/stalling), we use
 * a test-local copy of the simplex that does NOT have Bland fallback.
 * This is a minimal function used ONLY for testing; it is NOT production.
 */
SimplexStatus primal_simplex_dantzig_only(
    const Model& model,
    Basis& basis,
    std::vector<Float>& x,
    sankhya::numerics::BasisFactorization& factorizer
) {
    const Index m = model.A.rows;
    const Index n = model.A.cols;
    const Float obj_sign =
        (model.sense == OptimizationSense::Minimize) ? 1.0 : -1.0;

    // Same iteration limit as production code
    static constexpr Index kMaxIterations = 10000;

    factorizer.factorize(model.A, basis);

    for (Index iter = 0; iter < kMaxIterations; ++iter) {
        // BTRAN
        std::vector<Float> pi(static_cast<std::size_t>(m));
        for (Index i = 0; i < m; ++i) {
            const auto bi = static_cast<std::size_t>(
                basis.basic_indices[static_cast<std::size_t>(i)]);
            pi[static_cast<std::size_t>(i)] = obj_sign * model.obj[bi];
        }
        factorizer.btran(pi);

        // Dantzig pricing only (no Bland fallback)
        Index q = -1;
        Float min_rc = -sankhya::math::kDefaultFeasibilityTol;
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
            if (rc < min_rc) {
                min_rc = rc;
                q = j;
            }
        }

        if (q == -1) return SimplexStatus::Optimal;

        // FTRAN
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

        // Standard ratio test (no Bland tie-break)
        Index p = -1;
        Float theta = std::numeric_limits<Float>::infinity();
        for (Index i = 0; i < m; ++i) {
            if (d[static_cast<std::size_t>(i)] >
                sankhya::math::kDefaultPivotTol) {
                const Float xb = x[static_cast<std::size_t>(
                    basis.basic_indices[static_cast<std::size_t>(i)])];
                const Float ratio =
                    xb / d[static_cast<std::size_t>(i)];
                if (ratio < theta) {
                    theta = ratio;
                    p = i;
                }
            }
        }

        if (p == -1) return SimplexStatus::Unbounded;

        // Primal update
        for (Index i = 0; i < m; ++i) {
            x[static_cast<std::size_t>(
                basis.basic_indices[static_cast<std::size_t>(i)])] -=
                theta * d[static_cast<std::size_t>(i)];
        }
        const Index leaving_var =
            basis.basic_indices[static_cast<std::size_t>(p)];
        x[static_cast<std::size_t>(leaving_var)] = 0.0;
        basis.col_status[static_cast<std::size_t>(leaving_var)] =
            BasisStatus::AtLower;
        basis.basic_indices[static_cast<std::size_t>(p)] = q;
        basis.col_status[static_cast<std::size_t>(q)] = BasisStatus::Basic;
        x[static_cast<std::size_t>(q)] = theta;

        factorizer.update(p, q, Aq);
    }

    return SimplexStatus::IterationLimit;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static constexpr Float kTol = sankhya::math::kDefaultFeasibilityTol;

TEST_CASE("PrimalSimplex - 2-var LP reaches optimal status",
          "[primal_simplex][simplex]") {
    Model model = make_test_model();
    Basis basis = make_slack_basis();
    std::vector<Float> x = make_initial_x();
    NaiveDenseFactorization factorizer;

    SimplexStatus status = primal_simplex_phase2(model, basis, x, factorizer);

    REQUIRE(status == SimplexStatus::Optimal);
}

TEST_CASE("PrimalSimplex - 2-var LP optimal objective value",
          "[primal_simplex][simplex]") {
    Model model = make_test_model();
    Basis basis = make_slack_basis();
    std::vector<Float> x = make_initial_x();
    NaiveDenseFactorization factorizer;

    SimplexStatus status = primal_simplex_phase2(model, basis, x, factorizer);
    REQUIRE(status == SimplexStatus::Optimal);

    // Compute objective: c^T x = -3*x1 + -2*x2 + 0*s1 + 0*s2
    Float obj = 0.0;
    for (std::size_t j = 0; j < model.obj.size(); ++j) {
        obj += model.obj[j] * x[j];
    }

    // Hand-calculated optimum: -3(2) + -2(2) = -10
    REQUIRE(std::abs(obj - (-10.0)) < kTol);
}

TEST_CASE("PrimalSimplex - 2-var LP optimal variable values",
          "[primal_simplex][simplex]") {
    Model model = make_test_model();
    Basis basis = make_slack_basis();
    std::vector<Float> x = make_initial_x();
    NaiveDenseFactorization factorizer;

    SimplexStatus status = primal_simplex_phase2(model, basis, x, factorizer);
    REQUIRE(status == SimplexStatus::Optimal);

    // Expected optimal solution: x1=2, x2=2, s1=0, s2=0
    REQUIRE(std::abs(x[0] - 2.0) < kTol);
    REQUIRE(std::abs(x[1] - 2.0) < kTol);
    REQUIRE(std::abs(x[2] - 0.0) < kTol);
    REQUIRE(std::abs(x[3] - 0.0) < kTol);
}

TEST_CASE("PrimalSimplex - initial basis is not already optimal",
          "[primal_simplex][simplex]") {
    // Verify that the initial basis has negative reduced costs,
    // ensuring the simplex loop must actually pivot.
    Model model = make_test_model();
    Basis basis = make_slack_basis();
    std::vector<Float> x = make_initial_x();

    // At the slack basis, c_B = [0, 0], B = I, π = [0, 0]
    // r_0 = c_0 - A_0^T π = -3 - 0 = -3  (negative → not optimal)
    // r_1 = c_1 - A_1^T π = -2 - 0 = -2  (negative → not optimal)
    // So at least one pivot is required.

    NaiveDenseFactorization factorizer;
    SimplexStatus status = primal_simplex_phase2(model, basis, x, factorizer);
    REQUIRE(status == SimplexStatus::Optimal);

    // The basis must have changed from the initial slack basis.
    // After optimization, x1 and x2 should be basic:
    REQUIRE(basis.col_status[0] == BasisStatus::Basic);   // x1 basic
    REQUIRE(basis.col_status[1] == BasisStatus::Basic);   // x2 basic
    REQUIRE(basis.col_status[2] == BasisStatus::AtLower); // s1 nonbasic
    REQUIRE(basis.col_status[3] == BasisStatus::AtLower); // s2 nonbasic
}

TEST_CASE("PrimalSimplex - unbounded LP returns Unbounded",
          "[primal_simplex][simplex]") {
    // Construct an unbounded LP:
    //   minimize  -x1
    //   s.t.      x1 - s1 = 0     (i.e. x1 = s1)
    //             x1, s1 >= 0
    //
    // This is unbounded because x1 can grow without limit.
    //
    // A = [1, -1], b = [0]
    // Variables: x1(0), s1(1)
    // Initial basis: {s1=col1}, x = [0, 0]
    //
    // Actually, with x1 nonbasic at 0 and s1 basic at 0,
    // reduced cost of x1 = -1 - [1]*[0] = -1 < 0 → enters.
    // FTRAN: B d = A_0 → [-1] d = [1] → d = [-1]
    // d[0] = -1 < pivot_tol → no eligible leaving row → Unbounded.

    Model m;
    m.sense = OptimizationSense::Minimize;
    m.add_variable(-1.0);  // x1
    m.add_variable( 0.0);  // s1
    m.add_constraint({0, 1}, {1.0, -1.0}, 0.0);
    m.finalize();

    Basis basis;
    basis.col_status = {BasisStatus::AtLower, BasisStatus::Basic};
    basis.basic_indices = {1};

    std::vector<Float> x = {0.0, 0.0};

    NaiveDenseFactorization factorizer;
    SimplexStatus status = primal_simplex_phase2(m, basis, x, factorizer);

    REQUIRE(status == SimplexStatus::Unbounded);
}

// ---------------------------------------------------------------------------
// Step 5.1 — Chvatal cycling tests
// ---------------------------------------------------------------------------

TEST_CASE("PrimalSimplex - Chvatal LP Dantzig-only cycles to IterationLimit",
          "[primal_simplex][simplex][bland]") {
    // The classic Chvátal cycling LP under pure Dantzig pricing (no Bland
    // fallback) cycles indefinitely among degenerate pivots under smallest-index
    // tie-breaking. This test uses a test-local Dantzig-only simplex (identical
    // to the pre-5.1 Step 4.2 algorithm) to confirm that the iteration limit is reached.
    //
    // Engineering decision: a test-local copy of the Dantzig-only simplex
    // is used here to demonstrate the cycling behavior that motivates
    // Bland's rule. This avoids adding a public API flag to the production
    // code. The test-local function is defined only in this translation unit.
    //
    // The production code's iteration limit is 10000. The Chvátal cycling
    // LP under Dantzig pricing cycles with a cycle length of 6
    // degenerate pivots, so 10000 iterations is sufficient to confirm
    // the solver never finds optimality.

    Model model = make_chvatal_model();
    Basis basis = make_chvatal_basis();
    std::vector<Float> x = make_chvatal_initial_x();
    NaiveDenseFactorization factorizer;

    SimplexStatus status =
        primal_simplex_dantzig_only(model, basis, x, factorizer);

    REQUIRE(status == SimplexStatus::IterationLimit);
}

TEST_CASE("PrimalSimplex - Chvatal LP with Bland fallback reaches Optimal",
          "[primal_simplex][simplex][bland]") {
    // The production primal_simplex_phase2 includes the Step 5.1 Bland
    // fallback. After >50 consecutive degenerate pivots on the Chvátal LP,
    // Bland mode activates, breaks the cycle, and finds the optimum.
    //
    // Known Chvátal optimum:
    //   x = [1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0]
    //   Optimal objective value: -1.0

    Model model = make_chvatal_model();
    Basis basis = make_chvatal_basis();
    std::vector<Float> x = make_chvatal_initial_x();
    NaiveDenseFactorization factorizer;

    SimplexStatus status = primal_simplex_phase2(model, basis, x, factorizer);

    REQUIRE(status == SimplexStatus::Optimal);

    // Verify optimal objective value = -1.0
    Float obj = 0.0;
    for (std::size_t j = 0; j < model.obj.size(); ++j) {
        obj += model.obj[j] * x[j];
    }
    REQUIRE(std::abs(obj - (-1.0)) < kTol);

    // Verify key optimal variable values
    REQUIRE(std::abs(x[0] - 1.0) < kTol);   // x1 = 1
    REQUIRE(std::abs(x[1] - 0.0) < kTol);   // x2 = 0
    REQUIRE(std::abs(x[2] - 1.0) < kTol);   // x3 = 1
    REQUIRE(std::abs(x[3] - 0.0) < kTol);   // x4 = 0
    REQUIRE(std::abs(x[4] - 2.0) < kTol);   // x5 = 2
    REQUIRE(std::abs(x[5] - 0.0) < kTol);   // x6 = 0
    REQUIRE(std::abs(x[6] - 0.0) < kTol);   // x7 = 0
}
