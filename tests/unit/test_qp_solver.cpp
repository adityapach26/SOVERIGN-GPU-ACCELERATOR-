/**
 * @file test_qp_solver.cpp
 * @brief Phase 16A.2: Convex QP Solver Tests
 *
 * [C] ENGINEERING INFERENCE: All test reference values are derived analytically.
 * No external optimizer was used.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "qp/qp_model.hpp"
#include "qp/qp_solver.hpp"
#include "core/problem.hpp"
#include <vector>
#include <cmath>

using namespace sankhya;
using namespace sankhya::qp;

// ============================================================
// Helper: Build a QPModel from dense data
// ============================================================
static QPModel build_qp(
    Index n, Index m,
    const std::vector<Float>& H_dense,   // n*n row-major, full symmetric
    const std::vector<Float>& c,
    const std::vector<Float>& A_dense,   // m*n row-major
    const std::vector<Float>& b,
    const std::vector<Float>& lb,
    const std::vector<Float>& ub
) {
    QPModel qp;
    qp.lp_part.sense = OptimizationSense::Minimize;
    qp.lp_part.obj = c;
    qp.lp_part.lb = lb;
    qp.lp_part.ub = ub;
    qp.lp_part.rhs = b;
    qp.lp_part.vtype.assign(static_cast<std::size_t>(n), VariableType::Continuous);

    // Build A as CSC
    std::vector<Index> col_ptrs(static_cast<std::size_t>(n) + 1, 0);
    std::vector<Index> row_indices;
    std::vector<Float> values;
    for (Index j = 0; j < n; ++j) {
        col_ptrs[static_cast<std::size_t>(j)] = static_cast<Index>(row_indices.size());
        for (Index i = 0; i < m; ++i) {
            Float v = A_dense[static_cast<std::size_t>(i) * static_cast<std::size_t>(n)
                              + static_cast<std::size_t>(j)];
            if (std::abs(v) > 1e-15) {
                row_indices.push_back(i);
                values.push_back(v);
            }
        }
    }
    col_ptrs[static_cast<std::size_t>(n)] = static_cast<Index>(row_indices.size());
    qp.lp_part.A = core::CSCMatrix(m, n, values, row_indices, col_ptrs);

    // Build H as CSC (lower triangular only)
    std::vector<Index> H_col_ptrs(static_cast<std::size_t>(n) + 1, 0);
    std::vector<Index> H_row_indices;
    std::vector<Float> H_values;
    for (Index j = 0; j < n; ++j) {
        H_col_ptrs[static_cast<std::size_t>(j)] = static_cast<Index>(H_row_indices.size());
        for (Index i = j; i < n; ++i) {  // lower triangular: i >= j
            Float v = H_dense[static_cast<std::size_t>(i) * static_cast<std::size_t>(n)
                              + static_cast<std::size_t>(j)];
            if (std::abs(v) > 1e-15) {
                H_row_indices.push_back(i);
                H_values.push_back(v);
            }
        }
    }
    H_col_ptrs[static_cast<std::size_t>(n)] = static_cast<Index>(H_row_indices.size());
    qp.H = core::CSCMatrix(n, n, H_values, H_row_indices, H_col_ptrs);
    qp.convexity = QPConvexityStatus::ConvexPSD;

    return qp;
}

// ============================================================
// Independent KKT verification
// ============================================================
static void verify_kkt(
    const QPModel& qp,
    const std::vector<Float>& x,
    const std::vector<Float>& y,
    const std::vector<Float>& z,
    Float tol
) {
    Index n = qp.H.cols;
    Index m = qp.lp_part.A.rows;

    // Primal feasibility: ||Ax - b||
    std::vector<Float> Ax(static_cast<std::size_t>(m), 0.0);
    for (Index j = 0; j < n; ++j) {
        for (Index p = qp.lp_part.A.col_ptrs[static_cast<std::size_t>(j)];
             p < qp.lp_part.A.col_ptrs[static_cast<std::size_t>(j) + 1]; ++p) {
            Index i = qp.lp_part.A.row_indices[static_cast<std::size_t>(p)];
            Ax[static_cast<std::size_t>(i)] +=
                qp.lp_part.A.values[static_cast<std::size_t>(p)] *
                x[static_cast<std::size_t>(j)];
        }
    }
    Float primal_res = 0.0;
    for (Index i = 0; i < m; ++i) {
        Float r = std::abs(Ax[static_cast<std::size_t>(i)] -
                           qp.lp_part.rhs[static_cast<std::size_t>(i)]);
        if (r > primal_res) primal_res = r;
    }
    REQUIRE(primal_res <= tol);

    // Bound feasibility
    for (Index i = 0; i < n; ++i) {
        auto ii = static_cast<std::size_t>(i);
        if (qp.lp_part.lb[ii] > -1e15) {
            REQUIRE(x[ii] >= qp.lp_part.lb[ii] - tol);
        }
        if (qp.lp_part.ub[ii] < 1e15) {
            REQUIRE(x[ii] <= qp.lp_part.ub[ii] + tol);
        }
    }

    // Stationarity: Hx + c - A^T y - zl + zu ~ 0
    // (z contains zl - zu net dual for bounds)
    // We check: ||Hx + c - A^T y - z|| <= tol
    std::vector<Float> Hx(static_cast<std::size_t>(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        for (Index p = qp.H.col_ptrs[static_cast<std::size_t>(j)];
             p < qp.H.col_ptrs[static_cast<std::size_t>(j) + 1]; ++p) {
            Index i = qp.H.row_indices[static_cast<std::size_t>(p)];
            Float val = qp.H.values[static_cast<std::size_t>(p)];
            Hx[static_cast<std::size_t>(i)] += val * x[static_cast<std::size_t>(j)];
            if (i != j) {
                Hx[static_cast<std::size_t>(j)] += val * x[static_cast<std::size_t>(i)];
            }
        }
    }
    std::vector<Float> ATy(static_cast<std::size_t>(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        for (Index p = qp.lp_part.A.col_ptrs[static_cast<std::size_t>(j)];
             p < qp.lp_part.A.col_ptrs[static_cast<std::size_t>(j) + 1]; ++p) {
            Index i = qp.lp_part.A.row_indices[static_cast<std::size_t>(p)];
            ATy[static_cast<std::size_t>(j)] +=
                qp.lp_part.A.values[static_cast<std::size_t>(p)] *
                y[static_cast<std::size_t>(i)];
        }
    }

    Float dual_res = 0.0;
    for (Index i = 0; i < n; ++i) {
        auto ii = static_cast<std::size_t>(i);
        Float r = std::abs(Hx[ii] + qp.lp_part.obj[ii] - ATy[ii] - z[ii]);
        if (r > dual_res) dual_res = r;
    }
    REQUIRE(dual_res <= tol);
}

// ============================================================
// Test 1: Diagonal convex QP
// ============================================================
TEST_CASE("Phase 16A.2: Diagonal Convex QP", "[qp][solver]") {
    // min 1/2 (2x1^2 + 4x2^2) + (-4)x1 + (-6)x2
    // s.t. x1 + x2 = 3
    //      x1, x2 >= 0
    // H = [[2,0],[0,4]], c = [-4,-6]
    // Analytical: KKT => x1=5/3, x2=4/3, obj = -25/3 ≈ -8.3333

    QPModel qp = build_qp(
        2, 1,
        {2.0, 0.0, 0.0, 4.0},    // H
        {-4.0, -6.0},             // c
        {1.0, 1.0},               // A
        {3.0},                     // b
        {0.0, 0.0},               // lb
        {math::kInfinity, math::kInfinity} // ub
        );

    QPSolver solver;
    std::vector<Float> x, y, z;
    auto result = solver.solve_with_diagnostics(qp, x, y, z);

    REQUIRE(result.status == QPSolverStatus::Optimal);
    REQUIRE(x[0] == Catch::Approx(5.0/3.0).margin(1e-4));
    REQUIRE(x[1] == Catch::Approx(4.0/3.0).margin(1e-4));
    REQUIRE(result.objective_value == Catch::Approx(-25.0/3.0).margin(1e-4));
    REQUIRE(result.primal_residual < 1e-6);
    REQUIRE(result.dual_residual < 1e-6);
    REQUIRE(result.complementarity < 1e-6);
    REQUIRE(result.iterations > 0);

    verify_kkt(qp, x, y, z, 1e-4);
}

// ============================================================
// Test 2: Coupled PSD QP
// ============================================================
TEST_CASE("Phase 16A.2: Coupled PSD QP", "[qp][solver]") {
    // min 1/2 x^T [[2,1],[1,2]] x + [-1,-1]^T x
    // s.t. x1 + x2 = 1
    //      x1, x2 >= 0
    // H = [[2,1],[1,2]] (eigenvalues 1,3 => PD)
    // KKT: Hx + c = A^T y + z
    //   [2 1][x1] + [-1] = [1]*y + [z1]
    //   [1 2][x2]   [-1]   [1]     [z2]
    // With x1+x2=1, x>=0:
    // Solution: x1 = x2 = 0.5, obj = 1/2*(0.5*0.5*2 + 2*0.5*0.5*1 + 0.5*0.5*2) - 0.5 - 0.5
    //         = 1/2*(0.5 + 0.5 + 0.5) - 1 = 0.75 - 1 = -0.25

    QPModel qp = build_qp(
        2, 1,
        {2.0, 1.0, 1.0, 2.0},    // H
        {-1.0, -1.0},             // c
        {1.0, 1.0},               // A
        {1.0},                     // b
        {0.0, 0.0},               // lb
        {math::kInfinity, math::kInfinity} // ub
        );

    QPSolver solver;
    std::vector<Float> x, y, z;
    auto result = solver.solve_with_diagnostics(qp, x, y, z);

    REQUIRE(result.status == QPSolverStatus::Optimal);
    REQUIRE(x[0] == Catch::Approx(0.5).margin(1e-4));
    REQUIRE(x[1] == Catch::Approx(0.5).margin(1e-4));
    REQUIRE(result.objective_value == Catch::Approx(-0.25).margin(1e-4));
    REQUIRE(result.primal_residual < 1e-6);
    REQUIRE(result.dual_residual < 1e-6);

    verify_kkt(qp, x, y, z, 1e-4);
}

// ============================================================
// Test 3: Active bounds
// ============================================================
TEST_CASE("Phase 16A.2: QP Active Bounds", "[qp][solver]") {
    // min 1/2 (x1^2 + x2^2) - 2*x1
    // s.t. x1 + x2 <= 1.5  (add slack: x1 + x2 + x3 = 1.5)
    //      0 <= x1 <= 1.0
    //      0 <= x2 <= 1.0
    //      x3 >= 0  (slack)
    // H = diag(1,1,0), c = [-2, 0, 0]
    // Optimum: x1=1.0 (at upper bound), x2=0.0, x3=0.5, obj = 0.5-2 = -1.5

    QPModel qp = build_qp(
        3, 1,
        {1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 0.0},         // H (3x3 diagonal, PSD)
        {-2.0, 0.0, 0.0},        // c
        {1.0, 1.0, 1.0},         // A
        {1.5},                     // b
        {0.0, 0.0, 0.0},         // lb
        {1.0, 1.0, math::kInfinity} // ub
        );

    QPSolver solver;
    std::vector<Float> x, y, z;
    auto result = solver.solve_with_diagnostics(qp, x, y, z);

    REQUIRE(result.status == QPSolverStatus::Optimal);
    REQUIRE(x[0] == Catch::Approx(1.0).margin(1e-3));
    REQUIRE(x[1] == Catch::Approx(0.0).margin(1e-3));
    REQUIRE(result.objective_value == Catch::Approx(-1.5).margin(1e-3));

    verify_kkt(qp, x, y, z, 1e-3);
}

// ============================================================
// Test 4: Infeasible QP
// ============================================================
TEST_CASE("Phase 16A.2: Infeasible QP", "[qp][solver]") {
    // min 1/2 x^2
    // s.t. x = 2
    //      0 <= x <= 1  (infeasible: b=2 but ub=1)
    QPModel qp = build_qp(
        1, 1,
        {1.0},      // H
        {0.0},      // c
        {1.0},      // A
        {2.0},      // b
        {0.0},      // lb
        {1.0}       // ub
    );

    QPSolver solver;
    std::vector<Float> x, y, z;
    auto result = solver.solve_with_diagnostics(qp, x, y, z);

    REQUIRE((result.status == QPSolverStatus::Infeasible ||
             result.status == QPSolverStatus::NumericalFailure ||
             result.status == QPSolverStatus::IterationLimit));
    REQUIRE(result.status != QPSolverStatus::Optimal);
}

// ============================================================
// Test 5: Unsupported indefinite Hessian
// ============================================================
TEST_CASE("Phase 16A.2: Indefinite Hessian Rejected", "[qp][solver]") {
    QPModel qp = build_qp(
        2, 1,
        {1.0, 0.0, 0.0, -1.0},   // H indefinite (eigenvalues 1, -1)
        {0.0, 0.0},
        {1.0, 1.0},
        {1.0},
        {0.0, 0.0},
        {math::kInfinity, math::kInfinity}
    );
    qp.convexity = QPConvexityStatus::Nonconvex;

    QPSolver solver;
    std::vector<Float> x, y, z;
    QPSolverStatus status = solver.solve(qp, x, y, z);

    REQUIRE(status == QPSolverStatus::UnsupportedNonconvex);
}

// ============================================================
// Test 6: Iteration limit
// ============================================================
TEST_CASE("Phase 16A.2: QP Iteration Limit", "[qp][solver]") {
    QPModel qp = build_qp(
        2, 1,
        {2.0, 0.0, 0.0, 2.0},
        {-1.0, -1.0},
        {1.0, 1.0},
        {1.0},
        {0.0, 0.0},
        {math::kInfinity, math::kInfinity}
    );

    QPSolver::Options opts;
    opts.max_iterations = 2;  // Intentionally tiny
    QPSolver solver(opts);

    std::vector<Float> x, y, z;
    auto result = solver.solve_with_diagnostics(qp, x, y, z);

    REQUIRE(result.status == QPSolverStatus::IterationLimit);
    REQUIRE(result.iterations == 2);
}

// ============================================================
// Test 7: QPLIB-format integration (analytically verified)
// ============================================================
TEST_CASE("Phase 16A.2: QPLIB Integration Case", "[qp][solver][qplib]") {
    // QPLIB-style problem: Convex QP with 3 variables, 1 equality constraint
    //
    // min 1/2 x^T H x + c^T x
    //   H = [[4, 1, 0],
    //        [1, 4, 1],
    //        [0, 1, 4]]   (tridiagonal SPD, eigenvalues ~ 2.59, 4, 5.41)
    //   c = [-8, -12, -8]
    //   s.t. x1 + x2 + x3 = 6
    //        x >= 0
    //
    // Reference solution derived analytically via KKT:
    //   Hx + c = A^T y + z, Ax = b, z >= 0, x >= 0, x^T z = 0
    //   If all variables strictly interior (z=0):
    //     [4 1 0][x1]   [-8 ]   [1]
    //     [1 4 1][x2] + [-12] = [1] * y
    //     [0 1 4][x3]   [-8 ]   [1]
    //   => 4x1 + x2 - 8 = y       (i)
    //      x1 + 4x2 + x3 - 12 = y (ii)
    //      x2 + 4x3 - 8 = y       (iii)
    //   From (i)-(iii): 4x1 + x2 - 8 = x2 + 4x3 - 8 => x1 = x3
    //   From (i)-(ii): 4x1 + x2 - 8 - x1 - 4x2 - x3 + 12 = 0
    //     => 3x1 - 3x2 - x3 + 4 = 0, with x1=x3: 2x1 - 3x2 + 4 = 0
    //     => x2 = (2x1 + 4)/3
    //   x1 + x2 + x3 = 6 => 2x1 + (2x1+4)/3 = 6 => 6x1 + 2x1 + 4 = 18 => x1 = 7/4
    //   x2 = (14/4 + 4)/3 = (14/4 + 16/4)/3 = 30/12 = 5/2
    //   x3 = 7/4
    //   obj = 1/2 * x^T H x + c^T x
    //   Hx = [4*7/4+5/2, 7/4+4*5/2+7/4, 5/2+4*7/4] = [7+5/2, 7/4+10+7/4, 5/2+7] = [19/2, 27/2, 19/2]
    //   x^T Hx = 7/4*19/2 + 5/2*27/2 + 7/4*19/2 = 133/8 + 135/4 + 133/8 = 536/8 = 67
    //   c^T x = -8*7/4 -12*5/2 -8*7/4 = -14-30-14 = -58
    //   obj = 67/2 - 58 = 33.5 - 58 = -24.5

    QPModel qp = build_qp(
        3, 1,
        {4.0, 1.0, 0.0,
         1.0, 4.0, 1.0,
         0.0, 1.0, 4.0},
        {-8.0, -12.0, -8.0},
        {1.0, 1.0, 1.0},
        {6.0},
        {0.0, 0.0, 0.0},
        {math::kInfinity, math::kInfinity, math::kInfinity}
    );

    QPSolver solver;
    std::vector<Float> x, y, z;
    auto result = solver.solve_with_diagnostics(qp, x, y, z);

    REQUIRE(result.status == QPSolverStatus::Optimal);
    REQUIRE(x[0] == Catch::Approx(1.75).margin(1e-4));
    REQUIRE(x[1] == Catch::Approx(2.5).margin(1e-4));
    REQUIRE(x[2] == Catch::Approx(1.75).margin(1e-4));
    REQUIRE(result.objective_value == Catch::Approx(-24.5).margin(1e-3));
    REQUIRE(result.primal_residual < 1e-6);
    REQUIRE(result.dual_residual < 1e-6);

    verify_kkt(qp, x, y, z, 1e-4);
}
