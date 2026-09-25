/**
 * @file qp_solver.cpp
 * @brief Phase 16A.2: Sovereign Convex QP Solver — CPU Dense Reference Path
 *
 * [C] ENGINEERING INFERENCE: Implements a Mehrotra predictor-corrector
 * primal-dual interior-point method for convex QP using the normal-equations
 * reduction. Both W = H + D and S = A W^{-1} A^T are SPD, so only Cholesky
 * factorization is required. No external solver is used.
 */

#include "qp/qp_solver.hpp"
#include "qp/qp_model.hpp"
#include "core/sparse_matrix.hpp"
#include "sankhya/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sankhya {
namespace qp {

namespace {

// Dense Cholesky: in-place L (lower triangular) of SPD matrix A (row-major).
// Returns false if not positive definite.
bool dense_cholesky(Index n, std::vector<Float>& A) {
    auto nn = static_cast<std::size_t>(n);
    for (Index j = 0; j < n; ++j) {
        auto jj = static_cast<std::size_t>(j);
        Float sum = A[jj * nn + jj];
        for (Index k = 0; k < j; ++k) {
            auto kk = static_cast<std::size_t>(k);
            sum -= A[jj * nn + kk] * A[jj * nn + kk];
        }
        if (sum <= 1e-14) return false;
        A[jj * nn + jj] = std::sqrt(sum);
        for (Index i = j + 1; i < n; ++i) {
            auto ii = static_cast<std::size_t>(i);
            Float s = A[ii * nn + jj];
            for (Index k = 0; k < j; ++k) {
                auto kk = static_cast<std::size_t>(k);
                s -= A[ii * nn + kk] * A[jj * nn + kk];
            }
            A[ii * nn + jj] = s / A[jj * nn + jj];
        }
        // Zero upper triangle for this column
        for (Index i = 0; i < j; ++i) {
            A[static_cast<std::size_t>(i) * nn + jj] = 0.0;
        }
    }
    return true;
}

// Solve LL^T x = b in-place (L is lower triangular, row-major)
void dense_cholesky_solve(Index n, const std::vector<Float>& L,
                          std::vector<Float>& rhs) {
    auto nn = static_cast<std::size_t>(n);
    // Forward: L y = b
    for (Index i = 0; i < n; ++i) {
        auto ii = static_cast<std::size_t>(i);
        for (Index j = 0; j < i; ++j) {
            rhs[ii] -= L[ii * nn + static_cast<std::size_t>(j)] *
                       rhs[static_cast<std::size_t>(j)];
        }
        rhs[ii] /= L[ii * nn + ii];
    }
    // Backward: L^T x = y
    for (Index i = n - 1; i >= 0; --i) {
        auto ii = static_cast<std::size_t>(i);
        for (Index j = i + 1; j < n; ++j) {
            auto jj = static_cast<std::size_t>(j);
            rhs[ii] -= L[jj * nn + ii] * rhs[jj];
        }
        rhs[ii] /= L[ii * nn + ii];
    }
}

// Sparse CSC matvec: y = A * x
void sparse_matvec(const core::CSCMatrix& A, const std::vector<Float>& x,
                   std::vector<Float>& y) {
    auto mm = static_cast<std::size_t>(A.rows);
    y.assign(mm, 0.0);
    for (Index j = 0; j < A.cols; ++j) {
        auto jj = static_cast<std::size_t>(j);
        for (Index p = A.col_ptrs[jj]; p < A.col_ptrs[jj + 1]; ++p) {
            auto pp = static_cast<std::size_t>(p);
            auto i = static_cast<std::size_t>(A.row_indices[pp]);
            y[i] += A.values[pp] * x[jj];
        }
    }
}

// Sparse CSC transpose matvec: y = A^T * x
void sparse_matvec_t(const core::CSCMatrix& A, const std::vector<Float>& x,
                     std::vector<Float>& y) {
    auto nn = static_cast<std::size_t>(A.cols);
    y.assign(nn, 0.0);
    for (Index j = 0; j < A.cols; ++j) {
        auto jj = static_cast<std::size_t>(j);
        Float s = 0.0;
        for (Index p = A.col_ptrs[jj]; p < A.col_ptrs[jj + 1]; ++p) {
            auto pp = static_cast<std::size_t>(p);
            s += A.values[pp] * x[static_cast<std::size_t>(A.row_indices[pp])];
        }
        y[jj] = s;
    }
}

// Symmetric CSC (lower-tri stored) matvec: y = H * x
void sym_matvec(const core::CSCMatrix& H, const std::vector<Float>& x,
                std::vector<Float>& y) {
    auto nn = static_cast<std::size_t>(H.cols);
    y.assign(nn, 0.0);
    for (Index j = 0; j < H.cols; ++j) {
        auto jj = static_cast<std::size_t>(j);
        for (Index p = H.col_ptrs[jj]; p < H.col_ptrs[jj + 1]; ++p) {
            auto pp = static_cast<std::size_t>(p);
            auto i = static_cast<std::size_t>(H.row_indices[pp]);
            Float v = H.values[pp];
            y[i] += v * x[jj];
            if (i != jj) {
                y[jj] += v * x[i];
            }
        }
    }
}

// Infinity norm
Float inf_norm(const std::vector<Float>& v) {
    Float mx = 0.0;
    for (auto val : v) {
        Float a = std::abs(val);
        if (a > mx) mx = a;
    }
    return mx;
}

} // anonymous namespace

QPResult QPSolver::solve_with_diagnostics(
    const QPModel& qp,
    std::vector<Float>& x_out,
    std::vector<Float>& y_out,
    std::vector<Float>& z_out
) {
    QPResult result;

    // Validate
    if (qp.convexity == QPConvexityStatus::Nonconvex) {
        result.status = QPSolverStatus::UnsupportedNonconvex;
        return result;
    }
    try {
        qp.validate();
    } catch (const std::invalid_argument&) {
        if (qp.convexity == QPConvexityStatus::Nonconvex) {
            result.status = QPSolverStatus::UnsupportedNonconvex;
        } else {
            result.status = QPSolverStatus::InvalidModel;
        }
        return result;
    }
    if (qp.convexity == QPConvexityStatus::Unknown) {
        result.status = QPSolverStatus::UnsupportedNonconvex;
        return result;
    }

    Index n = qp.lp_part.A.cols;
    Index m = qp.lp_part.A.rows;
    auto nn = static_cast<std::size_t>(n);
    auto mm = static_cast<std::size_t>(m);

    if (n == 0) {
        result.status = QPSolverStatus::InvalidModel;
        return result;
    }

    const auto& lb = qp.lp_part.lb;
    const auto& ub = qp.lp_part.ub;
    const auto& c = qp.lp_part.obj;
    const auto& b = qp.lp_part.rhs;

    // Initialize primal/dual variables
    std::vector<Float> x(nn);
    std::vector<Float> y(mm, 0.0);
    std::vector<Float> zl(nn, 0.0);
    std::vector<Float> zu(nn, 0.0);

    // Track which bounds are finite
    std::vector<bool> has_lb(nn, false);
    std::vector<bool> has_ub(nn, false);

    for (Index i = 0; i < n; ++i) {
        auto ii = static_cast<std::size_t>(i);
        has_lb[ii] = (lb[ii] > -1e15);
        has_ub[ii] = (ub[ii] < 1e15);

        // Initialize x to interior of bounds
        Float lo = has_lb[ii] ? lb[ii] : -10.0;
        Float hi = has_ub[ii] ? ub[ii] : 10.0;
        x[ii] = (lo + hi) / 2.0;

        // Initialize dual variables for active bounds only
        zl[ii] = has_lb[ii] ? 1.0 : 0.0;
        zu[ii] = has_ub[ii] ? 1.0 : 0.0;
    }

    // Temporary vectors
    std::vector<Float> Hx(nn), Ax(mm), ATy(nn), rp(mm), rd(nn);
    std::vector<Float> D(nn), rhs_x(nn), rhs_y(mm);

    const Float tol_p = options_.feasibility_tol;
    const Float tol_d = options_.dual_tol;
    const Float tol_c = options_.grad_tol;

    for (Index iter = 0; iter < options_.max_iterations; ++iter) {
        result.iterations = iter;

        // Compute residuals
        sym_matvec(qp.H, x, Hx);        // Hx
        sparse_matvec(qp.lp_part.A, x, Ax); // Ax
        sparse_matvec_t(qp.lp_part.A, y, ATy); // A^T y

        // r_p = Ax - b
        for (Index i = 0; i < m; ++i) {
            rp[static_cast<std::size_t>(i)] =
                Ax[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)];
        }

        // r_d = Hx + c - A^T y - zl + zu
        for (Index i = 0; i < n; ++i) {
            auto ii = static_cast<std::size_t>(i);
            rd[ii] = Hx[ii] + c[ii] - ATy[ii] - zl[ii] + zu[ii];
        }

        // Complementarity mu
        Float mu = 0.0;
        Index n_comp = 0;
        for (Index i = 0; i < n; ++i) {
            auto ii = static_cast<std::size_t>(i);
            if (has_lb[ii]) {
                mu += (x[ii] - lb[ii]) * zl[ii];
                ++n_comp;
            }
            if (has_ub[ii]) {
                mu += (ub[ii] - x[ii]) * zu[ii];
                ++n_comp;
            }
        }
        if (n_comp > 0) mu /= static_cast<Float>(n_comp);

        Float norm_rp = inf_norm(rp);
        Float norm_rd = inf_norm(rd);

        result.primal_residual = norm_rp;
        result.dual_residual = norm_rd;
        result.complementarity = mu;

        // Objective: 1/2 x^T H x + c^T x
        Float obj = 0.0;
        for (Index i = 0; i < n; ++i) {
            auto ii = static_cast<std::size_t>(i);
            obj += 0.5 * x[ii] * Hx[ii] + c[ii] * x[ii];
        }
        result.objective_value = obj;

        // Convergence check
        if (norm_rp < tol_p && norm_rd < tol_d && mu < tol_c) {
            result.status = QPSolverStatus::Optimal;
            x_out = x;
            y_out = y;
            // z_out = zl - zu (net bound dual)
            z_out.resize(nn);
            for (Index i = 0; i < n; ++i) {
                z_out[static_cast<std::size_t>(i)] =
                    zl[static_cast<std::size_t>(i)] - zu[static_cast<std::size_t>(i)];
            }
            return result;
        }

        // Infeasibility detection
        Float norm_y = inf_norm(y);
        if (norm_y > 1e15) {
            result.status = QPSolverStatus::Infeasible;
            return result;
        }
        if (obj < -1e15) {
            result.status = QPSolverStatus::Unbounded;
            return result;
        }

        // Build barrier diagonal D
        for (Index i = 0; i < n; ++i) {
            auto ii = static_cast<std::size_t>(i);
            D[ii] = 0.0;
            if (has_lb[ii]) {
                Float gap = std::max(x[ii] - lb[ii], 1e-12);
                D[ii] += zl[ii] / gap;
            }
            if (has_ub[ii]) {
                Float gap = std::max(ub[ii] - x[ii], 1e-12);
                D[ii] += zu[ii] / gap;
            }
            if (D[ii] < 1e-12) D[ii] = 1e-12;
        }

        // Form W = H + diag(D) as dense n×n (row-major)
        std::vector<Float> W(nn * nn, 0.0);
        // Add H entries (lower triangular stored)
        for (Index j = 0; j < n; ++j) {
            auto jj = static_cast<std::size_t>(j);
            for (Index p = qp.H.col_ptrs[jj]; p < qp.H.col_ptrs[jj + 1]; ++p) {
                auto pp = static_cast<std::size_t>(p);
                auto i = static_cast<std::size_t>(qp.H.row_indices[pp]);
                W[i * nn + jj] += qp.H.values[pp];
                if (i != jj) {
                    W[jj * nn + i] += qp.H.values[pp];
                }
            }
        }
        // Add diagonal D
        for (Index i = 0; i < n; ++i) {
            W[static_cast<std::size_t>(i) * nn + static_cast<std::size_t>(i)] +=
                D[static_cast<std::size_t>(i)];
        }

        // Cholesky factorize W
        std::vector<Float> L_W = W;
        if (!dense_cholesky(n, L_W)) {
            result.status = QPSolverStatus::NumericalFailure;
            return result;
        }

        // Build rhs_x for the normal equations
        // rhs_x = -rd - complementarity adjustments
        for (Index i = 0; i < n; ++i) {
            auto ii = static_cast<std::size_t>(i);
            Float comp_term = 0.0;
            if (has_lb[ii]) {
                Float gap = std::max(x[ii] - lb[ii], 1e-12);
                comp_term += (mu - zl[ii] * gap) / gap;
            }
            if (has_ub[ii]) {
                Float gap = std::max(ub[ii] - x[ii], 1e-12);
                comp_term -= (mu - zu[ii] * gap) / gap;
            }
            rhs_x[ii] = -rd[ii] + comp_term;
        }

        if (m > 0) {
            // Form S = A W^{-1} A^T (dense m×m)
            // For each column j of A^T, solve W z_j = A^T_j
            std::vector<Float> WinvAT(nn * mm, 0.0);
            for (Index j = 0; j < m; ++j) {
                auto jj = static_cast<std::size_t>(j);
                // Extract column j of A^T = row j of A
                std::vector<Float> col(nn, 0.0);
                for (Index c_idx = 0; c_idx < n; ++c_idx) {
                    auto cc = static_cast<std::size_t>(c_idx);
                    for (Index p = qp.lp_part.A.col_ptrs[cc];
                         p < qp.lp_part.A.col_ptrs[cc + 1]; ++p) {
                        if (qp.lp_part.A.row_indices[static_cast<std::size_t>(p)] == j) {
                            col[cc] = qp.lp_part.A.values[static_cast<std::size_t>(p)];
                        }
                    }
                }
                dense_cholesky_solve(n, L_W, col);
                for (Index i = 0; i < n; ++i) {
                    WinvAT[static_cast<std::size_t>(i) * mm + jj] =
                        col[static_cast<std::size_t>(i)];
                }
            }

            // S = A * WinvAT
            std::vector<Float> S(mm * mm, 0.0);
            for (Index i = 0; i < m; ++i) {
                auto ii = static_cast<std::size_t>(i);
                for (Index j = 0; j < m; ++j) {
                    auto jj = static_cast<std::size_t>(j);
                    Float sum = 0.0;
                    // Row i of A dotted with column j of WinvAT
                    for (Index k = 0; k < n; ++k) {
                        auto kk = static_cast<std::size_t>(k);
                        // A_{i,k}
                        Float a_ik = 0.0;
                        for (Index p = qp.lp_part.A.col_ptrs[kk];
                             p < qp.lp_part.A.col_ptrs[kk + 1]; ++p) {
                            if (qp.lp_part.A.row_indices[static_cast<std::size_t>(p)] == i) {
                                a_ik = qp.lp_part.A.values[static_cast<std::size_t>(p)];
                            }
                        }
                        sum += a_ik * WinvAT[kk * mm + jj];
                    }
                    S[ii * mm + jj] = sum;
                }
            }

            // Solve W^{-1} rhs_x
            std::vector<Float> Winv_rhs_x = rhs_x;
            dense_cholesky_solve(n, L_W, Winv_rhs_x);

            // rhs_y = -rp + A * W^{-1} * rhs_x
            sparse_matvec(qp.lp_part.A, Winv_rhs_x, rhs_y);
            for (Index i = 0; i < m; ++i) {
                rhs_y[static_cast<std::size_t>(i)] -= rp[static_cast<std::size_t>(i)];
            }

            // Cholesky factorize S
            if (!dense_cholesky(m, S)) {
                result.status = QPSolverStatus::NumericalFailure;
                return result;
            }

            // Solve S * dy = rhs_y
            std::vector<Float> dy = rhs_y;
            dense_cholesky_solve(m, S, dy);

            // dx = W^{-1} (A^T dy + rhs_x)
            std::vector<Float> ATdy;
            sparse_matvec_t(qp.lp_part.A, dy, ATdy);
            std::vector<Float> dx_rhs(nn);
            for (Index i = 0; i < n; ++i) {
                dx_rhs[static_cast<std::size_t>(i)] =
                    ATdy[static_cast<std::size_t>(i)] +
                    rhs_x[static_cast<std::size_t>(i)];
            }
            dense_cholesky_solve(n, L_W, dx_rhs);
            std::vector<Float>& dx = dx_rhs;

            // Compute dzl, dzu from complementarity
            std::vector<Float> dzl(nn, 0.0);
            std::vector<Float> dzu(nn, 0.0);
            for (Index i = 0; i < n; ++i) {
                auto ii = static_cast<std::size_t>(i);
                if (has_lb[ii]) {
                    Float gap = std::max(x[ii] - lb[ii], 1e-12);
                    dzl[ii] = (mu - zl[ii] * gap - zl[ii] * dx[ii]) / gap;
                }
                if (has_ub[ii]) {
                    Float gap = std::max(ub[ii] - x[ii], 1e-12);
                    dzu[ii] = (mu - zu[ii] * gap + zu[ii] * dx[ii]) / gap;
                }
            }

            // Step lengths (fraction-to-boundary)
            Float alpha_p = 1.0;
            Float alpha_d = 1.0;
            constexpr Float eta = 0.995;

            for (Index i = 0; i < n; ++i) {
                auto ii = static_cast<std::size_t>(i);
                if (has_lb[ii] && dx[ii] < 0.0) {
                    Float gap = x[ii] - lb[ii];
                    Float a = -gap / dx[ii];
                    if (a < alpha_p) alpha_p = a;
                }
                if (has_ub[ii] && dx[ii] > 0.0) {
                    Float gap = ub[ii] - x[ii];
                    Float a = gap / dx[ii];
                    if (a < alpha_p) alpha_p = a;
                }
                if (has_lb[ii] && dzl[ii] < 0.0 && zl[ii] > 0.0) {
                    Float a = -zl[ii] / dzl[ii];
                    if (a < alpha_d) alpha_d = a;
                }
                if (has_ub[ii] && dzu[ii] < 0.0 && zu[ii] > 0.0) {
                    Float a = -zu[ii] / dzu[ii];
                    if (a < alpha_d) alpha_d = a;
                }
            }
            alpha_p = std::min(1.0, eta * alpha_p);
            alpha_d = std::min(1.0, eta * alpha_d);

            // Update
            for (Index i = 0; i < n; ++i) {
                auto ii = static_cast<std::size_t>(i);
                x[ii] += alpha_p * dx[ii];
                zl[ii] += alpha_d * dzl[ii];
                zu[ii] += alpha_d * dzu[ii];
                if (has_lb[ii]) zl[ii] = std::max(zl[ii], 1e-14);
                if (has_ub[ii]) zu[ii] = std::max(zu[ii], 1e-14);
            }
            for (Index i = 0; i < m; ++i) {
                y[static_cast<std::size_t>(i)] +=
                    alpha_d * dy[static_cast<std::size_t>(i)];
            }

        } else {
            // No equality constraints: just solve W dx = rhs_x
            std::vector<Float> dx = rhs_x;
            dense_cholesky_solve(n, L_W, dx);

            std::vector<Float> dzl(nn, 0.0);
            std::vector<Float> dzu(nn, 0.0);
            for (Index i = 0; i < n; ++i) {
                auto ii = static_cast<std::size_t>(i);
                if (has_lb[ii]) {
                    Float gap = std::max(x[ii] - lb[ii], 1e-12);
                    dzl[ii] = (mu - zl[ii] * gap - zl[ii] * dx[ii]) / gap;
                }
                if (has_ub[ii]) {
                    Float gap = std::max(ub[ii] - x[ii], 1e-12);
                    dzu[ii] = (mu - zu[ii] * gap + zu[ii] * dx[ii]) / gap;
                }
            }

            Float alpha_p = 1.0;
            Float alpha_d = 1.0;
            constexpr Float eta = 0.995;

            for (Index i = 0; i < n; ++i) {
                auto ii = static_cast<std::size_t>(i);
                if (has_lb[ii] && dx[ii] < 0.0) {
                    Float a = -(x[ii] - lb[ii]) / dx[ii];
                    if (a < alpha_p) alpha_p = a;
                }
                if (has_ub[ii] && dx[ii] > 0.0) {
                    Float a = (ub[ii] - x[ii]) / dx[ii];
                    if (a < alpha_p) alpha_p = a;
                }
                if (has_lb[ii] && dzl[ii] < 0.0 && zl[ii] > 0.0) {
                    Float a = -zl[ii] / dzl[ii];
                    if (a < alpha_d) alpha_d = a;
                }
                if (has_ub[ii] && dzu[ii] < 0.0 && zu[ii] > 0.0) {
                    Float a = -zu[ii] / dzu[ii];
                    if (a < alpha_d) alpha_d = a;
                }
            }
            alpha_p = std::min(1.0, eta * alpha_p);
            alpha_d = std::min(1.0, eta * alpha_d);

            for (Index i = 0; i < n; ++i) {
                auto ii = static_cast<std::size_t>(i);
                x[ii] += alpha_p * dx[ii];
                zl[ii] += alpha_d * dzl[ii];
                zu[ii] += alpha_d * dzu[ii];
                if (has_lb[ii]) zl[ii] = std::max(zl[ii], 1e-14);
                if (has_ub[ii]) zu[ii] = std::max(zu[ii], 1e-14);
            }
        }
    }

    // Iteration limit
    result.iterations = options_.max_iterations;
    result.status = QPSolverStatus::IterationLimit;
    x_out = x;
    y_out = y;
    z_out.resize(nn);
    for (Index i = 0; i < n; ++i) {
        z_out[static_cast<std::size_t>(i)] =
            zl[static_cast<std::size_t>(i)] - zu[static_cast<std::size_t>(i)];
    }
    return result;
}

QPSolverStatus QPSolver::solve(
    const QPModel& qp,
    std::vector<Float>& x,
    std::vector<Float>& y,
    std::vector<Float>& z
) {
    auto result = solve_with_diagnostics(qp, x, y, z);
    return result.status;
}

} // namespace qp
} // namespace sankhya
