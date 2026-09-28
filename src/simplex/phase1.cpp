/**
 * @file phase1.cpp
 * @brief Native CPU Phase-I LP solver implementation
 *
 * ALGORITHM OVERVIEW
 * ==================
 *
 * Given: A x = b,  lb <= x <= ub  (canonical equality form, m rows, n cols)
 *
 * Step 0 — Variable shifting
 *   Let x' = x - lb.  Then A x' = b - A lb := b'.  x' >= 0.
 *   Phase I operates on the shifted system.  The original model lower bounds
 *   are respected by construction; primal_simplex_phase2 treats nonbasic
 *   variables as being at their lower bound (shifted = 0).
 *
 *   NOTE: The current primal_simplex_phase2 sets x[leaving_var] = 0.0, so it
 *   correctly implements the shifted-variable convention.  Upper bounds are
 *   inherited from the original model and adjusted by the shift.
 *
 * Step 1 — Build Phase-I model
 *   The Phase-I model has n + n_art columns:
 *     columns 0 .. n-1      : original shifted variables (obj = 0)
 *     columns n .. n+n_art-1: artificial variables (obj = +1)
 *   Rows are the original m rows, possibly sign-flipped to make b_i' >= 0.
 *
 *   For row i:
 *     if b'_i >= 0: use A_i as-is, add artificial a_i >= 0 with coefficient +1
 *     if b'_i  < 0: multiply A_i and b'_i by -1, then add artificial a_i >= 0
 *                   (this ensures a_i = |b'_i| >= 0 is feasible initially)
 *
 *   The initial basis is the set of artificial variables (one per row).
 *   The initial solution sets each artificial a_i = |b'_i|.
 *
 * Step 2 — Run Phase I via primal_simplex_phase2
 *   The Phase-I objective is min sum(a_i).
 *   We use the existing primal_simplex_phase2() on the Phase-I model.
 *   This is valid because the Phase-I model has a known feasible basis.
 *
 * Step 3 — Check feasibility
 *   If sum(a_i) > kDefaultFeasibilityTol: original LP is infeasible.
 *
 * Step 4 — Remove artificial variables from basis
 *   For each row i where an artificial variable is still basic:
 *     - Scan original columns for a nonzero pivot element (via FTRAN).
 *     - If found: perform the pivot to drive the artificial out.
 *     - If no pivot exists: the row is redundant (handled by leaving the
 *       artificial at value 0, which is feasible but not recommended long-term;
 *       we pivot in the first available original column with any nonzero).
 *     Artificial variables MUST NOT remain basic.
 *
 * Step 5 — Extract original-variable basis
 *   Build Basis and x for the original model (n columns, m rows).
 *   Apply inverse shift: x_orig[j] = x'[j] + lb[j].
 *
 * Step 6 — Run Phase II
 *   Call primal_simplex_phase2(original_model, basis, x, factorizer).
 *
 * DETERMINISM
 *   All tie-breaks use the smallest variable or row index.
 *   No unordered container iteration determines pivot selection.
 */

#include "phase1.hpp"
#include "profiler.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace sankhya {
namespace simplex {

// ============================================================
// Internal helper: build a CSC matrix from accumulated COO data
// ============================================================
namespace {

struct COOEntry {
    Index row;
    Index col;
    Float val;
};

/// Build a CSC matrix from a list of (row, col, val) triples.
/// Entries need not be sorted; duplicate (row,col) entries are summed.
core::CSCMatrix build_csc(Index nrows, Index ncols,
                          std::vector<COOEntry>& entries) {
    // Sort by column, then row for deterministic order
    std::sort(entries.begin(), entries.end(),
              [](const COOEntry& a, const COOEntry& b) {
                  return a.col < b.col || (a.col == b.col && a.row < b.row);
              });

    std::vector<Index> col_ptrs(static_cast<std::size_t>(ncols) + 1, 0);

    for (const auto& e : entries) {
        col_ptrs[static_cast<std::size_t>(e.col) + 1]++;
    }
    for (Index j = 1; j <= ncols; ++j) {
        col_ptrs[static_cast<std::size_t>(j)] +=
            col_ptrs[static_cast<std::size_t>(j) - 1];
    }
    
    std::vector<Index> row_indices(entries.size());
    std::vector<Float> values(entries.size());

    std::vector<Index> fill_ptr(col_ptrs.begin(), col_ptrs.end());
    for (const auto& e : entries) {
        const auto pos = static_cast<std::size_t>(
            fill_ptr[static_cast<std::size_t>(e.col)]++);
        row_indices[pos] = e.row;
        values[pos] = e.val;
    }
    return core::CSCMatrix(nrows, ncols, std::move(values), std::move(row_indices), std::move(col_ptrs));
}

} // anonymous namespace

// ============================================================
// solve_with_phase1
// ============================================================
SimplexStatus solve_with_phase1(
    const core::Model& model,
    Basis& basis,
    std::vector<Float>& x,
    numerics::BasisFactorization& factorizer
) {
    const Index m = model.A.rows;
    const Index n = model.A.cols;

    // ------------------------------------------------------------------
    // Upper-bound contract audit
    // ------------------------------------------------------------------
    // primal_simplex_phase2() implements a lower-bound-only ratio test.
    // It sets nonbasic variables to 0.0 (i.e. at lb after shifting), and
    // does NOT flip variables at their upper bound.  This means finite
    // upper bounds are silently ignored, producing incorrect results.
    //
    // Policy (Option B per the architecture spec):
    //   If the model has any finite upper bound tighter than the shifted
    //   range, report this as an unsupported case.  Return IterationLimit
    //   so the caller's fallback logic handles it gracefully (e.g., logs
    //   the limitation) without claiming a wrong solution.
    //
    // Exception: FX variables (lb == ub) are handled correctly because
    //   they can be substituted / fixed at their value during shifting
    //   and their shifted range is [0,0].  A variable with ub-lb == 0
    //   is effectively a constant and the simplex never changes it.
    for (Index j = 0; j < n; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        const Float lbj = model.lb[jj];
        const Float ubj = model.ub[jj];
        const Float shifted_ub = ubj - lbj;
        if (shifted_ub < math::kInfinity * 0.5 && shifted_ub > math::kDefaultFeasibilityTol) {
            // This variable has a finite, non-trivial upper bound that
            // the current Phase-II simplex cannot respect.
            // Return IterationLimit (unsupported, not mathematically infeasible).
            return SimplexStatus::IterationLimit;
        }
    }

    // ------------------------------------------------------------------
    // Step 0 — Compute shifted RHS: b' = b - A * lb
    // ------------------------------------------------------------------
    std::vector<Float> b_shifted(static_cast<std::size_t>(m), 0.0);

    // b' = b
    for (Index i = 0; i < m; ++i) {
        b_shifted[static_cast<std::size_t>(i)] = model.rhs[static_cast<std::size_t>(i)];
    }

    // b' -= A * lb  (CSC format: iterate over each column j)
    for (Index j = 0; j < n; ++j) {
        const Float lbj = model.lb[static_cast<std::size_t>(j)];
        if (lbj == 0.0) continue;  // most common case for standard LPs
        const Index col_start = model.A.col_ptrs[static_cast<std::size_t>(j)];
        const Index col_end   = model.A.col_ptrs[static_cast<std::size_t>(j) + 1];
        for (Index k = col_start; k < col_end; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            b_shifted[static_cast<std::size_t>(model.A.row_indices[kk])] -=
                model.A.values[kk] * lbj;
        }
    }

    // ------------------------------------------------------------------
    // Step 1 — Build Phase-I model
    //
    // Phase-I columns: 0..n-1 = original vars (obj=0)
    //                  n..n+m-1 = artificial vars (obj=1)
    //
    // Phase-I rows are the original m rows, possibly sign-flipped.
    // We store the sign for each row (row_sign[i] = +1 or -1).
    // ------------------------------------------------------------------
    std::vector<Float> row_sign(static_cast<std::size_t>(m), 1.0);
    std::vector<Float> b_p1(static_cast<std::size_t>(m), 0.0);

    for (Index i = 0; i < m; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        if (b_shifted[ii] < 0.0) {
            row_sign[ii] = -1.0;
            b_p1[ii] = -b_shifted[ii];
        } else {
            row_sign[ii] = 1.0;
            b_p1[ii] = b_shifted[ii];
        }
    }

    // Build Phase-I CSC constraint matrix:
    // For original column j: coefficient in row i is sign[i] * A[i,j]
    // For artificial column n+i: coefficient in row i is +1
    const Index n_p1 = n + m; // total Phase-I variables
    std::vector<COOEntry> coo;
    coo.reserve(static_cast<std::size_t>(model.A.row_indices.size()) +
                static_cast<std::size_t>(m));

    // Original columns
    for (Index j = 0; j < n; ++j) {
        const Index col_start = model.A.col_ptrs[static_cast<std::size_t>(j)];
        const Index col_end   = model.A.col_ptrs[static_cast<std::size_t>(j) + 1];
        for (Index k = col_start; k < col_end; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const Index row = model.A.row_indices[kk];
            const Float val = model.A.values[kk] *
                              row_sign[static_cast<std::size_t>(row)];
            if (std::abs(val) > 0.0) {
                coo.push_back({row, j, val});
            }
        }
    }
    // Artificial columns: one per row
    for (Index i = 0; i < m; ++i) {
        coo.push_back({i, n + i, 1.0});
    }

    core::CSCMatrix A_p1 = build_csc(m, n_p1, coo);

    // Build Phase-I objective: 0 for original vars, 1 for artificials
    std::vector<Float> obj_p1(static_cast<std::size_t>(n_p1), 0.0);
    for (Index i = 0; i < m; ++i) {
        obj_p1[static_cast<std::size_t>(n + i)] = 1.0;
    }

    // Build Phase-I bounds:
    // Original shifted vars: lb' = 0 (shifted), ub' = ub - lb (shifted)
    // Artificial vars: lb = 0, ub = +inf
    std::vector<Float> lb_p1(static_cast<std::size_t>(n_p1), 0.0);
    std::vector<Float> ub_p1(static_cast<std::size_t>(n_p1), math::kInfinity);
    for (Index j = 0; j < n; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        Float shifted_ub = model.ub[jj] - model.lb[jj];
        if (shifted_ub < 0.0) shifted_ub = 0.0;  // safety clamp
        ub_p1[jj] = shifted_ub;
    }

    // Assemble the Phase-I Model object
    // We construct it manually since Model's add_* / finalize interface
    // is the correct way to build a Model, but we already have the CSC.
    // We construct a thin wrapper around the already-built CSC.
    core::Model model_p1;
    model_p1.sense = OptimizationSense::Minimize;
    model_p1.obj   = obj_p1;
    model_p1.lb    = lb_p1;
    model_p1.ub    = ub_p1;
    model_p1.rhs   = b_p1;
    model_p1.vtype.assign(static_cast<std::size_t>(n_p1), VariableType::Continuous);
    model_p1.A     = std::move(A_p1);

    // ------------------------------------------------------------------
    // Step 2 — Build initial Phase-I basis and solution
    //
    // Initial basis: artificial variable n+i is basic in row i.
    // Initial solution: a_i = b_p1[i] (which is >= 0 by construction).
    //                   all original variables = 0 (at shifted lower bound).
    // ------------------------------------------------------------------
    Basis basis_p1;
    basis_p1.col_status.assign(static_cast<std::size_t>(n_p1), BasisStatus::AtLower);
    basis_p1.basic_indices.resize(static_cast<std::size_t>(m));
    for (Index i = 0; i < m; ++i) {
        basis_p1.basic_indices[static_cast<std::size_t>(i)] = n + i;
        basis_p1.col_status[static_cast<std::size_t>(n + i)] = BasisStatus::Basic;
    }

    std::vector<Float> x_p1(static_cast<std::size_t>(n_p1), 0.0);
    for (Index i = 0; i < m; ++i) {
        x_p1[static_cast<std::size_t>(n + i)] = b_p1[static_cast<std::size_t>(i)];
    }

    // ------------------------------------------------------------------
    // Step 3 — Run Phase I (primal simplex on Phase-I model)
    // ------------------------------------------------------------------
    sankhya::profile::start_cpu("Simplex");
    SimplexStatus p1_status =
        primal_simplex_phase2(model_p1, basis_p1, x_p1, factorizer);
    sankhya::profile::stop_cpu("Simplex");

    if (p1_status == SimplexStatus::IterationLimit) {
        return SimplexStatus::IterationLimit;
    }

    // ------------------------------------------------------------------
    // Step 4 — Check feasibility: sum of artificial variables
    // ------------------------------------------------------------------
    Float art_sum = 0.0;
    for (Index i = 0; i < m; ++i) {
        const Float ai = x_p1[static_cast<std::size_t>(n + i)];
        if (ai > 0.0) art_sum += ai;
    }
    if (art_sum > math::kDefaultFeasibilityTol) {
        // Primal infeasibility confirmed: Phase I could not drive artificials to zero.
        return SimplexStatus::Infeasible;
    }

    // ------------------------------------------------------------------
    // Step 5 — Remove artificial variables from basis
    //
    // For each row i whose basic variable is an artificial (index >= n):
    //   Scan original columns j = 0..n-1 for a nonzero pivot element.
    //   The pivot element is d[i] = (B^{-1} A_j)[i] for an FTRAN'd column.
    //   Since the factorizer currently holds M(Phase-I basis), we can
    //   use it directly.
    //
    //   We find the best pivot by scanning column-by-column using FTRAN.
    //   Determinism: prefer smallest j with |d[i]| > kDefaultPivotTol.
    //
    //   If no original column can be pivoted in (truly redundant row),
    //   we note this as a degenerate-but-acceptable case: the artificial
    //   is zero, so the solution remains feasible.  We will remove it by
    //   substituting the first available original variable (even with a
    //   near-zero pivot, accepting the numerical risk for the degenerate
    //   case), OR report infeasibility if no nonzero exists in that row
    //   of A at all.
    // ------------------------------------------------------------------

    // Refactorize with the current Phase-I basis to get a clean factorizer
    factorizer.factorize(model_p1.A, basis_p1);

    for (Index i = 0; i < m; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        if (basis_p1.basic_indices[ii] < n) continue;  // already original var

        // Row i has an artificial in the basis. Find pivot from original cols.
        Index pivot_col = -1;

        for (Index j = 0; j < n; ++j) {
            const auto jj = static_cast<std::size_t>(j);
            // Skip if j is already basic
            if (basis_p1.col_status[jj] == BasisStatus::Basic) continue;

            // Extract column j of Phase-I model into a dense vector
            std::vector<Float> col_j(static_cast<std::size_t>(m), 0.0);
            const Index col_start = model_p1.A.col_ptrs[jj];
            const Index col_end   = model_p1.A.col_ptrs[jj + 1];
            for (Index k = col_start; k < col_end; ++k) {
                const auto kk = static_cast<std::size_t>(k);
                col_j[static_cast<std::size_t>(model_p1.A.row_indices[kk])] =
                    model_p1.A.values[kk];
            }
            // FTRAN: solve B d = col_j
            factorizer.ftran(col_j);
            const Float d_i = col_j[ii];

            if (std::abs(d_i) > math::kDefaultPivotTol) {
                // Accept first (smallest j) eligible pivot for determinism
                pivot_col = j;
                break;
            }
        }

        if (pivot_col == -1) {
            // No original column can pivot into this row because the row is
            // a linear combination of other rows in the basis (rank deficient).
            // The artificial is at value 0 (feasibility holds), but Phase II
            // requires a full-rank constraint matrix.
            //
            // OPTION B (architecture spec): Report IterationLimit to signal
            // "the recovery path does not support this rank-deficient model"
            // rather than falsely claiming mathematical infeasibility or
            // constructing a singular basis that will crash the factorizer.
            return SimplexStatus::IterationLimit;
        }

        // Perform the basis exchange: pivot_col enters row i.
        // x_p1[pivot_col] = x_p1[artificial] / pivot_val = 0 / pivot_val = 0
        const Index leaving_art = basis_p1.basic_indices[ii]; // index >= n
        x_p1[static_cast<std::size_t>(pivot_col)]   = 0.0;   // theta = 0
        x_p1[static_cast<std::size_t>(leaving_art)] = 0.0;

        basis_p1.basic_indices[ii] = pivot_col;
        basis_p1.col_status[static_cast<std::size_t>(pivot_col)]  = BasisStatus::Basic;
        basis_p1.col_status[static_cast<std::size_t>(leaving_art)] = BasisStatus::AtLower;

        // Update factorizer
        std::vector<Float> Aq(static_cast<std::size_t>(m), 0.0);
        const Index col_start2 = model_p1.A.col_ptrs[static_cast<std::size_t>(pivot_col)];
        const Index col_end2   = model_p1.A.col_ptrs[static_cast<std::size_t>(pivot_col) + 1];
        for (Index k = col_start2; k < col_end2; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            Aq[static_cast<std::size_t>(model_p1.A.row_indices[kk])] =
                model_p1.A.values[kk];
        }
        factorizer.update(i, pivot_col, Aq);
    }

    // Verify: no artificial should remain basic
    for (Index i = 0; i < m; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        if (basis_p1.basic_indices[ii] >= n) {
            // Still has an artificial; this should not happen after the loop above.
            // Return IterationLimit (unsupported/failure) rather than Infeasible.
            return SimplexStatus::IterationLimit;
        }
    }

    // ------------------------------------------------------------------
    // Step 6 — Build original-model basis and primal solution
    //
    // The Phase-I model has original variables at shifted values.
    // We now build:
    //   - basis: for the original n-column model
    //   - x: unshifted (x_orig = x'_shifted + lb)
    // ------------------------------------------------------------------
    basis.col_status.assign(static_cast<std::size_t>(n), BasisStatus::AtLower);
    basis.basic_indices.resize(static_cast<std::size_t>(m));

    for (Index i = 0; i < m; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        const Index orig_col = basis_p1.basic_indices[ii];
        // orig_col is in 0..n-1 (verified above)
        basis.basic_indices[ii] = orig_col;
        basis.col_status[static_cast<std::size_t>(orig_col)] = BasisStatus::Basic;
    }

    // Build unshifted x for original model
    x.assign(static_cast<std::size_t>(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        x[jj] = x_p1[jj] + model.lb[jj];  // unshift
    }

    // ------------------------------------------------------------------
    // Step 7 — Run Phase II on the original model
    // ------------------------------------------------------------------
    sankhya::profile::start_cpu("Simplex");
    auto res = primal_simplex_phase2(model, basis, x, factorizer);
    sankhya::profile::stop_cpu("Simplex");
    return res;
}

} // namespace simplex
} // namespace sankhya
