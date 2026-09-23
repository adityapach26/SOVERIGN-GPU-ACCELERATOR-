#include <vector>
/**
 * @file scaling.cpp
 * @brief Sovereign Presolve Pipeline - Ruiz Equilibrium Scaling implementation
 *
 * Implements Step 8.1.
 */

#include "scaling.hpp"

#include <cmath>
#include <cstddef>
#include <algorithm>

namespace sankhya {
namespace presolve {

void apply_ruiz_scaling(core::Model& model, int max_iterations) {
    const Index m = model.A.rows;
    const Index n = model.A.cols;

    if (m == 0 || n == 0) return;

    // Initialize scaling arrays if they are empty
    if (model.row_scale.empty()) {
        model.row_scale.assign(static_cast<std::size_t>(m), 1.0);
    }
    if (model.col_scale.empty()) {
        model.col_scale.assign(static_cast<std::size_t>(n), 1.0);
    }

    // Engineering decision: Stopping tolerance for row/column norms deviating from 1.0
    const Float tolerance = 1e-4;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Float max_row_deviation = 0.0;
        Float max_col_deviation = 0.0;

        // -----------------------------------------------------
        // 1. Row Scaling
        // -----------------------------------------------------
        std::vector<Float> r(static_cast<std::size_t>(m), 0.0);
        const std::size_t nnz = model.A.values.size();
        for (std::size_t k = 0; k < nnz; ++k) {
            Float val = std::abs(model.A.values[k]);
            Index row = model.A.row_indices[k];
            if (val > r[static_cast<std::size_t>(row)]) {
                r[static_cast<std::size_t>(row)] = val;
            }
        }

        std::vector<Float> s_r(static_cast<std::size_t>(m), 1.0);
        for (Index i = 0; i < m; ++i) {
            const auto ii = static_cast<std::size_t>(i);
            if (r[ii] > 1e-15) { // Skip zero rows
                s_r[ii] = 1.0 / std::sqrt(r[ii]);
                Float dev = std::abs(r[ii] - 1.0);
                if (dev > max_row_deviation) {
                    max_row_deviation = dev;
                }
                model.row_scale[ii] *= s_r[ii];
            }
        }

        // Apply row scaling to matrix A
        for (std::size_t k = 0; k < nnz; ++k) {
            model.A.values[k] *= s_r[static_cast<std::size_t>(model.A.row_indices[k])];
        }

        // -----------------------------------------------------
        // 2. Column Scaling
        // -----------------------------------------------------
        std::vector<Float> c(static_cast<std::size_t>(n), 0.0);
        for (Index j = 0; j < n; ++j) {
            const auto jj = static_cast<std::size_t>(j);
            Index start = model.A.col_ptrs[jj];
            Index end = model.A.col_ptrs[jj + 1];
            for (Index k = start; k < end; ++k) {
                Float val = std::abs(model.A.values[static_cast<std::size_t>(k)]);
                if (val > c[jj]) {
                    c[jj] = val;
                }
            }
        }

        std::vector<Float> s_c(static_cast<std::size_t>(n), 1.0);
        for (Index j = 0; j < n; ++j) {
            const auto jj = static_cast<std::size_t>(j);
            
            // Engineering decision: Do not scale integer columns.
            // Variable-coordinate scaling destroys integer semantics (e.g., x in {0,1} becomes x in {0, 1/D_c}).
            // To preserve MIP integrality, we bypass column scaling for these variables.
            if (model.vtype[jj] != VariableType::Continuous) {
                continue;
            }

            if (c[jj] > 1e-15) { // Skip zero columns
                s_c[jj] = 1.0 / std::sqrt(c[jj]);
                Float dev = std::abs(c[jj] - 1.0);
                if (dev > max_col_deviation) {
                    max_col_deviation = dev;
                }
                model.col_scale[jj] *= s_c[jj];
            }
        }

        // Apply column scaling to matrix A
        for (Index j = 0; j < n; ++j) {
            const auto jj = static_cast<std::size_t>(j);
            Index start = model.A.col_ptrs[jj];
            Index end = model.A.col_ptrs[jj + 1];
            for (Index k = start; k < end; ++k) {
                model.A.values[static_cast<std::size_t>(k)] *= s_c[jj];
            }
        }

        // Stopping criterion: If norms of all scaled rows/cols are sufficiently close to 1.0.
        if (max_row_deviation < tolerance && max_col_deviation < tolerance) {
            break;
        }
    }

    // -----------------------------------------------------
    // 3. Apply transformations to remaining Model components
    // -----------------------------------------------------
    // Constraint scaling: b_s = D_r * b
    for (Index i = 0; i < m; ++i) {
        model.rhs[static_cast<std::size_t>(i)] *= model.row_scale[static_cast<std::size_t>(i)];
    }

    // Variable coordinate scaling: x = D_c * x_s
    // Objective: c_s = D_c * c
    // Bounds: lb_s = D_c^{-1} * lb, ub_s = D_c^{-1} * ub
    for (Index j = 0; j < n; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        model.obj[jj] *= model.col_scale[jj];
        
        if (model.lb[jj] > -math::kInfinity) {
            model.lb[jj] /= model.col_scale[jj];
        }
        if (model.ub[jj] < math::kInfinity) {
            model.ub[jj] /= model.col_scale[jj];
        }
    }
}

} // namespace presolve
} // namespace sankhya

