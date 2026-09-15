/**
 * @file problem.cpp
 * @brief Linear Program data model implementation
 */

#include "problem.hpp"

#include <cstddef>
#include <stdexcept>

namespace sankhya {
namespace core {

Model::Model() : sense(OptimizationSense::Minimize), A(0, 0, {}, {}, {0}) {}

void Model::add_variable(Float obj_coeff,
                         Float lower,
                         Float upper,
                         VariableType vt) {
    obj.push_back(obj_coeff);
    lb.push_back(lower);
    ub.push_back(upper);
    vtype.push_back(vt);
}

void Model::add_constraint(const std::vector<Index>& cols,
                           const std::vector<Float>& vals,
                           Float b) {
    if (cols.size() != vals.size()) {
        throw std::invalid_argument(
            "add_constraint: cols and vals must have the same size");
    }
    const Index num_vars = static_cast<Index>(vtype.size());
    for (Index c : cols) {
        if (c < 0 || c >= num_vars) {
            throw std::invalid_argument(
                "add_constraint: column index out of range");
        }
    }
    constraint_cols_.push_back(cols);
    constraint_vals_.push_back(vals);
    rhs.push_back(b);
}

void Model::finalize() {
    const Index num_rows = static_cast<Index>(rhs.size());
    const Index num_cols = static_cast<Index>(vtype.size());

    if (obj.size() != vtype.size() ||
        lb.size() != vtype.size() ||
        ub.size() != vtype.size()) {
        throw std::invalid_argument(
            "finalize: objective/bounds/type vectors differ in size");
    }

    // Count non-zeros per column.
    std::vector<Index> col_count(static_cast<std::size_t>(num_cols), 0);
    for (Index r = 0; r < num_rows; ++r) {
        for (std::size_t e = 0; e < constraint_cols_[static_cast<std::size_t>(r)].size(); ++e) {
            const Index c = constraint_cols_[static_cast<std::size_t>(r)][e];
            col_count[static_cast<std::size_t>(c)] += 1;
        }
    }

    // Prefix sum to obtain col_ptrs (size num_cols + 1).
    std::vector<Index> col_ptrs(static_cast<std::size_t>(num_cols + 1), 0);
    for (Index c = 0; c < num_cols; ++c) {
        col_ptrs[static_cast<std::size_t>(c + 1)] =
            col_ptrs[static_cast<std::size_t>(c)] + col_count[static_cast<std::size_t>(c)];
    }

    // Scatter coefficients into their column buckets.
    const std::size_t nnz = static_cast<std::size_t>(col_ptrs.back());
    std::vector<Index> row_indices(nnz);
    std::vector<Float> values(nnz);
    std::vector<Index> fill(static_cast<std::size_t>(num_cols), 0);
    for (Index r = 0; r < num_rows; ++r) {
        const auto& cols_r = constraint_cols_[static_cast<std::size_t>(r)];
        const auto& vals_r = constraint_vals_[static_cast<std::size_t>(r)];
        for (std::size_t e = 0; e < cols_r.size(); ++e) {
            const Index c = cols_r[e];
            const Index pos = col_ptrs[static_cast<std::size_t>(c)] +
                              fill[static_cast<std::size_t>(c)];
            const std::size_t idx = static_cast<std::size_t>(pos);
            row_indices[idx] = r;
            values[idx] = vals_r[e];
            fill[static_cast<std::size_t>(c)] += 1;
        }
    }

    A = CSCMatrix(num_rows, num_cols, std::move(values),
                  std::move(row_indices), std::move(col_ptrs));
}

} // namespace core
} // namespace sankhya