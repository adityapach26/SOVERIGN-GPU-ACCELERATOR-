/**
 * @file sparse_matrix.cpp
 * @brief Compressed Sparse Column (CSC) matrix implementation
 */

#include "sparse_matrix.hpp"

#include <algorithm>

namespace sankhya {
namespace core {

CSCMatrix::CSCMatrix(Index rows,
                     Index cols,
                     std::vector<Float> vals,
                     std::vector<Index> rows_idx,
                     std::vector<Index> col_ptr)
    : rows(rows),
      cols(cols),
      col_ptrs(std::move(col_ptr)),
      row_indices(std::move(rows_idx)),
      values(std::move(vals)) {}

Index CSCMatrix::nnz() const {
    return static_cast<Index>(values.size());
}

Float CSCMatrix::get(Index r, Index c) const {
    if (r < 0 || c < 0 || r >= rows || c >= cols) {
        return 0.0;
    }
    // Entries of column c occupy [col_ptrs[c], col_ptrs[c + 1])
    const Index begin = col_ptrs[c];
    const Index end = col_ptrs[c + 1];
    // row_indices within a column are sorted ascending, so binary search
    const auto it = std::lower_bound(
        row_indices.begin() + begin,
        row_indices.begin() + end,
        r);
    if (it != row_indices.begin() + end && *it == r) {
        return values[static_cast<std::size_t>(it - row_indices.begin())];
    }
    return 0.0;
}

CSRMatrix to_csr(const CSCMatrix& csc) {
    CSRMatrix csr;
    csr.rows = csc.rows;
    csr.cols = csc.cols;
    csr.row_ptrs.assign(static_cast<std::size_t>(csc.rows + 1), 0);

    // Pass 1: count the non-zeros in each row.
    std::vector<Index> row_counts(static_cast<std::size_t>(csc.rows), 0);
    for (Index c = 0; c < csc.cols; ++c) {
        for (Index k = csc.col_ptrs[c]; k < csc.col_ptrs[c + 1]; ++k) {
            row_counts[static_cast<std::size_t>(csc.row_indices[k])] += 1;
        }
    }

    // Pass 2: prefix sum the counts to obtain row_ptrs.
    for (Index i = 0; i < csc.rows; ++i) {
        csr.row_ptrs[static_cast<std::size_t>(i + 1)] =
            csr.row_ptrs[static_cast<std::size_t>(i)] + row_counts[static_cast<std::size_t>(i)];
    }

    // Pass 3: scatter values and column indices into the per-row buckets.
    csr.col_indices.resize(static_cast<std::size_t>(csc.nnz()));
    csr.values.resize(static_cast<std::size_t>(csc.nnz()));
    std::vector<Index> fill(static_cast<std::size_t>(csc.rows), 0);
    for (Index c = 0; c < csc.cols; ++c) {
        for (Index k = csc.col_ptrs[c]; k < csc.col_ptrs[c + 1]; ++k) {
            const Index r = csc.row_indices[k];
            const Index pos = csr.row_ptrs[static_cast<std::size_t>(r)] + fill[static_cast<std::size_t>(r)];
            const std::size_t idx = static_cast<std::size_t>(pos);
            csr.col_indices[idx] = c;
            csr.values[idx] = csc.values[static_cast<std::size_t>(k)];
            fill[static_cast<std::size_t>(r)] += 1;
        }
    }

    return csr;
}

} // namespace core
} // namespace sankhya