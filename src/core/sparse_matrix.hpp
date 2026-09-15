/**
 * @file sparse_matrix.hpp
 * @brief Compressed Sparse Column (CSC) matrix representation for SANKHYA / VAJRA-OPT
 */

#pragma once

#include <cstddef>
#include <vector>

#include "sankhya/types.hpp"

namespace sankhya {
namespace core {

/**
 * @brief Compressed Sparse Column (CSC) matrix
 *
 * Stores a sparse matrix in CSC format using three parallel arrays:
 *   - col_ptrs:    size cols + 1; the entries of column c occupy the slice
 *                  [col_ptrs[c], col_ptrs[c + 1]) in row_indices / values
 *   - row_indices: the row index of each stored entry
 *   - values:      the numeric value of each stored entry
 *
 * Within each column, row_indices are assumed strictly increasing, enabling
 * binary search in get().
 */
struct CSCMatrix {
    Index rows;
    Index cols;
    std::vector<Index> col_ptrs;
    std::vector<Index> row_indices;
    std::vector<Float> values;

    CSCMatrix(Index rows,
              Index cols,
              std::vector<Float> vals,
              std::vector<Index> rows_idx,
              std::vector<Index> col_ptrs);

    Index nnz() const;

    Float get(Index r, Index c) const;
};

/**
 * @brief Compressed Sparse Row (CSR) matrix
 *
 * Stores a sparse matrix in CSR format using three parallel arrays:
 *   - row_ptrs:    size rows + 1; the entries of row i occupy the slice
 *                  [row_ptrs[i], row_ptrs[i + 1]) in col_indices / values
 *   - col_indices: the column index of each stored entry
 *   - values:      the numeric value of each stored entry
 *
 * Within each row, col_indices are assumed strictly increasing.
 */
struct CSRMatrix {
    Index rows;
    Index cols;
    std::vector<Index> row_ptrs;
    std::vector<Index> col_indices;
    std::vector<Float> values;
};

/**
 * @brief Convert a CSC matrix to CSR representation in O(nnz) time
 *
 * The conversion performs three passes over the stored entries:
 *   1. Count the number of non-zeros in each row.
 *   2. Prefix-sum the counts to produce row_ptrs.
 *   3. Scatter values and column indices into the per-row buckets.
 *
 * All Float values are preserved exactly.
 */
CSRMatrix to_csr(const CSCMatrix& csc);

} // namespace core
} // namespace sankhya