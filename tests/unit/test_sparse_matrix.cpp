/**
 * @file test_sparse_matrix.cpp
 * @brief Unit tests for the CSCMatrix sparse matrix representation
 */

#include <catch2/catch_test_macros.hpp>

#include "core/sparse_matrix.hpp"

using sankhya::Index;
using sankhya::core::CSCMatrix;
using sankhya::core::CSRMatrix;

// Test-only CSR lookup helper; mirrors CSC::get semantics for cross-checking.
static sankhya::Float csr_get(const CSRMatrix& csr, Index r, Index c) {
    if (r < 0 || c < 0 || r >= csr.rows || c >= csr.cols) {
        return 0.0;
    }
    for (Index k = csr.row_ptrs[r]; k < csr.row_ptrs[r + 1]; ++k) {
        if (csr.col_indices[k] == c) {
            return csr.values[k];
        }
    }
    return 0.0;
}

TEST_CASE("CSCMatrix - internal array sizes and values", "[sparse][core]") {
    // 3x3 matrix with entries:
    //   (0,0)=1.0  (1,0)=2.0  (2,0)=3.0
    //   (1,1)=4.0
    //   (0,2)=5.0  (2,2)=6.0
    // Columns are contiguous: col 0 -> {1,2,3}, col 1 -> {4}, col 2 -> {5,6}
    CSCMatrix m(3,
                3,
                {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                {0, 1, 2, 1, 0, 2},
                {0, 3, 4, 6});

    REQUIRE(m.rows == 3);
    REQUIRE(m.cols == 3);
    REQUIRE(m.nnz() == 6);

    // Internal array sizes
    REQUIRE(m.values.size() == 6);
    REQUIRE(m.row_indices.size() == 6);
    REQUIRE(m.col_ptrs.size() == 4);

    // Exact internal values
    REQUIRE(m.values == std::vector<sankhya::Float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    REQUIRE(m.row_indices == std::vector<Index>{0, 1, 2, 1, 0, 2});
    REQUIRE(m.col_ptrs == std::vector<Index>{0, 3, 4, 6});
}

TEST_CASE("CSCMatrix - get returns stored and zero entries", "[sparse][core]") {
    CSCMatrix m(3,
                3,
                {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                {0, 1, 2, 1, 0, 2},
                {0, 3, 4, 6});

    // Stored entries, column 0
    REQUIRE(m.get(0, 0) == 1.0);
    REQUIRE(m.get(1, 0) == 2.0);
    REQUIRE(m.get(2, 0) == 3.0);

    // Stored entries, column 1
    REQUIRE(m.get(1, 1) == 4.0);

    // Stored entries, column 2
    REQUIRE(m.get(0, 2) == 5.0);
    REQUIRE(m.get(2, 2) == 6.0);

    // Entries not stored must be 0.0
    REQUIRE(m.get(0, 1) == 0.0);
    REQUIRE(m.get(2, 1) == 0.0);
    REQUIRE(m.get(1, 2) == 0.0);

    // Out-of-bounds queries must be 0.0
    REQUIRE(m.get(3, 0) == 0.0);
    REQUIRE(m.get(0, 3) == 0.0);
    REQUIRE(m.get(-1, 0) == 0.0);
}

TEST_CASE("CSCMatrix - empty matrix", "[sparse][core]") {
    CSCMatrix m(3, 3, {}, {}, {0, 0, 0, 0});

    REQUIRE(m.nnz() == 0);
    REQUIRE(m.values.size() == 0);
    REQUIRE(m.row_indices.size() == 0);
    REQUIRE(m.col_ptrs.size() == 4);
    REQUIRE(m.get(0, 0) == 0.0);
    REQUIRE(m.get(2, 2) == 0.0);
}

TEST_CASE("CSCMatrix - to_csr representative conversion", "[sparse][core]") {
    // 4x4 matrix with a fully empty row (row 2):
    //   (0,0)=1.0  (0,2)=3.0
    //   (1,1)=2.0
    //   row 2: empty
    //   (3,0)=4.0  (3,3)=5.0
    CSCMatrix csc(4,
                  4,
                  {1.0, 4.0, 2.0, 3.0, 5.0},
                  {0, 3, 1, 0, 3},
                  {0, 2, 3, 4, 5});

    CSRMatrix csr = to_csr(csc);

    // Dimensions match the source CSC matrix
    REQUIRE(csr.rows == csc.rows);
    REQUIRE(csr.cols == csc.cols);
    REQUIRE(csr.rows == 4);
    REQUIRE(csr.cols == 4);

    // row_ptrs size is rows + 1
    REQUIRE(csr.row_ptrs.size() == 5);

    // col_indices and values are paired one-to-one
    REQUIRE(csr.col_indices.size() == csr.values.size());
    REQUIRE(csr.col_indices.size() == 5);

    // Exact array contents after conversion
    REQUIRE(csr.row_ptrs == std::vector<Index>{0, 2, 3, 3, 5});
    REQUIRE(csr.col_indices == std::vector<Index>{0, 2, 1, 0, 3});
    REQUIRE(csr.values == std::vector<sankhya::Float>{1.0, 3.0, 2.0, 4.0, 5.0});

    // Row 2 contains zero non-zeros; its row-pointer range must be empty
    REQUIRE(csr.row_ptrs[2] == 3);
    REQUIRE(csr.row_ptrs[3] == 3);
}

TEST_CASE("CSCMatrix - to_csr matches CSC semantics at every position", "[sparse][core]") {
    CSCMatrix csc(4,
                  4,
                  {1.0, 4.0, 2.0, 3.0, 5.0},
                  {0, 3, 1, 0, 3},
                  {0, 2, 3, 4, 5});
    CSRMatrix csr = to_csr(csc);

    // For every valid (r, c), the CSR representation matches the original CSC.
    for (Index r = 0; r < 4; ++r) {
        for (Index c = 0; c < 4; ++c) {
            REQUIRE(csr_get(csr, r, c) == csc.get(r, c));
        }
    }

    // Stored entries and implicit zeros, read explicitly through CSR.
    REQUIRE(csr_get(csr, 0, 0) == 1.0);
    REQUIRE(csr_get(csr, 0, 2) == 3.0);
    REQUIRE(csr_get(csr, 1, 1) == 2.0);
    REQUIRE(csr_get(csr, 3, 0) == 4.0);
    REQUIRE(csr_get(csr, 3, 3) == 5.0);

    // Implicit zeros remain zero.
    REQUIRE(csr_get(csr, 0, 1) == 0.0);
    REQUIRE(csr_get(csr, 0, 3) == 0.0);
    REQUIRE(csr_get(csr, 1, 0) == 0.0);
    REQUIRE(csr_get(csr, 2, 0) == 0.0);
    REQUIRE(csr_get(csr, 2, 3) == 0.0);
}

TEST_CASE("CSCMatrix - to_csr empty matrix", "[sparse][core]") {
    CSCMatrix csc(3, 3, {}, {}, {0, 0, 0, 0});
    CSRMatrix csr = to_csr(csc);

    REQUIRE(csr.rows == 3);
    REQUIRE(csr.cols == 3);
    REQUIRE(csr.row_ptrs.size() == 4);
    REQUIRE(csr.row_ptrs == std::vector<Index>{0, 0, 0, 0});
    REQUIRE(csr.col_indices.size() == 0);
    REQUIRE(csr.values.size() == 0);

    for (Index r = 0; r < 3; ++r) {
        for (Index c = 0; c < 3; ++c) {
            REQUIRE(csr_get(csr, r, c) == 0.0);
        }
    }
}