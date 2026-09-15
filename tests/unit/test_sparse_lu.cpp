/**
 * @file test_sparse_lu.cpp
 * @brief Unit tests for SparseLUFactorization (Step 6.1 and 6.2)
 *
 * Verifies:
 *   1. FTRAN correctness (Bx = b round-trip) on a sparse 10x10 matrix.
 *   2. BTRAN correctness (B^T x = b round-trip) on the same matrix.
 *   3. Markowitz-threshold pivoting exercises permutations.
 *   4. Numerical accuracy to 1e-9.
 *   5. Forrest-Tomlin updates (single and successive) match fresh factorizations.
 *   6. Residual monitoring correctly triggers needs_refactorization.
 */

#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "numerics/sparse_lu.hpp"
#include "core/problem.hpp"

using sankhya::Float;
using sankhya::Index;
using sankhya::core::CSCMatrix;
using sankhya::simplex::Basis;
using sankhya::simplex::BasisStatus;

namespace {

void dense_matvec(const std::vector<Float>& M, Index m, Index n,
                  const std::vector<Float>& x, std::vector<Float>& y) {
    y.assign(static_cast<std::size_t>(m), 0.0);
    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < n; ++j) {
            y[static_cast<std::size_t>(i)] +=
                M[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(j)] *
                x[static_cast<std::size_t>(j)];
        }
    }
}

void dense_matvec_t(const std::vector<Float>& M, Index m, Index n,
                    const std::vector<Float>& x, std::vector<Float>& y) {
    y.assign(static_cast<std::size_t>(n), 0.0);
    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < n; ++j) {
            y[static_cast<std::size_t>(j)] +=
                M[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(j)] *
                x[static_cast<std::size_t>(i)];
        }
    }
}

CSCMatrix make_test_10x10_csc() {
    std::vector<Float> values = {
        3.0, 1.0, 2.0, 1.0, 4.0, 1.0, 1.0, 5.0, 1.0, 3.0,
        1.0, 6.0, 2.0, 7.0, 8.0, 3.0, 1.0, 2.0, 5.0, 1.0, 9.0
    };
    std::vector<Index> row_indices = {
        1, 8, 0, 3, 2, 6, 1, 3, 0, 4,
        4, 5, 2, 6, 7, 9, 5, 7, 8, 7, 9
    };
    std::vector<Index> col_ptrs = {
        0, 2, 4, 6, 8, 10, 12, 14, 16, 19, 21
    };
    return CSCMatrix(10, 10, std::move(values), std::move(row_indices),
                     std::move(col_ptrs));
}

Basis make_identity_basis_10() {
    Basis b;
    b.col_status.resize(10, BasisStatus::Basic);
    b.basic_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    return b;
}

std::vector<Float> make_dense_B_10x10() {
    return {
        0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 2.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 9.0
    };
}

static constexpr Float kLUTol = 1e-9;

} // anonymous namespace

TEST_CASE("SparseLU - FTRAN round-trip on 10x10 sparse matrix", "[sparse_lu]") {
    CSCMatrix A = make_test_10x10_csc();
    Basis basis = make_identity_basis_10();
    std::vector<Float> dense_B = make_dense_B_10x10();
    sankhya::numerics::SparseLUFactorization lu;
    lu.factorize(A, basis);

    std::vector<Float> x_true = {1.0, -2.0, 3.0, -1.0, 0.5, 2.5, -0.5, 1.5, -3.0, 4.0};
    std::vector<Float> b;
    dense_matvec(dense_B, 10, 10, x_true, b);
    lu.ftran(b);

    for (Index i = 0; i < 10; ++i) {
        REQUIRE(std::abs(b[static_cast<std::size_t>(i)] - x_true[static_cast<std::size_t>(i)]) < kLUTol);
    }
}

TEST_CASE("SparseLU - BTRAN round-trip on 10x10 sparse matrix", "[sparse_lu]") {
    CSCMatrix A = make_test_10x10_csc();
    Basis basis = make_identity_basis_10();
    std::vector<Float> dense_B = make_dense_B_10x10();
    sankhya::numerics::SparseLUFactorization lu;
    lu.factorize(A, basis);

    std::vector<Float> y_true = {-1.5, 0.5, 2.0, -3.0, 1.0, 0.0, -2.0, 4.0, -0.5, 1.5};
    std::vector<Float> b;
    dense_matvec_t(dense_B, 10, 10, y_true, b);
    lu.btran(b);

    for (Index i = 0; i < 10; ++i) {
        REQUIRE(std::abs(b[static_cast<std::size_t>(i)] - y_true[static_cast<std::size_t>(i)]) < kLUTol);
    }
}

TEST_CASE("SparseLU - Forrest-Tomlin single update", "[sparse_lu]") {
    // We add extra columns to A so we can pivot them in.
    // Let's create A with 11 columns. Col 10 is [1, 0, ..., 0]^T
    std::vector<Float> vals = {1.0, 1.0, 2.0, 1.0, 1.0, 3.0};
    std::vector<Index> rows = {0, 1, 0, 1, 0, 1};
    std::vector<Index> cols = {0, 1, 2, 4, 6}; // 2x4 matrix
    CSCMatrix A(2, 4, std::move(vals), std::move(rows), std::move(cols));

    Basis basis;
    basis.col_status = {BasisStatus::Basic, BasisStatus::Basic, BasisStatus::AtLower, BasisStatus::AtLower};
    basis.basic_indices = {0, 1};

    sankhya::numerics::SparseLUFactorization lu_ft;
    lu_ft.factorize(A, basis);

    // FT update: replace basis col 0 with A col 2
    std::vector<Float> Aq = {2.0, 1.0};
    lu_ft.update(0, 2, Aq);
    REQUIRE(lu_ft.get_num_ft_updates() == 1u);

    // Fresh factorize for reference
    Basis basis_fresh = basis;
    basis_fresh.basic_indices[0] = 2;
    sankhya::numerics::SparseLUFactorization lu_fresh;
    lu_fresh.factorize(A, basis_fresh);

    // Compare FTRAN
    std::vector<Float> rhs_ft = {4.0, 3.0};
    std::vector<Float> rhs_fresh = {4.0, 3.0};
    lu_ft.ftran(rhs_ft);
    lu_fresh.ftran(rhs_fresh);
    for(Index i=0; i<2; ++i) {
        REQUIRE(std::abs(rhs_ft[i] - rhs_fresh[i]) < kLUTol);
    }

    // Compare BTRAN
    std::vector<Float> rhs_ft_b = {5.0, 2.0};
    std::vector<Float> rhs_fresh_b = {5.0, 2.0};
    lu_ft.btran(rhs_ft_b);
    lu_fresh.btran(rhs_fresh_b);
    for(Index i=0; i<2; ++i) {
        REQUIRE(std::abs(rhs_ft_b[i] - rhs_fresh_b[i]) < kLUTol);
    }
}

TEST_CASE("SparseLU - Forrest-Tomlin 5 successive updates", "[sparse_lu]") {
    // Create an LP with 5 constraints and 10 variables to have enough cols to swap
    // We'll use a diagonal initial basis (cols 0..4), and sequentially pivot in cols 5..9
    std::vector<Float> vals;
    std::vector<Index> rows;
    std::vector<Index> cols;
    cols.push_back(0);
    for(Index j=0; j<5; ++j) {
        vals.push_back(1.0); rows.push_back(j); cols.push_back(cols.back() + 1);
    }
    // Cols 5..9
    for(Index j=5; j<10; ++j) {
        vals.push_back(2.0); rows.push_back(j - 5);
        vals.push_back(-1.0); rows.push_back((j - 4) % 5);
        cols.push_back(cols.back() + 2);
    }
    CSCMatrix A(5, 10, std::move(vals), std::move(rows), std::move(cols));

    Basis basis;
    basis.col_status.resize(10, BasisStatus::AtLower);
    for(Index i=0; i<5; ++i) basis.col_status[i] = BasisStatus::Basic;
    basis.basic_indices = {0, 1, 2, 3, 4};

    sankhya::numerics::SparseLUFactorization lu_ft;
    lu_ft.factorize(A, basis);

    sankhya::numerics::SparseLUFactorization lu_fresh;

    for (Index step = 0; step < 5; ++step) {
        Index leaving_row = step;
        Index entering_col = step + 5;
        
        std::vector<Float> Aq(5, 0.0);
        Index start = A.col_ptrs[entering_col];
        Index end = A.col_ptrs[entering_col+1];
        for(Index k=start; k<end; ++k) Aq[A.row_indices[k]] = A.values[k];
        
        lu_ft.update(leaving_row, entering_col, Aq);
        REQUIRE(lu_ft.get_num_ft_updates() == static_cast<std::size_t>(step + 1));
        
        basis.basic_indices[leaving_row] = entering_col;
        lu_fresh.factorize(A, basis);
        
        std::vector<Float> b = {1.5, -2.1, 3.4, 0.8, -1.2};
        std::vector<Float> b_ft = b;
        std::vector<Float> b_fresh = b;
        
        lu_ft.ftran(b_ft);
        lu_fresh.ftran(b_fresh);
        for(Index i=0; i<5; ++i) {
            REQUIRE(std::abs(b_ft[i] - b_fresh[i]) < kLUTol);
        }
        
        std::vector<Float> c = {2.2, 0.4, -1.9, 3.1, -0.7};
        std::vector<Float> c_ft = c;
        std::vector<Float> c_fresh = c;
        
        lu_ft.btran(c_ft);
        lu_fresh.btran(c_fresh);
        for(Index i=0; i<5; ++i) {
            REQUIRE(std::abs(c_ft[i] - c_fresh[i]) < kLUTol);
        }
    }
}

TEST_CASE("SparseLU - Residual monitoring", "[sparse_lu]") {
    CSCMatrix A = make_test_10x10_csc();
    Basis basis = make_identity_basis_10();
    sankhya::numerics::SparseLUFactorization lu;
    lu.factorize(A, basis);

    std::vector<Float> x_B(10, 1.0);
    std::vector<Float> dense_B = make_dense_B_10x10();
    std::vector<Float> b_B;
    dense_matvec(dense_B, 10, 10, x_B, b_B);

    // Exact b_B means 0 residual
    REQUIRE(lu.needs_refactorization(x_B, b_B) == false);

    // Perturb b_B slightly
    b_B[3] += 1e-8;
    // Still within kResidualTol (1e-6)
    REQUIRE(lu.needs_refactorization(x_B, b_B) == false);

    // Perturb b_B by more than tolerance
    b_B[3] += 1e-5;
    REQUIRE(lu.needs_refactorization(x_B, b_B) == true);
}
