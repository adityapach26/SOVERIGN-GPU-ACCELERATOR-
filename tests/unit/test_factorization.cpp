/**
 * @file test_factorization.cpp
 * @brief Unit tests for the BasisFactorization interface (Step 4.1)
 *
 * Defines a test-only NaiveDenseFactorization backend that materializes
 * the basis as a small dense matrix.  Verifies that:
 *   1. The abstract interface compiles.
 *   2. A concrete backend can implement it.
 *   3. factorize, ftran, and btran have the intended mathematical behavior.
 *   4. update exists and can be overridden.
 */

#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "numerics/factorization.hpp"

using sankhya::Float;
using sankhya::Index;
using sankhya::core::CSCMatrix;
using sankhya::simplex::Basis;
using sankhya::simplex::BasisStatus;

// ---------------------------------------------------------------------------
// Test-only concrete implementation — NOT production code
// ---------------------------------------------------------------------------

namespace {

/**
 * @brief Naive dense basis factorization — test-only backend
 *
 * Stores the m×m basis matrix densely (row-major) after factorize().
 * Solves B x = rhs and B^T x = rhs via Gaussian elimination with
 * partial pivoting on small matrices.
 *
 * This is deliberately simple and exists solely to exercise the
 * BasisFactorization interface in unit tests.
 */
class NaiveDenseFactorization : public sankhya::numerics::BasisFactorization {
public:
    void factorize(
        const CSCMatrix& A,
        const Basis& basis
    ) override {
        m_ = static_cast<Index>(basis.basic_indices.size());

        // Materialize the basis columns of A into a dense m×m matrix
        // stored in row-major order.
        dense_.assign(static_cast<size_t>(m_) * static_cast<size_t>(m_), 0.0);

        for (Index j = 0; j < m_; ++j) {
            Index col = basis.basic_indices[static_cast<size_t>(j)];
            Index start = A.col_ptrs[static_cast<size_t>(col)];
            Index end   = A.col_ptrs[static_cast<size_t>(col) + 1];
            for (Index k = start; k < end; ++k) {
                Index row = A.row_indices[static_cast<size_t>(k)];
                if (row < m_) {
                    dense_[static_cast<size_t>(row) * static_cast<size_t>(m_)
                           + static_cast<size_t>(j)] =
                        A.values[static_cast<size_t>(k)];
                }
            }
        }
    }

    void ftran(std::vector<Float>& rhs) override {
        // Solve B x = rhs in place via Gaussian elimination with
        // partial pivoting on the stored dense copy.
        solve_dense(dense_, rhs, /*transpose=*/false);
    }

    void btran(std::vector<Float>& rhs) override {
        // Solve B^T x = rhs in place.
        solve_dense(dense_, rhs, /*transpose=*/true);
    }

    void update(
        Index leaving_row,
        Index /*entering_col*/,
        const std::vector<Float>& Aq
    ) override {
        // Minimal test-only implementation: replace the leaving_row-th
        // column of the dense basis with Aq.
        for (Index i = 0; i < m_; ++i) {
            dense_[static_cast<size_t>(i) * static_cast<size_t>(m_)
                   + static_cast<size_t>(leaving_row)] =
                Aq[static_cast<size_t>(i)];
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
     * Both solve in place, overwriting rhs with the solution.
     */
    static void solve_dense(
        const std::vector<Float>& matrix,
        std::vector<Float>& rhs,
        bool transpose
    ) {
        auto n = static_cast<Index>(rhs.size());
        if (n == 0) return;

        // Work on a copy of the matrix so factorize state is preserved.
        std::vector<Float> M = matrix;

        // If solving the transpose system, transpose M in place.
        if (transpose) {
            for (Index i = 0; i < n; ++i) {
                for (Index j = i + 1; j < n; ++j) {
                    auto ii = static_cast<size_t>(i);
                    auto jj = static_cast<size_t>(j);
                    auto nn = static_cast<size_t>(n);
                    std::swap(M[ii * nn + jj], M[jj * nn + ii]);
                }
            }
        }

        auto nn = static_cast<size_t>(n);

        // Forward elimination with partial pivoting
        for (Index col = 0; col < n; ++col) {
            auto cc = static_cast<size_t>(col);

            // Find pivot
            Index pivot_row = col;
            Float max_val = std::abs(M[cc * nn + cc]);
            for (Index row = col + 1; row < n; ++row) {
                Float v = std::abs(M[static_cast<size_t>(row) * nn + cc]);
                if (v > max_val) {
                    max_val = v;
                    pivot_row = row;
                }
            }

            if (max_val < 1e-15) {
                throw std::runtime_error("Singular matrix in NaiveDenseFactorization");
            }

            // Swap rows
            if (pivot_row != col) {
                for (Index j = col; j < n; ++j) {
                    auto jj = static_cast<size_t>(j);
                    std::swap(
                        M[cc * nn + jj],
                        M[static_cast<size_t>(pivot_row) * nn + jj]
                    );
                }
                std::swap(rhs[cc], rhs[static_cast<size_t>(pivot_row)]);
            }

            // Eliminate below
            Float diag = M[cc * nn + cc];
            for (Index row = col + 1; row < n; ++row) {
                auto rr = static_cast<size_t>(row);
                Float factor = M[rr * nn + cc] / diag;
                for (Index j = col + 1; j < n; ++j) {
                    M[rr * nn + static_cast<size_t>(j)] -=
                        factor * M[cc * nn + static_cast<size_t>(j)];
                }
                M[rr * nn + cc] = 0.0;
                rhs[rr] -= factor * rhs[cc];
            }
        }

        // Back substitution
        for (Index row = n - 1; row >= 0; --row) {
            auto rr = static_cast<size_t>(row);
            Float sum = rhs[rr];
            for (Index j = row + 1; j < n; ++j) {
                sum -= M[rr * nn + static_cast<size_t>(j)] * rhs[static_cast<size_t>(j)];
            }
            rhs[rr] = sum / M[rr * nn + rr];
        }
    }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static constexpr Float kTol = sankhya::math::kDefaultFeasibilityTol;

/**
 * Build a 3×3 CSCMatrix from a row-major dense array.
 *
 * The test basis B is:
 *     | 2  1  1 |
 *     | 4  3  3 |
 *     | 8  7  9 |
 *
 * det(B) = 2*(27-21) - 1*(36-24) + 1*(28-24) = 12-12+4 = 4  (nonsingular)
 */
static CSCMatrix make_test_csc() {
    // Dense row-major:
    //   row 0: 2 1 1
    //   row 1: 4 3 3
    //   row 2: 8 7 9
    //
    // CSC — column 0: rows {0,1,2} vals {2,4,8}
    //        column 1: rows {0,1,2} vals {1,3,7}
    //        column 2: rows {0,1,2} vals {1,3,9}
    return CSCMatrix(
        3, 3,
        {2.0, 4.0, 8.0,   1.0, 3.0, 7.0,   1.0, 3.0, 9.0},
        {0,   1,   2,      0,   1,   2,      0,   1,   2  },
        {0,              3,              6,              9  }
    );
}

/**
 * Build a Basis where all 3 columns are basic (identity mapping).
 */
static Basis make_test_basis() {
    Basis b;
    b.col_status = {BasisStatus::Basic, BasisStatus::Basic, BasisStatus::Basic};
    b.basic_indices = {0, 1, 2};
    return b;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("BasisFactorization - NaiveDense FTRAN solves B x = rhs",
          "[factorization][numerics]") {
    CSCMatrix A = make_test_csc();
    Basis basis = make_test_basis();

    NaiveDenseFactorization fac;
    fac.factorize(A, basis);

    // Known solution vector
    std::vector<Float> x_true = {1.0, 2.0, 3.0};

    // Compute rhs = B * x_true
    //   B = | 2 1 1 |    x = | 1 |    rhs = | 2+2+3  | = |  7 |
    //       | 4 3 3 |        | 2 |           | 4+6+9  |   | 19 |
    //       | 8 7 9 |        | 3 |           | 8+14+27|   | 49 |
    std::vector<Float> rhs = {7.0, 19.0, 49.0};

    fac.ftran(rhs);

    REQUIRE(std::abs(rhs[0] - x_true[0]) < kTol);
    REQUIRE(std::abs(rhs[1] - x_true[1]) < kTol);
    REQUIRE(std::abs(rhs[2] - x_true[2]) < kTol);
}

TEST_CASE("BasisFactorization - NaiveDense BTRAN solves B^T x = rhs",
          "[factorization][numerics]") {
    CSCMatrix A = make_test_csc();
    Basis basis = make_test_basis();

    NaiveDenseFactorization fac;
    fac.factorize(A, basis);

    // Known solution vector
    std::vector<Float> x_true = {1.0, 2.0, 3.0};

    // Compute rhs = B^T * x_true
    //   B^T = | 2 4 8 |    x = | 1 |    rhs = | 2+8+24 | = | 34 |
    //         | 1 3 7 |        | 2 |           | 1+6+21 |   | 28 |
    //         | 1 3 9 |        | 3 |           | 1+6+27 |   | 34 |
    std::vector<Float> rhs = {34.0, 28.0, 34.0};

    fac.btran(rhs);

    REQUIRE(std::abs(rhs[0] - x_true[0]) < kTol);
    REQUIRE(std::abs(rhs[1] - x_true[1]) < kTol);
    REQUIRE(std::abs(rhs[2] - x_true[2]) < kTol);
}

TEST_CASE("BasisFactorization - update can be called and overrides interface",
          "[factorization][numerics]") {
    CSCMatrix A = make_test_csc();
    Basis basis = make_test_basis();

    NaiveDenseFactorization fac;
    fac.factorize(A, basis);

    // Replace column 1 (leaving_row=1) with a new column Aq = {10, 20, 30}
    std::vector<Float> Aq = {10.0, 20.0, 30.0};
    fac.update(1, 5, Aq);

    // After update, the dense basis should be:
    //   col0: {2,4,8}   col1: {10,20,30}   col2: {1,3,9}
    // i.e. B' = | 2  10  1 |
    //           | 4  20  3 |
    //           | 8  30  9 |
    //
    // Verify by solving B' x = rhs for a known x_true = {1,1,1}
    //   rhs = B' * {1,1,1} = {2+10+1, 4+20+3, 8+30+9} = {13, 27, 47}
    std::vector<Float> rhs = {13.0, 27.0, 47.0};
    fac.ftran(rhs);

    REQUIRE(std::abs(rhs[0] - 1.0) < kTol);
    REQUIRE(std::abs(rhs[1] - 1.0) < kTol);
    REQUIRE(std::abs(rhs[2] - 1.0) < kTol);
}

TEST_CASE("BasisFactorization - interface can be used via base pointer",
          "[factorization][numerics]") {
    // Verify that the abstract interface works through a base-class pointer.
    std::unique_ptr<sankhya::numerics::BasisFactorization> fac =
        std::make_unique<NaiveDenseFactorization>();

    CSCMatrix A = make_test_csc();
    Basis basis = make_test_basis();

    fac->factorize(A, basis);

    std::vector<Float> rhs = {7.0, 19.0, 49.0};
    fac->ftran(rhs);

    REQUIRE(std::abs(rhs[0] - 1.0) < kTol);
    REQUIRE(std::abs(rhs[1] - 2.0) < kTol);
    REQUIRE(std::abs(rhs[2] - 3.0) < kTol);
}

