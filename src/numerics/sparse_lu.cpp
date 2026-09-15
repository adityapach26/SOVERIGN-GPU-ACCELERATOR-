#include "sparse_lu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sankhya {
namespace numerics {

void SparseLUFactorization::factorize(
    const core::CSCMatrix& A,
    const simplex::Basis& basis
) {
    m_ = static_cast<Index>(basis.basic_indices.size());
    const auto mm = static_cast<std::size_t>(m_);

    A_ptr_ = &A;
    basis_copy_ = basis;
    ft_updates_.clear();

    if (mm == 0) {
        factorized_ = true;
        return;
    }

    // Sparse active submatrix representations
    struct Entry {
        Index col;
        Float val;
    };
    // active_rows[orig_r] contains a list of active (orig_c, val)
    std::vector<std::vector<Entry>> active_rows(mm);
    // active_cols[orig_c] contains a list of orig_r that have a non-zero in this column
    std::vector<std::vector<Index>> active_cols(mm);
    
    std::vector<Index> row_nnz(mm, 0);
    std::vector<Index> col_nnz(mm, 0);

    // Extract sparse basis matrix B from A
    for (Index j = 0; j < m_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        const Index A_col = basis.basic_indices[jj];
        const Index start = A.col_ptrs[static_cast<std::size_t>(A_col)];
        const Index end = A.col_ptrs[static_cast<std::size_t>(A_col) + 1];
        
        for (Index k = start; k < end; ++k) {
            const Index r = A.row_indices[static_cast<std::size_t>(k)];
            const Float v = A.values[static_cast<std::size_t>(k)];
            if (r < m_ && std::abs(v) > 1e-15) {
                const auto rr = static_cast<std::size_t>(r);
                active_rows[rr].push_back({j, v});
                active_cols[jj].push_back(r);
                row_nnz[rr]++;
                col_nnz[jj]++;
            }
        }
    }

    std::vector<bool> is_active_row(mm, true);
    std::vector<bool> is_active_col(mm, true);

    perm_row_.resize(mm);
    perm_col_.resize(mm);

    std::vector<std::vector<std::pair<Index, Float>>> U_rows(mm);
    std::vector<std::vector<std::pair<Index, Float>>> L_cols(mm);

    // Sparse Accumulator (SPA) for row updates
    std::vector<Float> spa_val(mm, 0.0);
    std::vector<Index> spa_idx;
    std::vector<bool> in_spa(mm, false);

    // Engineering decision: numerical threshold = 0.1 * max_abs_in_active_submatrix
    static constexpr Float kMarkowitzThreshold = 0.1;
    static constexpr Float kDropTol = 1e-15;

    for (Index step = 0; step < m_; ++step) {
        const auto ss = static_cast<std::size_t>(step);

        // 1. Find max_abs for the numerical threshold
        Float max_abs = 0.0;
        for (Index r = 0; r < m_; ++r) {
            if (!is_active_row[static_cast<std::size_t>(r)]) continue;
            for (const auto& entry : active_rows[static_cast<std::size_t>(r)]) {
                if (is_active_col[static_cast<std::size_t>(entry.col)]) {
                    Float abs_val = std::abs(entry.val);
                    if (abs_val > max_abs) max_abs = abs_val;
                }
            }
        }

        Float threshold = kMarkowitzThreshold * max_abs;
        if (threshold < kDropTol) {
            throw std::runtime_error("SparseLUFactorization: singular or numerically unstable basis");
        }

        // 2. Markowitz pivot search
        Index pivot_r = -1;
        Index pivot_c = -1;
        Index best_cost = std::numeric_limits<Index>::max();
        Float best_abs = 0.0;

        for (Index r = 0; r < m_; ++r) {
            const auto rr = static_cast<std::size_t>(r);
            if (!is_active_row[rr]) continue;
            for (const auto& entry : active_rows[rr]) {
                const auto cc = static_cast<std::size_t>(entry.col);
                if (!is_active_col[cc]) continue;
                
                Float abs_val = std::abs(entry.val);
                if (abs_val >= threshold) {
                    Index cost = (row_nnz[rr] - 1) * (col_nnz[cc] - 1);
                    if (cost < best_cost || (cost == best_cost && abs_val > best_abs)) {
                        best_cost = cost;
                        best_abs = abs_val;
                        pivot_r = r;
                        pivot_c = entry.col;
                    }
                }
            }
        }

        if (pivot_r == -1) {
            throw std::runtime_error("SparseLUFactorization: singular or numerically unstable basis");
        }

        const auto pr = static_cast<std::size_t>(pivot_r);
        const auto pc = static_cast<std::size_t>(pivot_c);

        // Record permutations
        perm_row_[ss] = pivot_r;
        perm_col_[ss] = pivot_c;
        is_active_row[pr] = false;
        is_active_col[pc] = false;

        // 3. Extract pivot row
        Float pivot_val = 0.0;
        for (const auto& entry : active_rows[pr]) {
            if (entry.col == pivot_c) {
                pivot_val = entry.val;
                // Diagonal must be first in U
                U_rows[ss].push_back({pivot_c, pivot_val});
                break;
            }
        }

        std::vector<Entry> pivot_row_active;
        for (const auto& entry : active_rows[pr]) {
            if (entry.col != pivot_c && is_active_col[static_cast<std::size_t>(entry.col)]) {
                U_rows[ss].push_back({entry.col, entry.val});
                pivot_row_active.push_back({entry.col, entry.val});
            }
        }

        // The pivot row is removed from active matrices, so decrement col_nnz
        for (const auto& entry : pivot_row_active) {
            col_nnz[static_cast<std::size_t>(entry.col)]--;
        }
        col_nnz[pc] = 0;

        // 4. Sparse Elimination
        for (Index r : active_cols[pc]) {
            const auto rr = static_cast<std::size_t>(r);
            if (!is_active_row[rr]) continue;

            Float mult = 0.0;
            spa_idx.clear();

            // Scatter row r into SPA
            for (const auto& entry : active_rows[rr]) {
                const auto cc = static_cast<std::size_t>(entry.col);
                if (entry.col == pivot_c) {
                    mult = entry.val / pivot_val;
                } else if (is_active_col[cc]) {
                    spa_val[cc] = entry.val;
                    spa_idx.push_back(entry.col);
                    in_spa[cc] = true;
                    col_nnz[cc]--; // Temporarily remove from col counts
                }
            }

            if (mult != 0.0) {
                L_cols[ss].push_back({r, mult});

                // Apply elimination using pivot_row_active
                for (const auto& p_entry : pivot_row_active) {
                    const auto cc = static_cast<std::size_t>(p_entry.col);
                    if (!in_spa[cc]) {
                        in_spa[cc] = true;
                        spa_idx.push_back(p_entry.col);
                        active_cols[cc].push_back(r); // New fill-in
                    }
                    spa_val[cc] -= mult * p_entry.val;
                }
            }

            // Gather SPA back into active_rows
            active_rows[rr].clear();
            Index new_row_nnz = 0;
            for (Index c : spa_idx) {
                const auto cc = static_cast<std::size_t>(c);
                if (std::abs(spa_val[cc]) > kDropTol) {
                    active_rows[rr].push_back({c, spa_val[cc]});
                    col_nnz[cc]++; // Restore or add to col counts
                    new_row_nnz++;
                }
                spa_val[cc] = 0.0;
                in_spa[cc] = false;
            }
            row_nnz[rr] = new_row_nnz;
        }
    }

    // ---- Build inverse permutations ----
    inv_perm_row_.resize(mm);
    inv_perm_col_.resize(mm);
    for (Index i = 0; i < m_; ++i) {
        inv_perm_row_[static_cast<std::size_t>(perm_row_[static_cast<std::size_t>(i)])] = i;
        inv_perm_col_[static_cast<std::size_t>(perm_col_[static_cast<std::size_t>(i)])] = i;
    }

    // ---- Extract U (CSR, upper triangular) ----
    U_row_ptrs_.resize(mm + 1);
    U_vals_.clear();
    U_cols_.clear();
    for (Index k = 0; k < m_; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        U_row_ptrs_[kk] = static_cast<Index>(U_vals_.size());
        for (const auto& u_entry : U_rows[kk]) {
            U_vals_.push_back(u_entry.second);
            U_cols_.push_back(inv_perm_col_[static_cast<std::size_t>(u_entry.first)]);
        }
    }
    U_row_ptrs_[mm] = static_cast<Index>(U_vals_.size());

    // ---- Extract L (CSC, unit lower triangular) ----
    L_col_ptrs_.resize(mm + 1);
    L_vals_.clear();
    L_rows_.clear();
    for (Index k = 0; k < m_; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        L_col_ptrs_[kk] = static_cast<Index>(L_vals_.size());
        for (const auto& l_entry : L_cols[kk]) {
            L_vals_.push_back(l_entry.second);
            L_rows_.push_back(inv_perm_row_[static_cast<std::size_t>(l_entry.first)]);
        }
    }
    L_col_ptrs_[mm] = static_cast<Index>(L_vals_.size());

    factorized_ = true;
}

void SparseLUFactorization::ftran(std::vector<Float>& rhs) {
    if (!factorized_) {
        throw std::runtime_error("SparseLUFactorization::ftran: not factorized");
    }
    const auto mm = static_cast<std::size_t>(m_);
    std::vector<Float> w(mm);

    for (Index i = 0; i < m_; ++i) {
        w[static_cast<std::size_t>(i)] =
            rhs[static_cast<std::size_t>(perm_row_[static_cast<std::size_t>(i)])];
    }

    for (Index k = 0; k < m_; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        const Index start = L_col_ptrs_[kk];
        const Index end = L_col_ptrs_[kk + 1];
        for (Index p = start; p < end; ++p) {
            const auto pp = static_cast<std::size_t>(p);
            const auto row = static_cast<std::size_t>(L_rows_[pp]);
            w[row] -= L_vals_[pp] * w[kk];
        }
    }

    for (Index k = m_ - 1; k >= 0; --k) {
        const auto kk = static_cast<std::size_t>(k);
        const Index start = U_row_ptrs_[kk];
        const Index end = U_row_ptrs_[kk + 1];
        const Float diag = U_vals_[static_cast<std::size_t>(start)];
        if (std::abs(diag) < 1e-15) {
            throw std::runtime_error("SparseLUFactorization::ftran: zero diagonal in U");
        }
        for (Index p = start + 1; p < end; ++p) {
            const auto pp = static_cast<std::size_t>(p);
            w[kk] -= U_vals_[pp] * w[static_cast<std::size_t>(U_cols_[pp])];
        }
        w[kk] /= diag;
    }

    for (Index i = 0; i < m_; ++i) {
        rhs[static_cast<std::size_t>(perm_col_[static_cast<std::size_t>(i)])] =
            w[static_cast<std::size_t>(i)];
    }

    // Apply incremental Forrest-Tomlin / Eta updates (E_1^{-1} ... E_k^{-1} * rhs)
    for (const auto& ft : ft_updates_) {
        Index p = ft.leaving_row;
        Float x_p = rhs[static_cast<std::size_t>(p)];
        
        for (std::size_t j = 0; j < ft.eta_rows.size(); ++j) {
            Index row = ft.eta_rows[j];
            Float val = ft.eta_vals[j];
            if (row == p) {
                rhs[static_cast<std::size_t>(row)] = val * x_p;
            } else {
                rhs[static_cast<std::size_t>(row)] += val * x_p;
            }
        }
    }
}

void SparseLUFactorization::btran(std::vector<Float>& rhs) {
    if (!factorized_) {
        throw std::runtime_error("SparseLUFactorization::btran: not factorized");
    }

    // Apply incremental Forrest-Tomlin / Eta updates transposed (E_k^{-T} ... E_1^{-T} * rhs)
    for (auto it = ft_updates_.rbegin(); it != ft_updates_.rend(); ++it) {
        const auto& ft = *it;
        Index p = ft.leaving_row;
        
        Float dot = 0.0;
        for (std::size_t j = 0; j < ft.eta_rows.size(); ++j) {
            Index row = ft.eta_rows[j];
            Float val = ft.eta_vals[j];
            dot += val * rhs[static_cast<std::size_t>(row)];
        }
        rhs[static_cast<std::size_t>(p)] = dot;
    }

    const auto mm = static_cast<std::size_t>(m_);
    std::vector<Float> w(mm);

    for (Index i = 0; i < m_; ++i) {
        w[static_cast<std::size_t>(i)] =
            rhs[static_cast<std::size_t>(perm_col_[static_cast<std::size_t>(i)])];
    }

    for (Index k = 0; k < m_; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        const Index start = U_row_ptrs_[kk];
        const Index end = U_row_ptrs_[kk + 1];
        const Float diag = U_vals_[static_cast<std::size_t>(start)];
        if (std::abs(diag) < 1e-15) {
            throw std::runtime_error("SparseLUFactorization::btran: zero diagonal in U");
        }
        w[kk] /= diag;
        for (Index p = start + 1; p < end; ++p) {
            const auto pp = static_cast<std::size_t>(p);
            w[static_cast<std::size_t>(U_cols_[pp])] -= U_vals_[pp] * w[kk];
        }
    }

    for (Index k = m_ - 1; k >= 0; --k) {
        const auto kk = static_cast<std::size_t>(k);
        const Index start = L_col_ptrs_[kk];
        const Index end = L_col_ptrs_[kk + 1];
        for (Index p = start; p < end; ++p) {
            const auto pp = static_cast<std::size_t>(p);
            w[kk] -= L_vals_[pp] * w[static_cast<std::size_t>(L_rows_[pp])];
        }
    }

    for (Index i = 0; i < m_; ++i) {
        rhs[static_cast<std::size_t>(perm_row_[static_cast<std::size_t>(i)])] =
            w[static_cast<std::size_t>(i)];
    }
}

void SparseLUFactorization::update(
    Index leaving_row,
    Index entering_col,
    const std::vector<Float>& Aq
) {
    if (A_ptr_ == nullptr || !factorized_) {
        throw std::runtime_error("SparseLUFactorization::update: no prior factorize() call");
    }

    // 1. Construct the transformed entering column in factorized coordinates
    // We need d = B_{current}^{-1} A_q.
    std::vector<Float> d = Aq;
    ftran(d);

    Float d_p = d[static_cast<std::size_t>(leaving_row)];
    
    // Engineering decision: numerical pivot tolerance
    if (std::abs(d_p) < 1e-9) { 
        // Numerical instability, force refactorization
        basis_copy_.basic_indices[static_cast<std::size_t>(leaving_row)] = entering_col;
        factorize(*A_ptr_, basis_copy_);
        return;
    }

    // 2. Derive the Product-Form / Forrest-Tomlin update transformation (Eta vector)
    FTUpdate ft;
    ft.leaving_row = leaving_row;
    ft.entering_col = entering_col;

    // Eta vector \eta = E^{-1} e_p
    // \eta_p = 1 / d_p
    // \eta_i = -d_i / d_p  for i != p
    for (Index i = 0; i < m_; ++i) {
        Float val = (i == leaving_row) ? (1.0 / d_p) : (-d[static_cast<std::size_t>(i)] / d_p);
        if (std::abs(val) > 1e-15) {
            ft.eta_rows.push_back(i);
            ft.eta_vals.push_back(val);
        }
    }

    // 3. Update the maintained factor representation incrementally
    ft_updates_.push_back(std::move(ft));

    // 4. Update the cached basis to maintain structural consistency
    basis_copy_.basic_indices[static_cast<std::size_t>(leaving_row)] = entering_col;
}

bool SparseLUFactorization::needs_refactorization(
    const std::vector<Float>& x_B,
    const std::vector<Float>& b_B
) const {
    if (!factorized_) return true;
    
    // Evaluate max residual norm ||B x_B - b_B||_inf
    // We compute B * x_B
    std::vector<Float> B_xB(static_cast<std::size_t>(m_), 0.0);
    for (Index j = 0; j < m_; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        const Index col = basis_copy_.basic_indices[jj];
        const Index start = A_ptr_->col_ptrs[static_cast<std::size_t>(col)];
        const Index end = A_ptr_->col_ptrs[static_cast<std::size_t>(col) + 1];
        
        for (Index k = start; k < end; ++k) {
            const Index r = A_ptr_->row_indices[static_cast<std::size_t>(k)];
            if (r < m_) {
                B_xB[static_cast<std::size_t>(r)] += A_ptr_->values[static_cast<std::size_t>(k)] * x_B[jj];
            }
        }
    }
    
    Float max_res = 0.0;
    for (Index i = 0; i < m_; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        Float res = std::abs(B_xB[ii] - b_B[ii]);
        if (res > max_res) {
            max_res = res;
        }
    }
    
    // Engineering decision: default residual tolerance for refactorization
    return max_res > sankhya::math::kDefaultFeasibilityTol;
}

} // namespace numerics
} // namespace sankhya
