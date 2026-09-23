#include <cstddef>
#include <vector>
#include "ipm/symbolic.hpp"
#include <algorithm>
#include <stdexcept>
#include <set>

namespace sankhya {
namespace ipm {

std::vector<Index> compute_amd_ordering(const core::CSRMatrix& A) {
    const Index m = A.rows;
    const Index n = A.cols;

    // 1. Build A^T representation (column to rows mapping)
    std::vector<std::vector<Index>> col_to_rows(n);
    for (Index i = 0; i < m; ++i) {
        for (Index p = A.row_ptrs[static_cast<size_t>(i)]; p < A.row_ptrs[static_cast<size_t>(i+1)]; ++p) {
            col_to_rows[static_cast<size_t>(A.col_indices[static_cast<size_t>(p)])].push_back(i);
        }
    }

    // 2. Build AA^T structural adjacency graph
    std::vector<std::vector<Index>> adj(m);
    for (Index k = 0; k < n; ++k) {
        const auto& rows_k = col_to_rows[static_cast<size_t>(k)];
        for (size_t i = 0; i < rows_k.size(); ++i) {
            for (size_t j = i + 1; j < rows_k.size(); ++j) {
                adj[static_cast<size_t>(rows_k[i])].push_back(rows_k[j]);
                adj[static_cast<size_t>(rows_k[j])].push_back(rows_k[i]);
            }
        }
    }

    for (Index i = 0; i < m; ++i) {
        auto& row_adj = adj[static_cast<size_t>(i)];
        std::sort(row_adj.begin(), row_adj.end());
        row_adj.erase(std::unique(row_adj.begin(), row_adj.end()), row_adj.end());
        
        // Remove self-loops
        auto it = std::lower_bound(row_adj.begin(), row_adj.end(), i);
        if (it != row_adj.end() && *it == i) {
            row_adj.erase(it);
        }
    }

    // 3. Minimum degree elimination
    std::vector<Index> P;
    P.reserve(static_cast<size_t>(m));
    std::vector<bool> eliminated(m, false);

    for (Index step = 0; step < m; ++step) {
        Index min_deg_u = -1;
        size_t min_deg = m + 1;

        // Find active vertex with minimum degree
        // Tie-breaking: lower vertex index wins implicitly by looping 0 to m-1
        for (Index i = 0; i < m; ++i) {
            if (!eliminated[static_cast<size_t>(i)]) {
                size_t deg = adj[static_cast<size_t>(i)].size();
                if (deg < min_deg) {
                    min_deg = deg;
                    min_deg_u = i;
                }
            }
        }

        if (min_deg_u == -1) {
            throw std::runtime_error("compute_amd_ordering: Disconnected graph or invalid state");
        }

        eliminated[static_cast<size_t>(min_deg_u)] = true;
        P.push_back(min_deg_u);

        // Fill-in: connect all uneliminated neighbors to each other
        const auto& neighbors = adj[static_cast<size_t>(min_deg_u)];
        for (size_t i = 0; i < neighbors.size(); ++i) {
            Index v = neighbors[i];
            auto& adj_v = adj[static_cast<size_t>(v)];
            
            // Remove min_deg_u from adj[v]
            adj_v.erase(std::remove(adj_v.begin(), adj_v.end(), min_deg_u), adj_v.end());
            
            // Add other neighbors to adj[v] to create a clique
            for (size_t j = 0; j < neighbors.size(); ++j) {
                if (i != j) {
                    Index w = neighbors[j];
                    auto it = std::lower_bound(adj_v.begin(), adj_v.end(), w);
                    if (it == adj_v.end() || *it != w) {
                        adj_v.insert(it, w);
                    }
                }
            }
        }
    }

    return P;
}

void compute_symbolic_factorization(
    const core::CSRMatrix& A,
    const std::vector<Index>& P,
    SymbolicFactorization& out_factorization
) {
    const Index m = A.rows;
    const Index n = A.cols;

    out_factorization.P = P;
    
    // 1. Construct inverse permutation
    out_factorization.P_inv.resize(static_cast<size_t>(m));
    for (Index i = 0; i < m; ++i) {
        out_factorization.P_inv[static_cast<size_t>(P[static_cast<size_t>(i)])] = i;
    }

    // 2. Build A A^T structural adjacency (for the permuted matrix)
    // Permuted matrix M = P_inv * A A^T * P_inv^T
    // Wait, the convention is P[i] = original_index.
    // So row i of M corresponds to row P[i] of original A.
    std::vector<std::vector<Index>> M_adj(m);
    
    std::vector<std::vector<Index>> col_to_rows(n);
    for (Index i = 0; i < m; ++i) {
        for (Index p = A.row_ptrs[static_cast<size_t>(i)]; p < A.row_ptrs[static_cast<size_t>(i+1)]; ++p) {
            col_to_rows[static_cast<size_t>(A.col_indices[static_cast<size_t>(p)])].push_back(i);
        }
    }

    for (Index k = 0; k < n; ++k) {
        const auto& rows_k = col_to_rows[static_cast<size_t>(k)];
        for (size_t i = 0; i < rows_k.size(); ++i) {
            for (size_t j = i + 1; j < rows_k.size(); ++j) {
                Index orig_u = rows_k[i];
                Index orig_v = rows_k[j];
                Index perm_u = out_factorization.P_inv[static_cast<size_t>(orig_u)];
                Index perm_v = out_factorization.P_inv[static_cast<size_t>(orig_v)];
                
                // M_adj only needs strictly lower triangular part (i > j)
                if (perm_u > perm_v) {
                    M_adj[static_cast<size_t>(perm_v)].push_back(perm_u);
                } else if (perm_v > perm_u) {
                    M_adj[static_cast<size_t>(perm_u)].push_back(perm_v);
                }
            }
        }
    }

    for (Index j = 0; j < m; ++j) {
        auto& col_adj = M_adj[static_cast<size_t>(j)];
        std::sort(col_adj.begin(), col_adj.end());
        col_adj.erase(std::unique(col_adj.begin(), col_adj.end()), col_adj.end());
    }

    // 3. Symbolic Cholesky to compute elimination tree and fill pattern
    out_factorization.parent.assign(static_cast<size_t>(m), -1);
    
    std::vector<std::vector<Index>> L_cols(m);
    std::vector<Index> ancestor(m, -1);

    for (Index j = 0; j < m; ++j) {
        std::set<Index> struct_j;
        // Initial structure from M
        for (Index i : M_adj[static_cast<size_t>(j)]) {
            struct_j.insert(i);
        }
        
        // Find descendants in elimination tree whose first off-diagonal is j
        for (Index i = 0; i < j; ++i) {
            if (out_factorization.parent[static_cast<size_t>(i)] == j) {
                for (Index row : L_cols[static_cast<size_t>(i)]) {
                    if (row > j) {
                        struct_j.insert(row);
                    }
                }
            }
        }
        
        // The first off-diagonal becomes the parent in the elimination tree
        if (!struct_j.empty()) {
            out_factorization.parent[static_cast<size_t>(j)] = *struct_j.begin();
        }
        
        // Save L column structure
        L_cols[static_cast<size_t>(j)].assign(struct_j.begin(), struct_j.end());
    }

    // 4. Construct CSR pattern for L (strictly lower triangular)
    // Wait, the structure we computed is column-major. We must convert it to CSR.
    out_factorization.L_pattern.rows = m;
    out_factorization.L_pattern.cols = m;
    out_factorization.L_pattern.row_ptrs.assign(static_cast<size_t>(m + 1), 0);
    
    // Count nonzeros per row
    for (Index j = 0; j < m; ++j) {
        for (Index i : L_cols[static_cast<size_t>(j)]) {
            out_factorization.L_pattern.row_ptrs[static_cast<size_t>(i + 1)]++;
        }
    }
    
    // Prefix sum
    for (Index i = 0; i < m; ++i) {
        out_factorization.L_pattern.row_ptrs[static_cast<size_t>(i + 1)] += 
            out_factorization.L_pattern.row_ptrs[static_cast<size_t>(i)];
    }
    
    Index nnz = out_factorization.L_pattern.row_ptrs[static_cast<size_t>(m)];
    out_factorization.L_pattern.col_indices.resize(static_cast<size_t>(nnz));
    out_factorization.L_pattern.values.assign(static_cast<size_t>(nnz), 1.0); // Dummy values
    
    // Fill col_indices
    std::vector<Index> current_row_ptr = out_factorization.L_pattern.row_ptrs;
    for (Index j = 0; j < m; ++j) {
        for (Index i : L_cols[static_cast<size_t>(j)]) {
            Index dest = current_row_ptr[static_cast<size_t>(i)]++;
            out_factorization.L_pattern.col_indices[static_cast<size_t>(dest)] = j;
        }
    }
}

} // namespace ipm
} // namespace sankhya

