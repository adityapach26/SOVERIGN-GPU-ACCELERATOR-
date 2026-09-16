#include <catch2/catch_test_macros.hpp>
#include "ipm/symbolic.hpp"
#include <algorithm>
#include <set>

using namespace sankhya;

TEST_CASE("Phase 14.1: IPM CPU Symbolic Phase and AMD Ordering", "[symbolic][ipm]") {
    // 5x5 Matrix A
    // Designed so AA^T forms a star graph centered at Row 0.
    // Row 0 connects to Rows 1, 2, 3, 4.
    // Rows 1, 2, 3, 4 only connect to Row 0.
    core::CSRMatrix A;
    A.rows = 5;
    A.cols = 5;
    
    // Row 0: cols 0, 1, 2, 3
    // Row 1: col 0
    // Row 2: col 1
    // Row 3: col 2
    // Row 4: col 3
    A.row_ptrs = {0, 4, 5, 6, 7, 8};
    A.col_indices = {0, 1, 2, 3, 0, 1, 2, 3};
    A.values = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    
    // 1. Compute AMD Ordering
    std::vector<Index> P = ipm::compute_amd_ordering(A);
    
    // Requirement 2: Permutation size is exactly 5
    REQUIRE(P.size() == 5);
    
    // Requirement 3 & 4: Contains 0,1,2,3,4 exactly once, no out of range
    std::set<Index> p_set(P.begin(), P.end());
    REQUIRE(p_set.size() == 5);
    for (Index i = 0; i < 5; ++i) {
        REQUIRE(p_set.count(i) == 1);
    }
    
    // Requirement 7 & 8: 
    // Minimum-degree heuristic should eliminate leaves (1, 2, 3, 4) before the center (0)
    // to avoid O(V^2) fill-in clique creation.
    // The tie-breaker prefers lowest index, so the order should be 1, 2, 3, 4, 0.
    std::vector<Index> expected_P = {1, 2, 3, 4, 0};
    REQUIRE(P == expected_P);
    
    // Compute symbolic factorization
    ipm::SymbolicFactorization sym;
    ipm::compute_symbolic_factorization(A, P, sym);
    
    // Requirement 5: Inverse permutation consistency
    REQUIRE(sym.P_inv.size() == 5);
    for (Index i = 0; i < 5; ++i) {
        REQUIRE(sym.P_inv[static_cast<size_t>(P[static_cast<size_t>(i)])] == i);
    }
    
    // Requirement 6: Structural elimination
    // Since we used optimal AMD on a star graph, the fill-in should be exactly zero.
    // L_pattern should have exactly 4 non-zeros (the edges 1-0, 2-0, 3-0, 4-0 in the permuted graph).
    // In the permuted graph, original node 0 is now node 4.
    // Nodes 0, 1, 2, 3 (which were 1, 2, 3, 4) all connect to node 4.
    // So L_pattern (strictly lower triangular) has edges (4,0), (4,1), (4,2), (4,3).
    Index nnz_L = sym.L_pattern.row_ptrs[5];
    REQUIRE(nnz_L == 4);
    
    // Let's verify the parent array (elimination tree)
    // For j=0, 1, 2, 3, their only connection is to node 4, so parent is 4.
    // For j=4 (root), parent is -1.
    REQUIRE(sym.parent[0] == 4);
    REQUIRE(sym.parent[1] == 4);
    REQUIRE(sym.parent[2] == 4);
    REQUIRE(sym.parent[3] == 4);
    REQUIRE(sym.parent[4] == -1);
}

