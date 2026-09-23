#include <cstddef>
#include <vector>
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
    // Minimum-degree heuristic eliminates leaves to avoid O(V^2) fill-in.
    // Nodes 1, 2, 3 are eliminated first. This dynamically reduces the degree of the center node (0) to 1.
    // At step 4, Nodes 0 and 4 both have degree 1. The tie-breaker strictly prefers the lowest index, so 0 beats 4.
    // Therefore, the true elimination order is 1, 2, 3, 0, 4.
    std::vector<Index> expected_P = {1, 2, 3, 0, 4};
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
    // L_pattern should have exactly 4 non-zeros.
    // In the permuted graph, original node 0 is now node 3.
    // Original nodes 1, 2, 3 (permuted to 0, 1, 2) connect to permuted node 3.
    // Original node 4 (permuted to 4) connects to permuted node 3.
    // So L_pattern (strictly lower triangular) has edges (3,0), (3,1), (3,2), (4,3).
    Index nnz_L = sym.L_pattern.row_ptrs[5];
    REQUIRE(nnz_L == 4);
    
    // Let's verify the parent array (elimination tree)
    // For j=0, 1, 2, their only strictly lower connection is to node 3, so parent is 3.
    // For j=3, its strictly lower connection is to node 4, so parent is 4.
    // For j=4 (root), parent is -1.
    REQUIRE(sym.parent[0] == 3);
    REQUIRE(sym.parent[1] == 3);
    REQUIRE(sym.parent[2] == 3);
    REQUIRE(sym.parent[3] == 4);
    REQUIRE(sym.parent[4] == -1);
}

TEST_CASE("Phase 14.1: IPM CPU Symbolic Fill-in Verification", "[symbolic][ipm][fill]") {
    // 5x5 Matrix A
    // Designed so AA^T forms a 4-cycle among nodes 0, 1, 2, 3, and node 4 is isolated.
    // Cycle: 0-1-2-3-0.
    core::CSRMatrix A;
    A.rows = 5;
    A.cols = 5;
    
    // Row 0: cols 0, 3
    // Row 1: cols 0, 1
    // Row 2: cols 1, 2
    // Row 3: cols 2, 3
    // Row 4: col 4
    A.row_ptrs = {0, 2, 4, 6, 8, 9};
    A.col_indices = {0, 3, 0, 1, 1, 2, 2, 3, 4};
    A.values = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    
    // 1. Compute AMD Ordering
    std::vector<Index> P = ipm::compute_amd_ordering(A);
    
    // Minimum-degree heuristic trace:
    // Node 4 has degree 0 -> eliminated first (P[0] = 4).
    // Nodes 0, 1, 2, 3 all have degree 2. Tie-break (lowest index) -> Node 0 eliminated (P[1] = 0).
    // Eliminating 0 creates a FILL EDGE between 1 and 3.
    // Nodes 1, 2, 3 all have degree 2. Tie-break -> Node 1 eliminated (P[2] = 1).
    // Eliminating 1 (neighbors 2, 3) creates no new fill since 2-3 exists.
    // Node 2 has degree 1 (neighbor 3). Node 3 has degree 1 (neighbor 2).
    // Tie-break -> Node 2 eliminated (P[3] = 2).
    // Node 3 eliminated (P[4] = 3).
    std::vector<Index> expected_P = {4, 0, 1, 2, 3};
    REQUIRE(P == expected_P);
    
    // Compute symbolic factorization
    ipm::SymbolicFactorization sym;
    ipm::compute_symbolic_factorization(A, P, sym);
    
    // Verify L_pattern fill-in.
    // In the permuted graph M (ordered by elimination):
    // orig 4 -> perm 0
    // orig 0 -> perm 1
    // orig 1 -> perm 2
    // orig 2 -> perm 3
    // orig 3 -> perm 4
    // 
    // Original M strictly lower edges:
    // (2,1) [orig 1-0], (4,1) [orig 3-0]
    // (3,2) [orig 2-1]
    // (4,3) [orig 3-2]
    // Total original edges = 4.
    //
    // The symbolic elimination adds the fill edge (4,2) [orig 3-1].
    // Total non-zeros in L_pattern = 5.
    
    Index nnz_L = sym.L_pattern.row_ptrs[5];
    REQUIRE(nnz_L == 5);
    
    std::vector<Index> expected_row_ptrs = {0, 0, 0, 1, 2, 5};
    for (Index i = 0; i <= 5; ++i) {
        REQUIRE(sym.L_pattern.row_ptrs[i] == expected_row_ptrs[i]);
    }
    
    std::vector<Index> expected_col_indices = {1, 2, 1, 2, 3};
    for (size_t i = 0; i < 5; ++i) {
        REQUIRE(sym.L_pattern.col_indices[i] == expected_col_indices[i]);
    }
    
    // Verify elimination tree parent array
    std::vector<Index> expected_parent = {-1, 2, 3, 4, -1};
    for (size_t i = 0; i < 5; ++i) {
        REQUIRE(sym.parent[i] == expected_parent[i]);
    }
}
