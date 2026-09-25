#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "heuristics/rounding.hpp"
#include "heuristics/diving.hpp"
#include "core/problem.hpp"
#include "sankhya/types.hpp"

using namespace sankhya;

TEST_CASE("Phase 22.1: Rounding Heuristic - Strictly Feasible Candidate", "[heuristics][rounding]") {
    // Construct a small mock model:
    // x0 (continuous), x1 (integer)
    // x0 + x1 = 3
    // lb: 0, 0
    // ub: 10, 10
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    model.obj = {1.0, 1.0};
    model.lb = {0.0, 0.0};
    model.ub = {10.0, 10.0};
    model.vtype = {VariableType::Continuous, VariableType::Integer};
    model.A.rows = 1;
    model.A.cols = 2;
    model.A.col_ptrs = {0, 1, 2};
    model.A.row_indices = {0, 0};
    model.A.values = {1.0, 1.0};
    model.rhs = {3.0};

    // LP relaxation solution: x0 = 0.5, x1 = 2.5
    // Note: 0.5 + 2.5 = 3 (satisfies constraint)
    std::vector<Float> lp_x = {0.5, 2.5};

    // If we round x1 to nearest integer -> 3.0
    // But wait! If we round x1 to 3.0, the constraint x0 + x1 = 3 requires x0 = 0.0.
    // The simple rounding heuristic ONLY rounds integer variables, it does NOT solve for x0!
    // So if x0 remains 0.5 and x1 becomes 3.0, 0.5 + 3.0 = 3.5 != 3.0 -> INFEASIBLE.
    
    // To construct a STRICTLY FEASIBLE nearest integer point under pure rounding:
    // The continuous variables MUST happen to satisfy the constraint with the rounded integers.
    // Let's design a model where rounding produces an EXACTLY feasible point.
    // Model:
    // x0 (continuous), x1 (integer), x2 (integer)
    // Constraint: 1.0 * x0 + 0.0 * x1 + 0.0 * x2 = 0.5
    // So x0 MUST be 0.5. x1 and x2 can be anything.
    // Let's add another constraint: x1 + x2 = 3.
    // Wait, if x1 + x2 = 3, and LP x is x1 = 1.1, x2 = 1.9, they sum to 3.0.
    // Rounding gives x1 = 1.0, x2 = 2.0.
    // Sum = 1.0 + 2.0 = 3.0! This will be perfectly feasible!
    
    core::Model ok_model;
    ok_model.sense = OptimizationSense::Minimize;
    ok_model.obj = {1.0, 1.0, 1.0};
    ok_model.lb = {0.0, 0.0, 0.0};
    ok_model.ub = {10.0, 10.0, 10.0};
    ok_model.vtype = {VariableType::Continuous, VariableType::Integer, VariableType::Integer};
    ok_model.A.rows = 2;
    ok_model.A.cols = 3;
    // C0: x0 = 0.5 -> col0: {row0, 1.0}
    // C1: x1 + x2 = 3.0 -> col1: {row1, 1.0}, col2: {row1, 1.0}
    ok_model.A.col_ptrs = {0, 1, 2, 3};
    ok_model.A.row_indices = {0, 1, 1};
    ok_model.A.values = {1.0, 1.0, 1.0};
    ok_model.rhs = {0.5, 3.0};
    
    std::vector<Float> ok_lp_x = {0.5, 1.1, 1.9};
    std::vector<Float> incumbent;
    
    bool result = heuristics::apply_rounding_heuristic(ok_model, ok_lp_x, incumbent);
    
    REQUIRE(result == true); // Heuristic reports success
    REQUIRE(incumbent.size() == 3);
    REQUIRE(incumbent[0] == 0.5); // continuous variable remains unchanged
    REQUIRE(incumbent[1] == 1.0); // 1.1 rounded to 1.0
    REQUIRE(incumbent[2] == 2.0); // 1.9 rounded to 2.0
}

TEST_CASE("Phase 22.1: Rounding Heuristic - Infeasible Candidate", "[heuristics][rounding]") {
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    model.obj = {1.0, 1.0};
    model.lb = {0.0, 0.0};
    model.ub = {10.0, 10.0};
    model.vtype = {VariableType::Continuous, VariableType::Integer};
    model.A.rows = 1;
    model.A.cols = 2;
    model.A.col_ptrs = {0, 1, 2};
    model.A.row_indices = {0, 0};
    model.A.values = {1.0, 1.0};
    model.rhs = {3.0};

    // LP x: 0.5 + 2.5 = 3
    std::vector<Float> lp_x = {0.5, 2.5};
    std::vector<Float> incumbent;
    
    // Rounding makes x1=3.0, but leaves x0=0.5 -> 3.5 != 3.0 (INFEASIBLE)
    bool result = heuristics::apply_rounding_heuristic(model, lp_x, incumbent);
    
    REQUIRE(result == false);
}

