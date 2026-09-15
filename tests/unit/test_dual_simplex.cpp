/**
 * @file test_dual_simplex.cpp
 * @brief Unit tests for CPU Dual Simplex (Step 7.1)
 */

#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cmath>

#include "core/problem.hpp"
#include "simplex/basis.hpp"
#include "simplex/dual.hpp"
#include "numerics/sparse_lu.hpp"

using namespace sankhya;

TEST_CASE("Dual Simplex - Primal infeasible but dual feasible", "[dual_simplex][simplex]") {
    // Problem:
    // Min 2 x0 + 3 x1
    // s.t.
    // -x0 - x1 + s0 = -2
    //  x0 - 2 x1 + s1 = -1
    // x0, x1, s0, s1 >= 0
    //
    // Initial basis: {s0, s1} -> B = I
    // Initial basic solution: s0 = -2, s1 = -1 (primal infeasible)
    // Initial reduced costs: c_0 = 2 >= 0, c_1 = 3 >= 0 (dual feasible)
    
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    
    model.add_variable(2.0, 0.0, math::kInfinity, VariableType::Continuous); // x0
    model.add_variable(3.0, 0.0, math::kInfinity, VariableType::Continuous); // x1
    model.add_variable(0.0, 0.0, math::kInfinity, VariableType::Continuous); // s0
    model.add_variable(0.0, 0.0, math::kInfinity, VariableType::Continuous); // s1
    
    model.add_constraint({0, 1, 2}, {-1.0, -1.0, 1.0}, -2.0);
    model.add_constraint({0, 1, 3}, {1.0, -2.0, 1.0}, -1.0);
    
    model.finalize();
    
    simplex::Basis basis;
    basis.col_status = {
        simplex::BasisStatus::AtLower, 
        simplex::BasisStatus::AtLower, 
        simplex::BasisStatus::Basic, 
        simplex::BasisStatus::Basic
    };
    basis.basic_indices = {2, 3};
    
    std::vector<Float> x = {0.0, 0.0, -2.0, -1.0};
    
    numerics::SparseLUFactorization factorizer;
    
    // Step 1: Initial state is primal infeasible but dual feasible
    REQUIRE(x[2] < -math::kDefaultFeasibilityTol);
    
    simplex::SimplexStatus status = simplex::dual_simplex_phase2(model, basis, x, factorizer);
    
    REQUIRE(status == simplex::SimplexStatus::Optimal);
    
    // Check final solution
    REQUIRE(std::abs(x[0] - 1.0) < 1e-6);
    REQUIRE(std::abs(x[1] - 1.0) < 1e-6);
    REQUIRE(std::abs(x[2] - 0.0) < 1e-6);
    REQUIRE(std::abs(x[3] - 0.0) < 1e-6);
    
    // Check objective
    Float obj_val = 2.0 * x[0] + 3.0 * x[1];
    REQUIRE(std::abs(obj_val - 5.0) < 1e-6);
    
    // Ensure primal feasibility
    for (Float val : x) {
        REQUIRE(val >= -math::kDefaultFeasibilityTol);
    }
}

