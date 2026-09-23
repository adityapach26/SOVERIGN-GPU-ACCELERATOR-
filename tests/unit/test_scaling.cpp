#include <cstddef>
/**
 * @file test_scaling.cpp
 * @brief Unit tests for Sovereign Presolve Pipeline - Ruiz Equilibrium Scaling
 */

#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

#include "core/problem.hpp"
#include "presolve/scaling.hpp"

using namespace sankhya;

TEST_CASE("Ruiz Scaling - Conditioning and Reversibility", "[presolve][scaling]") {
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    
    model.add_variable(1.0, 10.0, 20.0, VariableType::Continuous); // x0
    model.add_variable(2.0, 0.0, math::kInfinity, VariableType::Continuous); // x1
    
    // Ill-conditioned constraints
    // R_before = 1e6 / 1e-4 = 1e10
    model.add_constraint({0, 1}, {1e6, 1e-4}, 1e3);
    model.add_constraint({0, 1}, {1e-4, 1e6}, 1e3);
    
    model.finalize();
    
    // Calculate R_before
    Float max_val_before = 0.0;
    Float min_val_before = math::kInfinity;
    for (std::size_t k = 0; k < model.A.values.size(); ++k) {
        Float val = std::abs(model.A.values[k]);
        if (val > max_val_before) max_val_before = val;
        if (val > 0.0 && val < min_val_before) min_val_before = val;
    }
    Float r_before = max_val_before / min_val_before;
    
    // Apply Ruiz Scaling
    presolve::apply_ruiz_scaling(model, 20); // allow up to 20 iterations
    
    // Calculate R_after
    Float max_val_after = 0.0;
    Float min_val_after = math::kInfinity;
    for (std::size_t k = 0; k < model.A.values.size(); ++k) {
        Float val = std::abs(model.A.values[k]);
        if (val > max_val_after) max_val_after = val;
        if (val > 0.0 && val < min_val_after) min_val_after = val;
    }
    Float r_after = max_val_after / min_val_after;
    
    // Test Conditioning: R_after < R_before
    REQUIRE(r_after < r_before);
    
    // Test Ruiz Behavior: Active norms approach 1.0
    // Col norms
    for (Index j = 0; j < model.A.cols; ++j) {
        Float col_norm = 0.0;
        Index start = model.A.col_ptrs[static_cast<std::size_t>(j)];
        Index end = model.A.col_ptrs[static_cast<std::size_t>(j)+1];
        for (Index k = start; k < end; ++k) {
            Float val = std::abs(model.A.values[static_cast<std::size_t>(k)]);
            if (val > col_norm) col_norm = val;
        }
        REQUIRE(std::abs(col_norm - 1.0) < 1e-2);
    }
    
    // Row norms
    std::vector<Float> row_norms(static_cast<std::size_t>(model.A.rows), 0.0);
    for (std::size_t k = 0; k < model.A.values.size(); ++k) {
        Float val = std::abs(model.A.values[k]);
        Index row = model.A.row_indices[k];
        if (val > row_norms[static_cast<std::size_t>(row)]) {
            row_norms[static_cast<std::size_t>(row)] = val;
        }
    }
    for (Index i = 0; i < model.A.rows; ++i) {
        REQUIRE(std::abs(row_norms[static_cast<std::size_t>(i)] - 1.0) < 1e-2);
    }
    
    // Test Reversibility
    // Given a hypothetical scaled solution x_s = (1.0, 1.0)
    std::vector<Float> x_s = {1.0, 1.0};
    
    // Original x = D_c * x_s
    std::vector<Float> x_orig = {
        model.col_scale[0] * x_s[0],
        model.col_scale[1] * x_s[1]
    };
    
    // Verify transformed RHS relationship: b_s = D_r * b
    // The original RHS was 1e3.
    REQUIRE(std::abs(model.rhs[0] - (model.row_scale[0] * 1e3)) < 1e-9);
    REQUIRE(std::abs(model.rhs[1] - (model.row_scale[1] * 1e3)) < 1e-9);
    
    // Verify transformed bounds: lb_s = D_c^{-1} * lb
    // Original lb[0] was 10.0
    Float expected_lb_s = 10.0 / model.col_scale[0];
    REQUIRE(std::abs(model.lb[0] - expected_lb_s) < 1e-9);
    
    // Verify transformed objective: c_s = D_c * c
    // Original c[0] was 1.0
    Float expected_c_s = 1.0 * model.col_scale[0];
    REQUIRE(std::abs(model.obj[0] - expected_c_s) < 1e-9);
}

TEST_CASE("Ruiz Scaling - Integer Variables Unchanged", "[presolve][scaling]") {
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    
    model.add_variable(1.0, 0.0, 1.0, VariableType::Binary); // integer
    model.add_variable(2.0, 0.0, math::kInfinity, VariableType::Continuous); // continuous
    
    model.add_constraint({0, 1}, {100.0, 100.0}, 50.0);
    model.finalize();
    
    presolve::apply_ruiz_scaling(model);
    
    // The integer variable (col 0) should have col_scale == 1.0 exactly
    REQUIRE(model.col_scale[0] == 1.0);
    // The continuous variable (col 1) should be scaled
    REQUIRE(model.col_scale[1] != 1.0);
}

