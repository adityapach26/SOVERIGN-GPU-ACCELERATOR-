/**
 * @file test_problem.cpp
 * @brief Unit tests for the LP data model (Phase 3, Step 3.1)
 */

#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include "core/problem.hpp"

using sankhya::Float;
using sankhya::Index;
using sankhya::OptimizationSense;
using sankhya::VariableType;
using sankhya::core::Model;

TEST_CASE("Model - populates objective, bounds, variable types", "[problem][core]") {
    Model m;
    m.sense = OptimizationSense::Maximize;
    m.add_variable(2.0, 0.0, 10.0, VariableType::Continuous);
    m.add_variable(3.0, -1.0, 5.0, VariableType::Integer);

    REQUIRE(m.sense == OptimizationSense::Maximize);
    REQUIRE(m.obj == std::vector<Float>{2.0, 3.0});
    REQUIRE(m.lb == std::vector<Float>{0.0, -1.0});
    REQUIRE(m.ub == std::vector<Float>{10.0, 5.0});
    REQUIRE(m.vtype.size() == 2);
    REQUIRE(m.vtype[0] == VariableType::Continuous);
    REQUIRE(m.vtype[1] == VariableType::Integer);
}

TEST_CASE("Model - default bounds and variable type", "[problem][core]") {
    Model m;
    m.add_variable(1.0);

    REQUIRE(m.obj.size() == 1);
    // default lower bound = 0, default upper bound = +infinity, default type Continuous
    REQUIRE(m.lb.back() == 0.0);
    REQUIRE(m.ub.back() == sankhya::math::kInfinity);
    REQUIRE(m.vtype.back() == VariableType::Continuous);
}

TEST_CASE("Model - finalize builds CSC constraint matrix", "[problem][core]") {
    Model m;
    m.add_variable(2.0, 0.0, 10.0, VariableType::Continuous);
    m.add_variable(3.0, 0.0, 5.0, VariableType::Continuous);

    // row 0: x0 + 2 x1 = 4
    m.add_constraint({0, 1}, {1.0, 2.0}, 4.0);
    // row 1: x1 = 1
    m.add_constraint({1}, {1.0}, 1.0);

    m.finalize();

    // Consistent dimensions
    REQUIRE(m.A.rows == 2);
    REQUIRE(m.A.cols == 2);
    REQUIRE(static_cast<Index>(m.rhs.size()) == m.A.rows);
    REQUIRE(m.A.cols == static_cast<Index>(m.obj.size()));
    REQUIRE(m.A.cols == static_cast<Index>(m.lb.size()));
    REQUIRE(m.A.cols == static_cast<Index>(m.ub.size()));
    REQUIRE(m.A.cols == static_cast<Index>(m.vtype.size()));

    // Exact CSC arrays: col 0 -> {(row0,1)}, col 1 -> {(row0,2),(row1,1)}
    REQUIRE(m.A.col_ptrs == std::vector<Index>{0, 1, 3});
    REQUIRE(m.A.row_indices == std::vector<Index>{0, 0, 1});
    REQUIRE(m.A.values == std::vector<Float>{1.0, 2.0, 1.0});
    REQUIRE(m.A.nnz() == 3);

    // Stored coefficients by position
    REQUIRE(m.A.get(0, 0) == 1.0);
    REQUIRE(m.A.get(0, 1) == 2.0);
    REQUIRE(m.A.get(1, 1) == 1.0);
    // Implicit zero
    REQUIRE(m.A.get(1, 0) == 0.0);
}

TEST_CASE("Model - empty model finalizes to empty matrix", "[problem][core]") {
    Model m;
    m.finalize();

    REQUIRE(m.A.rows == 0);
    REQUIRE(m.A.cols == 0);
    REQUIRE(m.A.col_ptrs == std::vector<Index>{0});
    REQUIRE(m.A.values.empty());
    REQUIRE(m.A.nnz() == 0);
}

TEST_CASE("Model - add_constraint rejects malformed input", "[problem][core]") {
    Model m;
    m.add_variable(1.0, 0.0, 10.0, VariableType::Continuous);

    // cols / vals size mismatch
    REQUIRE_THROWS_AS(m.add_constraint({0}, {1.0, 2.0}, 3.0), std::invalid_argument);
    // out-of-range column index
    REQUIRE_THROWS_AS(m.add_constraint({2}, {1.0}, 3.0), std::invalid_argument);
    // negative column index
    REQUIRE_THROWS_AS(m.add_constraint({-1}, {1.0}, 3.0), std::invalid_argument);
}

TEST_CASE("Model - finalize rejects inconsistent dimensions", "[problem][core]") {
    Model m;
    m.add_variable(1.0, 0.0, 10.0, VariableType::Continuous);
    m.add_constraint({0}, {1.0}, 1.0);

    // Corrupt one of the public vectors so sizes no longer agree.
    m.lb.push_back(0.0);

    REQUIRE_THROWS_AS(m.finalize(), std::invalid_argument);
}