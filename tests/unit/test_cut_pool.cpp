#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "cuts/cut_pool.hpp"

using namespace sankhya;
using namespace sankhya::cuts;

TEST_CASE("Phase 20.1: Required example - Violated Cut", "[cuts][cut_pool]") {
    CutPool pool;
    Cut cut;
    cut.coefficients = {1.0, 1.0};
    cut.rhs = 1.0;

    pool.add_cut_to_pool(cut);

    std::vector<Float> x = {0.6, 0.6}; // lhs = 1.2

    auto violated = pool.get_violated_cuts(x);
    REQUIRE(violated.size() == 1);
    REQUIRE(violated[0].coefficients == cut.coefficients);
    REQUIRE(violated[0].rhs == cut.rhs);
}

TEST_CASE("Phase 20.1: Satisfied cut", "[cuts][cut_pool]") {
    CutPool pool;
    Cut cut;
    cut.coefficients = {1.0, 1.0};
    cut.rhs = 1.0;

    pool.add_cut_to_pool(cut);

    std::vector<Float> x = {0.4, 0.4}; // lhs = 0.8

    auto violated = pool.get_violated_cuts(x);
    REQUIRE(violated.empty());
}

TEST_CASE("Phase 20.1: Boundary tolerance", "[cuts][cut_pool]") {
    CutPool pool;
    Cut cut;
    cut.coefficients = {1.0, 1.0};
    cut.rhs = 1.0;

    pool.add_cut_to_pool(cut);

    // Exactly satisfied
    std::vector<Float> x_exact = {0.5, 0.5}; // lhs = 1.0
    REQUIRE(pool.get_violated_cuts(x_exact).empty());

    // Violated by exactly tolerance
    // Construct this safely to prevent float associativity from pushing it 1 ULP over the boundary.
    // By setting x0 = 1.0 + tol and x1 = 0.0, lhs evaluates directly to 1.0 + tol.
    std::vector<Float> x_tol = {1.0 + math::kDefaultFeasibilityTol, 0.0};
    REQUIRE(pool.get_violated_cuts(x_tol).empty()); // Must be strictly greater than rhs + tol

    // Violated by slightly more than tolerance
    std::vector<Float> x_violated = {1.0 + math::kDefaultFeasibilityTol + 1e-7, 0.0};
    REQUIRE(pool.get_violated_cuts(x_violated).size() == 1);
}

TEST_CASE("Phase 20.1: Multiple cuts", "[cuts][cut_pool]") {
    CutPool pool;

    Cut c1{{1.0, 1.0}, 1.0}; // x0 + x1 <= 1.0
    Cut c2{{1.0, 0.0}, 0.8}; // x0 <= 0.8
    Cut c3{{0.0, 1.0}, 0.2}; // x1 <= 0.2

    pool.add_cut_to_pool(c1);
    pool.add_cut_to_pool(c2);
    pool.add_cut_to_pool(c3);

    std::vector<Float> x = {0.5, 0.5};
    // c1: 1.0 <= 1.0 (satisfied)
    // c2: 0.5 <= 0.8 (satisfied)
    // c3: 0.5 <= 0.2 (violated)

    auto violated = pool.get_violated_cuts(x);
    REQUIRE(violated.size() == 1);
    REQUIRE(violated[0].rhs == 0.2); // Identifies c3
}

TEST_CASE("Phase 20.1: Empty pool", "[cuts][cut_pool]") {
    CutPool pool;
    std::vector<Float> x = {1.0, 1.0};
    REQUIRE(pool.get_violated_cuts(x).empty());
}

TEST_CASE("Phase 20.1: Dimension mismatch and invalid input handling", "[cuts][cut_pool]") {
    CutPool pool;

    // Empty coefficient cut
    Cut empty_cut{{}, 1.0};
    REQUIRE_THROWS_AS(pool.add_cut_to_pool(empty_cut), std::invalid_argument);

    Cut cut{{1.0, 1.0}, 1.0};
    pool.add_cut_to_pool(cut);

    // Empty solution
    std::vector<Float> x_empty = {};
    REQUIRE_THROWS_AS(pool.get_violated_cuts(x_empty), std::invalid_argument);

    // Mismatched dimension
    std::vector<Float> x_mismatch = {1.0, 1.0, 1.0};
    REQUIRE_THROWS_AS(pool.get_violated_cuts(x_mismatch), std::invalid_argument);
}

TEST_CASE("Phase 20.1: Cut preservation", "[cuts][cut_pool]") {
    CutPool pool;
    Cut original_cut{{1.23, -4.56, 7.89}, 42.0};

    pool.add_cut_to_pool(original_cut);

    // Create a vector that will heavily violate the cut
    std::vector<Float> x = {100.0, -100.0, 100.0};

    auto violated = pool.get_violated_cuts(x);
    REQUIRE(violated.size() == 1);

    // Verify exact preservation without normalization or modification
    REQUIRE(violated[0].coefficients.size() == 3);
    REQUIRE(violated[0].coefficients[0] == 1.23);
    REQUIRE(violated[0].coefficients[1] == -4.56);
    REQUIRE(violated[0].coefficients[2] == 7.89);
    REQUIRE(violated[0].rhs == 42.0);
}
