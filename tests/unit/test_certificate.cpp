#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>

#include "verifier/certificate.hpp"
#include "core/problem.hpp"
#include "sankhya/types.hpp"

using namespace sankhya;

// Helper to construct a known LP
static core::Model build_test_lp() {
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    
    // min -x1 - x2
    // s.t. x1 + s1 = 1
    //      x2 + s2 = 1
    // x >= 0

    // x1 (0)
    model.add_variable(-1.0, 0.0, math::kInfinity, VariableType::Continuous);
    // x2 (1)
    model.add_variable(-1.0, 0.0, math::kInfinity, VariableType::Continuous);
    // s1 (2)
    model.add_variable(0.0, 0.0, math::kInfinity, VariableType::Continuous);
    // s2 (3)
    model.add_variable(0.0, 0.0, math::kInfinity, VariableType::Continuous);

    model.add_constraint({0, 2}, {1.0, 1.0}, 1.0);
    model.add_constraint({1, 3}, {1.0, 1.0}, 1.0);
    
    model.finalize();
    return model;
}

TEST_CASE("Phase 30.1: Certificate - Known optimal solution", "[verifier][certificate]") {
    core::Model model = build_test_lp();
    
    std::vector<Float> x = {1.0, 1.0, 0.0, 0.0};
    std::vector<Float> pi = {-1.0, -1.0};

    REQUIRE(verifier::verify_optimal(model, x, pi) == true);
}

TEST_CASE("Phase 30.1: Certificate - Corrupted primal solution", "[verifier][certificate]") {
    core::Model model = build_test_lp();
    
    std::vector<Float> pi = {-1.0, -1.0};

    // Case A: Violates Ax = b
    std::vector<Float> x_infeasible = {1.0, 0.5, 0.0, 0.0};
    REQUIRE(verifier::verify_optimal(model, x_infeasible, pi) == false);

    // Case B: Violates variable bounds
    std::vector<Float> x_bounds = {1.5, 1.0, -0.5, 0.0}; // x1 + s1 = 1.5 - 0.5 = 1 (Ax=b holds)
    REQUIRE(verifier::verify_optimal(model, x_bounds, pi) == false);
}

TEST_CASE("Phase 30.1: Certificate - Corrupted dual certificate", "[verifier][certificate]") {
    core::Model model = build_test_lp();
    
    std::vector<Float> x = {1.0, 1.0, 0.0, 0.0};
    
    // Corrupt pi so that reduced costs become negative for variables at lower bound
    // Let pi = {1.0, 1.0}
    // Then r_2 = c_2 - pi_0 = 0 - 1 = -1 < 0 (violation for lower bound variable s1)
    std::vector<Float> pi_corrupt = {1.0, 1.0};
    REQUIRE(verifier::verify_optimal(model, x, pi_corrupt) == false);
}

TEST_CASE("Phase 30.1: Certificate - Complementary-slackness failure", "[verifier][certificate]") {
    core::Model model = build_test_lp();
    
    // To fail complementary slackness without failing basic primal/dual feasibility:
    // A variable has r_i > 0, but it is NOT at its lower bound.
    // Let's set x to a feasible interior point.
    // x = {0.5, 0.5, 0.5, 0.5} (satisfies Ax=b and x >= 0)
    // Let pi = {-1.0, -1.0} (same as optimal, valid dual certificate)
    // Then r_2 = +1.0. 
    // But x_2 (s1) is 0.5, not at its lower bound (0.0).
    // Complementary slackness: r_2 * x_2 = 1.0 * 0.5 = 0.5 > 1e-6.
    
    std::vector<Float> x = {0.5, 0.5, 0.5, 0.5};
    std::vector<Float> pi = {-1.0, -1.0};
    
    REQUIRE(verifier::verify_optimal(model, x, pi) == false);
}
