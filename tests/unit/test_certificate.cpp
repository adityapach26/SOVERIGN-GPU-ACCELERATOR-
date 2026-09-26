#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "verifier/certificate.hpp"
#include "core/problem.hpp"
#include "sankhya/types.hpp"

using namespace sankhya;

// Helper: construct a 1-variable, 1-constraint LP:
//   min c*x  s.t.  x = rhs,  lb <= x <= ub
// For a valid certificate: pi = c (dual of the equality gives r = c - pi = 0).
// We override pi and x to drive specific KKT cases.
static core::Model make_single(double c_obj, double lb, double ub, double rhs_val) {
    core::Model m;
    m.sense = OptimizationSense::Minimize;
    m.add_variable(static_cast<Float>(c_obj),
                   static_cast<Float>(lb),
                   static_cast<Float>(ub),
                   VariableType::Continuous);
    m.add_constraint({0}, {1.0}, static_cast<Float>(rhs_val));
    m.finalize();
    return m;
}

// Helper: 2-var model for complementarity product tests.
// min c0*x0 + c1*x1
// s.t. x0 + x1 = b
// lb0 <= x0 <= ub0, lb1 <= x1 <= ub1
static core::Model make_two(double c0, double lb0, double ub0,
                             double c1, double lb1, double ub1,
                             double b) {
    core::Model m;
    m.sense = OptimizationSense::Minimize;
    m.add_variable(static_cast<Float>(c0), static_cast<Float>(lb0),
                   static_cast<Float>(ub0), VariableType::Continuous);
    m.add_variable(static_cast<Float>(c1), static_cast<Float>(lb1),
                   static_cast<Float>(ub1), VariableType::Continuous);
    m.add_constraint({0, 1}, {1.0, 1.0}, static_cast<Float>(b));
    m.finalize();
    return m;
}

// ============================================================
// Case 1: variable exactly at lower bound, positive reduced cost → PASS
// ============================================================
TEST_CASE("Certificate Case 1: x=lb, r>0 → PASS", "[verifier][certificate]") {
    // min 2x  s.t. x = 0,  0 <= x <= inf
    // optimal: x=0, pi=2 (dual of x=b gives r = c - pi = 2-2 = 0, but
    // we manually set pi=1 so r = 2-1 = 1 > 0, x=lb=0 → product 1*0=0 < eps)
    core::Model model = make_single(2.0, 0.0, math::kInfinity, 0.0);
    std::vector<Float> x  = {0.0};   // at lb
    std::vector<Float> pi = {1.0};   // r = 2 - 1*1 = 1 > 0
    // r*(x-lb) = 1*0 = 0 < 1e-6 → PASS
    REQUIRE(verifier::verify_optimal(model, x, pi) == true);
}

// ============================================================
// Case 2: variable slightly above lb, product r*(x-lb) < eps → PASS
// ============================================================
TEST_CASE("Certificate Case 2: x slightly above lb, r*(x-lb)<eps → PASS", "[verifier][certificate]") {
    // min 0*x  s.t. x = 5e-7,  0 <= x <= inf
    // Set pi so r = 0.0416, x-lb = 5.3e-6
    // product = 0.0416 * 5.3e-6 ≈ 2.2e-7 < 1e-6 → PASS
    core::Model model = make_single(0.0, 0.0, math::kInfinity, 5.3e-6);
    std::vector<Float> x  = {static_cast<Float>(5.3e-6)};
    // r = c - A^T pi = 0 - pi[0]*1 = -pi[0]
    // We want r = 0.0416 → pi[0] = -0.0416
    std::vector<Float> pi = {static_cast<Float>(-0.0416)};
    // product: 0.0416 * (5.3e-6 - 0) = 2.2e-7 < 1e-6 → PASS
    REQUIRE(verifier::verify_optimal(model, x, pi) == true);
}

// ============================================================
// Case 3: variable above lb, product r*(x-lb) > eps → FAIL
// ============================================================
TEST_CASE("Certificate Case 3: x above lb, r*(x-lb)>eps → FAIL", "[verifier][certificate]") {
    // r=1.0, x-lb=2e-5 → product=2e-5 > 1e-6 → FAIL
    core::Model model = make_single(0.0, 0.0, math::kInfinity, 2e-5);
    std::vector<Float> x  = {static_cast<Float>(2e-5)};
    // r = 0 - (-1)*1 = 1 → pi[0] = -1.0
    std::vector<Float> pi = {static_cast<Float>(-1.0)};
    // product: 1.0 * 2e-5 = 2e-5 >= 1e-6 → FAIL
    REQUIRE(verifier::verify_optimal(model, x, pi) == false);
}

// ============================================================
// Case 4: variable exactly at upper bound, negative reduced cost → PASS
// ============================================================
TEST_CASE("Certificate Case 4: x=ub, r<0 → PASS", "[verifier][certificate]") {
    // min -2x  s.t. x = 1,  0 <= x <= 1
    // At ub: x=1, pi s.t. r = c - A^T pi < 0
    // r = -2 - pi[0]*1; want r = -1 → pi[0] = -1
    // product: (-(-1))*(1-1) = 0 < eps → PASS
    core::Model model = make_single(-2.0, 0.0, 1.0, 1.0);
    std::vector<Float> x  = {1.0};   // at ub
    std::vector<Float> pi = {static_cast<Float>(-1.0)}; // r = -2 - (-1) = -1 < 0
    // product: (-r)*(ub-x) = 1*(1-1) = 0 < eps → PASS
    REQUIRE(verifier::verify_optimal(model, x, pi) == true);
}

// ============================================================
// Case 5: variable slightly below ub, product (-r)*(ub-x) < eps → PASS
// ============================================================
TEST_CASE("Certificate Case 5: x slightly below ub, (-r)*(ub-x)<eps → PASS", "[verifier][certificate]") {
    // ub=1, x=1-5e-7, r=-0.04
    // product: 0.04 * 5e-7 = 2e-8 < 1e-6 → PASS
    core::Model model = make_single(0.0, 0.0, 1.0, 1.0 - 5e-7);
    std::vector<Float> x  = {static_cast<Float>(1.0 - 5e-7)};
    // r = 0 - pi[0]; want r = -0.04 → pi[0] = 0.04
    std::vector<Float> pi = {static_cast<Float>(0.04)};
    // product: 0.04 * 5e-7 = 2e-8 < 1e-6 → PASS
    REQUIRE(verifier::verify_optimal(model, x, pi) == true);
}

// ============================================================
// Case 6: ub=+infinity, r < -eps → FAIL
// ============================================================
TEST_CASE("Certificate Case 6: ub=+inf, r<0 → FAIL", "[verifier][certificate]") {
    // min 0*x s.t. x = 0.5,  0 <= x < inf
    // r = 0 - pi[0]; want r = -0.1 → pi[0] = 0.1
    // ub = inf, r < -eps → dual infeasible → FAIL
    core::Model model = make_single(0.0, 0.0, math::kInfinity, 0.5);
    std::vector<Float> x  = {0.5};
    std::vector<Float> pi = {static_cast<Float>(0.1)}; // r = -0.1 < 0
    REQUIRE(verifier::verify_optimal(model, x, pi) == false);
}

// ============================================================
// Case 7: lb=-infinity, r > eps → FAIL
// ============================================================
TEST_CASE("Certificate Case 7: lb=-inf, r>0 → FAIL", "[verifier][certificate]") {
    // Free-below variable: lb=-inf, ub=+inf
    // r > 0 with no finite lower bound → dual infeasible → FAIL
    core::Model model = make_single(0.0, -math::kInfinity, math::kInfinity, 0.0);
    std::vector<Float> x  = {0.0};
    // r = 0 - (-0.1) = 0.1 > 0, lb = -inf → FAIL
    std::vector<Float> pi = {static_cast<Float>(-0.1)};
    REQUIRE(verifier::verify_optimal(model, x, pi) == false);
}

// ============================================================
// Case 8: free variable (lb=-inf, ub=+inf) with nonzero r → FAIL
// ============================================================
TEST_CASE("Certificate Case 8: free variable with r != 0 → FAIL", "[verifier][certificate]") {
    // Fully free variable: any r != 0 is dual infeasible
    core::Model model = make_single(0.0, -math::kInfinity, math::kInfinity, 0.0);
    std::vector<Float> x  = {0.0};
    // r = 0 - (-0.5) = 0.5 > eps → FAIL (lb=-inf)
    std::vector<Float> pi = {static_cast<Float>(-0.5)};
    REQUIRE(verifier::verify_optimal(model, x, pi) == false);
}

// ============================================================
// Existing regression: known optimal certificate → PASS
// ============================================================
TEST_CASE("Phase 30.1: Certificate - Known optimal solution", "[verifier][certificate]") {
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    // min -x1 - x2
    // s.t. x1 + s1 = 1,  x2 + s2 = 1
    model.add_variable(-1.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable(-1.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable( 0.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable( 0.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_constraint({0, 2}, {1.0, 1.0}, 1.0);
    model.add_constraint({1, 3}, {1.0, 1.0}, 1.0);
    model.finalize();

    // Optimal: x1=1, x2=1, s1=0, s2=0; pi = [-1, -1]
    // r[0] = -1 - (-1)*1 = 0 (basic), r[1] same, r[2] = 0-(-1) = 1>0 at lb=0 → product 0, r[3] same
    std::vector<Float> x  = {1.0, 1.0, 0.0, 0.0};
    std::vector<Float> pi = {-1.0, -1.0};
    REQUIRE(verifier::verify_optimal(model, x, pi) == true);
}

// ============================================================
// Existing regression: primal infeasibility → FAIL
// ============================================================
TEST_CASE("Phase 30.1: Certificate - Corrupted primal solution", "[verifier][certificate]") {
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    model.add_variable(-1.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable(-1.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable( 0.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable( 0.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_constraint({0, 2}, {1.0, 1.0}, 1.0);
    model.add_constraint({1, 3}, {1.0, 1.0}, 1.0);
    model.finalize();

    std::vector<Float> pi = {-1.0, -1.0};
    // Violates Ax = b
    std::vector<Float> x_infeasible = {1.0, 0.5, 0.0, 0.0};
    REQUIRE(verifier::verify_optimal(model, x_infeasible, pi) == false);
}

// ============================================================
// Existing regression: dual infeasibility → FAIL
// ============================================================
TEST_CASE("Phase 30.1: Certificate - Corrupted dual certificate", "[verifier][certificate]") {
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    model.add_variable(-1.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable(-1.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable( 0.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable( 0.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_constraint({0, 2}, {1.0, 1.0}, 1.0);
    model.add_constraint({1, 3}, {1.0, 1.0}, 1.0);
    model.finalize();

    std::vector<Float> x = {1.0, 1.0, 0.0, 0.0};
    // pi = {1,1}: r[2] = 0 - 1 = -1 < 0, but ub[2]=inf → FAIL
    std::vector<Float> pi_corrupt = {1.0, 1.0};
    REQUIRE(verifier::verify_optimal(model, x, pi_corrupt) == false);
}

// ============================================================
// Complementary slackness product failure → FAIL
// ============================================================
TEST_CASE("Phase 30.1: Certificate - CS product failure", "[verifier][certificate]") {
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    model.add_variable(-1.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable(-1.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable( 0.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_variable( 0.0, 0.0, math::kInfinity, VariableType::Continuous);
    model.add_constraint({0, 2}, {1.0, 1.0}, 1.0);
    model.add_constraint({1, 3}, {1.0, 1.0}, 1.0);
    model.finalize();

    // x = {0.5,0.5,0.5,0.5} feasible interior point (Ax=b holds)
    // pi = {-1,-1}: r[2] = 0-(-1)=1 > 0, x[2]=0.5, lb=0
    // product = 1 * 0.5 = 0.5 >> 1e-6 → FAIL
    std::vector<Float> x  = {0.5, 0.5, 0.5, 0.5};
    std::vector<Float> pi = {-1.0, -1.0};
    REQUIRE(verifier::verify_optimal(model, x, pi) == false);
}
