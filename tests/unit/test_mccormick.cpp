/**
 * @file test_mccormick.cpp
 * @brief Phase 25.1 unit tests: MRPL Bilinear McCormick Envelope Relaxation.
 *
 * All four constraint coefficients, RHS values, and sign conventions are verified
 * against independently derived formulas. No expected value is obtained by calling
 * the production function itself.
 *
 * Variable order in each Cut::coefficients: [x_coeff, y_coeff, w_coeff].
 *
 * Returned array ordering: [LB1, LB2, UB1, UB2].
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>

#include "mrpl/mccormick.hpp"

using namespace sankhya;
using namespace sankhya::mrpl;
using namespace sankhya::cuts;

// ---------------------------------------------------------------------------
// TEST 1 — Reference case: xL=0, xU=10, yL=0, yU=10
//
// Expected inequalities (derived manually, NOT from production code):
//
//  [LB1] w >= xL*y + yL*x - xL*yL = 0*y + 0*x - 0  → w >= 0
//        ⟺  yL*x + xL*y - w <= xL*yL
//        ⟺   0*x +  0*y - w <= 0
//        coeffs: [0, 0, -1], rhs: 0
//
//  [LB2] w >= xU*y + yU*x - xU*yU = 10y + 10x - 100
//        ⟺  yU*x + xU*y - w <= xU*yU
//        ⟺  10*x + 10*y - w <= 100
//        coeffs: [10, 10, -1], rhs: 100
//
//  [UB1] w <= xU*y + yL*x - xU*yL = 10y + 0 - 0
//        ⟺  -yL*x - xU*y + w <= -xU*yL
//        ⟺  -0*x - 10*y + w <= -0
//        coeffs: [0, -10, 1], rhs: 0
//
//  [UB2] w <= yU*x + xL*y - xL*yU = 10x + 0 - 0
//        ⟺  -yU*x - xL*y + w <= -xL*yU
//        ⟺  -10*x - 0*y + w <= -0
//        coeffs: [-10, 0, 1], rhs: 0
//
// ---------------------------------------------------------------------------
TEST_CASE("Phase 25.1: McCormick [0,10]x[0,10] - exact four constraints", "[mrpl][mccormick]") {
    const Float xL = 0.0, xU = 10.0, yL = 0.0, yU = 10.0;
    auto cuts = mccormick_envelopes(xL, xU, yL, yU);

    REQUIRE(cuts.size() == 4);

    // All cuts have 3 coefficients: [x, y, w]
    for (const auto& c : cuts) {
        REQUIRE(c.coefficients.size() == 3);
    }

    const double tol = 1e-12;

    // ---- [LB1]: 0*x + 0*y - w <= 0  (w >= 0) ----
    const auto& lb1 = cuts[0];
    REQUIRE(lb1.coefficients[0] == Catch::Approx(0.0).margin(tol));  // x coeff = yL
    REQUIRE(lb1.coefficients[1] == Catch::Approx(0.0).margin(tol));  // y coeff = xL
    REQUIRE(lb1.coefficients[2] == Catch::Approx(-1.0).margin(tol)); // w coeff
    REQUIRE(lb1.rhs             == Catch::Approx(0.0).margin(tol));  // xL*yL = 0

    // ---- [LB2]: 10*x + 10*y - w <= 100  (w >= 10x + 10y - 100) ----
    const auto& lb2 = cuts[1];
    REQUIRE(lb2.coefficients[0] == Catch::Approx(10.0).margin(tol));  // x coeff = yU
    REQUIRE(lb2.coefficients[1] == Catch::Approx(10.0).margin(tol));  // y coeff = xU
    REQUIRE(lb2.coefficients[2] == Catch::Approx(-1.0).margin(tol));  // w coeff
    REQUIRE(lb2.rhs             == Catch::Approx(100.0).margin(tol)); // xU*yU = 100

    // ---- [UB1]: 0*x - 10*y + w <= 0  (w <= 10y) ----
    const auto& ub1 = cuts[2];
    REQUIRE(ub1.coefficients[0] == Catch::Approx(0.0).margin(tol));   // x coeff = -yL = 0
    REQUIRE(ub1.coefficients[1] == Catch::Approx(-10.0).margin(tol)); // y coeff = -xU
    REQUIRE(ub1.coefficients[2] == Catch::Approx(1.0).margin(tol));   // w coeff
    REQUIRE(ub1.rhs             == Catch::Approx(0.0).margin(tol));   // -xU*yL = 0

    // ---- [UB2]: -10*x + 0*y + w <= 0  (w <= 10x) ----
    const auto& ub2 = cuts[3];
    REQUIRE(ub2.coefficients[0] == Catch::Approx(-10.0).margin(tol)); // x coeff = -yU
    REQUIRE(ub2.coefficients[1] == Catch::Approx(0.0).margin(tol));   // y coeff = -xL = 0
    REQUIRE(ub2.coefficients[2] == Catch::Approx(1.0).margin(tol));   // w coeff
    REQUIRE(ub2.rhs             == Catch::Approx(0.0).margin(tol));   // -xL*yU = 0
}

// ---------------------------------------------------------------------------
// TEST 2 — Non-zero bounds: x in [2,5], y in [3,7]
//   xL=2, xU=5, yL=3, yU=7
//
// [LB1]:  yL*x + xL*y - w <= xL*yL
//         3*x + 2*y - w <= 2*3 = 6
//         coeffs: [3, 2, -1], rhs: 6
//
// [LB2]:  yU*x + xU*y - w <= xU*yU
//         7*x + 5*y - w <= 5*7 = 35
//         coeffs: [7, 5, -1], rhs: 35
//
// [UB1]:  -yL*x - xU*y + w <= -xU*yL
//         -3*x - 5*y + w <= -5*3 = -15
//         coeffs: [-3, -5, 1], rhs: -15
//
// [UB2]:  -yU*x - xL*y + w <= -xL*yU
//         -7*x - 2*y + w <= -2*7 = -14
//         coeffs: [-7, -2, 1], rhs: -14
// ---------------------------------------------------------------------------
TEST_CASE("Phase 25.1: McCormick [2,5]x[3,7] - non-zero bounds", "[mrpl][mccormick]") {
    const Float xL = 2.0, xU = 5.0, yL = 3.0, yU = 7.0;
    auto cuts = mccormick_envelopes(xL, xU, yL, yU);

    REQUIRE(cuts.size() == 4);
    const double tol = 1e-12;

    // [LB1]: 3*x + 2*y - w <= 6
    const auto& lb1 = cuts[0];
    REQUIRE(lb1.coefficients[0] == Catch::Approx(3.0).margin(tol));
    REQUIRE(lb1.coefficients[1] == Catch::Approx(2.0).margin(tol));
    REQUIRE(lb1.coefficients[2] == Catch::Approx(-1.0).margin(tol));
    REQUIRE(lb1.rhs             == Catch::Approx(6.0).margin(tol));

    // [LB2]: 7*x + 5*y - w <= 35
    const auto& lb2 = cuts[1];
    REQUIRE(lb2.coefficients[0] == Catch::Approx(7.0).margin(tol));
    REQUIRE(lb2.coefficients[1] == Catch::Approx(5.0).margin(tol));
    REQUIRE(lb2.coefficients[2] == Catch::Approx(-1.0).margin(tol));
    REQUIRE(lb2.rhs             == Catch::Approx(35.0).margin(tol));

    // [UB1]: -3*x - 5*y + w <= -15
    const auto& ub1 = cuts[2];
    REQUIRE(ub1.coefficients[0] == Catch::Approx(-3.0).margin(tol));
    REQUIRE(ub1.coefficients[1] == Catch::Approx(-5.0).margin(tol));
    REQUIRE(ub1.coefficients[2] == Catch::Approx(1.0).margin(tol));
    REQUIRE(ub1.rhs             == Catch::Approx(-15.0).margin(tol));

    // [UB2]: -7*x - 2*y + w <= -14
    const auto& ub2 = cuts[3];
    REQUIRE(ub2.coefficients[0] == Catch::Approx(-7.0).margin(tol));
    REQUIRE(ub2.coefficients[1] == Catch::Approx(-2.0).margin(tol));
    REQUIRE(ub2.coefficients[2] == Catch::Approx(1.0).margin(tol));
    REQUIRE(ub2.rhs             == Catch::Approx(-14.0).margin(tol));
}

// ---------------------------------------------------------------------------
// TEST 3 — Bounds are not mutated by the call
// ---------------------------------------------------------------------------
TEST_CASE("Phase 25.1: McCormick - input bounds are not mutated", "[mrpl][mccormick]") {
    Float xL = 1.0, xU = 4.0, yL = 2.0, yU = 6.0;
    mccormick_envelopes(xL, xU, yL, yU);
    REQUIRE(xL == 1.0);
    REQUIRE(xU == 4.0);
    REQUIRE(yL == 2.0);
    REQUIRE(yU == 6.0);
}

// ---------------------------------------------------------------------------
// TEST 4 — Tight bounds: xL==xU, yL==yU (degenerate singleton case)
//   x in [3,3], y in [4,4] => w = 12 exactly
//   All four constraints should fix w = 12.
//
// [LB1]: yL*x + xL*y - w <= xL*yL  → 4x + 3y - w <= 12
// [LB2]: yU*x + xU*y - w <= xU*yU  → 4x + 3y - w <= 12  (same, since xL==xU, yL==yU)
// [UB1]: -yL*x - xU*y + w <= -xU*yL → -4x - 3y + w <= -12
// [UB2]: -yU*x - xL*y + w <= -xL*yU → -4x - 3y + w <= -12  (same)
// ---------------------------------------------------------------------------
TEST_CASE("Phase 25.1: McCormick - degenerate singleton bounds", "[mrpl][mccormick]") {
    const Float xL = 3.0, xU = 3.0, yL = 4.0, yU = 4.0;
    auto cuts = mccormick_envelopes(xL, xU, yL, yU);

    REQUIRE(cuts.size() == 4);
    const double tol = 1e-12;

    // Both LB constraints should be identical: 4x + 3y - w <= 12
    for (int i = 0; i < 2; ++i) {
        REQUIRE(cuts[static_cast<std::size_t>(i)].coefficients[0] == Catch::Approx(4.0).margin(tol));
        REQUIRE(cuts[static_cast<std::size_t>(i)].coefficients[1] == Catch::Approx(3.0).margin(tol));
        REQUIRE(cuts[static_cast<std::size_t>(i)].coefficients[2] == Catch::Approx(-1.0).margin(tol));
        REQUIRE(cuts[static_cast<std::size_t>(i)].rhs             == Catch::Approx(12.0).margin(tol));
    }

    // Both UB constraints should be identical: -4x - 3y + w <= -12
    for (int i = 2; i < 4; ++i) {
        REQUIRE(cuts[static_cast<std::size_t>(i)].coefficients[0] == Catch::Approx(-4.0).margin(tol));
        REQUIRE(cuts[static_cast<std::size_t>(i)].coefficients[1] == Catch::Approx(-3.0).margin(tol));
        REQUIRE(cuts[static_cast<std::size_t>(i)].coefficients[2] == Catch::Approx(1.0).margin(tol));
        REQUIRE(cuts[static_cast<std::size_t>(i)].rhs             == Catch::Approx(-12.0).margin(tol));
    }
}

// ---------------------------------------------------------------------------
// TEST 5 — Exactly four constraints are returned
// ---------------------------------------------------------------------------
TEST_CASE("Phase 25.1: McCormick - returns exactly four constraints", "[mrpl][mccormick]") {
    auto cuts = mccormick_envelopes(0.0, 5.0, 0.0, 5.0);
    REQUIRE(cuts.size() == 4);
    for (const auto& c : cuts) {
        REQUIRE(c.coefficients.size() == 3);
    }
}
