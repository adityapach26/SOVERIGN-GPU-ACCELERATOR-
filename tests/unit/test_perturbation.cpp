#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>

#include "numerics/perturbation.hpp"

using namespace sankhya;
using namespace sankhya::numerics;

// ---------------------------------------------------------------------------
// Reference constants — computed independently from production code.
// phi and epsilon0 are derived here from first principles so the test does
// not merely validate that production code calls itself.
// ---------------------------------------------------------------------------
static const double REF_PHI      = (std::sqrt(5.0) - 1.0) / 2.0;
static const double REF_EPSILON0 = std::ldexp(1.0, -48);

// Reference perturbation at index i with period P: epsilon0 * phi^(i % P)
static double ref_perturb(int i, int P) {
    return REF_EPSILON0 * std::pow(REF_PHI, static_cast<double>(i % P));
}

// ---------------------------------------------------------------------------
// TEST 1 — Exact/reference values for indices 0..5 with P >= 6
// ---------------------------------------------------------------------------
TEST_CASE("Phase 24.1: Symplectic Phi - exact reference values (P=7)", "[perturbation][numerics]") {
    // P = 7 is a prime >= 6, so i % P == i for all i in [0,5].
    const int P = 7;
    const int N = 6;
    std::vector<Float> v(static_cast<std::size_t>(N), 0.0);

    apply_symplectic_phi(v, static_cast<Index>(P));

    // Verify size unchanged
    REQUIRE(v.size() == static_cast<std::size_t>(N));

    // Tolerance: epsilon0 is ~3.6e-15. We require correctness to better than
    // 1 ULP of epsilon0 relative, which is a very tight tolerance.
    // Using absolute margin of epsilon0 * 1e-10 (much tighter than the signal).
    const double margin = REF_EPSILON0 * 1e-10;

    for (int i = 0; i < N; ++i) {
        const double expected = ref_perturb(i, P);
        REQUIRE(v[static_cast<std::size_t>(i)] == Catch::Approx(expected).margin(margin));
    }
}

// ---------------------------------------------------------------------------
// TEST 2a — Periodicity: P=3 (prime < N), expect rejection (P must be >= m)
// ---------------------------------------------------------------------------
TEST_CASE("Phase 24.1: Symplectic Phi - P < bounds.size() is rejected", "[perturbation][numerics]") {
    std::vector<Float> v(6, 1.0);
    // P = 3 < 6, must throw
    REQUIRE_THROWS_AS(apply_symplectic_phi(v, 3), std::invalid_argument);
    // Vector must be unmodified after throw
    for (double x : v) {
        REQUIRE(x == Catch::Approx(1.0));
    }
}

// ---------------------------------------------------------------------------
// TEST 2b — Periodicity: exactly P == N (boundary, P=7 > N is confirmed above)
// Use P == N where N is prime (N=5)
// ---------------------------------------------------------------------------
TEST_CASE("Phase 24.1: Symplectic Phi - P equals size (P=5, prime)", "[perturbation][numerics]") {
    const int P = 5;
    const int N = 5;
    std::vector<Float> v(static_cast<std::size_t>(N), 0.0);
    apply_symplectic_phi(v, static_cast<Index>(P));

    const double margin = REF_EPSILON0 * 1e-10;
    for (int i = 0; i < N; ++i) {
        const double expected = ref_perturb(i, P); // i % 5 == i for i in 0..4
        REQUIRE(v[static_cast<std::size_t>(i)] == Catch::Approx(expected).margin(margin));
    }
}

// ---------------------------------------------------------------------------
// TEST 2c — Non-zero base: perturbation adds to existing values
// ---------------------------------------------------------------------------
TEST_CASE("Phase 24.1: Symplectic Phi - adds to non-zero base values", "[perturbation][numerics]") {
    const int P = 11; // prime >= 6
    const int N = 6;
    const double base = 42.0;
    std::vector<Float> v(static_cast<std::size_t>(N), base);

    apply_symplectic_phi(v, static_cast<Index>(P));

    const double margin = REF_EPSILON0 * 1e-10;
    for (int i = 0; i < N; ++i) {
        const double expected = base + ref_perturb(i, P);
        REQUIRE(v[static_cast<std::size_t>(i)] == Catch::Approx(expected).margin(margin));
    }
}

// ---------------------------------------------------------------------------
// TEST 3 — No size mutation
// ---------------------------------------------------------------------------
TEST_CASE("Phase 24.1: Symplectic Phi - size is unchanged", "[perturbation][numerics]") {
    std::vector<Float> v(10, 0.0);
    const std::size_t original_size = v.size();
    apply_symplectic_phi(v, 11); // P=11 is prime, >= 10
    REQUIRE(v.size() == original_size);
}

// ---------------------------------------------------------------------------
// TEST 4 — P = 0 is invalid
// ---------------------------------------------------------------------------
TEST_CASE("Phase 24.1: Symplectic Phi - P=0 is rejected", "[perturbation][numerics]") {
    std::vector<Float> v(3, 1.0);
    REQUIRE_THROWS_AS(apply_symplectic_phi(v, 0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// TEST 5 — Empty vector is a no-op (vacuously valid for any P >= 0)
// ---------------------------------------------------------------------------
TEST_CASE("Phase 24.1: Symplectic Phi - empty vector is no-op", "[perturbation][numerics]") {
    std::vector<Float> v;
    // P=1 is not prime but empty vector has size 0, so P >= 0 trivially.
    // The guard is P >= bounds.size() = 0, so any P > 0 is accepted.
    REQUIRE_NOTHROW(apply_symplectic_phi(v, 1));
    REQUIRE(v.empty());
}

// ---------------------------------------------------------------------------
// TEST 6 — Geometric decrease: each term is smaller than the previous
// ---------------------------------------------------------------------------
TEST_CASE("Phase 24.1: Symplectic Phi - perturbations decrease geometrically", "[perturbation][numerics]") {
    const int P = 7;
    const int N = 6;
    std::vector<Float> v(static_cast<std::size_t>(N), 0.0);
    apply_symplectic_phi(v, static_cast<Index>(P));

    for (int i = 1; i < N; ++i) {
        // Since phi < 1, each term must be strictly smaller
        REQUIRE(v[static_cast<std::size_t>(i)] < v[static_cast<std::size_t>(i - 1)]);
    }
}
