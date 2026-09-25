#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "milp/branching.hpp"

using namespace sankhya;
using namespace sankhya::milp;

TEST_CASE("Phase 19.1: Pseudocost update and historical accumulation", "[milp][branching]") {
    PseudocostTracker tracker(2);

    // Test 1 & 2: Record up/down and verify mathematical consistency and accumulation
    tracker.record_up_observation(0, 1.5, 0.5); // P+ = 1.5 / 0.5 = 3.0
    tracker.record_down_observation(0, 0.8, 0.4); // P- = 0.8 / 0.4 = 2.0

    REQUIRE_THAT(tracker.get_up_pseudocost(0), Catch::Matchers::WithinAbs(3.0, 1e-6));
    REQUIRE_THAT(tracker.get_down_pseudocost(0), Catch::Matchers::WithinAbs(2.0, 1e-6));

    // Accumulate second observation for var 0
    tracker.record_up_observation(0, 2.5, 0.5); // P+ = 5.0 -> avg P+ = (3.0 + 5.0)/2 = 4.0
    tracker.record_down_observation(0, 1.2, 0.3); // P- = 4.0 -> avg P- = (2.0 + 4.0)/2 = 3.0

    REQUIRE_THAT(tracker.get_up_pseudocost(0), Catch::Matchers::WithinAbs(4.0, 1e-6));
    REQUIRE_THAT(tracker.get_down_pseudocost(0), Catch::Matchers::WithinAbs(3.0, 1e-6));
}

TEST_CASE("Phase 19.1: Branching score and highest candidate selected", "[milp][branching]") {
    PseudocostTracker tracker(3);
    
    // Var 0: P+ = 2.0, P- = 5.0  -> Score = 2.0
    tracker.record_up_observation(0, 1.0, 0.5); 
    tracker.record_down_observation(0, 2.5, 0.5);
    
    // Var 1: P+ = 4.0, P- = 3.5  -> Score = 3.5 (Best)
    tracker.record_up_observation(1, 2.0, 0.5);
    tracker.record_down_observation(1, 1.75, 0.5);
    
    // Var 2: P+ = 1.0, P- = 10.0 -> Score = 1.0
    tracker.record_up_observation(2, 0.5, 0.5);
    tracker.record_down_observation(2, 5.0, 0.5);

    // All fractional
    std::vector<Float> x_frac = {0.5, 0.5, 0.5};
    Index best = select_branching_variable(x_frac, tracker);
    
    REQUIRE(best == 1);
}

TEST_CASE("Phase 19.1: One-sided history", "[milp][branching]") {
    PseudocostTracker tracker(2);
    
    // Var 0 observed only UP
    tracker.record_up_observation(0, 2.0, 0.5); // P+ = 4.0

    // Var 0 DOWN should fallback to global uninitialized (since no global DOWN exists, 1.0)
    REQUIRE_THAT(tracker.get_down_pseudocost(0), Catch::Matchers::WithinAbs(1.0, 1e-6));

    // Var 1 DOWN should fallback to 1.0, UP should fallback to global avg UP = 4.0
    REQUIRE_THAT(tracker.get_up_pseudocost(1), Catch::Matchers::WithinAbs(4.0, 1e-6));
    REQUIRE_THAT(tracker.get_down_pseudocost(1), Catch::Matchers::WithinAbs(1.0, 1e-6));
    
    std::vector<Float> x_frac = {0.5, 0.5};
    // Var 0 score = min(4.0, 1.0) = 1.0
    // Var 1 score = min(4.0, 1.0) = 1.0
    // Equal scores -> deterministic tie-break selects smallest index (0)
    REQUIRE(select_branching_variable(x_frac, tracker) == 0);
}

TEST_CASE("Phase 19.1: No fractional candidate", "[milp][branching]") {
    PseudocostTracker tracker(2);
    // Not fractional (within feasibility tolerance)
    std::vector<Float> x_frac = {0.0, 1.0};
    
    Index best = select_branching_variable(x_frac, tracker);
    REQUIRE(best == -1); // API convention for no candidate
    
    x_frac = {};
    REQUIRE(select_branching_variable(x_frac, tracker) == -1);
}

TEST_CASE("Phase 19.1: Equal scores tie-breaking", "[milp][branching]") {
    PseudocostTracker tracker(3);
    tracker.record_up_observation(0, 1.0, 0.5);
    tracker.record_down_observation(0, 1.0, 0.5); // Score 2.0
    
    tracker.record_up_observation(1, 1.0, 0.5);
    tracker.record_down_observation(1, 1.0, 0.5); // Score 2.0
    
    tracker.record_up_observation(2, 1.0, 0.5);
    tracker.record_down_observation(2, 1.0, 0.5); // Score 2.0
    
    std::vector<Float> x_frac = {0.5, 0.5, 0.5};
    Index best = select_branching_variable(x_frac, tracker);
    
    // Should select smallest index deterministically
    REQUIRE(best == 0);
    
    // If Var 0 is integral, should pick Var 1
    x_frac = {0.0, 0.5, 0.5};
    REQUIRE(select_branching_variable(x_frac, tracker) == 1);
}

TEST_CASE("Phase 19.1: Invalid fractional distance", "[milp][branching]") {
    PseudocostTracker tracker(1);
    
    // Distance too small (<= 1e-6), should ignore
    tracker.record_up_observation(0, 5.0, 1e-7); 
    
    // Negative distance
    tracker.record_up_observation(0, 5.0, -0.1); 
    
    // Should remain uninitialized
    REQUIRE_THAT(tracker.get_up_pseudocost(0), Catch::Matchers::WithinAbs(1.0, 1e-6));
}
