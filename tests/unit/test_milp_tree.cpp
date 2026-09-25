#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <thread>
#include <vector>

#include "milp/tree.hpp"

using namespace sankhya;
using namespace sankhya::milp;

TEST_CASE("Phase 17.1: B&B Node Queue - Best-bound ordering", "[milp][tree]") {
    NodeQueue q;
    
    MILPNode a{1, 10.0, 2};
    MILPNode b{2, 3.0, 1};
    MILPNode c{3, 7.0, 3};
    
    q.push(a);
    q.push(b);
    q.push(c);
    
    // Expected pop order: B (3.0), C (7.0), A (10.0)
    MILPNode first = q.pop();
    REQUIRE(first.hbf_id == 2);
    REQUIRE(first.parent_bound == 3.0);
    
    MILPNode second = q.pop();
    REQUIRE(second.hbf_id == 3);
    REQUIRE(second.parent_bound == 7.0);
    
    MILPNode third = q.pop();
    REQUIRE(third.hbf_id == 1);
    REQUIRE(third.parent_bound == 10.0);
}

TEST_CASE("Phase 17.1: B&B Node Queue - Empty queue", "[milp][tree]") {
    NodeQueue q;
    
    REQUIRE(q.empty() == true);
    
    q.push(MILPNode{1, 5.0, 0});
    REQUIRE(q.empty() == false);
    
    q.pop();
    REQUIRE(q.empty() == true);
    
    // Verify exception on popping empty queue
    REQUIRE_THROWS_AS(q.pop(), std::out_of_range);
}

TEST_CASE("Phase 17.1: Global Incumbent - Initial and first update", "[milp][tree]") {
    GlobalIncumbent inc;
    
    // Verify initial objective is positive infinity
    REQUIRE(inc.get_obj() == math::kInfinity);
    
    std::vector<Float> x1 = {1.0, 2.0};
    inc.update(x1, 100.0);
    
    REQUIRE(inc.get_obj() == 100.0);
}

TEST_CASE("Phase 17.1: Global Incumbent - Strict improvement", "[milp][tree]") {
    GlobalIncumbent inc;
    
    std::vector<Float> x = {0.0};
    inc.update(x, 100.0);
    REQUIRE(inc.get_obj() == 100.0);
    
    // Strict improvement
    inc.update(x, 50.0);
    REQUIRE(inc.get_obj() == 50.0);
    
    // Same objective
    inc.update(x, 50.0);
    REQUIRE(inc.get_obj() == 50.0);
    
    // Worse objective
    inc.update(x, 75.0);
    REQUIRE(inc.get_obj() == 50.0);
}

TEST_CASE("Phase 17.1: Global Incumbent - Solution consistency", "[milp][tree]") {
    GlobalIncumbent inc;
    
    std::vector<Float> x1 = {1.0, 2.0};
    inc.update(x1, 100.0);
    
    REQUIRE(inc.get_x() == x1);
    
    std::vector<Float> x2 = {3.0, 4.0};
    inc.update(x2, 50.0);
    
    // Verify solution vector updated
    REQUIRE(inc.get_x() == x2);
    
    std::vector<Float> x3 = {5.0, 6.0};
    inc.update(x3, 75.0);
    
    // Verify solution vector did NOT update since 75.0 > 50.0
    REQUIRE(inc.get_x() == x2);
}

TEST_CASE("Phase 17.1: Global Incumbent - Concurrency", "[milp][tree]") {
    GlobalIncumbent inc;
    
    // Run multiple threads trying to update the incumbent concurrently
    auto worker = [&inc](Float obj, const std::vector<Float>& x) {
        for (int i = 0; i < 100; ++i) {
            inc.update(x, obj);
            inc.get_obj(); // Just read to test contention
        }
    };
    
    std::vector<std::thread> threads;
    threads.emplace_back(worker, 150.0, std::vector<Float>{1.0});
    threads.emplace_back(worker, 50.0, std::vector<Float>{2.0});
    threads.emplace_back(worker, 75.0, std::vector<Float>{3.0});
    threads.emplace_back(worker, 25.0, std::vector<Float>{4.0}); // Best
    
    for (auto& t : threads) {
        t.join();
    }
    
    // The final incumbent must be the best one
    REQUIRE(inc.get_obj() == 25.0);
    REQUIRE(inc.get_x() == std::vector<Float>{4.0});
}

