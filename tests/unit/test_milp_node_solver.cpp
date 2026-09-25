#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "milp/node_solver.hpp"
#include "milp/tree.hpp"
#include "core/problem.hpp"

using namespace sankhya;
using namespace sankhya::milp;

TEST_CASE("Phase 18.1: CPU Decision Logic - Mock Integrality and Bounds", "[milp][node_solver]") {
    GlobalIncumbent incumbent;
    // Set initial incumbent to 100.0
    incumbent.update({0.0, 0.0}, 100.0);

    MILPNode node{1, 50.0, 1};

    // Construct a tiny mock model with two variables: x0 continuous, x1 integer
    core::Model model;
    model.vtype = {VariableType::Continuous, VariableType::Integer};
    model.obj = {1.0, 1.0}; // Not actually used by process_node_result, but good for context
    
    // Path A: Optimal + worse/equal incumbent -> no update
    {
        std::vector<Float> host_x = {0.5, 1.0}; // Integer feasible
        Float host_obj = 150.0; // Worse than 100.0
        
        process_node_result(
            node, incumbent, host_x, host_obj, 
            gpu::DeviceSimplexStatus::Optimal, model
        );
        
        REQUIRE(incumbent.get_obj() == 100.0); // No update
    }
    
    // Path B: Optimal + better + integer-feasible -> update incumbent
    {
        std::vector<Float> host_x = {0.5, 2.0}; // Integer feasible because x0 is Continuous, x1 is Integer (2.0)
        Float host_obj = 80.0; // Better than 100.0
        
        process_node_result(
            node, incumbent, host_x, host_obj, 
            gpu::DeviceSimplexStatus::Optimal, model
        );
        
        REQUIRE(incumbent.get_obj() == 80.0); // Updated!
        REQUIRE(incumbent.get_x() == host_x);
    }
    
    // Path C: Optimal + better + fractional -> no update
    {
        std::vector<Float> host_x = {0.5, 2.5}; // Fractional because x1 (Integer) is 2.5
        Float host_obj = 60.0; // Better than 80.0
        
        process_node_result(
            node, incumbent, host_x, host_obj, 
            gpu::DeviceSimplexStatus::Optimal, model
        );
        
        REQUIRE(incumbent.get_obj() == 80.0); // Not updated, remains 80.0
    }
    
    // Path D: Infeasible -> no update
    {
        std::vector<Float> host_x = {0.5, 3.0}; 
        Float host_obj = 40.0;
        
        process_node_result(
            node, incumbent, host_x, host_obj, 
            gpu::DeviceSimplexStatus::Infeasible, model
        );
        
        REQUIRE(incumbent.get_obj() == 80.0); // Not updated
    }
}

