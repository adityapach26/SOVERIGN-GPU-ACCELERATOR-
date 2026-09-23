#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "ipm/mehrotra.hpp"
#include "core/problem.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

using namespace sankhya;
using sankhya::simplex::SimplexStatus;

TEST_CASE("Phase 15.2: GPU Mehrotra Predictor-Corrector IPM", "[cuda][ipm]") {
    // Construct a verified textbook LP
    // min -3x1 - 5x2
    // s.t. x1 + x2 + x3 = 4
    //      2x1 + 3x2 + x4 = 12
    //      x >= 0
    //
    // Optimal solution: x1 = 0, x2 = 4, x3 = 0, x4 = 0
    // Objective: -20
    
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        SKIP("CUDA Runtime: NOT AVAILABLE (Environment cannot execute GPU KKT solve)");
    }

    core::Model model;
    model.sense = OptimizationSense::Minimize;
    
    model.add_variable(-3.0); // x1
    model.add_variable(-5.0); // x2
    model.add_variable(0.0);  // x3 (slack 1)
    model.add_variable(0.0);  // x4 (slack 2)
    
    // x1 + x2 + x3 = 4
    model.add_constraint({0, 1, 2}, {1.0, 1.0, 1.0}, 4.0);
    
    // 2x1 + 3x2 + x4 = 12
    model.add_constraint({0, 1, 3}, {2.0, 3.0, 1.0}, 12.0);
    
    model.finalize();
    
    ipm::MehrotraSolver solver(model);
    ipm::MehrotraResult result;
    
    REQUIRE_NOTHROW(result = solver.solve());
    
    std::cout << "\n=== Phase 15.2 Validation Record ===" << std::endl;
    std::cout << "CUDA Device Availability: VERIFIED" << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Final Objective: " << result.objective_value << std::endl;
    std::cout << "Primal Residual: " << result.primal_residual << std::endl;
    std::cout << "Dual Residual: " << result.dual_residual << std::endl;
    std::cout << "Duality Gap: " << result.duality_gap << std::endl;
    std::cout << "Convergence Status: " << static_cast<int>(result.status) << std::endl;
    std::cout << "====================================\n" << std::endl;

    REQUIRE(result.status == SimplexStatus::Optimal);
    REQUIRE(result.iterations > 0);
    REQUIRE(result.primal_residual < 1e-6);
    REQUIRE(result.dual_residual < 1e-6);
    REQUIRE(result.duality_gap < 1e-6);
    
    // Expected optimal objective is -20
    REQUIRE(result.objective_value == Catch::Approx(-20.0).margin(1e-4));
}