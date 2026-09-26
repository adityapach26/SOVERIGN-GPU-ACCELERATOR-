#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>

#include "cuda/feasibility_pump.cuh"
#include "core/problem.hpp"
#include "gpu/device_model.cuh"
#include "gpu/vram_arena.cuh"

using namespace sankhya;

TEST_CASE("Phase 23.1: Feasibility Pump Tensor Core Projection", "[cuda][feasibility_pump]") {
    // We test the projection direction d = - A^T (A x_tilde - b).
    
    // Model: 2 rows, 3 columns
    // A = [ 1  0  2 ]
    //     [ 0  1 -1 ]
    // b = [ 4, 1 ]
    
    core::Model host_model;
    host_model.A.rows = 2;
    host_model.A.cols = 3;
    
    // CSC format for A:
    // Col 0: row 0, val 1
    // Col 1: row 1, val 1
    // Col 2: row 0, val 2; row 1, val -1
    host_model.A.col_ptrs = {0, 1, 2, 4};
    host_model.A.row_indices = {0, 1, 0, 1};
    host_model.A.values = {1.0, 1.0, 2.0, -1.0};
    host_model.rhs = {4.0, 1.0};
    
    host_model.obj = {0.0, 0.0, 0.0};
    host_model.lb = {0.0, 0.0, 0.0};
    host_model.ub = {10.0, 10.0, 10.0};
    
    // x_tilde = [1.0, 2.0, 1.0]
    std::vector<Float> x_tilde = {1.0, 2.0, 1.0};
    
    // Expected mathematical projection direction:
    // 1. A * x_tilde = [1(1)+0(2)+2(1), 0(1)+1(2)-1(1)] = [3, 1]
    // 2. r = A * x_tilde - b = [3 - 4, 1 - 1] = [-1, 0]
    // 3. d = - A^T * r = - [ 1  0 ] * [-1] = - [ -1 ] = [ 1 ]
    //                      [ 0  1 ]   [ 0]     [  0 ]   [ 0 ]
    //                      [ 2 -1 ]            [ -2 ]   [ 2 ]
    // Expected d = [1.0, 0.0, 2.0]
    
    gpu::VRAMArena arena(1024 * 1024);
    gpu::DeviceModel d_model = gpu::upload_to_device(host_model, arena);
    
    std::vector<Float> direction = gpu::compute_fp_projection_direction(d_model, x_tilde, arena);
    
    REQUIRE(direction.size() == 3);
    
    // Use a small tolerance appropriate for FP16 WMMA intermediates
    REQUIRE(direction[0] == Catch::Approx(1.0).margin(1e-3));
    REQUIRE(direction[1] == Catch::Approx(0.0).margin(1e-3));
    REQUIRE(direction[2] == Catch::Approx(2.0).margin(1e-3));
}
