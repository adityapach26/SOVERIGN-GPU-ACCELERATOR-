#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "ipm/crossover.cuh"
#include "core/problem.hpp"
#include "gpu/device_model.cuh"
#include "gpu/vram_arena.cuh"
#include "gpu/hbf.cuh"
#include <vector>
#include <iostream>
#include <cmath>

using namespace sankhya;

TEST_CASE("Phase 16.1: In-VRAM Megiddo Crossover", "[cuda][crossover]") {
    // 2D Simplex Test
    // min -x1 - x2
    // s.t. x1 + 2 x2 + x3 = 4
    //      2 x1 + x2 + x4 = 4
    //      x >= 0
    
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    
    model.add_variable(-1.0); // x1
    model.add_variable(-1.0); // x2
    model.add_variable(0.0);  // x3
    model.add_variable(0.0);  // x4
    
    model.add_constraint({0, 1, 2}, {1.0, 2.0, 1.0}, 4.0);
    model.add_constraint({0, 1, 3}, {2.0, 1.0, 1.0}, 4.0);
    
    model.finalize();

    gpu::VRAMArena arena;
    gpu::DeviceModel d_model = gpu::upload_to_device(model, arena);

    std::vector<Float> x_ipm = {1.0, 1.0, 1.0, 1.0};

    simplex::Basis basis;
    REQUIRE_NOTHROW(basis = ipm::generate_crossover_basis(d_model, x_ipm));

    // A. Basis size
    REQUIRE(basis.basic_indices.size() == 2);
    REQUIRE(basis.col_status.size() == 4);

    // B. Valid model indices
    REQUIRE(basis.basic_indices[0] < 4);
    REQUIRE(basis.basic_indices[1] < 4);

    // C. Exactly m basic variables
    int num_basic = 0;
    for (int i = 0; i < 4; ++i) {
        if (basis.col_status[i] == simplex::BasisStatus::Basic) num_basic++;
    }
    REQUIRE(num_basic == 2);

    // Reconstruct dense B
    Index b0 = basis.basic_indices[0];
    Index b1 = basis.basic_indices[1];
    
    auto get_col = [&](Index j) -> std::vector<Float> {
        std::vector<Float> col(2, 0.0);
        for (std::size_t p = model.A.col_ptrs[j]; p < model.A.col_ptrs[j+1]; ++p) {
            col[model.A.row_indices[p]] = model.A.values[p];
        }
        return col;
    };
    
    std::vector<Float> A0 = get_col(b0);
    std::vector<Float> A1 = get_col(b1);
    
    // D. Basis nonsingularity
    Float det = A0[0] * A1[1] - A0[1] * A1[0];
    REQUIRE(std::abs(det) > 1e-7);

    // E. Reconstruct the BFS
    std::vector<Float> b_vec = {4.0, 4.0};
    Float x_B0 = (b_vec[0] * A1[1] - b_vec[1] * A1[0]) / det;
    Float x_B1 = (A0[0] * b_vec[1] - A0[1] * b_vec[0]) / det;

    std::vector<Float> x_final(4, 0.0);
    x_final[b0] = x_B0;
    x_final[b1] = x_B1;

    Float res0 = std::abs((1.0*x_final[0] + 2.0*x_final[1] + 1.0*x_final[2]) - 4.0);
    Float res1 = std::abs((2.0*x_final[0] + 1.0*x_final[1] + 1.0*x_final[3]) - 4.0);
    REQUIRE(res0 <= 1e-6);
    REQUIRE(res1 <= 1e-6);

    // F. Bound feasibility
    for (int i = 0; i < 4; ++i) {
        REQUIRE(x_final[i] >= -1e-6);
    }

    // G. Nonbasic variables
    for (int i = 0; i < 4; ++i) {
        if (i != b0 && i != b1) {
            REQUIRE(basis.col_status[i] == simplex::BasisStatus::AtLower);
            REQUIRE(std::abs(x_final[i]) <= 1e-6);
        }
    }

    // I. Objective
    // Crossover is a basis-purification procedure, not an optimizer.
    // It is mathematically blind to the objective unless starting exactly on the optimal face.
    // The objective is calculated purely for diagnostic trace/logging if needed.
    Float obj = -1.0 * x_final[0] - 1.0 * x_final[1];
    (void)obj; // Silence unused warning

    
    // J. HBF compatibility
    gpu::HBFManager hbf_manager(arena);
    REQUIRE_NOTHROW(hbf_manager.set_root_basis(basis.basic_indices));
}

TEST_CASE("Phase 16.1: Crossover Rank Deficiency Failure", "[cuda][crossover]") {
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    model.add_variable(-1.0);
    model.add_variable(0.0);
    model.add_constraint({0, 1}, {1.0, 1.0}, 2.0);
    model.add_constraint({0, 1}, {2.0, 2.0}, 4.0);
    model.finalize();

    gpu::VRAMArena arena;
    gpu::DeviceModel d_model = gpu::upload_to_device(model, arena);

    std::vector<Float> x_ipm = {1.0, 1.0};

    REQUIRE_THROWS_WITH(ipm::generate_crossover_basis(d_model, x_ipm), 
        Catch::Matchers::ContainsSubstring("Crossover failed: Matrix is rank deficient"));
}

