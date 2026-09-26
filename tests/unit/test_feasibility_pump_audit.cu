/**
 * @file test_feasibility_pump_audit.cu
 * @brief Phase 23.2 unit tests: Strict FP64 CUDA-core incumbent auditor.
 *
 * Tests that verify_incumbent_fp64() correctly:
 *   1. Accepts a genuinely feasible candidate (max |Ax-b| == 0).
 *   2. Rejects a candidate whose true FP64 residual is ~1e-4 >> 1e-6.
 *
 * The infeasible candidate is carefully constructed so that in FP16 the
 * residual could be rounded away to zero (FP16 epsilon ~1e-3), but in FP64
 * it remains a clear violation.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>

#include "cuda/feasibility_pump.cuh"
#include "core/problem.hpp"
#include "gpu/device_model.cuh"
#include "gpu/vram_arena.cuh"

using namespace sankhya;

// ---------------------------------------------------------------------------
// Helper: build a small LP model and upload to device
// ---------------------------------------------------------------------------
static gpu::DeviceModel build_and_upload(
    core::Model& host_model,
    gpu::VRAMArena& arena
) {
    return gpu::upload_to_device(host_model, arena);
}

// ---------------------------------------------------------------------------
// TEST 1 — Genuinely feasible candidate
// ---------------------------------------------------------------------------
TEST_CASE("Phase 23.2: FP64 auditor accepts feasible candidate", "[cuda][feasibility_pump][audit]") {
    //
    // Model: 2 rows, 3 cols
    //   A = [ 1  0  2 ]
    //       [ 0  1 -1 ]
    //   b = [ 4, 1 ]
    //
    // Feasible candidate: x = [2, 0, 1]
    //   Row 0: 1*2 + 0*0 + 2*1 = 4 == b[0]  (residual = 0)
    //   Row 1: 0*2 + 1*0 - 1*1 = -1 != 1     wait — let's pick x = [2, 2, 1]
    //   Row 1: 0*2 + 1*2 - 1*1 = 1  == b[1]  (residual = 0)
    //
    // x = [2, 2, 1] is exactly feasible.
    //
    core::Model host_model;
    host_model.A.rows = 2;
    host_model.A.cols = 3;
    host_model.A.col_ptrs    = {0, 1, 2, 4};
    host_model.A.row_indices = {0, 1, 0, 1};
    host_model.A.values      = {1.0, 1.0, 2.0, -1.0};
    host_model.rhs = {4.0, 1.0};
    host_model.obj = {0.0, 0.0, 0.0};
    host_model.lb  = {0.0, 0.0, 0.0};
    host_model.ub  = {10.0, 10.0, 10.0};

    gpu::VRAMArena arena(1024 * 1024);
    gpu::DeviceModel d_model = build_and_upload(host_model, arena);

    // Exactly feasible integer candidate
    std::vector<Float> x_feasible = {2.0, 2.0, 1.0};
    // Verify analytically:
    // Row 0: 1*2 + 2*1 = 4 = b[0]
    // Row 1: 1*2 - 1*1 = 1 = b[1]
    bool result = gpu::verify_incumbent_fp64(d_model, x_feasible);
    REQUIRE(result == true);
}

// ---------------------------------------------------------------------------
// TEST 2 — FP64 auditor catches precision-loss infeasibility
// ---------------------------------------------------------------------------
TEST_CASE("Phase 23.2: FP64 auditor rejects infeasible candidate (residual ~1e-4)", "[cuda][feasibility_pump][audit]") {
    //
    // Construct a model where a candidate looks feasible in FP16 but is
    // actually infeasible in FP64.
    //
    // Model: 1 row, 2 cols
    //   A = [ 1.0   1.0 ]
    //   b = [ 1.0 ]
    //
    // True feasible point: x = [0.5, 0.5]
    //
    // Infeasible candidate: x = [0.5, 0.5 + 1e-4]
    //   Ax = 0.5 + 0.5 + 1e-4 = 1.0001
    //   residual = |1.0001 - 1.0| = 1e-4
    //
    // In FP16 (epsilon ~9.77e-4), 1e-4 is within the roundoff error and the
    // residual can appear as 0. In FP64 (epsilon ~2.2e-16), 1e-4 is exactly
    // representable and clearly exceeds kDefaultFeasibilityTol = 1e-6.
    //
    core::Model host_model;
    host_model.A.rows = 1;
    host_model.A.cols = 2;
    host_model.A.col_ptrs    = {0, 1, 2};
    host_model.A.row_indices = {0, 0};
    host_model.A.values      = {1.0, 1.0};
    host_model.rhs = {1.0};
    host_model.obj = {0.0, 0.0};
    host_model.lb  = {0.0, 0.0};
    host_model.ub  = {10.0, 10.0};

    gpu::VRAMArena arena(1024 * 1024);
    gpu::DeviceModel d_model = build_and_upload(host_model, arena);

    // Infeasible candidate: residual ~1e-4 >> kDefaultFeasibilityTol (1e-6)
    std::vector<Float> x_infeasible = {0.5, 0.5 + 1e-4};
    bool result = gpu::verify_incumbent_fp64(d_model, x_infeasible);

    // FP64 audit MUST catch this: 1e-4 > 1e-6
    REQUIRE(result == false);
}

// ---------------------------------------------------------------------------
// TEST 3 — Safety: zero-dimension model returns true (vacuously feasible)
// ---------------------------------------------------------------------------
TEST_CASE("Phase 23.2: FP64 auditor handles empty model", "[cuda][feasibility_pump][audit]") {
    core::Model host_model;
    host_model.A.rows = 0;
    host_model.A.cols = 0;
    host_model.A.col_ptrs = {0};
    host_model.rhs = {};
    host_model.obj = {};
    host_model.lb  = {};
    host_model.ub  = {};

    gpu::VRAMArena arena(1024 * 1024);
    gpu::DeviceModel d_model = build_and_upload(host_model, arena);

    std::vector<Float> x_empty = {};
    bool result = gpu::verify_incumbent_fp64(d_model, x_empty);
    REQUIRE(result == true);
}
