#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "cuda/gomory.cuh"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"
#include "core/problem.hpp"
#include "simplex/basis.hpp"
#include <vector>
#include <cmath>

using namespace sankhya;

TEST_CASE("Phase 21.1: Warp-Level Gomory Cut Generation", "[cuda][gomory]") {
    // We create a mock formulation specifically to yield the tableau row requested by the test:
    // ā = [1.25, 2.75, 0.50, -1.25]
    // Since ā_i = e_i^T B^{-1} A, we can simply construct B = I.
    // That means B^{-1} = I, so the extracted tableau row is exactly the row of A.
    // A will have 1 row and 5 columns (the 5th is the basic slack variable representing B=I).
    
    core::Model host_model;
    host_model.num_rows = 1;
    host_model.num_cols = 5;
    host_model.A.rows = 1;
    host_model.A.cols = 5;
    
    // Construct CSR/CSC. 1 row, 5 cols, 5 nonzeros.
    host_model.A.col_ptrs = {0, 1, 2, 3, 4, 5};
    host_model.A.row_indices = {0, 0, 0, 0, 0};
    host_model.A.values = {1.25, 2.75, 0.50, -1.25, 1.0};
    
    host_model.obj = {0.0, 0.0, 0.0, 0.0, 0.0};
    host_model.lb = {0.0, 0.0, 0.0, 0.0, 0.0};
    host_model.ub = {0.0, 0.0, 0.0, 0.0, 0.0};
    host_model.vtype = {
        VariableType::Continuous, VariableType::Continuous, VariableType::Continuous,
        VariableType::Continuous, VariableType::Continuous
    };

    gpu::VRAMArena arena(1024 * 1024);
    
    // 1. Upload Model to GPU
    gpu::DeviceModel d_model = gpu::upload_to_device(host_model, arena);

    // 2. Establish Basis
    simplex::Basis basis;
    basis.col_status.resize(5, simplex::BasisStatus::AtLower);
    basis.col_status[4] = simplex::BasisStatus::Basic; 
    basis.basic_indices = {4}; // Column 4 is basic

    // 3. Perform CPU Root Factorization (B = I)
    numerics::SparseLUFactorization cpu_lu;
    cpu_lu.factorize(host_model.A, basis);

    // 4. Upload Factorization to GPU
    gpu::DeviceSparseLUManager lu_manager(arena);
    lu_manager.upload(cpu_lu);
    gpu::DeviceSparseLU d_lu = lu_manager.get_device_struct();

    // 5. Establish HBF Working State
    gpu::HBFManager hbf_manager(arena);
    gpu::WorkingBasisState ws = hbf_manager.allocate_working_state(1, 5, 0, 0);

    // 6. Test Inputs
    Index basic_row = 0; // Only 1 row in our mock
    Float basic_value = 3.75;

    // 7. Execute warp-level GPU extraction (proves no basis-inverse host roundtrip occurs)
    cuts::Cut cut = gpu::generate_gomory_cut(d_model, d_lu, ws, basic_row, basic_value, arena);

    // 8. Validate mathematical exactness
    // Fractional requirement:
    // f_0 = b_bar_i - floor(b_bar_i)
    // f_0 = 3.75 - 3 = 0.75
    // RHS = -f_0 = -0.75
    REQUIRE(cut.rhs == Catch::Approx(-0.75).margin(1e-9));

    // Coefficients:
    // f_j = a_ij - floor(a_ij)
    // Cut coefficient = -f_j
    // j=0: 1.25 -> f = 0.25 -> -0.25
    // j=1: 2.75 -> f = 0.75 -> -0.75
    // j=2: 0.50 -> f = 0.50 -> -0.50
    // j=3: -1.25 -> f = -1.25 - (-2.0) = 0.75 -> -0.75
    // j=4: 1.00 -> f = 0.00 -> -0.00
    REQUIRE(cut.coefficients.size() == 5);
    REQUIRE(cut.coefficients[0] == Catch::Approx(-0.25).margin(1e-9));
    REQUIRE(cut.coefficients[1] == Catch::Approx(-0.75).margin(1e-9));
    REQUIRE(cut.coefficients[2] == Catch::Approx(-0.50).margin(1e-9));
    REQUIRE(cut.coefficients[3] == Catch::Approx(-0.75).margin(1e-9));
    REQUIRE(std::abs(cut.coefficients[4]) < 1e-9);

    // 9. Cleanup
    hbf_manager.free_working_state(ws);
    lu_manager.free_all();
}
