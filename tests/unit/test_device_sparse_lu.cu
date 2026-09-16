#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "gpu/device_sparse_lu.cuh"
#include "numerics/sparse_lu.hpp"
#include "gpu/vram_arena.cuh"
#include "core/sparse_matrix.hpp"
#include "simplex/basis.hpp"
#include <vector>
#include <stdexcept>
#include <string>

using namespace sankhya;

// Helper for testing
static void check_cuda_test_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

TEST_CASE("Pre-Phase 13.1 Foundation 0 - DeviceSparseLU Representation", "[cuda][sparse_lu]") {
    // 1. Construct a small nontrivial matrix and basis
    core::CSCMatrix A;
    A.rows = 4;
    A.cols = 4;
    
    // Nontrivial 4x4 matrix
    // [  1   2   0   0 ]
    // [  0   3   4   0 ]
    // [  0   0   5   6 ]
    // [  7   0   0   8 ]
    A.col_ptrs = {0, 2, 4, 6, 8};
    A.row_indices = {0, 3,  0, 1,  1, 2,  2, 3};
    A.values =      {1.0, 7.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0};

    simplex::Basis basis;
    basis.col_status.resize(4, simplex::BasisStatus::AtLower);
    basis.basic_indices = {0, 1, 2, 3}; // Select all columns to form a basis
    for(Index idx : basis.basic_indices) basis.col_status[idx] = simplex::BasisStatus::Basic;

    // 2. CPU Factorization
    numerics::SparseLUFactorization cpu_lu;
    cpu_lu.factorize(A, basis);

    REQUIRE(cpu_lu.is_factorized() == true);
    REQUIRE(cpu_lu.get_m() == 4);
    REQUIRE(cpu_lu.get_L_vals().size() > 0); // Non-trivial L
    REQUIRE(cpu_lu.get_U_vals().size() > 0); // Non-trivial U

    // 3. GPU VRAM Arena and Upload
    gpu::VRAMArena arena(1024 * 1024); // 1 MB
    gpu::DeviceSparseLUManager manager(arena);
    
    manager.upload(cpu_lu);

    gpu::DeviceSparseLU d_lu = manager.get_device_struct();

    // 4. Validate Metadata
    REQUIRE(d_lu.m == 4);
    REQUIRE(d_lu.L_nnz == static_cast<Index>(cpu_lu.get_L_vals().size()));
    REQUIRE(d_lu.U_nnz == static_cast<Index>(cpu_lu.get_U_vals().size()));

    // 5. Verify Complete Arrays (Test-only device-to-host copy)
    // [ENGINEERING DECISION] 
    // We explicitly use cudaMemcpy to copy the entire arrays back to host vectors 
    // to validate them element-by-element. This avoids raw cudaMalloc/cudaFree 
    // for temporary test buffers and ensures 100% verification of the device state.
    // This host round-trip is strictly for testing and does not bypass or pollute 
    // the production architecture.
    
    Index m = cpu_lu.get_m();
    Index L_nnz = cpu_lu.get_L_vals().size();
    Index U_nnz = cpu_lu.get_U_vals().size();

    std::vector<Float> h_L_vals(L_nnz), h_U_vals(U_nnz);
    std::vector<Index> h_L_rows(L_nnz), h_U_cols(U_nnz);
    std::vector<Index> h_L_col_ptrs(m + 1), h_U_row_ptrs(m + 1);
    std::vector<Index> h_perm_row(m), h_perm_col(m);
    std::vector<Index> h_inv_perm_row(m), h_inv_perm_col(m);

    check_cuda_test_error(cudaMemcpy(h_L_vals.data(), d_lu.L_vals, L_nnz * sizeof(Float), cudaMemcpyDeviceToHost), "L_vals");
    check_cuda_test_error(cudaMemcpy(h_L_rows.data(), d_lu.L_rows, L_nnz * sizeof(Index), cudaMemcpyDeviceToHost), "L_rows");
    check_cuda_test_error(cudaMemcpy(h_L_col_ptrs.data(), d_lu.L_col_ptrs, (m + 1) * sizeof(Index), cudaMemcpyDeviceToHost), "L_col_ptrs");

    check_cuda_test_error(cudaMemcpy(h_U_vals.data(), d_lu.U_vals, U_nnz * sizeof(Float), cudaMemcpyDeviceToHost), "U_vals");
    check_cuda_test_error(cudaMemcpy(h_U_cols.data(), d_lu.U_cols, U_nnz * sizeof(Index), cudaMemcpyDeviceToHost), "U_cols");
    check_cuda_test_error(cudaMemcpy(h_U_row_ptrs.data(), d_lu.U_row_ptrs, (m + 1) * sizeof(Index), cudaMemcpyDeviceToHost), "U_row_ptrs");

    check_cuda_test_error(cudaMemcpy(h_perm_row.data(), d_lu.perm_row, m * sizeof(Index), cudaMemcpyDeviceToHost), "perm_row");
    check_cuda_test_error(cudaMemcpy(h_perm_col.data(), d_lu.perm_col, m * sizeof(Index), cudaMemcpyDeviceToHost), "perm_col");
    check_cuda_test_error(cudaMemcpy(h_inv_perm_row.data(), d_lu.inv_perm_row, m * sizeof(Index), cudaMemcpyDeviceToHost), "inv_perm_row");
    check_cuda_test_error(cudaMemcpy(h_inv_perm_col.data(), d_lu.inv_perm_col, m * sizeof(Index), cudaMemcpyDeviceToHost), "inv_perm_col");

    // L Arrays
    for(size_t i = 0; i < static_cast<size_t>(L_nnz); ++i) {
        REQUIRE(h_L_vals[i] == Catch::Approx(cpu_lu.get_L_vals()[i]));
        REQUIRE(h_L_rows[i] == cpu_lu.get_L_rows()[i]);
    }
    for(size_t i = 0; i <= static_cast<size_t>(m); ++i) {
        REQUIRE(h_L_col_ptrs[i] == cpu_lu.get_L_col_ptrs()[i]);
    }

    // U Arrays
    for(size_t i = 0; i < static_cast<size_t>(U_nnz); ++i) {
        REQUIRE(h_U_vals[i] == Catch::Approx(cpu_lu.get_U_vals()[i]));
        REQUIRE(h_U_cols[i] == cpu_lu.get_U_cols()[i]);
    }
    for(size_t i = 0; i <= static_cast<size_t>(m); ++i) {
        REQUIRE(h_U_row_ptrs[i] == cpu_lu.get_U_row_ptrs()[i]);
    }

    // Permutations
    for(size_t i = 0; i < static_cast<size_t>(m); ++i) {
        REQUIRE(h_perm_row[i] == cpu_lu.get_perm_row()[i]);
        REQUIRE(h_perm_col[i] == cpu_lu.get_perm_col()[i]);
        REQUIRE(h_inv_perm_row[i] == cpu_lu.get_inv_perm_row()[i]);
        REQUIRE(h_inv_perm_col[i] == cpu_lu.get_inv_perm_col()[i]);
    }

    // 6. Memory Cleanup
    manager.free_all();
    REQUIRE(arena.occupancy_percentage() == 0);
}
