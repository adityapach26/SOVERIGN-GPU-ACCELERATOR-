#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "gpu/device_sparse_lu.cuh"
#include "numerics/sparse_lu.hpp"
#include "gpu/vram_arena.cuh"
#include "core/sparse_matrix.hpp"
#include "simplex/basis.hpp"
#include <vector>

using namespace sankhya;

// Small kernel to copy one value out of device arrays to prove residency
__global__ void verify_device_sparse_lu_kernel(
    gpu::DeviceSparseLU d_lu,
    Float* out_L_val,
    Float* out_U_val,
    Index* out_perm) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (d_lu.L_nnz > 0) *out_L_val = d_lu.L_vals[0];
        if (d_lu.U_nnz > 0) *out_U_val = d_lu.U_vals[0];
        if (d_lu.m > 0) *out_perm = d_lu.perm_row[0];
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

    // 4. Verify Representation on Device
    REQUIRE(d_lu.m == 4);
    REQUIRE(d_lu.L_nnz == cpu_lu.get_L_vals().size());
    REQUIRE(d_lu.U_nnz == cpu_lu.get_U_vals().size());

    // Allocate host-accessible device memory for verification output
    Float* d_out_L;
    Float* d_out_U;
    Index* d_out_perm;
    cudaMalloc(&d_out_L, sizeof(Float));
    cudaMalloc(&d_out_U, sizeof(Float));
    cudaMalloc(&d_out_perm, sizeof(Index));

    verify_device_sparse_lu_kernel<<<1, 1>>>(d_lu, d_out_L, d_out_U, d_out_perm);
    cudaDeviceSynchronize();

    Float h_out_L = 0, h_out_U = 0;
    Index h_out_perm = 0;
    cudaMemcpy(&h_out_L, d_out_L, sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_out_U, d_out_U, sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_out_perm, d_out_perm, sizeof(Index), cudaMemcpyDeviceToHost);

    // Verify values match the CPU representation
    REQUIRE(h_out_L == Catch::Approx(cpu_lu.get_L_vals()[0]));
    REQUIRE(h_out_U == Catch::Approx(cpu_lu.get_U_vals()[0]));
    REQUIRE(h_out_perm == cpu_lu.get_perm_row()[0]);

    // Clean up
    cudaFree(d_out_L);
    cudaFree(d_out_U);
    cudaFree(d_out_perm);

    manager.free_all();
    REQUIRE(arena.occupancy_percentage() == 0);
}

