#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "gpu/device_ftran.cuh"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"
#include "numerics/sparse_lu.hpp"
#include "core/sparse_matrix.hpp"
#include "simplex/basis.hpp"
#include <vector>

using namespace sankhya;

// Test kernel wrapper to execute device FTRAN/BTRAN
__global__ void execute_device_ftran_kernel(
    gpu::DeviceSparseLU lu,
    gpu::WorkingBasisState ws,
    Float* vec)
{
    gpu::device_ftran(lu, ws, vec);
}

__global__ void execute_device_btran_kernel(
    gpu::DeviceSparseLU lu,
    gpu::WorkingBasisState ws,
    Float* vec)
{
    gpu::device_btran(lu, ws, vec);
}

__global__ void execute_device_update_kernel(
    gpu::WorkingBasisState ws,
    Index leaving_row,
    Index entering_col,
    const Float* Aq,
    bool* success_out)
{
    bool ok = gpu::device_update_basis(ws, leaving_row, entering_col, Aq, 1e-9);
    if (threadIdx.x == 0) {
        *success_out = ok;
    }
}

TEST_CASE("Pre-Phase 13.1 Foundation A - Device FTRAN/BTRAN", "[cuda][ftran]") {
    // 1. Construct non-trivial matrix and basis
    // [  1   2   0   0 ]
    // [  0   3   4   0 ]
    // [  0   0   5   6 ]
    // [  7   0   0   8 ]
    core::CSCMatrix A(
        4, 4,
        {1.0, 7.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0}, // vals
        {0, 3,  0, 1,  1, 2,  2, 3}, // rows_idx
        {0, 2, 4, 6, 8} // col_ptrs
    );

    simplex::Basis basis;
    basis.col_status.resize(4, simplex::BasisStatus::AtLower);
    basis.basic_indices = {0, 1, 2, 3}; 
    for(Index idx : basis.basic_indices) basis.col_status[idx] = simplex::BasisStatus::Basic;

    numerics::SparseLUFactorization cpu_lu;
    cpu_lu.factorize(A, basis);

    gpu::VRAMArena arena(1024 * 1024);
    gpu::DeviceSparseLUManager manager(arena);
    manager.upload(cpu_lu);
    gpu::DeviceSparseLU d_lu = manager.get_device_struct();

    gpu::HBFManager hbf_manager(arena);
    
    // Standard state for success tests
    gpu::WorkingBasisState ws = hbf_manager.allocate_working_state(4, 4, 100, 10);
    
    int zero_int = 0;
    Index zero_idx = 0;
    cudaMemcpy(ws.error_code, &zero_int, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.num_eta_cols, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.eta_nnz, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.eta_col_starts, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);

    SECTION("Base FTRAN") {
        std::vector<Float> h_rhs = {1.0, 2.0, 3.0, 4.0};
        std::vector<Float> h_rhs_cpu = h_rhs;
        
        cpu_lu.ftran(h_rhs_cpu);

        cudaMemcpy(ws.work_vec, h_rhs.data(), 4 * sizeof(Float), cudaMemcpyHostToDevice);
        execute_device_ftran_kernel<<<1, 32>>>(d_lu, ws, ws.work_vec);
        cudaDeviceSynchronize();
        
        std::vector<Float> h_rhs_gpu(4);
        cudaMemcpy(h_rhs_gpu.data(), ws.work_vec, 4 * sizeof(Float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 4; ++i) {
            REQUIRE(h_rhs_gpu[i] == Catch::Approx(h_rhs_cpu[i]).margin(1e-9));
        }
    }

    SECTION("Base BTRAN") {
        std::vector<Float> h_rhs = {1.0, 2.0, 3.0, 4.0};
        std::vector<Float> h_rhs_cpu = h_rhs;
        
        cpu_lu.btran(h_rhs_cpu);

        cudaMemcpy(ws.work_vec, h_rhs.data(), 4 * sizeof(Float), cudaMemcpyHostToDevice);
        execute_device_btran_kernel<<<1, 32>>>(d_lu, ws, ws.work_vec);
        cudaDeviceSynchronize();
        
        std::vector<Float> h_rhs_gpu(4);
        cudaMemcpy(h_rhs_gpu.data(), ws.work_vec, 4 * sizeof(Float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 4; ++i) {
            REQUIRE(h_rhs_gpu[i] == Catch::Approx(h_rhs_cpu[i]).margin(1e-9));
        }
    }

    SECTION("FTRAN/BTRAN after FT Update") {
        // [ENGINEERING DECISION] CPU update contract resolution
        // CPU SparseLU::update(Aq) takes the original column Aq and calls ftran(Aq) internally
        // to produce d_q. To provide mathematically equivalent data to device_update_basis
        // (which expects d_q), we run device_ftran(Aq) first, then pass the result to device_update_basis.
        
        std::vector<Float> Aq = {0.5, 0.0, 1.5, -0.5}; // original column
        Index leaving_row = 1;
        Index entering_col = 4;

        // Apply update to CPU
        cpu_lu.update(leaving_row, entering_col, Aq);

        // Apply update to GPU
        // 1. Compute d_q = B^-1 A_q natively on device
        cudaMemcpy(ws.work_vec, Aq.data(), 4 * sizeof(Float), cudaMemcpyHostToDevice);
        execute_device_ftran_kernel<<<1, 32>>>(d_lu, ws, ws.work_vec);
        cudaDeviceSynchronize();

        // 2. Perform the update with the computed d_q (now in work_vec)
        bool* d_success = static_cast<bool*>(arena.allocate(sizeof(bool)));
        execute_device_update_kernel<<<1, 32>>>(ws, leaving_row, entering_col, ws.work_vec, d_success);
        cudaDeviceSynchronize();

        bool h_success = false;
        cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);
        REQUIRE(h_success == true);

        // Verify successful metadata update
        Index h_num_cols = 0;
        cudaMemcpy(&h_num_cols, ws.num_eta_cols, sizeof(Index), cudaMemcpyDeviceToHost);
        REQUIRE(h_num_cols == 1);

        // Test FTRAN post-update
        std::vector<Float> h_rhs_f = {1.0, 2.0, 3.0, 4.0};
        std::vector<Float> h_cpu_f = h_rhs_f;
        cpu_lu.ftran(h_cpu_f);

        cudaMemcpy(ws.work_vec, h_rhs_f.data(), 4 * sizeof(Float), cudaMemcpyHostToDevice);
        execute_device_ftran_kernel<<<1, 32>>>(d_lu, ws, ws.work_vec);
        cudaDeviceSynchronize();

        std::vector<Float> h_gpu_f(4);
        cudaMemcpy(h_gpu_f.data(), ws.work_vec, 4 * sizeof(Float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 4; ++i) {
            REQUIRE(h_gpu_f[i] == Catch::Approx(h_cpu_f[i]).margin(1e-9));
        }

        // Test BTRAN post-update
        std::vector<Float> h_rhs_b = {1.0, 2.0, 3.0, 4.0};
        std::vector<Float> h_cpu_b = h_rhs_b;
        cpu_lu.btran(h_cpu_b);

        cudaMemcpy(ws.work_vec, h_rhs_b.data(), 4 * sizeof(Float), cudaMemcpyHostToDevice);
        execute_device_btran_kernel<<<1, 32>>>(d_lu, ws, ws.work_vec);
        cudaDeviceSynchronize();

        std::vector<Float> h_gpu_b(4);
        cudaMemcpy(h_gpu_b.data(), ws.work_vec, 4 * sizeof(Float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 4; ++i) {
            REQUIRE(h_gpu_b[i] == Catch::Approx(h_cpu_b[i]).margin(1e-9));
        }

        arena.free(d_success);
    }

    SECTION("Update Capacity Failure Verification") {
        // Allocate a severely constrained working state to force overflow
        // Capacity: 0 columns, 0 nnz
        gpu::WorkingBasisState tiny_ws = hbf_manager.allocate_working_state(4, 4, 0, 0);
        
        cudaMemcpy(tiny_ws.error_code, &zero_int, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tiny_ws.num_eta_cols, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);
        cudaMemcpy(tiny_ws.eta_nnz, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);
        cudaMemcpy(tiny_ws.eta_col_starts, &zero_idx, sizeof(Index), cudaMemcpyHostToDevice);

        std::vector<Float> d_q = {0.5, 0.0, 1.5, -0.5}; // Non-trivial d_q
        cudaMemcpy(tiny_ws.work_vec, d_q.data(), 4 * sizeof(Float), cudaMemcpyHostToDevice);

        bool* d_success = static_cast<bool*>(arena.allocate(sizeof(bool)));
        execute_device_update_kernel<<<1, 32>>>(tiny_ws, 1, 4, tiny_ws.work_vec, d_success);
        cudaDeviceSynchronize();

        bool h_success = true;
        cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);
        
        // 1. Must return false
        REQUIRE(h_success == false);

        // 2. Must set error code 3
        int h_error = 0;
        cudaMemcpy(&h_error, tiny_ws.error_code, sizeof(int), cudaMemcpyDeviceToHost);
        REQUIRE(h_error == 3);

        // 3. Verify failed update does NOT mutate metadata
        Index h_num_cols = 1;
        Index h_eta_nnz = 1;
        Index h_start = 1;
        cudaMemcpy(&h_num_cols, tiny_ws.num_eta_cols, sizeof(Index), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_eta_nnz, tiny_ws.eta_nnz, sizeof(Index), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_start, tiny_ws.eta_col_starts, sizeof(Index), cudaMemcpyDeviceToHost);

        REQUIRE(h_num_cols == 0);
        REQUIRE(h_eta_nnz == 0);
        REQUIRE(h_start == 0);

        arena.free(d_success);
        hbf_manager.free_working_state(tiny_ws);
    }

    hbf_manager.free_working_state(ws);
    manager.free_all();
    REQUIRE(arena.occupancy_percentage() == 0);
}
