/**
 * @file test_hbf.cu
 * @brief Unit tests for Hierarchical Basis Forest (Steps 11.1 & 11.2)
 */

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <vector>

#include "gpu/hbf.cuh"
#include "numerics/sparse_lu.hpp"
#include "core/sparse_matrix.hpp"

using namespace sankhya;

// Small CUDA kernel verifying struct readability on device.
__global__ void inspect_hbf_node_kernel(
    gpu::HBFNode node,
    int* d_flags)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (node.num_deltas > 0 && node.deltas != nullptr) {
            if (node.deltas[0].var_idx == 1 && node.deltas[0].new_lb == 5.0) {
                d_flags[0] = 1;
            }
        }
    }
}

TEST_CASE("HBFNode - Lightweight Representation & VRAMArena Management", "[gpu][hbf]") {
    gpu::VRAMArena arena(1024 * 1024);
    REQUIRE(arena.occupancy_percentage() == 0);

    gpu::HBFManager hbf_manager(arena);

    // Root: parent_id == node_id
    std::vector<gpu::BoundDelta> root_deltas = {{0, 0.0, 10.0}};
    gpu::HBFNode root = hbf_manager.create_node(100, 100, root_deltas);
    REQUIRE(root.node_id == 100);
    REQUIRE(root.parent_id == 100);
    REQUIRE(root.num_deltas == 1);
    REQUIRE(root.deltas != nullptr);
    REQUIRE(root.is_fathomed == false);
    REQUIRE(arena.occupancy_percentage() > 0);

    std::vector<gpu::BoundDelta> child1_deltas = {{1, 5.0, 15.0}};
    std::vector<gpu::BoundDelta> child2_deltas = {{1, 0.0, 4.0}};

    gpu::HBFNode child1 = hbf_manager.create_node(101, root.node_id, child1_deltas);
    gpu::HBFNode child2 = hbf_manager.create_node(102, root.node_id, child2_deltas);

    REQUIRE(child1.parent_id == 100);
    REQUIRE(child2.parent_id == 100);
    REQUIRE(sizeof(gpu::HBFNode) <= 64);

    REQUIRE(child1.deltas != nullptr);
    REQUIRE(child2.deltas != nullptr);
    REQUIRE(child1.deltas != child2.deltas);
    REQUIRE(child1.deltas != root.deltas);

    int* d_flags = static_cast<int*>(arena.allocate(sizeof(int)));
    int h_init = 0;
    cudaMemcpy(d_flags, &h_init, sizeof(int), cudaMemcpyHostToDevice);
    inspect_hbf_node_kernel<<<1, 1>>>(child1, d_flags);
    cudaDeviceSynchronize();
    int h_flags = 0;
    cudaMemcpy(&h_flags, d_flags, sizeof(int), cudaMemcpyDeviceToHost);
    REQUIRE(h_flags == 1);
    arena.free(d_flags);

    hbf_manager.free_all_nodes();
    REQUIRE(arena.occupancy_percentage() == 0);
}

TEST_CASE("HBF - Actual Forrest-Tomlin Basis Inheritance (Step 11.2)", "[gpu][hbf][inherit]") {
    /**
     * Mathematical Setup
     * ==================
     * We construct a deterministic nonsingular basis B and test incremental
     * updates. We then build an HBF chain mimicking those updates, run
     * GPU inheritance, and compare the GPU-resident FTRAN result to a fresh
     * CPU factorization of the same final basis.
     */

    gpu::VRAMArena arena(4 * 1024 * 1024);

    Index m = 5;
    Index num_cols = 10;
    
    // Create an LP with 5 constraints and 10 variables
    std::vector<Float> vals;
    std::vector<Index> rows;
    std::vector<Index> cols;
    cols.push_back(0);
    // Identity for first 5 columns
    for(Index j=0; j<5; ++j) {
        vals.push_back(1.0); rows.push_back(j); cols.push_back(cols.back() + 1);
    }
    // Additional columns for pivoting
    for(Index j=5; j<10; ++j) {
        vals.push_back(2.0); rows.push_back(j - 5);
        vals.push_back(-1.0); rows.push_back((j - 4) % 5);
        cols.push_back(cols.back() + 2);
    }
    core::CSCMatrix A(5, 10, std::move(vals), std::move(rows), std::move(cols));

    simplex::Basis basis;
    basis.col_status.resize(10, simplex::BasisStatus::AtLower);
    basis.basic_indices = {1, 0, 2, 3, 4};
    for(Index idx : basis.basic_indices) basis.col_status[idx] = simplex::BasisStatus::Basic;

    // Factorize base
    numerics::SparseLUFactorization lu_base;
    lu_base.factorize(A, basis);

    gpu::DeviceModel d_model;
    d_model.cols = num_cols;
    d_model.lb = static_cast<Float*>(arena.allocate(num_cols * sizeof(Float)));
    d_model.ub = static_cast<Float*>(arena.allocate(num_cols * sizeof(Float)));
    
    std::vector<Float> h_orig_lb(10, 0.0);
    std::vector<Float> h_orig_ub(10, 100.0);
    cudaMemcpy(d_model.lb, h_orig_lb.data(), num_cols * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model.ub, h_orig_ub.data(), num_cols * sizeof(Float), cudaMemcpyHostToDevice);

    gpu::HBFManager manager(arena);
    manager.set_root_basis({1, 0, 2, 3, 4});

    // Pivot 1 (Root node 10): replace col 0 with col 5
    Index entering_col_1 = 5;
    Index leaving_row_1 = 0;
    std::vector<Float> Aq_1(5, 0.0);
    for(Index k=A.col_ptrs[entering_col_1]; k<A.col_ptrs[entering_col_1+1]; ++k) Aq_1[A.row_indices[k]] = A.values[k];
    
    std::vector<Float> d_1 = Aq_1;
    lu_base.ftran(d_1); // FTRAN to get eta vector

    std::vector<gpu::FTUpdateHost> fts_1;
    gpu::FTUpdateHost ft1;
    ft1.leaving_row = leaving_row_1;
    ft1.entering_col = entering_col_1;
    Float d_p1 = d_1[leaving_row_1];
    for (Index i = 0; i < m; ++i) {
        Float val = (i == leaving_row_1) ? (1.0 / d_p1) : (-d_1[i] / d_p1);
        if (std::abs(val) > 1e-15) {
            ft1.eta_indices.push_back(i);
            ft1.eta_values.push_back(val);
        }
    }
    fts_1.push_back(ft1);
    
    std::vector<gpu::BoundDelta> deltas_1 = {{0, 5.0, 95.0}};
    manager.create_node(10, 10, deltas_1, fts_1);
    
    basis.basic_indices[leaving_row_1] = entering_col_1;
    numerics::SparseLUFactorization lu_1;
    lu_1.factorize(A, basis);

    // Pivot 2 (Parent node 11): replace col 1 with col 6
    Index entering_col_2 = 6;
    Index leaving_row_2 = 1;
    std::vector<Float> Aq_2(5, 0.0);
    for(Index k=A.col_ptrs[entering_col_2]; k<A.col_ptrs[entering_col_2+1]; ++k) Aq_2[A.row_indices[k]] = A.values[k];
    
    std::vector<Float> d_2 = Aq_2;
    lu_1.ftran(d_2); // FTRAN on previous basis

    std::vector<gpu::FTUpdateHost> fts_2;
    gpu::FTUpdateHost ft2;
    ft2.leaving_row = leaving_row_2;
    ft2.entering_col = entering_col_2;
    Float d_p2 = d_2[leaving_row_2];
    for (Index i = 0; i < m; ++i) {
        Float val = (i == leaving_row_2) ? (1.0 / d_p2) : (-d_2[i] / d_p2);
        if (std::abs(val) > 1e-15) {
            ft2.eta_indices.push_back(i);
            ft2.eta_values.push_back(val);
        }
    }
    fts_2.push_back(ft2);
    
    std::vector<gpu::BoundDelta> deltas_2 = {{1, 10.0, 90.0}};
    manager.create_node(11, 10, deltas_2, fts_2);
    
    basis.basic_indices[leaving_row_2] = entering_col_2;
    numerics::SparseLUFactorization lu_2;
    lu_2.factorize(A, basis);

    // Pivot 3 (Child node 12): replace col 2 with col 7
    Index entering_col_3 = 7;
    Index leaving_row_3 = 2;
    std::vector<Float> Aq_3(5, 0.0);
    for(Index k=A.col_ptrs[entering_col_3]; k<A.col_ptrs[entering_col_3+1]; ++k) Aq_3[A.row_indices[k]] = A.values[k];
    
    std::vector<Float> d_3 = Aq_3;
    lu_2.ftran(d_3); // FTRAN on previous basis

    std::vector<gpu::FTUpdateHost> fts_3;
    gpu::FTUpdateHost ft3;
    ft3.leaving_row = leaving_row_3;
    ft3.entering_col = entering_col_3;
    Float d_p3 = d_3[leaving_row_3];
    for (Index i = 0; i < m; ++i) {
        Float val = (i == leaving_row_3) ? (1.0 / d_p3) : (-d_3[i] / d_p3);
        if (std::abs(val) > 1e-15) {
            ft3.eta_indices.push_back(i);
            ft3.eta_values.push_back(val);
        }
    }
    fts_3.push_back(ft3);
    
    std::vector<gpu::BoundDelta> deltas_3 = {{2, 2.0, 8.0}};
    manager.create_node(12, 11, deltas_3, fts_3);
    
    basis.basic_indices[leaving_row_3] = entering_col_3;
    numerics::SparseLUFactorization lu_3;
    lu_3.factorize(A, basis);

    // ---- Allocate working state ----
    gpu::WorkingBasisState ws = manager.allocate_working_state(m, num_cols, 100, 20);

    // We want to test FTRAN on the GPU. 
    // The GPU kernel inherently applies E_1 * E_2 * ... * work_vec.
    // So if we seed work_vec with B_0^{-1} * rhs, the kernel will produce B_2^{-1} * rhs.
    std::vector<Float> rhs = {1.5, -2.1, 3.4, 0.8, -1.2};
    std::vector<Float> base_ftran_rhs = rhs;
    
    // CPU base LU ftran
    numerics::SparseLUFactorization lu_base_for_test;
    simplex::Basis base_basis;
    base_basis.col_status.resize(10, simplex::BasisStatus::AtLower);
    base_basis.basic_indices = {1, 0, 2, 3, 4};
    for(Index idx : base_basis.basic_indices) base_basis.col_status[idx] = simplex::BasisStatus::Basic;
    lu_base_for_test.factorize(A, base_basis);
    lu_base_for_test.ftran(base_ftran_rhs);

    // Initialize working state with B_0^{-1} rhs
    cudaMemcpy(ws.work_vec, base_ftran_rhs.data(), m * sizeof(Float), cudaMemcpyHostToDevice);

    // ---- Execute GPU-resident inheritance ----
    manager.inherit_basis(12, d_model, ws);

    // Verify GPU FTRAN result matches FRESH FACTORIZATION FTRAN result
    std::vector<Float> h_result_work(m);
    cudaMemcpy(h_result_work.data(), ws.work_vec, m * sizeof(Float), cudaMemcpyDeviceToHost);

    std::vector<Float> fresh_rhs = rhs;
    lu_3.ftran(fresh_rhs);

    for (Index i = 0; i < m; ++i) {
        REQUIRE(std::abs(h_result_work[i] - fresh_rhs[i]) < 1e-9);
    }

    // Verify basis_indices were updated
    std::vector<Index> h_result_basis(m);
    cudaMemcpy(h_result_basis.data(), ws.basis_indices, m * sizeof(Index), cudaMemcpyDeviceToHost);
    // Updated positions
    REQUIRE(h_result_basis[0] == 5); // Node 10 (Pivot 1)
    REQUIRE(h_result_basis[1] == 6); // Node 11 (Pivot 2)
    REQUIRE(h_result_basis[2] == 7); // Node 12 (Pivot 3)
    // Unchanged positions (should mirror initial basis {1, 0, 2, 3, 4})
    REQUIRE(h_result_basis[3] == 3);
    REQUIRE(h_result_basis[4] == 4);

    // Verify cumulative BoundDeltas
    std::vector<Float> h_working_lb(num_cols);
    std::vector<Float> h_working_ub(num_cols);
    cudaMemcpy(h_working_lb.data(), ws.lb, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_working_ub.data(), ws.ub, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);

    REQUIRE(h_working_lb[0] == 5.0);
    REQUIRE(h_working_ub[0] == 95.0);
    REQUIRE(h_working_lb[1] == 10.0);
    REQUIRE(h_working_ub[1] == 90.0);
    REQUIRE(h_working_lb[2] == 2.0);
    REQUIRE(h_working_ub[2] == 8.0);

    // Verify original DeviceModel bounds unchanged
    std::vector<Float> h_pristine_lb(num_cols);
    cudaMemcpy(h_pristine_lb.data(), d_model.lb, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);
    REQUIRE(h_pristine_lb[0] == 0.0);
    REQUIRE(h_pristine_lb[1] == 0.0);
    REQUIRE(h_pristine_lb[2] == 0.0);

    // ---- Error handling tests ----
    // Cycle: node 13 -> 14 -> 13
    manager.create_node(13, 14, {}, {});
    manager.create_node(14, 13, {}, {});
    REQUIRE_THROWS_AS(manager.inherit_basis(14, d_model, ws), std::runtime_error);

    manager.free_working_state(ws);
    arena.free(d_model.lb);
    arena.free(d_model.ub);

    // ---- Test set_root_basis single-allocation safety ----
    // 1. Same size (reuse existing allocation)
    manager.set_root_basis({1, 0, 2, 3, 4}); 
    // 2. Different size (release old, allocate new)
    manager.set_root_basis({1, 0, 2, 3, 4, 5, 6, 7, 8, 9});

    manager.free_all_nodes();
    REQUIRE(arena.occupancy_percentage() == 0);
}
