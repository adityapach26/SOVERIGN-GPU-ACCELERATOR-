/**
 * @file test_hbf.cu
 * @brief Unit tests for Hierarchical Basis Forest (Steps 11.1 & 11.2)
 */

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <vector>

#include "gpu/hbf.cuh"

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
     * We construct a 3-level HBF chain (root → parent → child) where
     * each edge carries a Forrest-Tomlin eta transformation.
     *
     * The working vector starts at work_vec = [10.0, 20.0, 30.0].
     *
     * Root edge FT update:
     *   leaving_row = 0, entering_col = 5
     *   eta column (sparse): row 0 -> 0.5 (pivot), row 1 -> 2.0
     *   Mathematical operation E_root * work_vec:
     *     x_p = work_vec[0] = 10.0
     *     work_vec[0] = 0.5 * 10.0 = 5.0    (pivot row: scale)
     *     work_vec[1] = 20.0 + 2.0 * 10.0 = 40.0  (off-pivot: add)
     *     work_vec[2] = 30.0  (untouched)
     *   After root: [5.0, 40.0, 30.0]
     *
     * Parent edge FT update:
     *   leaving_row = 1, entering_col = 6
     *   eta column (sparse): row 1 -> -0.25 (pivot), row 2 -> 3.0
     *   Mathematical operation E_parent * work_vec:
     *     x_p = work_vec[1] = 40.0
     *     work_vec[1] = -0.25 * 40.0 = -10.0  (pivot row: scale)
     *     work_vec[2] = 30.0 + 3.0 * 40.0 = 150.0  (off-pivot: add)
     *     work_vec[0] = 5.0  (untouched)
     *   After parent: [5.0, -10.0, 150.0]
     *
     * Child edge FT update:
     *   leaving_row = 2, entering_col = 7
     *   eta column (sparse): row 0 -> -1.0, row 2 -> 2.0 (pivot)
     *   Mathematical operation E_child * work_vec:
     *     x_p = work_vec[2] = 150.0
     *     work_vec[0] = 5.0 + (-1.0) * 150.0 = -145.0  (off-pivot)
     *     work_vec[2] = 2.0 * 150.0 = 300.0  (pivot row: scale)
     *     work_vec[1] = -10.0  (untouched)
     *   After child: [-145.0, -10.0, 300.0]
     *
     * These expected values are UNIQUELY determined by the actual eta
     * transformation. Any implementation that merely appends/copies eta
     * data WITHOUT applying the transformation will NOT produce these
     * values, causing the test to FAIL.
     */

    gpu::VRAMArena arena(4 * 1024 * 1024);

    // Mock DeviceModel with 3 variables
    Index m = 3;
    Index num_cols = 3;
    std::vector<Float> h_orig_lb = {0.0, 0.0, 0.0};
    std::vector<Float> h_orig_ub = {100.0, 100.0, 100.0};

    gpu::DeviceModel d_model;
    d_model.cols = num_cols;
    d_model.lb = static_cast<Float*>(arena.allocate(num_cols * sizeof(Float)));
    d_model.ub = static_cast<Float*>(arena.allocate(num_cols * sizeof(Float)));

    cudaMemcpy(d_model.lb, h_orig_lb.data(), num_cols * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model.ub, h_orig_ub.data(), num_cols * sizeof(Float), cudaMemcpyHostToDevice);

    gpu::HBFManager manager(arena);

    // ---- Build 3-level chain ----
    // Root (node 10, self-parent): bound delta on var 0, one FT update
    std::vector<gpu::BoundDelta> root_deltas = {{0, 5.0, 95.0}};
    std::vector<gpu::FTUpdateHost> root_fts = {
        {0, 5, {0.5, 2.0}, {0, 1}}  // leaving_row=0, entering_col=5, eta: r0->0.5, r1->2.0
    };
    manager.create_node(10, 10, root_deltas, root_fts);

    // Parent (node 11, parent=10): bound deltas, one FT update
    std::vector<gpu::BoundDelta> parent_deltas = {{1, 10.0, 90.0}, {0, 15.0, 85.0}};
    std::vector<gpu::FTUpdateHost> parent_fts = {
        {1, 6, {-0.25, 3.0}, {1, 2}}  // leaving_row=1, entering_col=6, eta: r1->-0.25, r2->3.0
    };
    manager.create_node(11, 10, parent_deltas, parent_fts);

    // Child (node 12, parent=11): bound deltas, one FT update
    std::vector<gpu::BoundDelta> child_deltas = {{2, 20.0, 80.0}, {1, 30.0, 70.0}};
    std::vector<gpu::FTUpdateHost> child_fts = {
        {2, 7, {-1.0, 2.0}, {0, 2}}  // leaving_row=2, entering_col=7, eta: r0->-1.0, r2->2.0
    };
    manager.create_node(12, 11, child_deltas, child_fts);

    // ---- Allocate working state ----
    gpu::WorkingBasisState ws = manager.allocate_working_state(m, num_cols, 100, 20);

    // Initialize working state with known initial values
    std::vector<Index> h_basis = {0, 1, 2};       // initial basis columns
    std::vector<Float> h_work = {10.0, 20.0, 30.0}; // initial working vector

    cudaMemcpy(ws.basis_indices, h_basis.data(), m * sizeof(Index), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.work_vec, h_work.data(), m * sizeof(Float), cudaMemcpyHostToDevice);

    // ---- Execute GPU-resident inheritance ----
    manager.inherit_basis(12, d_model, ws);

    // ============================================================
    // VERIFY ACTUAL MATHEMATICAL FT TRANSFORMATION
    // ============================================================
    // These assertions will FAIL if FT is merely appended/copied.

    // (A) Verify work_vec was mathematically transformed
    std::vector<Float> h_result_work(m);
    cudaMemcpy(h_result_work.data(), ws.work_vec, m * sizeof(Float), cudaMemcpyDeviceToHost);

    REQUIRE(h_result_work[0] == -145.0);
    REQUIRE(h_result_work[1] == -10.0);
    REQUIRE(h_result_work[2] == 300.0);

    // (B) Verify basis_indices were updated by FT leaving_row/entering_col
    std::vector<Index> h_result_basis(m);
    cudaMemcpy(h_result_basis.data(), ws.basis_indices, m * sizeof(Index), cudaMemcpyDeviceToHost);

    REQUIRE(h_result_basis[0] == 5);  // Root FT: leaving_row=0 → entering_col=5
    REQUIRE(h_result_basis[1] == 6);  // Parent FT: leaving_row=1 → entering_col=6
    REQUIRE(h_result_basis[2] == 7);  // Child FT: leaving_row=2 → entering_col=7

    // (C) Verify eta-file was built with correct structure
    Index h_num_eta_cols = 0;
    cudaMemcpy(&h_num_eta_cols, ws.num_eta_cols, sizeof(Index), cudaMemcpyDeviceToHost);
    REQUIRE(h_num_eta_cols == 3); // 3 FT updates across root→parent→child

    Index h_eta_nnz = 0;
    cudaMemcpy(&h_eta_nnz, ws.eta_nnz, sizeof(Index), cudaMemcpyDeviceToHost);
    REQUIRE(h_eta_nnz == 6); // 2 + 2 + 2 non-zeros total

    // Verify eta_col_starts consistency
    std::vector<Index> h_col_starts(4);
    cudaMemcpy(h_col_starts.data(), ws.eta_col_starts, 4 * sizeof(Index), cudaMemcpyDeviceToHost);
    REQUIRE(h_col_starts[0] == 0);
    REQUIRE(h_col_starts[1] == 2);
    REQUIRE(h_col_starts[2] == 4);
    REQUIRE(h_col_starts[3] == 6);

    // Verify eta_pivot_row records
    std::vector<Index> h_pivot_rows(3);
    cudaMemcpy(h_pivot_rows.data(), ws.eta_pivot_row, 3 * sizeof(Index), cudaMemcpyDeviceToHost);
    REQUIRE(h_pivot_rows[0] == 0);  // Root pivot row
    REQUIRE(h_pivot_rows[1] == 1);  // Parent pivot row
    REQUIRE(h_pivot_rows[2] == 2);  // Child pivot row

    // Verify actual eta values stored in the file
    std::vector<Float> h_eta_vals(6);
    std::vector<Index> h_eta_rows(6);
    cudaMemcpy(h_eta_vals.data(), ws.eta_vals, 6 * sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eta_rows.data(), ws.eta_rows, 6 * sizeof(Index), cudaMemcpyDeviceToHost);

    // Root eta column: {0.5 at row 0, 2.0 at row 1}
    REQUIRE(h_eta_vals[0] == 0.5);
    REQUIRE(h_eta_rows[0] == 0);
    REQUIRE(h_eta_vals[1] == 2.0);
    REQUIRE(h_eta_rows[1] == 1);
    // Parent eta column: {-0.25 at row 1, 3.0 at row 2}
    REQUIRE(h_eta_vals[2] == -0.25);
    REQUIRE(h_eta_rows[2] == 1);
    REQUIRE(h_eta_vals[3] == 3.0);
    REQUIRE(h_eta_rows[3] == 2);
    // Child eta column: {-1.0 at row 0, 2.0 at row 2}
    REQUIRE(h_eta_vals[4] == -1.0);
    REQUIRE(h_eta_rows[4] == 0);
    REQUIRE(h_eta_vals[5] == 2.0);
    REQUIRE(h_eta_rows[5] == 2);

    // (D) Verify cumulative BoundDeltas
    std::vector<Float> h_working_lb(num_cols);
    std::vector<Float> h_working_ub(num_cols);
    cudaMemcpy(h_working_lb.data(), ws.lb, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_working_ub.data(), ws.ub, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);

    // Root: var0 [5,95]; Parent: var1 [10,90], var0 [15,85]; Child: var2 [20,80], var1 [30,70]
    REQUIRE(h_working_lb[0] == 15.0);
    REQUIRE(h_working_ub[0] == 85.0);
    REQUIRE(h_working_lb[1] == 30.0);
    REQUIRE(h_working_ub[1] == 70.0);
    REQUIRE(h_working_lb[2] == 20.0);
    REQUIRE(h_working_ub[2] == 80.0);

    // (E) Verify original DeviceModel bounds unchanged
    std::vector<Float> h_pristine_lb(num_cols);
    cudaMemcpy(h_pristine_lb.data(), d_model.lb, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);
    REQUIRE(h_pristine_lb[0] == 0.0);
    REQUIRE(h_pristine_lb[1] == 0.0);
    REQUIRE(h_pristine_lb[2] == 0.0);

    // ---- Error handling tests ----
    // Cycle: node 13 → 14 → 13
    manager.create_node(13, 14, {}, {});
    manager.create_node(14, 13, {}, {});
    REQUIRE_THROWS_AS(manager.inherit_basis(14, d_model, ws), std::runtime_error);

    // ---- Reclamation ----
    manager.free_working_state(ws);
    arena.free(d_model.lb);
    arena.free(d_model.ub);
    manager.free_all_nodes();
    REQUIRE(arena.occupancy_percentage() == 0);
}
