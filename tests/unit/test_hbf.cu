/**
 * @file test_hbf.cu
 * @brief Unit tests for Hierarchical Basis Forest Node Representation (Step 11.1)
 */

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <vector>

#include "gpu/hbf.cuh"

using namespace sankhya;

// Small CUDA kernel verifying that pointers and struct contents are readable exactly 
// as intended from the GPU executing environment.
__global__ void inspect_hbf_node_kernel(
    gpu::HBFNode node,
    int* d_flags)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (node.num_deltas > 0 && node.deltas != nullptr) {
            // Check the exact node-local delta data resident natively in GPU VRAM
            if (node.deltas[0].var_idx == 1 && node.deltas[0].new_lb == 5.0) {
                d_flags[0] = 1;
            }
        }
    }
}

TEST_CASE("HBFNode - Lightweight Representation & VRAMArena Management", "[gpu][hbf]") {
    // 1. Create a VRAMArena.
    gpu::VRAMArena arena(1024 * 1024); // 1 MB is more than enough for testing
    std::size_t initial_occupancy = arena.occupancy_percentage();
    REQUIRE(initial_occupancy == 0);

    // 2. Create an HBFManager natively using that persistent arena.
    gpu::HBFManager hbf_manager(arena);

    // 3. Create one root node dynamically mapped out of the persistent arena.
    // Root convention: parent_id == node_id
    std::vector<gpu::BoundDelta> root_deltas = {
        {0, 0.0, 10.0}
    };
    gpu::HBFNode root = hbf_manager.create_node(100, 100, root_deltas);
    REQUIRE(root.node_id == 100);
    REQUIRE(root.parent_id == 100);
    REQUIRE(root.num_deltas == 1);
    REQUIRE(root.deltas != nullptr);
    REQUIRE(root.is_fathomed == false);

    // Confirm the arena correctly recognized the reservation.
    std::size_t occupancy_after_root = arena.occupancy_percentage();
    REQUIRE(occupancy_after_root > 0);

    // 4. Create two child nodes linking back to the root.
    std::vector<gpu::BoundDelta> child1_deltas = {
        {1, 5.0, 15.0} // Branching constraint delta (e.g., x1 >= 5)
    };
    std::vector<gpu::BoundDelta> child2_deltas = {
        {1, 0.0, 4.0}  // Branching constraint delta (e.g., x1 <= 4)
    };
    
    gpu::HBFNode child1 = hbf_manager.create_node(101, root.node_id, child1_deltas);
    gpu::HBFNode child2 = hbf_manager.create_node(102, root.node_id, child2_deltas);

    // 5. Strict Property Verifications
    // The two children correctly reference the lightweight parent ID.
    REQUIRE(child1.parent_id == 100);
    REQUIRE(child2.parent_id == 100);
    
    // Child nodes do NOT contain duplicated full basis matrices.
    // We enforce a strict sizeof limit to aggressively block accidental architecture leaks.
    // Step 11.2 expanded it slightly for FT updates, but it strictly remains <= 64 bytes.
    REQUIRE(sizeof(gpu::HBFNode) <= 64); 
    
    // Pointers are completely distinct, ensuring node-local bound isolation.
    REQUIRE(child1.deltas != nullptr);
    REQUIRE(child2.deltas != nullptr);
    REQUIRE(child1.deltas != child2.deltas);
    REQUIRE(child1.deltas != root.deltas);

    // 6. Native CUDA Kernel Proof
    int* d_flags = static_cast<int*>(arena.allocate(1 * sizeof(int)));
    int h_init_flags[1] = {0};
    REQUIRE(cudaMemcpy(d_flags, h_init_flags, sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    
    inspect_hbf_node_kernel<<<1, 1>>>(child1, d_flags);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    
    int h_flags[1] = {0};
    REQUIRE(cudaMemcpy(h_flags, d_flags, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    
    // Target struct bytes were flawlessly decoded and validated inside the GPU execution domain.
    REQUIRE(h_flags[0] == 1);
    
    arena.free(d_flags);

    // 7. Explicit Lifecycle Reclamation Proof
    hbf_manager.free_all_nodes();
    // Validates that zero memory leaked and all pointers successfully returned to the VRAMArena broker.
    REQUIRE(arena.occupancy_percentage() == 0);
}

TEST_CASE("HBFNode - Basis State Inheritance (Step 11.2)", "[gpu][hbf][inherit]") {
    // 1. Setup VRAMArena and DeviceModel (mocked manually for test)
    gpu::VRAMArena arena(1024 * 1024);
    
    // Mock 3 variables
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
    
    // 2. Build a 3-level chain: Root -> Parent -> Child
    // Root modifies var 0
    std::vector<gpu::BoundDelta> root_deltas = {{0, 5.0, 95.0}};
    std::vector<gpu::FTUpdateHost> root_fts = {{0, 0, {1.0, 2.0}, {0, 1}}};
    manager.create_node(10, 10, root_deltas, root_fts); // Root ID = 10, Parent ID = 10
    
    // Parent modifies var 1 and var 0 (tightening var 0 from Root)
    std::vector<gpu::BoundDelta> parent_deltas = {{1, 10.0, 90.0}, {0, 15.0, 85.0}};
    manager.create_node(11, 10, parent_deltas, {}); // Parent ID = 11, Parent ID = 10
    
    // Child modifies var 2 and var 1 (tightening var 1 from Parent)
    std::vector<gpu::BoundDelta> child_deltas = {{2, 20.0, 80.0}, {1, 30.0, 70.0}};
    manager.create_node(12, 11, child_deltas, {}); // Child ID = 12, Parent ID = 11
    
    // 3. Allocate working state
    gpu::WorkingBasisState wstate = manager.allocate_working_state(num_cols);
    
    // 4. Execute entirely GPU-resident inheritance traversal
    manager.inherit_basis(12, d_model, wstate);
    
    // 5. Verify cumulative bound modifications and original bound immutability
    std::vector<Float> h_working_lb(num_cols);
    std::vector<Float> h_working_ub(num_cols);
    
    cudaMemcpy(h_working_lb.data(), wstate.lb, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_working_ub.data(), wstate.ub, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);
    
    // Original bounds were [0, 100]
    // Root -> var 0: [5, 95]
    // Parent -> var 1: [10, 90], var 0: [15, 85]
    // Child -> var 2: [20, 80], var 1: [30, 70]
    // Expected Cumulative:
    // var 0: [15, 85]
    // var 1: [30, 70]
    // var 2: [20, 80]
    
    REQUIRE(h_working_lb[0] == 15.0);
    REQUIRE(h_working_ub[0] == 85.0);
    
    REQUIRE(h_working_lb[1] == 30.0);
    REQUIRE(h_working_ub[1] == 70.0);
    
    REQUIRE(h_working_lb[2] == 20.0);
    REQUIRE(h_working_ub[2] == 80.0);
    
    // Verify original bounds remained pristine and immutable
    std::vector<Float> h_pristine_lb(num_cols);
    cudaMemcpy(h_pristine_lb.data(), d_model.lb, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);
    REQUIRE(h_pristine_lb[0] == 0.0);
    REQUIRE(h_pristine_lb[1] == 0.0);
    REQUIRE(h_pristine_lb[2] == 0.0);
    
    // 6. Test Error Handling (Malformed Chain / Depth Exceeded)
    // Create a circular chain: 13 -> 14 -> 13
    manager.create_node(13, 14, {}, {});
    manager.create_node(14, 13, {}, {});
    
    REQUIRE_THROWS_AS(manager.inherit_basis(14, d_model, wstate), std::runtime_error);
    
    // 7. Test Explicit Working State Reclamation
    manager.free_working_state(wstate);
    
    // Clean up mock DeviceModel bounds
    arena.free(d_model.lb);
    arena.free(d_model.ub);
    
    manager.free_all_nodes();
    REQUIRE(arena.occupancy_percentage() == 0);
}

