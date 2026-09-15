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

    // 3. Create one parent node dynamically mapped out of the persistent arena.
    std::vector<gpu::BoundDelta> parent_deltas = {
        {0, 0.0, 10.0}
    };
    gpu::HBFNode parent = hbf_manager.create_node(100, 0, parent_deltas);
    REQUIRE(parent.node_id == 100);
    REQUIRE(parent.parent_id == 0);
    REQUIRE(parent.num_deltas == 1);
    REQUIRE(parent.deltas != nullptr);
    REQUIRE(parent.is_fathomed == false);

    // Confirm the arena correctly recognized the reservation.
    std::size_t occupancy_after_parent = arena.occupancy_percentage();
    REQUIRE(occupancy_after_parent > 0);

    // 4. Create two child nodes linking back to the parent.
    std::vector<gpu::BoundDelta> child1_deltas = {
        {1, 5.0, 15.0} // Branching constraint delta (e.g., x1 >= 5)
    };
    std::vector<gpu::BoundDelta> child2_deltas = {
        {1, 0.0, 4.0}  // Branching constraint delta (e.g., x1 <= 4)
    };
    
    gpu::HBFNode child1 = hbf_manager.create_node(101, parent.node_id, child1_deltas);
    gpu::HBFNode child2 = hbf_manager.create_node(102, parent.node_id, child2_deltas);

    // 5. Strict Property Verifications
    // The two children correctly reference the lightweight parent ID.
    REQUIRE(child1.parent_id == 100);
    REQUIRE(child2.parent_id == 100);
    
    // Child nodes do NOT contain duplicated full basis matrices.
    // We enforce a strict sizeof limit to aggressively block accidental architecture leaks.
    REQUIRE(sizeof(gpu::HBFNode) <= 32); 
    
    // Pointers are completely distinct, ensuring node-local bound isolation.
    REQUIRE(child1.deltas != nullptr);
    REQUIRE(child2.deltas != nullptr);
    REQUIRE(child1.deltas != child2.deltas);
    REQUIRE(child1.deltas != parent.deltas);

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
