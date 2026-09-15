/**
 * @file hbf.cu
 * @brief Hierarchical Basis Forest (HBF) Implementation (Step 11.1)
 */

#include "hbf.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace sankhya {
namespace gpu {

static void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

HBFManager::HBFManager(VRAMArena& arena) : arena_(arena) {}

HBFManager::~HBFManager() {
    free_all_nodes();
}

HBFNode HBFManager::create_node(uint32_t node_id, uint32_t parent_id, const std::vector<BoundDelta>& host_deltas) {
    HBFNode node;
    node.node_id = node_id;
    node.parent_id = parent_id;
    node.num_deltas = static_cast<uint32_t>(host_deltas.size());
    node.is_fathomed = false;
    node.deltas = nullptr;

    if (node.num_deltas > 0) {
        std::size_t bytes = host_deltas.size() * sizeof(BoundDelta);
        
        // Ownership & Residency Rule: 
        // Allocations originate strictly from the persistent VRAMArena.
        // No cudaMalloc/cudaFree is executed for HBF node representation.
        void* ptr = arena_.allocate(bytes);
        
        // Engineering Decision:
        // Record the arena-offset pointer in the manager so its lifetime is managed deterministically.
        tracked_allocations_.push_back(ptr);
        
        node.deltas = static_cast<BoundDelta*>(ptr);
        
        // Initialize the device-side delta payload.
        check_cuda_error(
            cudaMemcpy(node.deltas, host_deltas.data(), bytes, cudaMemcpyHostToDevice),
            "Failed to copy bound deltas to device"
        );
    }

    // Notice: node structure returned here acts simply as a metadata view linking the lightweight parent relationship 
    // and its bound deltas. It inherently omits duplicating basis factorization.
    return node;
}

void HBFManager::free_all_nodes() {
    // Explicit reclamation rule: return blocks directly to VRAMArena.
    // Zero individual cudaFree calls.
    for (void* ptr : tracked_allocations_) {
        arena_.free(ptr);
    }
    tracked_allocations_.clear();
}

} // namespace gpu
} // namespace sankhya
