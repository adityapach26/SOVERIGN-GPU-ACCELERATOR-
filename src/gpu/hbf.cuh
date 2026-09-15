/**
 * @file hbf.cuh
 * @brief Hierarchical Basis Forest (HBF) Node Representation (Step 11.1)
 */

#pragma once

#include "sankhya/types.hpp"
#include "gpu/vram_arena.cuh"
#include <cstdint>
#include <vector>

namespace sankhya {
namespace gpu {

/**
 * @brief Represents a bound modification on a single variable relative to parent bounds.
 */
struct BoundDelta {
    Index var_idx;
    Float new_lb;
    Float new_ub;
};

/**
 * @brief Represents a lightweight node in the Hierarchical Basis Forest.
 * 
 * Strict Architectural Invariant: 
 * Node contains only local bound information (deltas) and hierarchy (parent_id).
 * It explicitly does NOT contain a full basis matrix, factorization, or duplicate problem.
 */
struct HBFNode {
    uint32_t node_id;
    uint32_t parent_id;
    BoundDelta* deltas;
    uint32_t num_deltas;
    bool is_fathomed;
};

/**
 * @brief Manages HBF nodes using the persistent VRAMArena.
 *
 * Memory Ownership & Architecture:
 * 1. VRAMArena is the exclusive, persistent owner of the underlying device memory.
 * 2. HBFManager acts solely as an allocation broker (making zero cudaMalloc/cudaFree API calls).
 * 3. HBFManager tracks device-allocated blocks assigned to nodes.
 * 4. HBFManager's `free_all()` (or its destructor) cleanly returns allocations to the VRAMArena.
 * 5. Device residency is strictly accounted for by the VRAMArena occupancy metrics.
 */
class HBFManager {
public:
    // HBFManager operates securely over an established persistent arena
    explicit HBFManager(VRAMArena& arena);
    
    // Auto-reclaims allocations back to the arena
    ~HBFManager();

    // Prevent implicit copies which would double-free tracked arena allocations
    HBFManager(const HBFManager&) = delete;
    HBFManager& operator=(const HBFManager&) = delete;

    /**
     * @brief Allocates and initializes an HBFNode representation in device memory.
     * 
     * @param node_id Unique node ID
     * @param parent_id The parent node ID
     * @param host_deltas Bound deltas local to this node (host-side)
     * @return HBFNode Struct containing the device-resident pointers
     */
    HBFNode create_node(uint32_t node_id, uint32_t parent_id, const std::vector<BoundDelta>& host_deltas);

    /**
     * @brief Releases all node-associated blocks currently tracked by the manager back to the arena.
     * 
     * If the architecture doesn't have an explicit node-by-node reclamation scheme yet, 
     * this guarantees bulk safety during Step 11.1.
     */
    void free_all_nodes();

private:
    VRAMArena& arena_;
    
    // Engineering Decision: Track device pointer allocations yielded by the arena
    // so they can be explicitly relinquished back to the arena during destruction.
    std::vector<void*> tracked_allocations_;
};

} // namespace gpu
} // namespace sankhya
