/**
 * @file hbf.cuh
 * @brief Hierarchical Basis Forest (HBF) Node Representation (Step 11.1)
 */

#pragma once

#include "sankhya/types.hpp"
#include "gpu/vram_arena.cuh"
#include "gpu/device_model.cuh"
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
 * @brief Represents a Forrest-Tomlin update along an HBF edge.
 * 
 * Lightweight representation: stores only the leaving row, entering column, 
 * and the sparse eta vector, explicitly avoiding dense matrix duplication.
 */
struct FTUpdate {
    Index leaving_row;
    Index entering_col;
    Float* eta_values;
    Index* eta_indices;
    uint32_t num_eta_elements;
};

/**
 * @brief Host-side definition of FTUpdate to facilitate O(1) device transfer.
 */
struct FTUpdateHost {
    Index leaving_row;
    Index entering_col;
    std::vector<Float> eta_values;
    std::vector<Index> eta_indices;
};

/**
 * @brief Represents a lightweight node in the Hierarchical Basis Forest.
 * 
 * Strict Architectural Invariant: 
 * Node contains only local bound information (deltas), edge FT updates, and hierarchy.
 * It explicitly does NOT contain a full basis matrix, factorization, or duplicate problem.
 */
struct HBFNode {
    uint32_t node_id;
    uint32_t parent_id;
    BoundDelta* deltas;
    uint32_t num_deltas;
    FTUpdate* ft_updates;
    uint32_t num_ft_updates;
    bool is_fathomed;
};

/**
 * @brief Represents temporary working basis state used during inheritance.
 * 
 * Allocated exclusively through VRAMArena. Its lifetime ends precisely 
 * when the orchestrating component completes the operation, enforcing
 * strict residency boundaries (no per-node permanent duplicate states).
 *
 * The working LU state uses an eta-file representation consistent with
 * Step 6.2 Forrest-Tomlin semantics:
 *   For a basis update B_new = B * E_i, the stored sparse eta vector
 *   represents the INVERSE transformation E_i^{-1} = I + (eta_col_i - e_p) * e_p^T
 *   FTRAN applies the inverse transformations: x <- E_k^{-1} * ... * E_2^{-1} * E_1^{-1} * x (root-to-child order)
 *
 * The basis_indices array tracks which original column of A occupies each
 * basis position, mirroring Step 6.2's basis_copy_.basic_indices.
 */
struct WorkingBasisState {
    Float* lb;
    Float* ub;
    int* error_code; // 0=ok, 1=max_depth, 2=invalid parent, 3=capacity overflow

    // Dimensions
    Index m;             // basis dimension (= number of rows/constraints)
    Index num_cols;      // total number of variables

    // Basis column tracking (mirrors basis_copy_.basic_indices from Step 6.2)
    // basis_indices[i] = column index of A in basis position i
    Index* basis_indices;

    // Eta-file: sequence of FT elementary column transformations
    // eta_vals[eta_col_starts[t] .. eta_col_starts[t+1]-1] = non-zero values
    // eta_rows[eta_col_starts[t] .. eta_col_starts[t+1]-1] = row indices
    // eta_pivot_row[t] = pivot row for eta transformation t
    Float* eta_vals;
    Index* eta_rows;
    Index* eta_col_starts;   // length = num_eta_cols + 1
    Index* eta_pivot_row;    // length = num_eta_cols
    Index* num_eta_cols;     // device scalar: current number of eta columns
    Index* eta_nnz;          // device scalar: current total non-zeros in eta file
    Index  eta_capacity;     // host-side: max non-zeros allocated
    Index  eta_col_capacity; // host-side: max eta columns allocated

    // Dense working vector for FTRAN/BTRAN operations
    Float* work_vec;
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
    // Engineering Decision: Bounded node registry to enable O(1) device traversal
    static constexpr uint32_t kMaxNodes = 4096;

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
     * @param host_fts Forrest-Tomlin updates along this edge (host-side)
     * @return HBFNode Struct containing the device-resident pointers
     */
    HBFNode create_node(
        uint32_t node_id, 
        uint32_t parent_id, 
        const std::vector<BoundDelta>& host_deltas,
        const std::vector<FTUpdateHost>& host_fts = {}
    );

    /**
     * @brief Allocates temporary working state for inheritance.
     * @param m Basis dimension (number of constraints)
     * @param num_cols Total number of variables
     * @param max_eta_nnz Maximum total non-zeros across all eta columns
     * @param max_eta_cols Maximum number of eta column transformations
     */
    WorkingBasisState allocate_working_state(
        Index m, Index num_cols, 
        Index max_eta_nnz = 10000, Index max_eta_cols = 1000
    );

    /**
     * @brief Explicitly releases working state back to the arena.
     */
    void free_working_state(WorkingBasisState& state);

    /**
     * @brief Orchestrates root-to-child inheritance entirely on the GPU.
     * 
     * Validates chain, enforces bounded traversal, and applies FT updates
     * and bound deltas to the working state sequentially.
     */
    void inherit_basis(uint32_t child_id, const DeviceModel& device_model, WorkingBasisState& state);

    /**
     * @brief Releases all node-associated blocks currently tracked by the manager back to the arena.
     */
    void free_all_nodes();

    /**
     * @brief Sets the immutable root basis from which all nodes inherit.
     */
    void set_root_basis(const std::vector<Index>& host_root_basis);

    const HBFNode* get_node_registry() const { return d_node_registry_; }
    const Index* get_root_basis() const { return d_root_basis_; }

private:
    VRAMArena& arena_;
    
    // Array of nodes resident in VRAM for direct device traversal
    HBFNode* d_node_registry_;

    // Root basis for inheritance
    Index* d_root_basis_ = nullptr;
    Index root_basis_size_ = 0;

    // Engineering Decision: Track device pointer allocations yielded by the arena
    // so they can be explicitly relinquished back to the arena during destruction.
    std::vector<void*> tracked_allocations_;
};

/**
 * @brief Device-callable inheritance function for batched execution.
 */
__device__ void inherit_basis_device(
    uint32_t child_id,
    const HBFNode* node_registry,
    const Float* orig_lb,
    const Float* orig_ub,
    const Index* orig_basis,
    WorkingBasisState ws
);

} // namespace gpu
} // namespace sankhya

