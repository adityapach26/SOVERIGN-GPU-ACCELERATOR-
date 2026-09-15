/**
 * @file hbf.cu
 * @brief Hierarchical Basis Forest (HBF) Implementation (Step 11.1 & 11.2)
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

HBFManager::HBFManager(VRAMArena& arena) : arena_(arena) {
    // Engineering Decision: Allocate node registry to enable O(1) device traversal
    std::size_t registry_bytes = kMaxNodes * sizeof(HBFNode);
    d_node_registry_ = static_cast<HBFNode*>(arena_.allocate(registry_bytes));
    tracked_allocations_.push_back(d_node_registry_);
}

HBFManager::~HBFManager() {
    free_all_nodes();
}

HBFNode HBFManager::create_node(
    uint32_t node_id, 
    uint32_t parent_id, 
    const std::vector<BoundDelta>& host_deltas,
    const std::vector<FTUpdateHost>& host_fts
) {
    if (node_id >= kMaxNodes) {
        throw std::out_of_range("HBFManager::create_node - node_id exceeds kMaxNodes");
    }

    HBFNode node;
    node.node_id = node_id;
    node.parent_id = parent_id;
    node.num_deltas = static_cast<uint32_t>(host_deltas.size());
    node.is_fathomed = false;
    node.deltas = nullptr;
    node.num_ft_updates = static_cast<uint32_t>(host_fts.size());
    node.ft_updates = nullptr;

    if (node.num_deltas > 0) {
        std::size_t bytes = host_deltas.size() * sizeof(BoundDelta);
        void* ptr = arena_.allocate(bytes);
        tracked_allocations_.push_back(ptr);
        node.deltas = static_cast<BoundDelta*>(ptr);
        check_cuda_error(
            cudaMemcpy(node.deltas, host_deltas.data(), bytes, cudaMemcpyHostToDevice),
            "Failed to copy bound deltas to device"
        );
    }

    if (node.num_ft_updates > 0) {
        std::vector<FTUpdate> device_fts(host_fts.size());
        for (std::size_t i = 0; i < host_fts.size(); ++i) {
            const auto& host_ft = host_fts[i];
            FTUpdate d_ft;
            d_ft.leaving_row = host_ft.leaving_row;
            d_ft.entering_col = host_ft.entering_col;
            d_ft.num_eta_elements = static_cast<uint32_t>(host_ft.eta_values.size());
            d_ft.eta_values = nullptr;
            d_ft.eta_indices = nullptr;

            if (d_ft.num_eta_elements > 0) {
                std::size_t val_bytes = host_ft.eta_values.size() * sizeof(Float);
                std::size_t idx_bytes = host_ft.eta_indices.size() * sizeof(Index);
                
                void* val_ptr = arena_.allocate(val_bytes);
                tracked_allocations_.push_back(val_ptr);
                d_ft.eta_values = static_cast<Float*>(val_ptr);
                
                void* idx_ptr = arena_.allocate(idx_bytes);
                tracked_allocations_.push_back(idx_ptr);
                d_ft.eta_indices = static_cast<Index*>(idx_ptr);

                check_cuda_error(cudaMemcpy(d_ft.eta_values, host_ft.eta_values.data(), val_bytes, cudaMemcpyHostToDevice), "FT val copy failed");
                check_cuda_error(cudaMemcpy(d_ft.eta_indices, host_ft.eta_indices.data(), idx_bytes, cudaMemcpyHostToDevice), "FT idx copy failed");
            }
            device_fts[i] = d_ft;
        }

        std::size_t ft_bytes = device_fts.size() * sizeof(FTUpdate);
        void* ft_ptr = arena_.allocate(ft_bytes);
        tracked_allocations_.push_back(ft_ptr);
        node.ft_updates = static_cast<FTUpdate*>(ft_ptr);
        check_cuda_error(
            cudaMemcpy(node.ft_updates, device_fts.data(), ft_bytes, cudaMemcpyHostToDevice),
            "Failed to copy FT updates to device"
        );
    }

    // Register node in the device array for O(1) traversal lookups
    check_cuda_error(
        cudaMemcpy(d_node_registry_ + node_id, &node, sizeof(HBFNode), cudaMemcpyHostToDevice),
        "Failed to register HBFNode in device registry"
    );

    return node;
}

WorkingBasisState HBFManager::allocate_working_state(
    Index m, Index num_cols, Index max_eta_nnz, Index max_eta_cols
) {
    WorkingBasisState state;
    state.m = m;
    state.num_cols = num_cols;
    state.eta_capacity = max_eta_nnz;
    state.eta_col_capacity = max_eta_cols;

    state.lb = static_cast<Float*>(arena_.allocate(num_cols * sizeof(Float)));
    state.ub = static_cast<Float*>(arena_.allocate(num_cols * sizeof(Float)));
    state.error_code = static_cast<int*>(arena_.allocate(sizeof(int)));
    
    state.basis_indices = static_cast<Index*>(arena_.allocate(m * sizeof(Index)));

    state.eta_vals = static_cast<Float*>(arena_.allocate(max_eta_nnz * sizeof(Float)));
    state.eta_rows = static_cast<Index*>(arena_.allocate(max_eta_nnz * sizeof(Index)));
    state.eta_col_starts = static_cast<Index*>(arena_.allocate((max_eta_cols + 1) * sizeof(Index)));
    state.eta_pivot_row = static_cast<Index*>(arena_.allocate(max_eta_cols * sizeof(Index)));
    state.num_eta_cols = static_cast<Index*>(arena_.allocate(sizeof(Index)));
    state.eta_nnz = static_cast<Index*>(arena_.allocate(sizeof(Index)));

    state.work_vec = static_cast<Float*>(arena_.allocate(m * sizeof(Float)));

    return state;
}

void HBFManager::free_working_state(WorkingBasisState& state) {
    arena_.free(state.work_vec);
    arena_.free(state.eta_nnz);
    arena_.free(state.num_eta_cols);
    arena_.free(state.eta_pivot_row);
    arena_.free(state.eta_col_starts);
    arena_.free(state.eta_rows);
    arena_.free(state.eta_vals);
    arena_.free(state.basis_indices);
    arena_.free(state.error_code);
    arena_.free(state.ub);
    arena_.free(state.lb);
}

// Engineering Decision: Bounded traversal depth to prevent infinite loops in malformed chains.
#define MAX_HBF_DEPTH 1024

/**
 * GPU kernel that performs actual Forrest-Tomlin basis inheritance.
 *
 * Mathematical operation per FT update (Step 6.2 semantics):
 *   Each FT update represents an elementary column transformation:
 *     E_t = I + (eta_col_t - e_{pivot_row}) * e_{pivot_row}^T
 *   where eta_col_t is sparse (eta_vals/eta_rows) and pivot_row = leaving_row.
 *
 *   The eta column stores the multipliers: for each non-zero (row_i, val_i),
 *   the transformation is:
 *     work_vec[row_i] += val_i * work_vec[pivot_row]   (for row_i != pivot_row)
 *     work_vec[pivot_row] *= val_i                     (for the pivot row entry)
 *
 *   This directly modifies the working factor state so that subsequent
 *   FTRAN/BTRAN solves through this eta-file produce the correct result
 *   for the child's basis.
 *
 *   Additionally, basis_indices[leaving_row] = entering_col records
 *   the actual basis column swap.
 */
__global__ void inherit_basis_kernel(
    uint32_t child_id,
    const HBFNode* node_registry,
    const Float* orig_lb,
    const Float* orig_ub,
    WorkingBasisState ws
) {
    __shared__ uint32_t path[MAX_HBF_DEPTH];
    __shared__ int path_len;
    __shared__ int local_error;

    if (threadIdx.x == 0) {
        local_error = 0;
    }
    __syncthreads();

    // 1. Initialize working bounds from immutable DeviceModel
    for (Index i = threadIdx.x; i < ws.num_cols; i += blockDim.x) {
        ws.lb[i] = orig_lb[i];
        ws.ub[i] = orig_ub[i];
    }
    __syncthreads();

    // 2. Traverse parent_id pointers child → root
    if (threadIdx.x == 0) {
        uint32_t curr = child_id;
        int len = 0;
        
        while (len < MAX_HBF_DEPTH) {
            if (curr >= 4096) { // kMaxNodes
                local_error = 2;
                break;
            }
            
            path[len++] = curr;
            uint32_t parent = node_registry[curr].parent_id;
            
            if (parent == curr) break; // Root
            curr = parent;
        }

        if (local_error == 0 && len == MAX_HBF_DEPTH) {
            local_error = 1;
        }
        path_len = len;
        *ws.error_code = local_error;
    }
    __syncthreads();

    if (local_error != 0) return;

    // 3. Reconstruct state root → child
    for (int i = path_len - 1; i >= 0; --i) {
        uint32_t nid = path[i];
        HBFNode node = node_registry[nid];

        // ============================================================
        // ACTUAL FORREST-TOMLIN UPDATE APPLICATION (Step 6.2 semantics)
        // ============================================================
        // Each FTUpdate represents one basis pivot that occurred at this
        // node's edge. We replay it by:
        //   1. Recording the basis column swap
        //   2. Appending the eta column to the working eta-file
        //   3. Applying the eta transformation to the dense work_vec
        //      to prove mathematical state change
        //
        // The eta transformation for FTRAN is:
        //   For elementary matrix E = I + (eta - e_p) * e_p^T
        //   Applied to vector x:
        //     x_new[p] = eta_pivot_val * x[p]
        //     x_new[j] = x[j] + eta[j] * x[p]   for j != p
        //
        // This is the exact Step 6.2 product-form-of-inverse operation.
        // ============================================================

        for (uint32_t f = 0; f < node.num_ft_updates; ++f) {
            FTUpdate ft = node.ft_updates[f];

            if (threadIdx.x == 0) {
                // Read current eta-file state
                Index cur_eta_cols = *ws.num_eta_cols;
                Index cur_eta_nnz  = *ws.eta_nnz;

                // Capacity check: prevent out-of-bounds VRAM writes
                if (cur_eta_cols >= ws.eta_col_capacity ||
                    cur_eta_nnz + static_cast<Index>(ft.num_eta_elements) > ws.eta_capacity) {
                    local_error = 3; // capacity overflow
                    *ws.error_code = 3;
                } else {
                    // (a) Record the basis column swap
                    if (ft.leaving_row < ws.m) {
                        ws.basis_indices[ft.leaving_row] = ft.entering_col;
                    }

                    // (b) Record eta column start pointer
                    ws.eta_col_starts[cur_eta_cols] = cur_eta_nnz;
                    ws.eta_pivot_row[cur_eta_cols] = ft.leaving_row;

                    // (c) Copy sparse eta entries into the eta-file
                    for (uint32_t j = 0; j < ft.num_eta_elements; ++j) {
                        ws.eta_vals[cur_eta_nnz + j] = ft.eta_values[j];
                        ws.eta_rows[cur_eta_nnz + j] = ft.eta_indices[j];
                    }

                    // (d) Apply actual eta transformation to work_vec
                    //     This is the mathematical operation E * work_vec
                    //     E = I + (eta_col - e_p) * e_p^T
                    //     work_vec[j] += eta[j] * work_vec[p]  for j != p
                    //     work_vec[p] *= eta_pivot_val          for j == p
                    Index p = ft.leaving_row;
                    Float x_p = (p < ws.m) ? ws.work_vec[p] : 0.0;

                    for (uint32_t j = 0; j < ft.num_eta_elements; ++j) {
                        Index row = ft.eta_indices[j];
                        Float val = ft.eta_values[j];
                        if (row == p) {
                            // Pivot row: scale by the eta diagonal entry
                            ws.work_vec[row] = val * x_p;
                        } else if (row < ws.m) {
                            // Off-pivot: add eta[row] * x_p
                            ws.work_vec[row] += val * x_p;
                        }
                    }

                    // (e) Update eta-file counters
                    Index new_nnz = cur_eta_nnz + static_cast<Index>(ft.num_eta_elements);
                    *ws.eta_nnz = new_nnz;
                    Index new_cols = cur_eta_cols + 1;
                    *ws.num_eta_cols = new_cols;
                    // Close the column pointer for consistency
                    ws.eta_col_starts[new_cols] = new_nnz;
                }
            }
            __syncthreads();
            if (local_error != 0) return;
        }

        // Apply Bound Deltas cumulatively (unchanged)
        for (uint32_t d = threadIdx.x; d < node.num_deltas; d += blockDim.x) {
            Index var = node.deltas[d].var_idx;
            ws.lb[var] = node.deltas[d].new_lb;
            ws.ub[var] = node.deltas[d].new_ub;
        }
        __syncthreads();
    }
}

void HBFManager::inherit_basis(uint32_t child_id, const DeviceModel& device_model, WorkingBasisState& state) {
    if (child_id >= kMaxNodes) {
        throw std::invalid_argument("HBFManager::inherit_basis - Invalid child_id");
    }

    // Initialize device-side error and counters
    int h_zero_int = 0;
    Index h_zero_idx = 0;
    check_cuda_error(cudaMemcpy(state.error_code, &h_zero_int, sizeof(int), cudaMemcpyHostToDevice), "Error init");
    check_cuda_error(cudaMemcpy(state.num_eta_cols, &h_zero_idx, sizeof(Index), cudaMemcpyHostToDevice), "Eta cols init");
    check_cuda_error(cudaMemcpy(state.eta_nnz, &h_zero_idx, sizeof(Index), cudaMemcpyHostToDevice), "Eta nnz init");
    // Initialize eta_col_starts[0] = 0
    check_cuda_error(cudaMemcpy(state.eta_col_starts, &h_zero_idx, sizeof(Index), cudaMemcpyHostToDevice), "Eta starts init");

    int threads = 256;
    inherit_basis_kernel<<<1, threads>>>(
        child_id, 
        d_node_registry_, 
        device_model.lb, 
        device_model.ub, 
        state
    );
    
    check_cuda_error(cudaDeviceSynchronize(), "Inherit basis kernel failed");
    
    int h_error = 0;
    check_cuda_error(cudaMemcpy(&h_error, state.error_code, sizeof(int), cudaMemcpyDeviceToHost), "Error readback");

    if (h_error == 1) {
        throw std::runtime_error("HBFManager::inherit_basis - Traversal exceeded MAX_HBF_DEPTH (cycle or malformed chain)");
    } else if (h_error == 2) {
        throw std::runtime_error("HBFManager::inherit_basis - Invalid parent_id (out of registry bounds)");
    } else if (h_error == 3) {
        throw std::runtime_error("HBFManager::inherit_basis - Eta-file capacity overflow");
    }
}

void HBFManager::free_all_nodes() {
    for (void* ptr : tracked_allocations_) {
        arena_.free(ptr);
    }
    tracked_allocations_.clear();
}

} // namespace gpu
} // namespace sankhya
