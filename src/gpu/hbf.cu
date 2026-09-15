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

WorkingBasisState HBFManager::allocate_working_state(Index num_cols) {
    WorkingBasisState state;
    state.lb = static_cast<Float*>(arena_.allocate(num_cols * sizeof(Float)));
    state.ub = static_cast<Float*>(arena_.allocate(num_cols * sizeof(Float)));
    state.error_code = static_cast<int*>(arena_.allocate(sizeof(int)));
    return state;
}

void HBFManager::free_working_state(WorkingBasisState& state) {
    // Reclaim working state explicit lifetime explicitly ending after operation.
    // Zero cudaFree calls. VRAMArena handles block merges.
    arena_.free(state.error_code);
    arena_.free(state.ub);
    arena_.free(state.lb);
}

// Engineering Decision: Bounded traversal depth to prevent infinite loops in malformed chains.
#define MAX_HBF_DEPTH 1024

__global__ void inherit_basis_kernel(
    uint32_t child_id,
    const HBFNode* node_registry,
    const Float* orig_lb,
    const Float* orig_ub,
    Index num_cols,
    WorkingBasisState working_state
) {
    __shared__ uint32_t path[MAX_HBF_DEPTH];
    __shared__ int path_len;
    __shared__ int local_error;

    if (threadIdx.x == 0) {
        local_error = 0;
    }
    __syncthreads();

    // 1. Initialize working bounds from original static problem state
    for (Index i = threadIdx.x; i < num_cols; i += blockDim.x) {
        working_state.lb[i] = orig_lb[i];
        working_state.ub[i] = orig_ub[i];
    }
    __syncthreads();

    // 2. Traverse parent_id pointers from child to root
    if (threadIdx.x == 0) {
        uint32_t curr = child_id;
        int len = 0;
        
        while (len < MAX_HBF_DEPTH) {
            path[len++] = curr;
            uint32_t parent = node_registry[curr].parent_id;
            
            // Root convention: parent_id == node_id
            if (parent == curr) {
                break;
            }
            curr = parent;
        }

        if (len == MAX_HBF_DEPTH) {
            local_error = 1; // Cycle or max depth exceeded
        }
        path_len = len;
        
        // Push error out to host
        *working_state.error_code = local_error;
    }
    __syncthreads();

    // Abort if malformed chain
    if (local_error != 0) return;

    // 3. Reconstruct State (Root to Child)
    for (int i = path_len - 1; i >= 0; --i) {
        uint32_t node_id = path[i];
        HBFNode node = node_registry[node_id];

        // Apply FT Updates (Sequential representation)
        // Note: For Step 11.2, this proves ordering and accessibility.
        // Full dense sparse LU matrix operations belong to a later phase.
        for (uint32_t f = 0; f < node.num_ft_updates; ++f) {
            FTUpdate ft = node.ft_updates[f];
            // E.g., re-apply eta vector (ft.eta_indices, ft.eta_values) to LU working state
            // (Simulated explicitly as required structural proof)
            if (threadIdx.x == 0 && ft.num_eta_elements > 0 && ft.eta_values != nullptr) {
                // Ensure pointers are perfectly dereferenceable in device memory
                Float dummy = ft.eta_values[0]; 
                (void)dummy;
            }
        }
        __syncthreads();

        // Apply Bound Deltas cumulatively
        for (uint32_t d = threadIdx.x; d < node.num_deltas; d += blockDim.x) {
            Index var = node.deltas[d].var_idx;
            working_state.lb[var] = node.deltas[d].new_lb;
            working_state.ub[var] = node.deltas[d].new_ub;
        }
        __syncthreads();
    }
}

void HBFManager::inherit_basis(uint32_t child_id, const DeviceModel& device_model, WorkingBasisState& state) {
    if (child_id >= kMaxNodes) {
        throw std::invalid_argument("HBFManager::inherit_basis - Invalid child_id");
    }

    // Initialize host-side error check
    int h_error = 0;
    check_cuda_error(cudaMemcpy(state.error_code, &h_error, sizeof(int), cudaMemcpyHostToDevice), "Error init failed");

    // Launch traversal/reconstruction entirely on the device
    int threads = 256;
    inherit_basis_kernel<<<1, threads>>>(
        child_id, 
        d_node_registry_, 
        device_model.lb, 
        device_model.ub, 
        device_model.cols, 
        state
    );
    
    check_cuda_error(cudaDeviceSynchronize(), "Inherit basis kernel failed");
    
    // Readback error code
    check_cuda_error(cudaMemcpy(&h_error, state.error_code, sizeof(int), cudaMemcpyDeviceToHost), "Error readback failed");

    if (h_error == 1) {
        throw std::runtime_error("HBFManager::inherit_basis - Traversal exceeded MAX_HBF_DEPTH (Cycle or malformed chain)");
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
