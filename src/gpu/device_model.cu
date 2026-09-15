/**
 * @file device_model.cu
 * @brief Immutable Device-Resident Problem State Implementation (Step 10.1)
 */

#include "device_model.cuh"
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

DeviceModel upload_to_device(const core::Model& host_model, VRAMArena& arena) {
    DeviceModel d_model;
    
    d_model.rows = host_model.A.rows;
    d_model.cols = host_model.A.cols;
    d_model.nnz = static_cast<Index>(host_model.A.values.size());
    
    if (d_model.cols == 0 && d_model.rows == 0) {
        return d_model;
    }

    // Calculate strict byte sizes natively
    const std::size_t size_col_ptrs = static_cast<std::size_t>(d_model.cols + 1) * sizeof(Index);
    const std::size_t size_row_indices = static_cast<std::size_t>(d_model.nnz) * sizeof(Index);
    const std::size_t size_values = static_cast<std::size_t>(d_model.nnz) * sizeof(Float);
    const std::size_t size_obj = static_cast<std::size_t>(d_model.cols) * sizeof(Float);
    const std::size_t size_lb = static_cast<std::size_t>(d_model.cols) * sizeof(Float);
    const std::size_t size_ub = static_cast<std::size_t>(d_model.cols) * sizeof(Float);

    try {
        // Exclusively use VRAMArena for device allocations (ZERO cudaMalloc calls here).
        // Each requested chunk strictly represents a pointer securely inside the persistent arena buffer.
        d_model.col_ptrs = static_cast<Index*>(arena.allocate(size_col_ptrs));
        
        if (d_model.nnz > 0) {
            d_model.row_indices = static_cast<Index*>(arena.allocate(size_row_indices));
            d_model.values = static_cast<Float*>(arena.allocate(size_values));
        }

        if (d_model.cols > 0) {
            d_model.obj = static_cast<Float*>(arena.allocate(size_obj));
            d_model.lb = static_cast<Float*>(arena.allocate(size_lb));
            d_model.ub = static_cast<Float*>(arena.allocate(size_ub));
        }

        // Perform initialization-only device transfers (O(1) host-device upload stage).
        // Actual values are perfectly and flawlessly copied via explicit synchronization.
        check_cuda_error(
            cudaMemcpy(d_model.col_ptrs, host_model.A.col_ptrs.data(), size_col_ptrs, cudaMemcpyHostToDevice),
            "Failed to upload col_ptrs"
        );
        
        if (d_model.nnz > 0) {
            check_cuda_error(
                cudaMemcpy(d_model.row_indices, host_model.A.row_indices.data(), size_row_indices, cudaMemcpyHostToDevice),
                "Failed to upload row_indices"
            );
            check_cuda_error(
                cudaMemcpy(d_model.values, host_model.A.values.data(), size_values, cudaMemcpyHostToDevice),
                "Failed to upload values"
            );
        }

        if (d_model.cols > 0) {
            check_cuda_error(
                cudaMemcpy(d_model.obj, host_model.obj.data(), size_obj, cudaMemcpyHostToDevice),
                "Failed to upload obj"
            );
            check_cuda_error(
                cudaMemcpy(d_model.lb, host_model.lb.data(), size_lb, cudaMemcpyHostToDevice),
                "Failed to upload lb"
            );
            check_cuda_error(
                cudaMemcpy(d_model.ub, host_model.ub.data(), size_ub, cudaMemcpyHostToDevice),
                "Failed to upload ub"
            );
        }
        
    } catch (...) {
        // Strict safety rollback: Release allocated pointers exclusively via the arena allocator.
        // DeviceModel does not own these blocks independently.
        arena.free(d_model.col_ptrs);
        arena.free(d_model.row_indices);
        arena.free(d_model.values);
        arena.free(d_model.obj);
        arena.free(d_model.lb);
        arena.free(d_model.ub);
        throw; // Escalate failure cleanly
    }

    return d_model;
}

} // namespace gpu
} // namespace sankhya

