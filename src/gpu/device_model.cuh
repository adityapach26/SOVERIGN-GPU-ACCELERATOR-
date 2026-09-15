/**
 * @file device_model.cuh
 * @brief Immutable Device-Resident Problem State (Step 10.1)
 */

#pragma once

#include "sankhya/types.hpp"
#include "core/problem.hpp"
#include "gpu/vram_arena.cuh"

namespace sankhya {
namespace gpu {

/**
 * @brief Device-resident representation of the optimization problem.
 * 
 * VRAMArena must strictly outlive DeviceModel.
 * DeviceModel does not own the memory pointers; they are allocated from VRAMArena.
 * To safely enforce this non-ownership, DeviceModel acts only as a trivial view 
 * struct onto the arena allocations. No individual cudaFree calls are made.
 * It strictly treats the uploaded state as immutable initialization data.
 */
struct DeviceModel {
    // Structural metadata
    Index rows = 0;
    Index cols = 0;
    Index nnz = 0;

    // Device pointers to CSC structure
    Index* col_ptrs = nullptr;    // size: cols + 1
    Index* row_indices = nullptr; // size: nnz
    Float* values = nullptr;      // size: nnz

    // Device pointers to variables
    Float* obj = nullptr;         // size: cols
    Float* lb = nullptr;          // size: cols
    Float* ub = nullptr;          // size: cols
};

/**
 * @brief Performs an O(1) host-device upload to initialize immutable static problem state.
 *
 * Exclusively uses `VRAMArena::allocate()` for device allocations. Zero `cudaMalloc` calls.
 * Does not silently change numerical values.
 * 
 * @param host_model The fully finalized host problem model
 * @param arena The VRAMArena instance that will hold the allocations
 * @return DeviceModel The initialized device model view
 */
DeviceModel upload_to_device(const core::Model& host_model, VRAMArena& arena);

} // namespace gpu
} // namespace sankhya

