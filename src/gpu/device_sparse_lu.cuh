/**
 * @file device_sparse_lu.cuh
 * @brief Device-resident representation of the Sparse LU root factorization
 *
 * This component provides a device-resident struct (DeviceSparseLU) that
 * mirrors the established CPU SparseLUFactorization. It owns pointers to
 * the arrays uploaded from the CPU for the single root basis factorization.
 *
 * [ENGINEERING DECISION] 
 * We encapsulate the device arrays in a new struct, DeviceSparseLU, 
 * managed by a host class, DeviceSparseLUManager.
 * - Why? It prevents polluting the immutable DeviceModel with solve-specific 
 *   factor state, and avoids overcomplicating HBFManager which manages 
 *   incremental FT updates.
 * - This creates a clear boundary: DeviceSparseLU is the initial factorization,
 *   while HBF WorkingBasisState stores the incremental updates.
 */

#pragma once

#include "sankhya/types.hpp"
#include "numerics/sparse_lu.hpp"
#include "gpu/vram_arena.cuh"
#include <vector>

namespace sankhya {
namespace gpu {

/**
 * @brief Device-side representation of the root Sparse LU factorization.
 * 
 * Preserves the exact semantics of CPU SparseLUFactorization:
 * - L is CSC format, unit diagonal implicitly 1.
 * - U is CSR format, including explicit diagonal.
 * - P and Q are row and column permutations (and their inverses).
 */
struct DeviceSparseLU {
    Index m;
    
    // L factor (CSC)
    Float* L_vals;
    Index* L_rows;
    Index* L_col_ptrs; // size m + 1
    
    // U factor (CSR)
    Float* U_vals;
    Index* U_cols;
    Index* U_row_ptrs; // size m + 1
    
    // Permutations
    Index* perm_row;
    Index* perm_col;
    Index* inv_perm_row;
    Index* inv_perm_col;

    // Array sizes stored for validation
    Index L_nnz;
    Index U_nnz;
};

/**
 * @brief Host-side manager responsible for allocating and uploading 
 *        the Sparse LU root factorization to the GPU via VRAMArena.
 */
class DeviceSparseLUManager {
public:
    DeviceSparseLUManager(VRAMArena& arena);
    ~DeviceSparseLUManager();

    /**
     * @brief Uploads the given CPU factorization to the device.
     */
    void upload(const numerics::SparseLUFactorization& cpu_lu);

    /**
     * @brief Returns the device-resident struct for use in kernels.
     */
    DeviceSparseLU get_device_struct() const;

    /**
     * @brief Frees all tracked allocations from the arena.
     */
    void free_all();

private:
    VRAMArena& arena_;
    DeviceSparseLU d_struct_;
    std::vector<void*> tracked_allocations_;
    bool is_uploaded_ = false;

    template <typename T>
    T* allocate_and_upload(const std::vector<T>& host_vec);
};

} // namespace gpu
} // namespace sankhya

