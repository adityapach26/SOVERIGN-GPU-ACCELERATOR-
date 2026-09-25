#include "cuda/gomory.cuh"
#include "gpu/device_ftran.cuh"
#include <stdexcept>

namespace sankhya {
namespace gpu {

__global__ void gomory_cut_kernel(
    DeviceModel model,
    DeviceSparseLU lu,
    WorkingBasisState ws,
    Index basic_row,
    Float basic_value,
    Float* d_cut_coeffs,
    Float* d_cut_rhs
) {
    // 1. Initialize BTRAN RHS (e_i) in the working state's pi vector
    for (Index k = threadIdx.x; k < ws.m; k += blockDim.x) {
        ws.d_pi[k] = (k == basic_row) ? 1.0 : 0.0;
    }
    __syncthreads();

    // 2. Perform BTRAN cooperatively: solves B^T pi = e_i natively on device
    device_btran(lu, ws, ws.d_pi);
    __syncthreads();

    // 3. Compute fractional coefficients for all columns
    // tableau row: ā_ij = pi^T A_j
    for (Index j = threadIdx.x; j < model.cols; j += blockDim.x) {
        Float a_ij = 0.0;
        Index start = model.col_ptrs[j];
        Index end = model.col_ptrs[j+1];
        
        for (Index p = start; p < end; ++p) {
            a_ij += ws.d_pi[model.row_indices[p]] * model.values[p];
        }
        
        // Generate Gomory fractional coefficient mathematically: f_j = a_ij - floor(a_ij)
        Float f_j = a_ij - floor(a_ij);
        
        // The CutPool represents cuts as: a^T x <= rhs.
        // The mathematical GMI cut is: Σ f_j x_j >= f_0.
        // Negating both sides yields: Σ (-f_j) x_j <= -f_0.
        d_cut_coeffs[j] = -f_j;
    }

    // 4. Compute RHS fractional part
    if (threadIdx.x == 0) {
        Float f_0 = basic_value - floor(basic_value);
        *d_cut_rhs = -f_0;
    }
}

cuts::Cut generate_gomory_cut(
    const DeviceModel& model,
    const DeviceSparseLU& lu,
    WorkingBasisState& ws,
    Index basic_row,
    Float basic_value,
    VRAMArena& arena
) {
    std::size_t coeffs_size = model.cols * sizeof(Float);
    
    // Transient allocation for the cut result payload (not a full basis copy)
    Float* d_cut_coeffs = static_cast<Float*>(arena.allocate(coeffs_size));
    Float* d_cut_rhs = static_cast<Float*>(arena.allocate(sizeof(Float)));

    if (!d_cut_coeffs || !d_cut_rhs) {
        if (d_cut_coeffs) arena.free(d_cut_coeffs);
        if (d_cut_rhs) arena.free(d_cut_rhs);
        throw std::runtime_error("Failed to allocate transient Gomory cut arrays on device");
    }

    // Launch warp-parallel kernel
    // Thread block cooperates to perform BTRAN and parallel coefficient extraction
    gomory_cut_kernel<<<1, 256>>>(model, lu, ws, basic_row, basic_value, d_cut_coeffs, d_cut_rhs);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        arena.free(d_cut_coeffs);
        arena.free(d_cut_rhs);
        throw std::runtime_error("Failed to launch Gomory cut kernel");
    }

    // Wait for the single block to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        arena.free(d_cut_coeffs);
        arena.free(d_cut_rhs);
        throw std::runtime_error("Gomory cut kernel execution failed");
    }

    // Hand off cut to the CPU CutPool representation
    cuts::Cut cut;
    cut.coefficients.resize(model.cols);
    
    err = cudaMemcpy(cut.coefficients.data(), d_cut_coeffs, coeffs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        arena.free(d_cut_coeffs);
        arena.free(d_cut_rhs);
        throw std::runtime_error("Failed to copy cut coefficients to host");
    }
    
    err = cudaMemcpy(&cut.rhs, d_cut_rhs, sizeof(Float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        arena.free(d_cut_coeffs);
        arena.free(d_cut_rhs);
        throw std::runtime_error("Failed to copy cut rhs to host");
    }

    // Clean up transient device buffers immediately
    arena.free(d_cut_rhs);
    arena.free(d_cut_coeffs);

    return cut;
}

} // namespace gpu
} // namespace sankhya
