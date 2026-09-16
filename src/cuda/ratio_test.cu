#include "ratio_test.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace sankhya {
namespace cuda {

// ============================================================================
// PASS 1: HARRIS EXPANDED STEP-LENGTH BOUND
// ============================================================================

__global__ void harris_pass1_block_kernel(
    const Float* __restrict__ d,
    const Float* __restrict__ x,
    const Index* __restrict__ basic_indices,
    Index m,
    Float* __restrict__ block_deltas
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    Float my_delta = math::kInfinity;

    // 1. Parallel thread evaluation
    for (int i = tid; i < m; i += blockDim.x * gridDim.x) {
        Float di = d[i];
        if (di > math::kDefaultPivotTol) {
            Float xb = x[basic_indices[i]];
            Float relaxed_ratio = (xb + math::kDefaultFeasibilityTol) / di;
            if (relaxed_ratio < my_delta) {
                my_delta = relaxed_ratio;
            }
        }
    }

    // 2. Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        Float other_delta = __shfl_down_sync(mask, my_delta, offset);
        if (other_delta < my_delta) {
            my_delta = other_delta;
        }
    }

    // 3. Block-level reduction in shared memory
    extern __shared__ double s_data_p1[];
    double* s_deltas = s_data_p1;
    
    if (threadIdx.x % 32 == 0) {
        s_deltas[threadIdx.x / 32] = my_delta;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = blockDim.x / 32;
        Float block_min = math::kInfinity;
        for (int i = 0; i < num_warps; ++i) {
            if (s_deltas[i] < block_min) {
                block_min = s_deltas[i];
            }
        }
        block_deltas[blockIdx.x] = block_min;
    }
}

__global__ void harris_pass1_global_kernel(
    const Float* __restrict__ block_deltas,
    int num_blocks,
    Float* __restrict__ global_delta
) {
    Float my_delta = math::kInfinity;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        if (block_deltas[i] < my_delta) {
            my_delta = block_deltas[i];
        }
    }

    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        Float other_delta = __shfl_down_sync(mask, my_delta, offset);
        if (other_delta < my_delta) {
            my_delta = other_delta;
        }
    }

    extern __shared__ double s_data_p1g[];
    if (threadIdx.x % 32 == 0) {
        s_data_p1g[threadIdx.x / 32] = my_delta;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = blockDim.x / 32;
        Float final_min = math::kInfinity;
        for (int i = 0; i < num_warps; ++i) {
            if (s_data_p1g[i] < final_min) {
                final_min = s_data_p1g[i];
            }
        }
        *global_delta = final_min;
    }
}

// ============================================================================
// PASS 2: LARGEST PIVOT MAGNITUDE
// ============================================================================

__global__ void harris_pass2_block_kernel(
    const Float* __restrict__ d,
    const Float* __restrict__ x,
    const Index* __restrict__ basic_indices,
    Index m,
    const Float* __restrict__ global_delta,
    Float* __restrict__ block_best_d,
    Index* __restrict__ block_best_idx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    Float Delta = *global_delta;
    Float my_best_d = -1.0;
    Index my_best_idx = 0x7FFFFFFF;

    // 1. Parallel thread evaluation
    for (int i = tid; i < m; i += blockDim.x * gridDim.x) {
        Float di = d[i];
        if (di > math::kDefaultPivotTol) {
            Float xb = x[basic_indices[i]];
            Float ratio = xb / di;
            if (ratio <= Delta) {
                Float abs_d = std::abs(di);
                // Deterministic Tie Rule: smaller row index `i` wins.
                if (abs_d > my_best_d || (abs_d == my_best_d && i < my_best_idx)) {
                    my_best_d = abs_d;
                    my_best_idx = i;
                }
            }
        }
    }

    // 2. Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        Float other_d = __shfl_down_sync(mask, my_best_d, offset);
        Index other_idx = __shfl_down_sync(mask, my_best_idx, offset);
        
        if (other_d > my_best_d || (other_d == my_best_d && other_idx < my_best_idx)) {
            my_best_d = other_d;
            my_best_idx = other_idx;
        }
    }

    // 3. Block-level reduction in shared memory
    extern __shared__ double s_data_p2[];
    double* s_best_d = s_data_p2;
    Index* s_best_idx = (Index*)(s_data_p2 + (blockDim.x / 32));
    
    if (threadIdx.x % 32 == 0) {
        s_best_d[threadIdx.x / 32] = my_best_d;
        s_best_idx[threadIdx.x / 32] = my_best_idx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = blockDim.x / 32;
        Float block_max_d = -1.0;
        Index block_min_idx = 0x7FFFFFFF;
        for (int i = 0; i < num_warps; ++i) {
            Float bd = s_best_d[i];
            Index bi = s_best_idx[i];
            if (bd > block_max_d || (bd == block_max_d && bi < block_min_idx)) {
                block_max_d = bd;
                block_min_idx = bi;
            }
        }
        block_best_d[blockIdx.x] = block_max_d;
        block_best_idx[blockIdx.x] = block_min_idx;
    }
}

__global__ void harris_pass2_global_kernel(
    const Float* __restrict__ block_best_d,
    const Index* __restrict__ block_best_idx,
    int num_blocks,
    Index* __restrict__ final_idx
) {
    Float my_best_d = -1.0;
    Index my_best_idx = 0x7FFFFFFF;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        Float bd = block_best_d[i];
        Index bi = block_best_idx[i];
        if (bd > my_best_d || (bd == my_best_d && bi < my_best_idx)) {
            my_best_d = bd;
            my_best_idx = bi;
        }
    }

    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        Float other_d = __shfl_down_sync(mask, my_best_d, offset);
        Index other_idx = __shfl_down_sync(mask, my_best_idx, offset);
        
        if (other_d > my_best_d || (other_d == my_best_d && other_idx < my_best_idx)) {
            my_best_d = other_d;
            my_best_idx = other_idx;
        }
    }

    extern __shared__ double s_data_p2g[];
    double* s_best_d = s_data_p2g;
    Index* s_best_idx = (Index*)(s_data_p2g + (blockDim.x / 32));

    if (threadIdx.x % 32 == 0) {
        s_best_d[threadIdx.x / 32] = my_best_d;
        s_best_idx[threadIdx.x / 32] = my_best_idx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = blockDim.x / 32;
        Float final_max_d = -1.0;
        Index final_min_idx = 0x7FFFFFFF;
        for (int i = 0; i < num_warps; ++i) {
            Float bd = s_best_d[i];
            Index bi = s_best_idx[i];
            if (bd > final_max_d || (bd == final_max_d && bi < final_min_idx)) {
                final_max_d = bd;
                final_min_idx = bi;
            }
        }
        *final_idx = (final_max_d >= 0.0) ? final_min_idx : -1;
    }
}

// ============================================================================
// HOST LAUNCHER
// ============================================================================

/**
 * Engineering Decision (Multi-stage GPU Reduction without Atomics):
 * To strictly follow the "Do not copy 12.1 atomic design" directive while 
 * keeping all reduction stages on the GPU, I used a four-kernel pipeline:
 * 1. Pass 1 Block: Finds block-local min Delta.
 * 2. Pass 1 Global: Single block kernel reduces to global Delta.
 * 3. Pass 2 Block: Uses global Delta to find block-local best row.
 * 4. Pass 2 Global: Single block kernel reduces to final leaving row index.
 * This guarantees strict double-precision evaluation and deterministic
 * tie-breaking entirely on the GPU without any atomics.
 */
void launch_harris_ratio_test(
    const Float* d_d,
    const Float* d_x,
    const Index* d_basic_indices,
    Index m,
    Float* d_block_deltas,
    Float* d_global_delta,
    Float* d_block_best_d,
    Index* d_block_best_idx,
    Index* d_final_idx,
    int blocks,
    int threads
) {
    int warps_per_block = threads / 32;
    
    // Pass 1
    size_t s_mem_p1_block = warps_per_block * sizeof(double);
    harris_pass1_block_kernel<<<blocks, threads, s_mem_p1_block>>>(
        d_d, d_x, d_basic_indices, m, d_block_deltas
    );

    int g_threads = (blocks < 256) ? ((blocks + 31)/32)*32 : 256;
    if (g_threads == 0) g_threads = 32;
    size_t s_mem_p1_global = (g_threads / 32) * sizeof(double);
    
    harris_pass1_global_kernel<<<1, g_threads, s_mem_p1_global>>>(
        d_block_deltas, blocks, d_global_delta
    );

    // Pass 2
    size_t s_mem_p2_block = warps_per_block * (sizeof(double) + sizeof(Index));
    harris_pass2_block_kernel<<<blocks, threads, s_mem_p2_block>>>(
        d_d, d_x, d_basic_indices, m, d_global_delta, d_block_best_d, d_block_best_idx
    );

    size_t s_mem_p2_global = (g_threads / 32) * (sizeof(double) + sizeof(Index));
    harris_pass2_global_kernel<<<1, g_threads, s_mem_p2_global>>>(
        d_block_best_d, d_block_best_idx, blocks, d_final_idx
    );
}

} // namespace cuda
} // namespace sankhya
