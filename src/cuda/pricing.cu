#include "pricing.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace sankhya {
namespace cuda {

/**
 * @brief CUDA kernel for Step 12.1 Warp-Synchronous Devex Pricing (Pass 1)
 * 
 * Computes Devex scores in parallel and reduces the maximum score at the block level.
 * 
 * Engineering Decision (Precision & Global Reduction):
 * The source guideline mentioned "atomicMax to global memory block."
 * However, casting double-precision Devex scores to float to pack them into a 64-bit 
 * atomicMax fundamentally changes the valid ordering of scores by discarding 29 bits of mantissa.
 * To enforce strict double-precision evaluation and deterministic tie-breaking without
 * introducing global spinlock deadlocks, the architecture uses a two-pass reduction.
 * Pass 1 reduces values to block-level outputs. Pass 2 finalizes the GPU-native selection.
 */
__global__ void devex_pricing_block_kernel(
    const Float* __restrict__ reduced_costs,
    const Float* __restrict__ devex_weights,
    const bool* __restrict__ is_eligible,
    Index n,
    Float* __restrict__ block_scores,
    Index* __restrict__ block_indices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    Float max_score = -1.0;
    Index best_index = 0x7FFFFFFF;

    // 1. Parallel thread evaluation (Double Precision)
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (is_eligible[i]) {
            Float r = reduced_costs[i];
            Float gamma = devex_weights[i];
            
            // Minimum mathematically necessary validity condition
            if (gamma > 0.0) {
                Float score = (r * r) / gamma;
                if (score > max_score || (score == max_score && i < best_index)) {
                    max_score = score;
                    best_index = i;
                }
            }
        }
    }

    // 2. Warp-level reduction
    // Using shared memory for intra-block reduction since __shfl_down_sync with double requires two registers
    // and to safely handle the index together.
    extern __shared__ double s_data[]; // size: 2 * blockDim.x 
    double* s_scores = s_data;
    Index* s_indices = (Index*)(s_data + blockDim.x);
    
    s_scores[threadIdx.x] = max_score;
    s_indices[threadIdx.x] = best_index;
    __syncthreads();

    // 3. Block-level reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            Float other_score = s_scores[threadIdx.x + s];
            Index other_index = s_indices[threadIdx.x + s];
            
            if (other_score > s_scores[threadIdx.x] || 
               (other_score == s_scores[threadIdx.x] && other_index < s_indices[threadIdx.x])) {
                s_scores[threadIdx.x] = other_score;
                s_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    // 4. Output block maximum to global memory
    if (threadIdx.x == 0) {
        block_scores[blockIdx.x] = s_scores[0];
        block_indices[blockIdx.x] = s_indices[0];
    }
}

/**
 * @brief CUDA kernel for Step 12.1 Warp-Synchronous Devex Pricing (Pass 2)
 * 
 * Performs final GPU-native selection from the block-level outputs.
 */
__global__ void devex_pricing_global_kernel(
    const Float* __restrict__ block_scores,
    const Index* __restrict__ block_indices,
    int num_blocks,
    Float* __restrict__ global_score,
    Index* __restrict__ global_index
) {
    Float max_score = -1.0;
    Index best_index = 0x7FFFFFFF;

    // Single warp/block sequential reduction over num_blocks (typically very small, e.g. < 1000)
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        Float score = block_scores[i];
        Index idx = block_indices[i];
        
        if (score > max_score || (score == max_score && idx < best_index)) {
            max_score = score;
            best_index = idx;
        }
    }

    extern __shared__ double s_data[]; // size: 2 * blockDim.x 
    double* s_scores = s_data;
    Index* s_indices = (Index*)(s_data + blockDim.x);
    
    s_scores[threadIdx.x] = max_score;
    s_indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            Float other_score = s_scores[threadIdx.x + s];
            Index other_index = s_indices[threadIdx.x + s];
            
            if (other_score > s_scores[threadIdx.x] || 
               (other_score == s_scores[threadIdx.x] && other_index < s_indices[threadIdx.x])) {
                s_scores[threadIdx.x] = other_score;
                s_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *global_score = s_scores[0];
        *global_index = s_indices[0];
    }
}

void launch_devex_pricing(
    const Float* d_reduced_costs,
    const Float* d_devex_weights,
    const bool* d_is_eligible,
    Index n,
    Float* d_block_scores,
    Index* d_block_indices,
    Float* d_global_score,
    Index* d_global_index,
    int blocks,
    int threads
) {
    size_t shared_mem_bytes = threads * (sizeof(double) + sizeof(Index));
    
    devex_pricing_block_kernel<<<blocks, threads, shared_mem_bytes>>>(
        d_reduced_costs,
        d_devex_weights,
        d_is_eligible,
        n,
        d_block_scores,
        d_block_indices
    );
    
    // Final GPU-native reduction (using 1 block of 256 threads)
    int final_threads = (blocks < 256) ? blocks : 256;
    // Round up to next power of 2 for shared memory reduction if needed, but the loop handles non-powers if padded
    // Wait, standard block reduction requires power of 2 blockDim. Let's force power of 2.
    int p2 = 1;
    while(p2 < final_threads) p2 *= 2;
    final_threads = p2;
    
    size_t final_shared_bytes = final_threads * (sizeof(double) + sizeof(Index));
    
    devex_pricing_global_kernel<<<1, final_threads, final_shared_bytes>>>(
        d_block_scores,
        d_block_indices,
        blocks,
        d_global_score,
        d_global_index
    );
}

} // namespace cuda
} // namespace sankhya
