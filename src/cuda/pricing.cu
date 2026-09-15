#include "pricing.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace sankhya {
namespace cuda {

/**
 * @brief CUDA kernel for Step 12.1 Warp-Synchronous Devex Pricing (Pass 1)
 * 
 * Computes Devex scores, uses __shfl_down_sync for warp reduction, and staged 
 * atomicMax/atomicMin for block and global maximum score tracking.
 * 
 * Engineering Decision (Staged Atomics for Exact Double Precision):
 * To preserve exact double precision without packing into float, we execute a staged 
 * atomic mechanism. First, atomicMax identifies the exact maximum double score 
 * (via __double_as_longlong). Second, any thread matching that exact maximum uses 
 * atomicMin to deterministically record its index. This strictly guarantees lower-index 
 * tie-breaking natively on the GPU without spinlocks.
 */
__global__ void devex_pricing_kernel(
    const Float* __restrict__ reduced_costs,
    const Float* __restrict__ devex_weights,
    const bool* __restrict__ is_eligible,
    Index n,
    Float* __restrict__ block_scores,
    Index* __restrict__ block_indices,
    unsigned long long int* __restrict__ global_score_ull
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    
    double max_score = -1.0;
    Index best_index = 0x7FFFFFFF;

    // 1. Parallel thread evaluation
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (is_eligible[i]) {
            Float r = reduced_costs[i];
            Float gamma = devex_weights[i];
            
            // Minimum mathematically necessary validity condition
            if (gamma > 0.0) {
                double score = (r * r) / gamma;
                if (score > max_score || (score == max_score && i < best_index)) {
                    max_score = score;
                    best_index = i;
                }
            }
        }
    }

    // 2. Warp-level reduction using actual __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        double other_score = __shfl_down_sync(mask, max_score, offset);
        Index other_index = __shfl_down_sync(mask, best_index, offset);
        
        if (other_score > max_score || (other_score == max_score && other_index < best_index)) {
            max_score = other_score;
            best_index = other_index;
        }
    }

    // 3. Staged atomic operation for Block-level reduction
    __shared__ unsigned long long int s_block_max_score_ull;
    __shared__ Index s_block_best_index;

    if (threadIdx.x == 0) {
        s_block_max_score_ull = 0;
        s_block_best_index = 0x7FFFFFFF;
    }
    __syncthreads();

    // Stage 3A: atomicMax on the score
    if (lane == 0 && max_score >= 0.0) {
        atomicMax(&s_block_max_score_ull, __double_as_longlong(max_score));
    }
    __syncthreads();

    // Stage 3B: atomicMin on the index for ties
    if (lane == 0 && max_score >= 0.0) {
        if (__double_as_longlong(max_score) == s_block_max_score_ull) {
            atomicMin(&s_block_best_index, best_index);
        }
    }
    __syncthreads();

    // 4. Staged atomic operation for Global-level score tracking
    if (threadIdx.x == 0) {
        unsigned long long int b_score_ull = s_block_max_score_ull;
        Index b_index = s_block_best_index;
        
        if (b_score_ull > 0) {
            atomicMax(global_score_ull, b_score_ull);
        }
        
        // Save block winner for Kernel 2 to resolve the global index tie
        block_scores[blockIdx.x] = __longlong_as_double(b_score_ull);
        block_indices[blockIdx.x] = b_index;
    }
}

/**
 * @brief CUDA kernel for Step 12.1 Warp-Synchronous Devex Pricing (Pass 2)
 * 
 * Performs Stage 2 of the global staged atomic operation: any block that achieved
 * the final global maximum score uses atomicMin to record its index.
 */
__global__ void devex_pricing_resolve_index_kernel(
    const Float* __restrict__ block_scores,
    const Index* __restrict__ block_indices,
    int num_blocks,
    const unsigned long long int* __restrict__ global_score_ull,
    Index* __restrict__ global_index
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_blocks) {
        double global_max = __longlong_as_double(*global_score_ull);
        
        if (block_scores[tid] == global_max) {
            atomicMin(global_index, block_indices[tid]);
        }
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
    // We bit-cast the double pointer to ULL for the atomicMax target
    unsigned long long int* d_global_score_ull = reinterpret_cast<unsigned long long int*>(d_global_score);

    devex_pricing_kernel<<<blocks, threads>>>(
        d_reduced_costs,
        d_devex_weights,
        d_is_eligible,
        n,
        d_block_scores,
        d_block_indices,
        d_global_score_ull
    );
    
    int resolve_threads = 256;
    int resolve_blocks = (blocks + resolve_threads - 1) / resolve_threads;
    
    devex_pricing_resolve_index_kernel<<<resolve_blocks, resolve_threads>>>(
        d_block_scores,
        d_block_indices,
        blocks,
        d_global_score_ull,
        d_global_index
    );
}

} // namespace cuda
} // namespace sankhya
