#include "pricing.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace sankhya {
namespace cuda {

/**
 * @brief CUDA kernel for Step 12.1 Warp-Synchronous Devex Pricing.
 * 
 * Computes Devex scores in parallel across CUDA threads and reduces
 * the maximum score at the warp level using __shfl_down_sync.
 * The warp winner writes its result to global memory using atomicMax.
 * 
 * Engineering Decision (Numerical Safety & Tie-Breaking):
 * To guarantee deterministic tie-breaking (smaller variable index wins) 
 * via a single hardware-atomic operation without locking:
 * 1. The score (r_t^2 / gamma_t) is cast to single-precision (float).
 * 2. The float is bit-cast to uint32_t (safe for non-negative scores).
 * 3. The 32-bit index is bitwise-inverted (0xFFFFFFFF - index).
 * 4. These are packed into a 64-bit unsigned long long int.
 * atomicMax on this 64-bit value natively maximizes the score and
 * resolves ties by maximizing the inverted index (i.e. minimizing the true index).
 */
__global__ void devex_pricing_kernel(
    const Float* __restrict__ reduced_costs,
    const Float* __restrict__ devex_weights,
    const bool* __restrict__ is_eligible,
    Index n,
    unsigned long long int* __restrict__ block_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float max_score = -1.0f;
    Index best_index = 0x7FFFFFFF;

    // 1. Parallel thread evaluation
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (is_eligible[i]) {
            Float r = reduced_costs[i];
            Float gamma = devex_weights[i];
            
            // Numerical safety: guard against division by zero or negative weights
            if (gamma > 1e-12) {
                float score = static_cast<float>((r * r) / gamma);
                if (score > max_score || (score == max_score && i < best_index)) {
                    max_score = score;
                    best_index = i;
                }
            }
        }
    }

    // 2. Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_score = __shfl_down_sync(mask, max_score, offset);
        Index other_index = __shfl_down_sync(mask, best_index, offset);
        
        if (other_score > max_score || (other_score == max_score && other_index < best_index)) {
            max_score = other_score;
            best_index = other_index;
        }
    }

    // 3. The first thread in each warp writes the warp's maximum to the block output
    if (threadIdx.x % 32 == 0 && max_score >= 0.0f) {
        uint32_t score_bits = __float_as_uint(max_score);
        uint32_t index_bits = ~static_cast<uint32_t>(best_index); // Bitwise invert so larger is smaller index
        unsigned long long int packed = (static_cast<unsigned long long int>(score_bits) << 32) | index_bits;
        
        atomicMax(&block_out[blockIdx.x], packed);
    }
}

void launch_devex_pricing(
    const Float* d_reduced_costs,
    const Float* d_devex_weights,
    const bool* d_is_eligible,
    Index n,
    unsigned long long int* d_block_out,
    int blocks,
    int threads
) {
    devex_pricing_kernel<<<blocks, threads>>>(
        d_reduced_costs,
        d_devex_weights,
        d_is_eligible,
        n,
        d_block_out
    );
}

} // namespace cuda
} // namespace sankhya
