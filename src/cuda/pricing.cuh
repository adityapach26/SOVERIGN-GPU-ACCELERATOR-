#pragma once

#include "sankhya/types.hpp"

namespace sankhya {
namespace cuda {

/**
 * @brief Step 12.1 Warp-Synchronous Devex Pricing Kernel.
 *
 * @param d_reduced_costs Array of reduced costs (length n).
 * @param d_devex_weights Array of Devex weights (length n).
 * @param d_is_eligible   Array indicating eligibility for entering (length n).
 * @param n               Total number of variables.
 * @param d_block_out     Global memory block-level output array for packed atomic max results.
 *                        Must be pre-initialized to 0. Length: gridDim.x.
 */
void launch_devex_pricing(
    const Float* d_reduced_costs,
    const Float* d_devex_weights,
    const bool* d_is_eligible,
    Index n,
    unsigned long long int* d_block_out,
    int blocks,
    int threads
);

} // namespace cuda
} // namespace sankhya
