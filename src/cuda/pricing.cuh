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
 * @param d_block_scores  Global memory array for block-level maximum scores (length: blocks).
 * @param d_block_indices Global memory array for block-level best indices (length: blocks).
 * @param d_global_score  Global memory output for final maximum score (length: 1).
 * @param d_global_index  Global memory output for final best index (length: 1).
 * @param blocks          Number of blocks for the first pass.
 * @param threads         Number of threads per block for the first pass.
 */
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
);

} // namespace cuda
} // namespace sankhya
