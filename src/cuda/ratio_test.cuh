#pragma once

#include "sankhya/types.hpp"

namespace sankhya {
namespace cuda {

/**
 * @brief Step 12.2 Warp-Synchronous Harris Two-Pass Ratio Test.
 * 
 * @param d_d             FTRAN'd entering column (length m).
 * @param d_x             Current primal basic solution (length n, indexed by basic_indices).
 * @param d_basic_indices Current basis (length m).
 * @param m               Number of rows.
 * @param d_block_deltas  Workspace for Pass 1 block minimums (length blocks).
 * @param d_global_delta  Workspace for Pass 1 global minimum (length 1).
 * @param d_block_best_d  Workspace for Pass 2 block best abs_d (length blocks).
 * @param d_block_best_idx Workspace for Pass 2 block best index (length blocks).
 * @param d_final_idx     Global memory output for final selected leaving row index (length 1).
 * @param blocks          Number of blocks.
 * @param threads         Number of threads per block.
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
);

} // namespace cuda
} // namespace sankhya
