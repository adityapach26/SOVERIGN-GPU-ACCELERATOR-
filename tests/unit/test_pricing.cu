#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include "cuda/pricing.cuh"
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>

using namespace sankhya;

TEST_CASE("Step 12.1 Warp-Synchronous Devex Pricing Kernel (Hardened)", "[cuda][pricing]") {
    Index n = 1024; // Use large enough for multiple blocks
    
    std::vector<Float> h_rc(n, 0.0);
    std::vector<Float> h_weight(n, 1.0);
    std::vector<bool> h_eligible(n, true);
    
    // TEST 1 — BASIC MAXIMUM
    h_rc[42] = 10.0;
    h_weight[42] = 2.0; // score = 100 / 2 = 50.0
    
    // TEST 2 — DETERMINISTIC TIE (CROSS-BLOCK)
    // block A (threads 0-127, covers index 42) has score 50 at index 42.
    // block B (threads 128-255, covers index 150) will have identical score 50.
    h_rc[150] = 10.0;
    h_weight[150] = 2.0; // score = 50.0
    
    // TEST 3 — PRECISION SENSITIVITY
    // Float (double) precision test. 
    // Float cast loses 29 bits. We create a score that differs ONLY in the lower bits of double precision.
    // Score at 42: 50.0
    // Let's set index 8 (which is smaller than 42, so it WOULD win if scores were equal).
    // Make its score just SLIGHTLY smaller than 50.0 in double precision, but identical in float precision.
    // 50.0 in binary is 110010.
    // Float precision has 24 bits. Double has 53.
    // 50.0 * (1 - 1e-12) will be identical in float, but smaller in double.
    h_rc[8] = std::sqrt(50.0 * (1.0 - 1e-12));
    h_weight[8] = 1.0; 
    // If precision is lost (cast to float), score[8] == score[42], and 8 < 42, so 8 wins.
    // With strict double precision, score[8] < score[42], so 42 wins.
    
    Float* d_rc;
    Float* d_weight;
    bool* d_eligible;
    
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    
    Float* d_block_scores;
    Index* d_block_indices;
    Float* d_global_score;
    Index* d_global_index;
    
    cudaMalloc(&d_rc, n * sizeof(Float));
    cudaMalloc(&d_weight, n * sizeof(Float));
    cudaMalloc(&d_eligible, n * sizeof(bool));
    cudaMalloc(&d_block_scores, blocks * sizeof(Float));
    cudaMalloc(&d_block_indices, blocks * sizeof(Index));
    cudaMalloc(&d_global_score, sizeof(Float));
    cudaMalloc(&d_global_index, sizeof(Index));
    
    cudaMemcpy(d_rc, h_rc.data(), n * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), n * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eligible, h_eligible.data(), n * sizeof(bool), cudaMemcpyHostToDevice);
    
    cuda::launch_devex_pricing(d_rc, d_weight, d_eligible, n, 
                               d_block_scores, d_block_indices, 
                               d_global_score, d_global_index, 
                               blocks, threads);
    cudaDeviceSynchronize();
    
    Float final_score = 0.0;
    Index final_index = 0;
    
    cudaMemcpy(&final_score, d_global_score, sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_index, d_global_index, sizeof(Index), cudaMemcpyDeviceToHost);
    
    // We expect index 42.
    // It beats 150 by index tie-breaker (42 < 150, score 50 == 50).
    // It beats 8 by strict double precision score (50.0 > 50.0 - epsilon).
    REQUIRE(final_score == 50.0);
    REQUIRE(final_index == 42);
    
    cudaFree(d_rc);
    cudaFree(d_weight);
    cudaFree(d_eligible);
    cudaFree(d_block_scores);
    cudaFree(d_block_indices);
    cudaFree(d_global_score);
    cudaFree(d_global_index);
}
