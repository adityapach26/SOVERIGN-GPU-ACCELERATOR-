#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include "cuda/pricing.cuh"
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>

using namespace sankhya;

TEST_CASE("Step 12.1 Warp-Synchronous Devex Pricing Kernel (Source Fidelity)", "[cuda][pricing]") {
    Index n = 1024; // Use large enough for multiple blocks
    
    std::vector<Float> h_rc(n, 0.0);
    std::vector<Float> h_weight(n, 1.0);
    std::vector<bool> h_eligible(n, true);
    
    // TEST 1 — BASIC MAXIMUM
    h_rc[42] = 10.0;
    h_weight[42] = 2.0; // score = 100 / 2 = 50.0
    
    // TEST 2 — EXACT EQUAL-SCORE TIE
    h_rc[105] = 10.0;
    h_weight[105] = 2.0; // score = 50.0
    
    // TEST 3 — CROSS-BLOCK EQUAL-SCORE TIE
    // Block boundaries depend on threads per block.
    // We use threads=128. So block 0 handles 0-127 (includes 42).
    // Block 1 handles 128-255 (includes 150).
    h_rc[150] = 10.0;
    h_weight[150] = 2.0; // score = 50.0
    
    // TEST 4 — PRECISION SENSITIVITY
    // Score at 42: 50.0
    // Let's set index 8 (which is smaller than 42, so it WOULD win if scores were exactly equal).
    // Make its score just SLIGHTLY smaller than 50.0 in double precision, but identical in float precision.
    h_rc[8] = std::sqrt(50.0 * (1.0 - 1e-12));
    h_weight[8] = 1.0; 
    
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
    
    unsigned long long int zero = 0;
    Index max_idx = 0x7FFFFFFF;
    cudaMemcpy(d_global_score, &zero, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_index, &max_idx, sizeof(Index), cudaMemcpyHostToDevice);
    
    cuda::launch_devex_pricing(d_rc, d_weight, d_eligible, n, 
                               d_block_scores, d_block_indices, 
                               d_global_score, d_global_index, 
                               blocks, threads);
    cudaDeviceSynchronize();
    
    Float final_score = 0.0;
    Index final_index = 0;
    
    cudaMemcpy(&final_score, d_global_score, sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_index, d_global_index, sizeof(Index), cudaMemcpyDeviceToHost);
    
    // We expect index 42 to win against 105 and 150 because it has the exact same max score (50.0) but lowest index.
    // We expect index 42 to win against 8 because index 8 has a slightly strictly lower double-precision score.
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
