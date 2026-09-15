#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include "cuda/pricing.cuh"
#include <vector>
#include <iostream>
#include <cstring>

using namespace sankhya;

TEST_CASE("Step 12.1 Warp-Synchronous Devex Pricing Kernel", "[cuda][pricing]") {
    Index n = 128; // Using enough for multiple warps
    
    std::vector<Float> h_rc(n, 0.0);
    std::vector<Float> h_weight(n, 1.0);
    std::vector<bool> h_eligible(n, true);
    
    // TEST 1 — BASIC MAXIMUM
    // Set a known unique maximum score
    h_rc[42] = 10.0;
    h_weight[42] = 2.0; // score = 100 / 2 = 50
    
    // TEST 2 — DETERMINISTIC TIE
    // Set another variable with exactly the SAME maximum score but higher index
    h_rc[105] = 10.0;
    h_weight[105] = 2.0; // score = 50
    
    // And one with the same score but even higher index
    h_rc[77] = 5.0; // score = 25
    
    Float* d_rc;
    Float* d_weight;
    bool* d_eligible;
    unsigned long long int* d_block_out;
    
    int blocks = 2; // TEST 3 — CROSS-WARP / PARALLEL COVERAGE
    
    cudaMalloc(&d_rc, n * sizeof(Float));
    cudaMalloc(&d_weight, n * sizeof(Float));
    cudaMalloc(&d_eligible, n * sizeof(bool));
    cudaMalloc(&d_block_out, blocks * sizeof(unsigned long long int));
    
    cudaMemcpy(d_rc, h_rc.data(), n * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), n * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eligible, h_eligible.data(), n * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemset(d_block_out, 0, blocks * sizeof(unsigned long long int));
    
    int threads = 64; // 2 warps per block
    
    cuda::launch_devex_pricing(d_rc, d_weight, d_eligible, n, d_block_out, blocks, threads);
    cudaDeviceSynchronize();
    
    std::vector<unsigned long long int> h_block_out(blocks);
    cudaMemcpy(h_block_out.data(), d_block_out, blocks * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    
    // The maximum should be aggregated from block outputs
    unsigned long long int global_max_packed = 0;
    for (int i = 0; i < blocks; ++i) {
        if (h_block_out[i] > global_max_packed) {
            global_max_packed = h_block_out[i];
        }
    }
    
    uint32_t best_index_inverted = static_cast<uint32_t>(global_max_packed & 0xFFFFFFFF);
    Index best_index = ~best_index_inverted;
    
    uint32_t score_bits = static_cast<uint32_t>(global_max_packed >> 32);
    float best_score;
    memcpy(&best_score, &score_bits, sizeof(float));
    
    // We expect index 42 because it ties with 105 but is lower
    REQUIRE(best_score == 50.0f);
    REQUIRE(best_index == 42);
    
    cudaFree(d_rc);
    cudaFree(d_weight);
    cudaFree(d_eligible);
    cudaFree(d_block_out);
}
