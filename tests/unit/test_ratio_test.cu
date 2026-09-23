#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include "cuda/ratio_test.cuh"
#include <vector>
#include <cmath>
#include <iostream>

using namespace sankhya;

TEST_CASE("Step 12.2 Warp-Synchronous Harris Two-Pass Ratio Test", "[cuda][ratio_test]") {
    Index m = 1024; // Multi-warp size
    
    std::vector<Float> h_d(m, 0.0);
    std::vector<Float> h_x(m, 0.0);
    std::vector<Index> h_basic(m, 0);
    
    for (Index i = 0; i < m; ++i) {
        h_basic[i] = i; // 1:1 mapping for simplicity
    }
    
    // TEST 1 — BASIC HARRIS SELECTION
    // TEST 2 — LARGEST PIVOT MAGNITUDE
    // TEST 3 — HARRIS EXCLUSION
    // TEST 4 — TIE / DETERMINISM
    // TEST 5 — MULTI-WARP (handled by spreading indices)
    // TEST 6 — NUMERICAL PRECISION (exercising math::kDefaultFeasibilityTol)
    
    // Baseline: index 10 (ratio = 10.0 / 1.0 = 10.0)
    h_x[10] = 10.0;
    h_d[10] = 1.0;
    
    // Index 20: (ratio = 8.0 / 1.0 = 8.0). This sets Delta.
    // Delta = (8.0 + 1e-6) / 1.0 = 8.000001
    h_x[20] = 8.0;
    h_d[20] = 1.0;
    
    // Harris exclusion:
    // Index 30: strong pivot, but ratio = 15.0 / 2.0 = 7.5. Wait!
    // Ratio 7.5 is LESS than Delta 8.000001, so it is valid.
    // Let's make index 30 fail the Harris bound.
    // To fail bound, ratio must be > Delta.
    // Index 30: ratio = 20.0 / 2.0 = 10.0 > 8.000001. Strong pivot (2.0) but fails bound.
    h_x[30] = 20.0;
    h_d[30] = 2.0;
    
    // Index 40: strong pivot, ratio = 16.0 / 2.0 = 8.0.
    // Ratio 8.0 <= Delta (8.000001). It passes!
    // Pivot is 2.0. So this beats index 20 (pivot 1.0).
    h_x[40] = 16.0;
    h_d[40] = 2.0;
    
    // Tie / Determinism:
    // Index 50: same ratio, same pivot as 40.
    h_x[50] = 16.0;
    h_d[50] = 2.0;
    // Lower index (40) should win over 50.
    
    // Precision:
    // Index 60: ratio = 8.0000005. 
    // Is it <= Delta (8.0000010)? Yes.
    // Pivot is 3.0. This beats index 40!
    h_x[60] = 8.0000005 * 3.0;
    h_d[60] = 3.0;
    
    // Tie / Determinism cross-warp:
    // Index 150 (different warp than 60): exact same values as 60.
    h_x[150] = 8.0000005 * 3.0;
    h_d[150] = 3.0;
    
    // Setup CUDA data
    Float* d_d;
    Float* d_x;
    Index* d_basic;
    
    int threads = 128;
    int blocks = (m + threads - 1) / threads;
    
    Float* d_block_deltas;
    Float* d_global_delta;
    Float* d_block_best_d;
    Index* d_block_best_idx;
    Index* d_final_idx;
    
    cudaMalloc(&d_d, m * sizeof(Float));
    cudaMalloc(&d_x, m * sizeof(Float));
    cudaMalloc(&d_basic, m * sizeof(Index));
    cudaMalloc(&d_block_deltas, blocks * sizeof(Float));
    cudaMalloc(&d_global_delta, sizeof(Float));
    cudaMalloc(&d_block_best_d, blocks * sizeof(Float));
    cudaMalloc(&d_block_best_idx, blocks * sizeof(Index));
    cudaMalloc(&d_final_idx, sizeof(Index));
    
    cudaMemcpy(d_d, h_d.data(), m * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), m * sizeof(Float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_basic, h_basic.data(), m * sizeof(Index), cudaMemcpyHostToDevice);
    
    cuda::launch_harris_ratio_test(
        d_d, d_x, d_basic, m,
        d_block_deltas, d_global_delta,
        d_block_best_d, d_block_best_idx, d_final_idx,
        blocks, threads
    );
    cudaDeviceSynchronize();
    
    Index final_idx = -1;
    Float final_delta = -1.0;
    cudaMemcpy(&final_idx, d_final_idx, sizeof(Index), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_delta, d_global_delta, sizeof(Float), cudaMemcpyDeviceToHost);
    
    // Verify Pass 1 (Delta)
    // The min relaxed_ratio is at index 20: (8.0 + 1e-6) / 1.0 = 8.000001
    // Wait, what about index 60? relaxed_ratio = (8.0000005*3 + 1e-6) / 3 = 8.0000005 + 1e-6/3 = 8.0000005 + 0.000000333 = 8.000000833
    // So index 20 still sets Delta?
    // Wait: 
    // index 20 relaxed ratio = 8.000001
    // index 60 relaxed ratio = 8.000000833.
    // Ah! 8.000000833 < 8.000001. So index 60 will actually set the Delta!
    // Delta = 8.000000833...
    
    // Let's print or calculate the exact Delta.
    Float expected_delta = (h_x[40] + 1e-6) / 2.0;
    REQUIRE(std::abs(final_delta - expected_delta) < 1e-12);
    
    // Verify Pass 2 (Selection)
    // Candidates passing Delta check (<= expected_delta):
    // index 20: ratio = 8.0 <= 8.000000833 (Passes, pivot = 1.0)
    // index 40: ratio = 8.0 <= 8.000000833 (Passes, pivot = 2.0)
    // index 50: ratio = 8.0 <= 8.000000833 (Passes, pivot = 2.0)
    // index 60: ratio = 8.0000005 <= 8.000000833 (Passes, pivot = 3.0)
    // index 150: ratio = 8.0000005 <= 8.000000833 (Passes, pivot = 3.0)
    // index 30: ratio = 10.0 > 8.000000833 (Fails, pivot = 2.0 excluded)
    // Winner: largest pivot is 3.0 (indices 60 and 150).
    // Tie breaker: lower index wins -> 60.
    
    REQUIRE(final_idx == 60);
    
    cudaFree(d_d);
    cudaFree(d_x);
    cudaFree(d_basic);
    cudaFree(d_block_deltas);
    cudaFree(d_global_delta);
    cudaFree(d_block_best_d);
    cudaFree(d_block_best_idx);
    cudaFree(d_final_idx);
}
