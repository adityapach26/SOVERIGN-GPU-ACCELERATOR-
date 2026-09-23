#include <cstddef>
/**
 * @file test_device_model.cu
 * @brief Unit tests for Immutable Device-Resident Problem State (Step 10.1)
 */

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <vector>

#include "core/problem.hpp"
#include "gpu/device_model.cuh"

using namespace sankhya;

// A simple CUDA kernel to mathematically verify we can access the device pointers on the GPU
__global__ void inspect_device_model_kernel(
    gpu::DeviceModel d_model,
    int* d_verification_flags)
{
    // A single thread reads values and sets verification flags
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Inspect objective
        if (d_model.obj[0] > 0.0) {
            d_verification_flags[0] = 1;
        }
        
        // Inspect bounds
        if (d_model.ub[1] > 100.0) { // Should be +infinity
            d_verification_flags[1] = 1;
        }

        // Inspect CSC structure
        if (d_model.col_ptrs[0] == 0) {
            d_verification_flags[2] = 1;
        }
    }
}

TEST_CASE("DeviceModel - Immutable Problem Upload", "[gpu][device_model]") {
    // 1. Construct deterministic Model
    core::Model model;
    model.sense = OptimizationSense::Minimize;
    
    // Add variables: nontrivial obj, finite and infinite bounds
    model.add_variable(1.5, 0.0, 10.0);           // x0
    model.add_variable(-2.5, -5.0, math::kInfinity); // x1
    model.add_variable(3.0, 0.0, 1.0);            // x2 (Empty column test)
    
    // Add constraints
    // row 0: 2.0 * x0 + 1.0 * x1 = 5.0
    model.add_constraint({0, 1}, {2.0, 1.0}, 5.0);
    // row 1: 1.0 * x0 = 2.0
    model.add_constraint({0}, {1.0}, 2.0);
    
    model.finalize();
    
    // 2. Initialize VRAMArena
    gpu::VRAMArena arena(1024 * 16); // 16 KB test arena
    std::size_t initial_occupancy = arena.occupancy_percentage();
    REQUIRE(initial_occupancy == 0);

    // 3. Upload to Device
    gpu::DeviceModel d_model = gpu::upload_to_device(model, arena);

    // 4. Verify occupancy increased (O(1) allocation path works)
    std::size_t final_occupancy = arena.occupancy_percentage();
    REQUIRE(final_occupancy > 0);

    // 5. Explicitly verify device pointers are valid and contain exact data via cudaMemcpy (Device-to-Host)
    // CSC arrays
    std::vector<Index> h_col_ptrs(model.A.cols + 1);
    REQUIRE(cudaMemcpy(h_col_ptrs.data(), d_model.col_ptrs, (model.A.cols + 1) * sizeof(Index), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (Index j = 0; j <= model.A.cols; ++j) {
        REQUIRE(h_col_ptrs[j] == model.A.col_ptrs[j]);
    }

    std::vector<Index> h_row_indices(model.A.values.size());
    std::vector<Float> h_values(model.A.values.size());
    REQUIRE(cudaMemcpy(h_row_indices.data(), d_model.row_indices, model.A.values.size() * sizeof(Index), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(h_values.data(), d_model.values, model.A.values.size() * sizeof(Float), cudaMemcpyDeviceToHost) == cudaSuccess);
    
    for (std::size_t k = 0; k < model.A.values.size(); ++k) {
        REQUIRE(h_row_indices[k] == model.A.row_indices[k]);
        REQUIRE(h_values[k] == model.A.values[k]);
    }

    // Objective & Bounds
    std::vector<Float> h_obj(model.A.cols);
    std::vector<Float> h_lb(model.A.cols);
    std::vector<Float> h_ub(model.A.cols);
    
    REQUIRE(cudaMemcpy(h_obj.data(), d_model.obj, model.A.cols * sizeof(Float), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(h_lb.data(), d_model.lb, model.A.cols * sizeof(Float), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(h_ub.data(), d_model.ub, model.A.cols * sizeof(Float), cudaMemcpyDeviceToHost) == cudaSuccess);
    
    for (Index j = 0; j < model.A.cols; ++j) {
        REQUIRE(h_obj[j] == model.obj[j]);
        REQUIRE(h_lb[j] == model.lb[j]);
        REQUIRE(h_ub[j] == model.ub[j]);
    }
    
    // Explicit infinity check
    REQUIRE(h_ub[1] == math::kInfinity);

    // 6. Execute direct device-side verification via CUDA kernel
    int* d_flags = static_cast<int*>(arena.allocate(3 * sizeof(int)));
    int h_init_flags[3] = {0, 0, 0};
    REQUIRE(cudaMemcpy(d_flags, h_init_flags, 3 * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    
    inspect_device_model_kernel<<<1, 1>>>(d_model, d_flags);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    
    int h_flags[3] = {0, 0, 0};
    REQUIRE(cudaMemcpy(h_flags, d_flags, 3 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    
    // Validate the kernel correctly accessed and verified data natively on the GPU
    REQUIRE(h_flags[0] == 1);
    REQUIRE(h_flags[1] == 1);
    REQUIRE(h_flags[2] == 1);

    // Clean up our custom test flag array
    arena.free(d_flags);
    
    // Explicit manual cleanup of device model arrays to avoid memory leak inside test context.
    // In production, the VRAMArena outlives the DeviceModel view, freeing everything perfectly safely upon destruction.
    arena.free(d_model.col_ptrs);
    arena.free(d_model.row_indices);
    arena.free(d_model.values);
    arena.free(d_model.obj);
    arena.free(d_model.lb);
    arena.free(d_model.ub);
}

