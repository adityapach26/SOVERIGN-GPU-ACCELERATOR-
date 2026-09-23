#include <cstddef>
#include <new>
#include <stdexcept>
/**
 * @file test_vram_arena.cu
 * @brief Unit tests for CUDA VRAM Arena
 */

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <vector>

#include "gpu/vram_arena.cuh"

using namespace sankhya;

// A simple deterministic kernel to write into the allocated memory
__global__ void write_pattern_kernel(int* data, std::size_t num_elements, int offset_val) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = static_cast<int>(idx) + offset_val;
    }
}

TEST_CASE("VRAMArena - Construction and Initialization", "[gpu][vram_arena]") {
    // Test 1 - Construction (1 MB)
    gpu::VRAMArena arena(1024 * 1024);
    REQUIRE(arena.occupancy_percentage() == 0);
}

TEST_CASE("VRAMArena - Allocation and Free", "[gpu][vram_arena]") {
    gpu::VRAMArena arena(1024 * 1024); // 1 MB
    
    // Test 2 - Device allocation
    void* ptr = arena.allocate(1024);
    REQUIRE(ptr != nullptr);
    REQUIRE(arena.occupancy_percentage() > 0);
    
    // Test 3 & 4 - Kernel write and read
    std::size_t num_elements = 1024 / sizeof(int);
    int* d_data = static_cast<int*>(ptr);
    
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    write_pattern_kernel<<<blocks, threads>>>(d_data, num_elements, 42);
    
    // Explicit sync and check
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    
    std::vector<int> h_data(num_elements);
    REQUIRE(cudaMemcpy(h_data.data(), d_data, num_elements * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    
    for (std::size_t i = 0; i < num_elements; ++i) {
        REQUIRE(h_data[i] == static_cast<int>(i) + 42);
    }
    
    // Test 5 - Free
    arena.free(ptr);
    REQUIRE(arena.occupancy_percentage() == 0);
    
    // Test 6 - Reuse
    void* ptr2 = arena.allocate(1024);
    REQUIRE(ptr2 != nullptr);
    arena.free(ptr2);
    REQUIRE(arena.occupancy_percentage() == 0);
}

TEST_CASE("VRAMArena - Multiple Allocations and Coalescing", "[gpu][vram_arena]") {
    // Use a small arena (1024 bytes). Alignment is 256 bytes, so we have exactly 4 blocks of 256 bytes.
    gpu::VRAMArena arena(1024);
    
    // Test 7 - Multiple allocations
    void* ptrA = arena.allocate(200); // Takes 256 bytes (aligned)
    void* ptrB = arena.allocate(200); // Takes 256 bytes
    void* ptrC = arena.allocate(200); // Takes 256 bytes
    
    REQUIRE(ptrA != nullptr);
    REQUIRE(ptrB != nullptr);
    REQUIRE(ptrC != nullptr);
    
    REQUIRE(ptrA != ptrB);
    REQUIRE(ptrB != ptrC);
    REQUIRE(ptrA != ptrC);
    
    REQUIRE(arena.occupancy_percentage() == 75); // 768 / 1024 * 100 = 75
    
    // Test 8 - Coalescing
    // Free B and C
    arena.free(ptrB);
    arena.free(ptrC);
    
    REQUIRE(arena.occupancy_percentage() == 25);
    
    // We should now have 768 bytes contiguous free space.
    // If we request 512 bytes, it should succeed.
    void* ptr_large = arena.allocate(500); // Takes 512 bytes
    REQUIRE(ptr_large != nullptr);
    
    REQUIRE(arena.occupancy_percentage() == 75);
    
    arena.free(ptrA);
    arena.free(ptr_large);
    REQUIRE(arena.occupancy_percentage() == 0);
}

TEST_CASE("VRAMArena - Edge Cases", "[gpu][vram_arena]") {
    gpu::VRAMArena arena(1024 * 1024);
    
    // Test 9 - Zero allocation
    void* ptr_zero = arena.allocate(0);
    REQUIRE(ptr_zero == nullptr);
    REQUIRE(arena.occupancy_percentage() == 0);
    
    // Test 10 - Oversized allocation
    REQUIRE_THROWS_AS(arena.allocate(2 * 1024 * 1024), std::bad_alloc);
    REQUIRE(arena.occupancy_percentage() == 0);
    
    // Double free / Invalid free
    void* ptr = arena.allocate(100);
    arena.free(ptr);
    REQUIRE_THROWS_AS(arena.free(ptr), std::invalid_argument);
    
    // Free nullptr (safe)
    REQUIRE_NOTHROW(arena.free(nullptr));
}

TEST_CASE("VRAMArena - Occupancy tracking", "[gpu][vram_arena]") {
    // 4 blocks of 256 bytes
    gpu::VRAMArena arena(1024);
    REQUIRE(arena.occupancy_percentage() == 0);
    
    void* p1 = arena.allocate(100); // 256
    REQUIRE(arena.occupancy_percentage() == 25);
    
    void* p2 = arena.allocate(100); // 256
    REQUIRE(arena.occupancy_percentage() == 50);
    
    void* p3 = arena.allocate(100); // 256
    REQUIRE(arena.occupancy_percentage() == 75);
    
    void* p4 = arena.allocate(100); // 256
    REQUIRE(arena.occupancy_percentage() == 100);
    
    arena.free(p2); // frees 256
    REQUIRE(arena.occupancy_percentage() == 75);
    
    arena.free(p1);
    arena.free(p3);
    arena.free(p4);
    
    REQUIRE(arena.occupancy_percentage() == 0);
}

