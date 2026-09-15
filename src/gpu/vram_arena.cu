/**
 * @file vram_arena.cu
 * @brief CUDA Memory Pool / Persistent VRAM Arena implementation (Step 9.1)
 */

#include "vram_arena.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace sankhya {
namespace gpu {

// Helper to check CUDA errors
static void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

VRAMArena::VRAMArena(std::size_t capacity_bytes)
    : d_base_ptr_(nullptr)
    , total_capacity_(capacity_bytes)
    , currently_allocated_(0)
{
    // The architecture strictly requires one contiguous pre-allocated device buffer.
    check_cuda_error(cudaMalloc(&d_base_ptr_, total_capacity_), "VRAMArena constructor failed to allocate persistent buffer");
    
    // Initialize the free list with a single block covering the entire arena
    free_list_.push_back({0, total_capacity_});
}

VRAMArena::~VRAMArena() {
    // The destructor releases the single backing allocation.
    // This is the only place cudaFree is called for the arena buffer.
    if (d_base_ptr_) {
        cudaFree(d_base_ptr_);
        d_base_ptr_ = nullptr;
    }
}

void* VRAMArena::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Align size to satisfy memory access alignments
    std::size_t aligned_size = align_size(bytes);

    if (currently_allocated_ + aligned_size > total_capacity_) {
        throw std::bad_alloc();
    }

    // Engineering decision: First-fit allocation policy
    auto it = free_list_.begin();
    while (it != free_list_.end()) {
        if (it->size >= aligned_size) {
            std::size_t offset = it->offset;
            
            // If there's leftover space, split the block; otherwise remove it entirely
            if (it->size > aligned_size) {
                it->offset += aligned_size;
                it->size -= aligned_size;
            } else {
                free_list_.erase(it);
            }
            
            currently_allocated_ += aligned_size;
            allocated_blocks_[offset] = aligned_size;
            
            return static_cast<char*>(d_base_ptr_) + offset;
        }
        ++it;
    }

    // No sufficiently large contiguous block found (fragmentation)
    throw std::bad_alloc();
}

void VRAMArena::free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Validate the pointer lies within the backing allocation
    const char* p = static_cast<const char*>(ptr);
    const char* base = static_cast<const char*>(d_base_ptr_);
    
    if (p < base || p >= base + total_capacity_) {
        // Invalid free (pointer outside arena)
        // Engineering decision: Ignore silently or throw. We throw.
        throw std::invalid_argument("VRAMArena::free - Pointer is outside the persistent device buffer.");
    }
    
    std::size_t offset = p - base;
    
    auto alloc_it = allocated_blocks_.find(offset);
    if (alloc_it == allocated_blocks_.end()) {
        // Double-free or invalid pointer protection
        throw std::invalid_argument("VRAMArena::free - Invalid or double-free pointer.");
    }
    
    std::size_t size = alloc_it->second;
    allocated_blocks_.erase(alloc_it);
    currently_allocated_ -= size;
    
    // Return block to free list and coalesce
    FreeBlock new_block{offset, size};
    
    // Insert in sorted order by offset
    auto it = std::upper_bound(free_list_.begin(), free_list_.end(), new_block);
    it = free_list_.insert(it, new_block);
    
    // Coalesce with next block if adjacent
    auto next_it = std::next(it);
    if (next_it != free_list_.end() && it->offset + it->size == next_it->offset) {
        it->size += next_it->size;
        free_list_.erase(next_it);
    }
    
    // Coalesce with previous block if adjacent
    if (it != free_list_.begin()) {
        auto prev_it = std::prev(it);
        if (prev_it->offset + prev_it->size == it->offset) {
            prev_it->size += it->size;
            free_list_.erase(it);
        }
    }
}

std::size_t VRAMArena::occupancy_percentage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total_capacity_ == 0) return 0;
    
    // Occupancy based on the actual reserved aligned block sizes
    double pct = (static_cast<double>(currently_allocated_) / total_capacity_) * 100.0;
    return static_cast<std::size_t>(std::round(pct));
}

} // namespace gpu
} // namespace sankhya

