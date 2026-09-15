/**
 * @file vram_arena.cuh
 * @brief CUDA Memory Pool / Persistent VRAM Arena (Step 9.1)
 *
 * Provides a block-based free-list allocator managing a single large persistent 
 * CUDA device buffer to avoid cudaMalloc/cudaFree overhead during runtime operations.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <limits>
#include <new>

namespace sankhya {
namespace gpu {

class VRAMArena {
public:
    // Constructor initializes the persistent device pool.
    // Engineering decision: Default size 256 MB.
    explicit VRAMArena(std::size_t capacity_bytes = 256ULL * 1024ULL * 1024ULL);

    // Releases the persistent device pool.
    ~VRAMArena();

    // Prevent copy/move to avoid double frees of the backing buffer
    VRAMArena(const VRAMArena&) = delete;
    VRAMArena& operator=(const VRAMArena&) = delete;

    // Host-callable allocation returning a device pointer.
    void* allocate(std::size_t bytes);

    // Host-callable deallocation of a pointer previously returned by allocate().
    void free(void* ptr);

    // Percentage of pool capacity currently allocated.
    std::size_t occupancy_percentage() const;

private:
    struct FreeBlock {
        std::size_t offset;
        std::size_t size;
        
        bool operator<(const FreeBlock& other) const {
            return offset < other.offset;
        }
    };

    void* d_base_ptr_;
    std::size_t total_capacity_;
    std::size_t currently_allocated_;
    std::vector<FreeBlock> free_list_;
    
    // Engineering decision: Track allocated blocks mapping offset -> size
    // Provides protection against double frees and invalid pointer frees
    std::unordered_map<std::size_t, std::size_t> allocated_blocks_;
    
    // Engineering decision: Mutex for thread-safe host-side allocation
    mutable std::mutex mutex_;

    // Engineering decision: 256-byte alignment boundary for allocations
    static constexpr std::size_t kAlignment = 256;
    
    // Internal helper to align size
    static std::size_t align_size(std::size_t size) {
        if (size > std::numeric_limits<std::size_t>::max() - (kAlignment - 1)) {
            throw std::bad_alloc(); // Deterministic failure on overflow
        }
        return (size + kAlignment - 1) & ~(kAlignment - 1);
    }
};

} // namespace gpu
} // namespace sankhya
