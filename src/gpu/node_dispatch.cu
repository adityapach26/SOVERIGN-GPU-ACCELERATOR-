#include "gpu/node_dispatch.cuh"
#include "gpu/batched_dual_simplex.cuh"
#include "gpu/vram_arena.cuh"
#include "gpu/device_model.cuh"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace sankhya {
namespace gpu {

NodeDispatchResult dispatch_single_node(
    const DeviceModel& model,
    const DeviceSparseLU& lu,
    WorkingBasisState& ws,
    Float obj_sign,
    Index max_iterations,
    VRAMArena& arena,
    Index num_cols
) {
    // Allocate single-element arrays for batched dispatcher
    WorkingBasisState* d_ws_array = static_cast<WorkingBasisState*>(
        arena.allocate(sizeof(WorkingBasisState)));
    DeviceSimplexStatus* d_status = static_cast<DeviceSimplexStatus*>(
        arena.allocate(sizeof(DeviceSimplexStatus)));
    Index* d_iter_count = static_cast<Index*>(
        arena.allocate(sizeof(Index)));

    if (!d_ws_array || !d_status || !d_iter_count) {
        if (d_ws_array) arena.free(d_ws_array);
        if (d_status) arena.free(d_status);
        if (d_iter_count) arena.free(d_iter_count);
        throw std::runtime_error("Failed to allocate node dispatch arrays in VRAMArena");
    }

    cudaError_t err = cudaMemcpy(d_ws_array, &ws, sizeof(WorkingBasisState), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy to d_ws_array failed");
    }

    // Trigger dual simplex on GPU
    launch_batched_dual_simplex(
        model,
        lu,
        d_ws_array,
        obj_sign,
        max_iterations,
        d_status,
        d_iter_count,
        1
    );

    // Synchronize to ensure kernel finishes before reading back results
    cudaDeviceSynchronize();

    NodeDispatchResult result;
    err = cudaMemcpy(&result.status, d_status, sizeof(DeviceSimplexStatus), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy from d_status failed");
    }

    result.x.resize(static_cast<std::size_t>(num_cols));
    err = cudaMemcpy(result.x.data(), ws.x, num_cols * sizeof(Float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy from ws.x failed");
    }

    // Cleanup device allocations
    arena.free(d_iter_count);
    arena.free(d_status);
    arena.free(d_ws_array);

    return result;
}

} // namespace gpu
} // namespace sankhya
