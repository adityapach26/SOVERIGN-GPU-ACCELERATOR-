#include <cstdint>
#include "gpu/batched_dual_simplex.cuh"

namespace sankhya {
namespace gpu {

__global__ void batched_dual_simplex_kernel(
    DeviceModel model,
    DeviceSparseLU lu,
    WorkingBasisState* d_ws_array,
    Float obj_sign,
    Index max_iterations,
    DeviceSimplexStatus* d_status_array,
    Index* d_iter_count_array,
    uint32_t num_nodes
) {
    uint32_t node_idx = blockIdx.x;
    if (node_idx >= num_nodes) return;

    // Isolate independent state for this block (node)
    WorkingBasisState ws = d_ws_array[node_idx];
    DeviceSimplexStatus* status_out = &d_status_array[node_idx];
    Index* iter_count_out = nullptr;
    if (d_iter_count_array != nullptr) {
        iter_count_out = &d_iter_count_array[node_idx];
    }

    // Reuse the exact validated foundation sequentially in this block
    device_dual_simplex_iteration_loop(
        model, 
        lu, 
        ws, 
        obj_sign, 
        max_iterations, 
        status_out, 
        iter_count_out
    );
}

void launch_batched_dual_simplex(
    const DeviceModel& model,
    const DeviceSparseLU& lu,
    WorkingBasisState* d_ws_array,
    Float obj_sign,
    Index max_iterations,
    DeviceSimplexStatus* d_status_array,
    Index* d_iter_count_array,
    uint32_t num_nodes
) {
    if (num_nodes == 0) return;

    // [ENGINEERING DECISION] Grid mapping: 1 block per node
    // This allows each block to execute its independent node solve concurrently.
    // 32 threads per block are used to maintain parity with the foundation's warp configuration.
    dim3 grid(num_nodes);
    dim3 block(32);

    batched_dual_simplex_kernel<<<grid, block>>>(
        model,
        lu,
        d_ws_array,
        obj_sign,
        max_iterations,
        d_status_array,
        d_iter_count_array,
        num_nodes
    );
}

} // namespace gpu
} // namespace sankhya

