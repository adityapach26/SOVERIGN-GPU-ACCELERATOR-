#pragma once

#include "sankhya/types.hpp"
#include <vector>

// Forward declarations
namespace sankhya {
namespace gpu {
    struct DeviceModel;
    struct DeviceSparseLU;
    struct WorkingBasisState;
    class VRAMArena;
    enum class DeviceSimplexStatus;

    // Bridge struct for the result
    struct NodeDispatchResult {
        DeviceSimplexStatus status;
        std::vector<Float> x;
    };

    /**
     * @brief CPU-to-GPU bridge to launch a single node solve.
     * 
     * Handles the minimal host-to-device memory transfers for node execution
     * without polluting CPU-only translation units with CUDA APIs.
     */
    NodeDispatchResult dispatch_single_node(
        const DeviceModel& model,
        const DeviceSparseLU& lu,
        WorkingBasisState& ws, // Note: ws contains device pointers but the struct itself is host-side
        Float obj_sign,
        Index max_iterations,
        VRAMArena& arena,
        Index num_cols
    );

    /**
     * @brief CPU-to-GPU bridge to fetch bounds from a WorkingBasisState.
     */
    void fetch_working_bounds(
        const WorkingBasisState& ws,
        Index num_cols,
        std::vector<Float>& out_lb,
        std::vector<Float>& out_ub
    );
}
}
