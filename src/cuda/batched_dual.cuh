#pragma once

#include "sankhya/types.hpp"
#include "gpu/hbf.cuh"
#include "gpu/device_model.cuh"
#include <vector>
#include <cstdint>

namespace sankhya {
namespace cuda {

/**
 * @brief Step 13.1 Batched Dual Simplex Launch
 */
class BatchedDualSolver {
public:
    BatchedDualSolver(gpu::HBFManager& hbf_manager, const gpu::DeviceModel& device_model)
        : hbf_manager_(hbf_manager), device_model_(device_model) {}

    /**
     * @brief Required Public API: Launch K sibling nodes concurrently.
     */
    void batched_dual_simplex(const std::vector<uint32_t>& hbf_node_ids);

    const std::vector<Float>& get_objectives() const { return out_objectives_; }
    const std::vector<int>& get_statuses() const { return out_statuses_; }

private:
    gpu::HBFManager& hbf_manager_;
    const gpu::DeviceModel& device_model_;
    std::vector<Float> out_objectives_;
    std::vector<int> out_statuses_;
};

} // namespace cuda
} // namespace sankhya
