#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <memory>
#include "sankhya/types.hpp"
#include "core/problem.hpp"

namespace sankhya {
namespace milp {

struct MILPNode {
    uint32_t hbf_id;
    Float parent_bound;
    int depth;

    // Optional standalone model for spatial B&B coefficient updates
    std::shared_ptr<core::Model> spatial_model;

    // For priority queue ordering: lowest parent_bound = highest priority
    // std::priority_queue outputs the largest element first by default.
    // So we define operator< such that a node with a larger bound is "less" in priority,
    // causing the smaller bound to be at the top of the heap.
    bool operator<(const MILPNode& other) const {
        return parent_bound > other.parent_bound;
    }
};

class NodeQueue {
public:
    void push(MILPNode node);
    MILPNode pop();
    bool empty();

private:
    std::priority_queue<MILPNode> queue_;
};

class GlobalIncumbent {
public:
    GlobalIncumbent();

    void update(const std::vector<Float>& x, Float obj);
    Float get_obj();

    // To verify internal state safely for tests
    std::vector<Float> get_x();

private:
    mutable std::mutex mutex_;
    std::vector<Float> incumbent_x_;
    Float incumbent_obj_;
};

} // namespace milp
} // namespace sankhya
