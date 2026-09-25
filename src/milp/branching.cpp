#include "milp/branching.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace sankhya {
namespace milp {

PseudocostTracker::PseudocostTracker(Index num_vars) {
    if (num_vars < 0) {
        throw std::invalid_argument("PseudocostTracker size cannot be negative");
    }
    stats_.resize(static_cast<std::size_t>(num_vars));
}

void PseudocostTracker::record_up_observation(Index var_idx, Float delta_obj, Float fractional_distance) {
    if (var_idx < 0 || static_cast<std::size_t>(var_idx) >= stats_.size()) {
        throw std::out_of_range("Invalid var_idx in record_up_observation");
    }
    // Prevent division by zero / negative distance corruption
    if (fractional_distance <= math::kDefaultFeasibilityTol) {
        return;
    }
    
    Float p = delta_obj / fractional_distance;
    
    // Accumulate history (average convention)
    stats_[static_cast<std::size_t>(var_idx)].sum_p_up += p;
    stats_[static_cast<std::size_t>(var_idx)].count_up++;
    
    global_sum_p_up_ += p;
    global_count_up_++;
}

void PseudocostTracker::record_down_observation(Index var_idx, Float delta_obj, Float fractional_distance) {
    if (var_idx < 0 || static_cast<std::size_t>(var_idx) >= stats_.size()) {
        throw std::out_of_range("Invalid var_idx in record_down_observation");
    }
    // Prevent division by zero / negative distance corruption
    if (fractional_distance <= math::kDefaultFeasibilityTol) {
        return;
    }
    
    Float p = delta_obj / fractional_distance;
    
    stats_[static_cast<std::size_t>(var_idx)].sum_p_down += p;
    stats_[static_cast<std::size_t>(var_idx)].count_down++;
    
    global_sum_p_down_ += p;
    global_count_down_++;
}

Float PseudocostTracker::get_up_pseudocost(Index var_idx) const {
    const auto& st = stats_[static_cast<std::size_t>(var_idx)];
    if (st.count_up > 0) {
        return st.sum_p_up / static_cast<Float>(st.count_up);
    }
    // Deterministic fallback if variable is unobserved in this direction
    if (global_count_up_ > 0) {
        return global_sum_p_up_ / static_cast<Float>(global_count_up_);
    }
    return kUninitializedDefault;
}

Float PseudocostTracker::get_down_pseudocost(Index var_idx) const {
    const auto& st = stats_[static_cast<std::size_t>(var_idx)];
    if (st.count_down > 0) {
        return st.sum_p_down / static_cast<Float>(st.count_down);
    }
    // Deterministic fallback if variable is unobserved in this direction
    if (global_count_down_ > 0) {
        return global_sum_p_down_ / static_cast<Float>(global_count_down_);
    }
    return kUninitializedDefault;
}

Index select_branching_variable(
    const std::vector<Float>& x_frac,
    const PseudocostTracker& tracker
) {
    if (x_frac.empty()) {
        return -1;
    }
    
    if (x_frac.size() != static_cast<std::size_t>(tracker.size())) {
        throw std::invalid_argument("Size mismatch between x_frac and tracker");
    }

    Index best_var = -1;
    Float max_score = -math::kInfinity;

    for (std::size_t i = 0; i < x_frac.size(); ++i) {
        Float val = x_frac[i];
        
        // Compute integer fractional distances
        Float dist_down = val - std::floor(val);
        Float dist_up = std::ceil(val) - val;

        // Candidate is eligible only if it is actually fractional
        if (dist_down > math::kDefaultFeasibilityTol && dist_up > math::kDefaultFeasibilityTol) {
            Index var_idx = static_cast<Index>(i);
            
            Float p_up = tracker.get_up_pseudocost(var_idx);
            Float p_down = tracker.get_down_pseudocost(var_idx);
            
            // Score = min(P_j+, P_j-)
            Float score = std::min(p_up, p_down);
            
            // Break ties deterministically (first encountered/smallest index)
            if (score > max_score) {
                max_score = score;
                best_var = var_idx;
            }
        }
    }

    return best_var;
}

} // namespace milp
} // namespace sankhya
