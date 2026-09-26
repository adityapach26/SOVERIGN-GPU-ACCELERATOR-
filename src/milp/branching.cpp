#include "milp/branching.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include "mrpl/mccormick.hpp"

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

void branch_spatial(
    const MILPNode& parent_node,
    const core::Model& parent_model,
    const core::BilinearTerm& term,
    Index branch_var,
    Float split_val,
    NodeQueue& queue
) {
    if (branch_var != term.x_col && branch_var != term.y_col) {
        throw std::invalid_argument("branch_spatial: branch_var must be x_col or y_col of the bilinear term");
    }

    Float current_lb = parent_model.lb[static_cast<std::size_t>(branch_var)];
    Float current_ub = parent_model.ub[static_cast<std::size_t>(branch_var)];

    // Do not branch if split_val is numerically at the boundary
    if (split_val <= current_lb + math::kDefaultFeasibilityTol ||
        split_val >= current_ub - math::kDefaultFeasibilityTol) {
        return;
    }

    // Helper to create a child
    auto create_child = [&](Float new_lb, Float new_ub) {
        MILPNode child = parent_node;
        child.depth = parent_node.depth + 1;
        
        // Create an independent model for this child because coefficients change
        auto child_model = std::make_shared<core::Model>(parent_model);
        child_model->lb[static_cast<std::size_t>(branch_var)] = new_lb;
        child_model->ub[static_cast<std::size_t>(branch_var)] = new_ub;

        Float xL = child_model->lb[static_cast<std::size_t>(term.x_col)];
        Float xU = child_model->ub[static_cast<std::size_t>(term.x_col)];
        Float yL = child_model->lb[static_cast<std::size_t>(term.y_col)];
        Float yU = child_model->ub[static_cast<std::size_t>(term.y_col)];

        // Regenerate McCormick constraints for the child's bounds
        auto cuts = mrpl::mccormick_envelopes(xL, xU, yL, yU);

        for (int i = 0; i < 4; ++i) {
            Index row_idx = term.mccormick_rows[static_cast<std::size_t>(i)];
            const auto& cut = cuts[static_cast<std::size_t>(i)];
            
            std::vector<Index> cols = {term.x_col, term.y_col, term.w_col};
            
            // Re-finalize must be called if the matrix A is used
            child_model->update_constraint(row_idx, cols, cut.coefficients, cut.rhs);
        }
        
        child_model->finalize();
        child.spatial_model = child_model;
        queue.push(child);
    };

    // Child L: x <= split_val
    create_child(current_lb, split_val);
    
    // Child R: x >= split_val
    create_child(split_val, current_ub);
}

} // namespace milp
} // namespace sankhya
