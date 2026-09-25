#pragma once

#include <vector>
#include "sankhya/types.hpp"

namespace sankhya {
namespace milp {

/**
 * @brief Maintains historical pseudocost statistics for reliability branching (Step 19.1)
 */
class PseudocostTracker {
public:
    explicit PseudocostTracker(Index num_vars);

    void record_up_observation(Index var_idx, Float delta_obj, Float fractional_distance);
    void record_down_observation(Index var_idx, Float delta_obj, Float fractional_distance);

    Float get_up_pseudocost(Index var_idx) const;
    Float get_down_pseudocost(Index var_idx) const;

    Index size() const { return static_cast<Index>(stats_.size()); }

private:
    struct VarStats {
        Float sum_p_up = 0.0;
        Index count_up = 0;
        Float sum_p_down = 0.0;
        Index count_down = 0;
    };
    std::vector<VarStats> stats_;

    // Global tracking to provide robust unobserved defaults
    Float global_sum_p_up_ = 0.0;
    Index global_count_up_ = 0;
    Float global_sum_p_down_ = 0.0;
    Index global_count_down_ = 0;

    // Default value when no observations exist globally
    static constexpr Float kUninitializedDefault = 1.0;
};

/**
 * @brief Selects the best fractional integer variable for branching
 * 
 * @param x_frac Values of integer variables to consider. Variables that are
 *               already integral (within tolerance) are skipped.
 * @param tracker The historical pseudocost tracker.
 * @return Index of the chosen variable, or -1 if no candidates exist.
 */
Index select_branching_variable(
    const std::vector<Float>& x_frac,
    const PseudocostTracker& tracker
);

} // namespace milp
} // namespace sankhya
