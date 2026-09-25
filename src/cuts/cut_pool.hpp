#pragma once

#include <vector>
#include "sankhya/types.hpp"

namespace sankhya {
namespace cuts {

/**
 * @brief Represents a single linear inequality cut of the form: a^T x <= rhs
 */
struct Cut {
    std::vector<Float> coefficients;
    Float rhs;
};

/**
 * @brief Global Cut Pool manager (Phase 20.1)
 * 
 * Stores generated linear cuts globally on the CPU and identifies
 * which cuts are violated by a given LP solution vector.
 */
class CutPool {
public:
    /**
     * @brief Adds a linear cut to the pool
     */
    void add_cut_to_pool(const Cut& cut);

    /**
     * @brief Identifies and returns all stored cuts violated by the solution x
     * 
     * A cut a^T x <= rhs is violated if (a^T x) > rhs + tolerance,
     * using sankhya::math::kDefaultFeasibilityTol.
     * 
     * @param x LP solution vector
     * @return std::vector<Cut> List of violated cuts
     */
    std::vector<Cut> get_violated_cuts(const std::vector<Float>& x) const;

private:
    std::vector<Cut> pool_;
};

} // namespace cuts
} // namespace sankhya
