#include "cuts/cut_pool.hpp"
#include <stdexcept>

namespace sankhya {
namespace cuts {

void CutPool::add_cut_to_pool(const Cut& cut) {
    if (cut.coefficients.empty()) {
        throw std::invalid_argument("Cannot add a cut with empty coefficients");
    }
    pool_.push_back(cut);
}

std::vector<Cut> CutPool::get_violated_cuts(const std::vector<Float>& x) const {
    if (x.empty()) {
        throw std::invalid_argument("Solution vector x is empty");
    }

    std::vector<Cut> violated_cuts;

    for (const auto& cut : pool_) {
        if (cut.coefficients.size() != x.size()) {
            throw std::invalid_argument(
                "Dimension mismatch: cut coefficient size differs from solution vector size");
        }

        Float lhs = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            lhs += cut.coefficients[i] * x[i];
        }

        // A cut is violated when lhs > rhs + tolerance
        if (lhs > cut.rhs + math::kDefaultFeasibilityTol) {
            violated_cuts.push_back(cut);
        }
    }

    return violated_cuts;
}

} // namespace cuts
} // namespace sankhya
