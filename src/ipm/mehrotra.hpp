#pragma once

#include "core/problem.hpp"
#include "sankhya/types.hpp"
#include "simplex/primal.hpp"

namespace sankhya {
namespace ipm {

struct MehrotraResult {
    simplex::SimplexStatus status;
    Float objective_value;
    Index iterations;
    Float primal_residual;
    Float dual_residual;
    Float duality_gap;
};

class MehrotraSolver {
public:
    MehrotraSolver(const core::Model& model);
    ~MehrotraSolver();

    MehrotraResult solve();

private:
    class Impl;
    Impl* impl_;
};

} // namespace ipm
} // namespace sankhya