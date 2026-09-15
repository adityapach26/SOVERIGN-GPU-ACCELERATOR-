#pragma once

#include <cstdint>
#include <limits>

namespace sankhya {

using Float = double;
using Index = int32_t;

enum class VariableType { Continuous, Integer, Binary };
enum class OptimizationSense { Minimize, Maximize };

namespace math {

constexpr Float kInfinity = std::numeric_limits<Float>::infinity();
constexpr Float kDefaultFeasibilityTol = 1e-6;
constexpr Float kDefaultPivotTol = 1e-7;

} // namespace math

} // namespace sankhya
