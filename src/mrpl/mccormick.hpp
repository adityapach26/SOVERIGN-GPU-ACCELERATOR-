/**
 * @file mccormick.hpp
 * @brief MRPL Bilinear McCormick Envelope Relaxation (Phase 25.1)
 *
 * SOURCE CLASSIFICATION: [A] SOURCE-SPECIFIED
 *
 * Generates the convex polyhedral relaxation for a bilinear pooling term w = x * y,
 * as required by the MRPL crude-blending / pooling specialization.
 *
 * For bounds xL <= x <= xU and yL <= y <= yU, the four classical McCormick
 * inequalities bound the bilinear product w = x*y from above and below.
 *
 * The constraints are returned as sankhya::cuts::Cut objects (a^T x <= rhs),
 * preserving full compatibility with the existing cut-pool infrastructure.
 * The variable ordering in the coefficient vector is always:
 *   [ x_coeff, y_coeff, w_coeff ]
 */

#pragma once

#include "sankhya/types.hpp"
#include "cuts/cut_pool.hpp"

#include <array>

namespace sankhya {
namespace mrpl {

/**
 * @brief Generate the four McCormick envelope inequalities for w = x * y.
 *
 * Given bounds [xL, xU] and [yL, yU], returns four linear inequality cuts
 * in the form a^T [x, y, w] <= rhs, expressed as sankhya::cuts::Cut.
 *
 * Variable ordering in each cut's coefficient vector:
 *   index 0 = x coefficient
 *   index 1 = y coefficient
 *   index 2 = w coefficient
 *
 * The four generated inequalities (before converting to <= form) are:
 *   [LB1] w >= xL*y + yL*x - xL*yL   →   yL*x + xL*y - w <=  xL*yL
 *   [LB2] w >= xU*y + yU*x - xU*yU   →   yU*x + xU*y - w <=  xU*yU
 *   [UB1] w <= xU*y + yL*x - xU*yL   →  -yL*x - xU*y + w <= -xU*yL
 *   [UB2] w <= yU*x + xL*y - xL*yU   →  -yU*x - xL*y + w <= -xL*yU
 *
 * @param xL  Lower bound on x.
 * @param xU  Upper bound on x.
 * @param yL  Lower bound on y.
 * @param yU  Upper bound on y.
 * @return    Array of exactly 4 cuts, ordered [LB1, LB2, UB1, UB2].
 */
std::array<cuts::Cut, 4> mccormick_envelopes(Float xL, Float xU, Float yL, Float yU);

} // namespace mrpl
} // namespace sankhya
