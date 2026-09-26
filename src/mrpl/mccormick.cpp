/**
 * @file mccormick.cpp
 * @brief MRPL Bilinear McCormick Envelope Relaxation implementation (Phase 25.1)
 *
 * SOURCE CLASSIFICATION: [A] SOURCE-SPECIFIED
 *
 * Implements the four classical McCormick linearization inequalities for the
 * bilinear pooling term w = x * y, used in MRPL crude-blending optimisation.
 *
 * This step does NOT implement a full pooling solver, nonlinear solver,
 * or GPU/CUDA kernel. It is a pure CPU numerics utility.
 */

#include "mrpl/mccormick.hpp"

namespace sankhya {
namespace mrpl {

std::array<cuts::Cut, 4> mccormick_envelopes(Float xL, Float xU, Float yL, Float yU) {
    // -----------------------------------------------------------------------
    // The four McCormick inequalities for w = x*y with
    //   xL <= x <= xU,  yL <= y <= yU
    //
    // All returned as  a^T [x, y, w] <= rhs  (sankhya::cuts::Cut convention).
    // Variable ordering in coefficient vector: [x_coeff, y_coeff, w_coeff].
    //
    // Derivations:
    //
    // [LB1]  w >= xL*y + yL*x - xL*yL
    //        ⟺   yL*x + xL*y - w <= xL*yL
    //
    // [LB2]  w >= xU*y + yU*x - xU*yU
    //        ⟺   yU*x + xU*y - w <= xU*yU
    //
    // [UB1]  w <= xU*y + yL*x - xU*yL
    //        ⟺  -yL*x - xU*y + w <= -xU*yL
    //
    // [UB2]  w <= yU*x + xL*y - xL*yU
    //        ⟺  -yU*x - xL*y + w <= -xL*yU
    // -----------------------------------------------------------------------

    cuts::Cut lb1, lb2, ub1, ub2;

    // [LB1]: yL*x + xL*y - w <= xL*yL
    lb1.coefficients = { yL,  xL, Float{-1} };
    lb1.rhs          = xL * yL;

    // [LB2]: yU*x + xU*y - w <= xU*yU
    lb2.coefficients = { yU,  xU, Float{-1} };
    lb2.rhs          = xU * yU;

    // [UB1]: -yL*x - xU*y + w <= -xU*yL
    ub1.coefficients = { -yL, -xU, Float{1} };
    ub1.rhs          = -(xU * yL);

    // [UB2]: -yU*x - xL*y + w <= -xL*yU
    ub2.coefficients = { -yU, -xL, Float{1} };
    ub2.rhs          = -(xL * yU);

    return { lb1, lb2, ub1, ub2 };
}

} // namespace mrpl
} // namespace sankhya
