/**
 * @file scaling.hpp
 * @brief Sovereign Presolve Pipeline - Ruiz Equilibrium Scaling
 *
 * Implements matrix scaling to reduce coefficient magnitude imbalance 
 * and improve conditioning before factorization.
 */

#pragma once

#include "core/problem.hpp"

namespace sankhya {
namespace presolve {

/**
 * @brief Apply Ruiz equilibrium scaling to the optimization model.
 * 
 * Iteratively scales rows and columns using infinity norms to move them towards 1.0.
 * Updates the model's constraint matrix, RHS, objective, and bounds, while
 * preserving scaling factors in the model to allow unscaling later.
 *
 * @param model          The finalized optimization model to scale in-place.
 * @param max_iterations Maximum number of Ruiz iterations to perform (default: 10).
 */
void apply_ruiz_scaling(core::Model& model, int max_iterations = 10);

} // namespace presolve
} // namespace sankhya

