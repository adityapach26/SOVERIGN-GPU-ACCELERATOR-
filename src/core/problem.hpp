/**
 * @file problem.hpp
 * @brief Linear Program data model for SANKHYA / VAJRA-OPT
 *
 * Represents an LP in equality form:
 *
 *   min/max  c^T x
 *   s.t.     A x = b
 *            lb <= x <= ub
 *
 * with variable types according to VariableType.
 */

#pragma once

#include <stdexcept>
#include <vector>

#include "sankhya/types.hpp"
#include "core/sparse_matrix.hpp"

namespace sankhya {
namespace core {

/**
 * @brief Linear Program data model in canonical equality form
 *
 * The model is populated with add_variable / add_constraint and then converted
 * to its CSC constraint matrix A via finalize(). Dimensional consistency is
 * the responsibility of the caller and is validated at finalize time.
 */
class Model {
public:
    OptimizationSense sense;
    std::vector<Float> obj;
    std::vector<Float> lb;
    std::vector<Float> ub;
    std::vector<VariableType> vtype;
    std::vector<Float> rhs;
    CSCMatrix A;

    // Presolve equilibrium scaling factors (Step 8.1)
    // A_scaled = D_r * A * D_c, where D_r = diag(row_scale), D_c = diag(col_scale).
    // Original x = D_c * x_scaled.
    // If empty, it implies the model is not scaled.
    std::vector<Float> row_scale;
    std::vector<Float> col_scale;

    Model();

    /**
     * @brief Append a variable to the model
     * @param obj_coeff objective coefficient c_j
     * @param lower     lower bound (default 0)
     * @param upper     upper bound (default +infinity)
     * @param vt        variable type (default Continuous)
     */
    void add_variable(Float obj_coeff,
                      Float lower = 0.0,
                      Float upper = math::kInfinity,
                      VariableType vt = VariableType::Continuous);

    /**
     * @brief Append an equality constraint row  (sum_j a_j x_j = b)
     * @param cols variable indices of the non-zero coefficients
     * @param vals coefficients, one per entry in cols
     * @param b    right-hand side value
     */
    void add_constraint(const std::vector<Index>& cols,
                        const std::vector<Float>& vals,
                        Float b);

    /**
     * @brief Materialize the accumulated constraints into the CSC matrix A
     *
     * Validates that the objective/bounds/type vectors are mutually consistent
     * in size, then builds A in compressed sparse column form, preserving the
     * exact (row, column, value) of every stored coefficient.
     *
     * Throws std::invalid_argument on malformed dimensional state.
     */
    void finalize();

private:
    // Accumulated sparse rows awaiting CSC construction.
    std::vector<std::vector<Index>> constraint_cols_;
    std::vector<std::vector<Float>> constraint_vals_;
};

} // namespace core
} // namespace sankhya