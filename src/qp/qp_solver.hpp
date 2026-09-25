/**
 * @file qp_solver.hpp
 * @brief Quadratic Program solver interface for SANKHYA / VAJRA-OPT
 */

#pragma once

#include <vector>
#include "sankhya/types.hpp"
#include "qp/qp_model.hpp"

namespace sankhya {
namespace qp {

/**
 * @brief Termination status of the QP solver
 */
enum class QPSolverStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    NumericalFailure,
    UnsupportedNonconvex,
    InvalidModel
};

/**
 * @brief Detailed result from QP solver
 */
struct QPResult {
    QPSolverStatus status = QPSolverStatus::InvalidModel;
    Float objective_value = 0.0;
    Index iterations = 0;
    Float primal_residual = 0.0;
    Float dual_residual = 0.0;
    Float complementarity = 0.0;
};

/**
 * @brief Solver interface for Convex Quadratic Programs
 *
 * Implements the mathematical contract defined in ADR-QP-001.
 * Uses the normal-equations reduction of the QP Newton system.
 * This class does NOT wrap external solvers.
 */
class QPSolver {
public:
    struct Options {
        Float feasibility_tol = math::kDefaultFeasibilityTol;
        Float dual_tol = math::kDefaultFeasibilityTol;
        Float grad_tol = 1e-6;
        Index max_iterations = 200;
    };

    QPSolver() = default;
    explicit QPSolver(const Options& options) : options_(options) {}

    QPSolverStatus solve(
        const QPModel& qp,
        std::vector<Float>& x,
        std::vector<Float>& y,
        std::vector<Float>& z
    );

    QPResult solve_with_diagnostics(
        const QPModel& qp,
        std::vector<Float>& x,
        std::vector<Float>& y,
        std::vector<Float>& z
    );

private:
    Options options_;
};

} // namespace qp
} // namespace sankhya
