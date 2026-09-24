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
 * @brief Solver interface for Convex Quadratic Programs
 *
 * Implements the mathematical contract defined in ADR-QP-001.
 * This class establishes the API boundary and does NOT wrap external solvers.
 */
class QPSolver {
public:
    struct Options {
        Float feasibility_tol = math::kDefaultFeasibilityTol;
        Float dual_tol = math::kDefaultFeasibilityTol;
        Float grad_tol = 1e-6;
        Index max_iterations = 200;
    };

    QPSolver(const Options& options = Options{}) : options_(options) {}

    /**
     * @brief Solves the given Quadratic Program
     *
     * @param qp The quadratic program model
     * @param x  On exit, the primal solution vector
     * @param y  On exit, the dual solution vector for equality constraints
     * @param z  On exit, the dual solution vector for variable bounds
     * @return The solver termination status
     */
    QPSolverStatus solve(
        const QPModel& qp,
        std::vector<Float>& x,
        std::vector<Float>& y,
        std::vector<Float>& z
    ) {
        try {
            qp.validate();
        } catch (const std::invalid_argument&) {
            if (qp.convexity == QPConvexityStatus::Nonconvex) {
                return QPSolverStatus::UnsupportedNonconvex;
            }
            return QPSolverStatus::InvalidModel;
        }
        
        if (qp.convexity == QPConvexityStatus::Unknown) {
            // Note: In 16A.2, eigenvalue decomposition or LDLT inertia
            // would dynamically check PSD status here.
            // For 16A.1, if it's unknown we just return unsupported as a strict contract.
            return QPSolverStatus::UnsupportedNonconvex;
        }

        // Implementation of CPU KKT assembly and factorization
        // is deferred to Step 16A.2.
        
        return QPSolverStatus::UnsupportedNonconvex;
    }

private:
    Options options_;
};

} // namespace qp
} // namespace sankhya
