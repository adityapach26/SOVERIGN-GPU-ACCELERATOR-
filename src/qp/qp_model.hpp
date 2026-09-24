/**
 * @file qp_model.hpp
 * @brief Quadratic Program data model for SANKHYA / VAJRA-OPT
 */

#pragma once

#include <stdexcept>
#include <vector>

#include "sankhya/types.hpp"
#include "core/problem.hpp"
#include "core/sparse_matrix.hpp"

namespace sankhya {
namespace qp {

/**
 * @brief Convexity status of the Quadratic Program
 */
enum class QPConvexityStatus {
    Unknown,        // Convexity has not been checked
    ConvexPSD,      // H is positive semidefinite (Supported)
    Nonconvex       // H is indefinite or negative definite (Unsupported)
};

/**
 * @brief Quadratic Program Model
 *
 * Canonical representation:
 *   minimize    1/2 x^T H x + c^T x
 *   subject to  A x = b
 *               lb <= x <= ub
 *
 * H must be symmetric. The implementation expects a core::CSCMatrix 
 * containing only the lower triangular elements (including the diagonal).
 */
class QPModel {
public:
    core::Model lp_part;
    core::CSCMatrix H;
    QPConvexityStatus convexity = QPConvexityStatus::Unknown;

    QPModel() : H(0, 0, {}, {}, {0}) {}

    /**
     * @brief Validates dimensional and structural consistency of the QP.
     * @throws std::invalid_argument if dimensions mismatch, bounds are invalid,
     *         sense is not minimize, or if explicitly marked nonconvex.
     */
    void validate() const {
        if (lp_part.sense != OptimizationSense::Minimize) {
            throw std::invalid_argument("QPModel: Only Minimization is supported for QP canonical form.");
        }

        if (H.rows != lp_part.A.cols || H.cols != lp_part.A.cols) {
            throw std::invalid_argument("QPModel: Hessian dimensions must match number of variables.");
        }

        if (lp_part.lb.size() != static_cast<std::size_t>(lp_part.A.cols) || 
            lp_part.ub.size() != static_cast<std::size_t>(lp_part.A.cols)) {
            throw std::invalid_argument("QPModel: Bounds arrays size must match number of variables.");
        }

        for (std::size_t i = 0; i < lp_part.lb.size(); ++i) {
            if (lp_part.lb[i] > lp_part.ub[i]) {
                throw std::invalid_argument("QPModel: lower bound cannot be strictly greater than upper bound.");
            }
        }

        if (convexity == QPConvexityStatus::Nonconvex) {
            throw std::invalid_argument("QPModel: Nonconvex (indefinite/negative definite) QP is explicitly unsupported.");
        }
    }
};

} // namespace qp
} // namespace sankhya

