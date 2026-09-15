/**
 * @file basis.hpp
 * @brief Basis status representation for the SANKHYA / VAJRA-OPT simplex core
 *
 * This step defines only the structural representation of a basis:
 * a per-variable status vector plus an explicit list of basic-variable
 * indices. It intentionally carries no validation, construction, or
 * initialization logic.
 */

#pragma once

#include <vector>

#include "sankhya/types.hpp"

namespace sankhya {
namespace simplex {

/**
 * @brief Status of each variable with respect to the basis
 */
enum class BasisStatus {
    Basic,
    AtLower,
    AtUpper,
    Superbasic
};

/**
 * @brief A basis of an LP in the form min/max c^T x s.t. A x = b.
 *
 *  - col_status: structural status (Basic / AtLower / AtUpper / Superbasic)
 *    of every variable.
 *  - basic_indices: indices of the basic variables; the i-th entry is the
 *    basic variable associated with row i.
 *
 * Structural invariant (not enforced here): exactly m variables are Basic,
 * where m is the number of constraints.
 */
struct Basis {
    std::vector<BasisStatus> col_status;
    std::vector<Index> basic_indices;
};

} // namespace simplex
} // namespace sankhya