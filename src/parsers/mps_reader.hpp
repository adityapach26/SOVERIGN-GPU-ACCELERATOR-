#pragma once

#include <string>
#include "core/problem.hpp"

namespace sankhya {
namespace parsers {

/**
 * @brief Reads a standard MPS formulation into the internal Model representation.
 *
 * Supports free-format MPS parsing of the NAME, ROWS, COLUMNS, RHS, and BOUNDS
 * sections. Handles integer markers ('INTORG', 'INTEND') and translates L/G rows 
 * into equality constraints with corresponding slack/surplus variables.
 *
 * @param filename The path to the .mps file.
 * @return A populated, finalized core::Model.
 * @throws std::runtime_error on parsing failures or unsupported MPS features.
 */
core::Model read_mps(const std::string& filename);

} // namespace parsers
} // namespace sankhya
