#include "parsers/mps_reader.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace sankhya {
namespace parsers {

namespace {

enum class Section {
    NONE,
    NAME,
    ROWS,
    COLUMNS,
    RHS,
    BOUNDS
};

struct RowInfo {
    char type; // 'N', 'L', 'G', 'E'
    Index index;
    Float rhs;
    bool has_rhs;
    std::vector<Index> cols;
    std::vector<Float> vals;
    Index slack_var_index; // -1 if none
};

struct ColInfo {
    std::string name;
    Index index;
    Float obj_coeff;
    Float lb;
    Float ub;
    VariableType vtype;
    bool has_lb;
    bool has_ub;
};

std::vector<std::string> split_whitespace(const std::string& line) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : line) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        } else {
            token += c;
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}

} // anonymous namespace

core::Model read_mps(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("MPS parser error: Could not open file " + filename);
    }

    Section current_section = Section::NONE;
    
    std::vector<RowInfo> rows;
    std::unordered_map<std::string, Index> row_map;
    Index obj_row_index = -1;
    
    std::vector<ColInfo> cols;
    std::unordered_map<std::string, Index> col_map;
    
    bool in_integer_block = false;
    std::string active_rhs_name = "";
    std::string active_bound_name = "";

    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        line_number++;
        
        // Remove trailing carriage return if any
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Handle empty lines or full comment lines
        if (line.empty() || line[0] == '*') {
            continue;
        }

        // Check if section header
        if (!std::isspace(static_cast<unsigned char>(line[0]))) {
            auto tokens = split_whitespace(line);
            if (tokens.empty()) continue;
            
            std::string header = tokens[0];
            if (header == "NAME") {
                current_section = Section::NAME;
            } else if (header == "ROWS") {
                current_section = Section::ROWS;
            } else if (header == "COLUMNS") {
                current_section = Section::COLUMNS;
            } else if (header == "RHS") {
                current_section = Section::RHS;
            } else if (header == "BOUNDS") {
                current_section = Section::BOUNDS;
            } else if (header == "ENDATA") {
                break;
            } else {
                // Ignore unknown sections safely (e.g., RANGES, SOS) as required, 
                // but actually the prompt says: "If sections outside the supported 
                // set appear, handle them explicitly... safely skip only when documented".
                // Since RANGES/SOS are unsupported, we throw.
                throw std::runtime_error("MPS parser error: Unsupported section '" + header + "' at line " + std::to_string(line_number));
            }
            continue;
        }

        // Data record
        auto tokens = split_whitespace(line);
        if (tokens.empty()) continue;

        if (current_section == Section::ROWS) {
            if (tokens.size() < 2) {
                throw std::runtime_error("MPS parser error: Malformed ROWS record at line " + std::to_string(line_number));
            }
            char type = tokens[0][0];
            std::string name = tokens[1];
            
            if (type != 'N' && type != 'L' && type != 'G' && type != 'E') {
                throw std::runtime_error("MPS parser error: Unknown row type '" + std::string(1, type) + "' at line " + std::to_string(line_number));
            }
            
            if (type == 'N' && obj_row_index == -1) {
                obj_row_index = static_cast<Index>(rows.size());
            }
            
            row_map[name] = static_cast<Index>(rows.size());
            rows.push_back({type, static_cast<Index>(rows.size()), 0.0, false, {}, {}, -1});
            
        } else if (current_section == Section::COLUMNS) {
            if (tokens.size() < 3) {
                throw std::runtime_error("MPS parser error: Malformed COLUMNS record at line " + std::to_string(line_number));
            }
            std::string col_name = tokens[0];
            
            // Check for integer markers
            if (tokens.size() >= 3 && tokens[1] == "'MARKER'") {
                std::string marker = tokens[2];
                if (marker == "'INTORG'") {
                    in_integer_block = true;
                } else if (marker == "'INTEND'") {
                    in_integer_block = false;
                }
                continue;
            }
            
            // Create column if it doesn't exist
            if (col_map.find(col_name) == col_map.end()) {
                col_map[col_name] = static_cast<Index>(cols.size());
                cols.push_back({
                    col_name,
                    static_cast<Index>(cols.size()),
                    0.0, 0.0, math::kInfinity,
                    in_integer_block ? VariableType::Integer : VariableType::Continuous,
                    false, false
                });
            }
            Index c_idx = col_map[col_name];
            
            // Parse row/value pairs
            for (size_t i = 1; i < tokens.size(); i += 2) {
                if (i + 1 >= tokens.size()) {
                    throw std::runtime_error("MPS parser error: Missing value for row in COLUMNS at line " + std::to_string(line_number));
                }
                std::string r_name = tokens[i];
                Float val;
                try {
                    val = std::stod(tokens[i+1]);
                } catch (...) {
                    throw std::runtime_error("MPS parser error: Invalid numeric value in COLUMNS at line " + std::to_string(line_number));
                }
                
                if (row_map.find(r_name) == row_map.end()) {
                    throw std::runtime_error("MPS parser error: Reference to unknown row '" + r_name + "' in COLUMNS");
                }
                Index r_idx = row_map[r_name];
                
                if (r_idx == obj_row_index) {
                    cols[static_cast<std::size_t>(c_idx)].obj_coeff = val;
                } else {
                    rows[static_cast<std::size_t>(r_idx)].cols.push_back(c_idx);
                    rows[static_cast<std::size_t>(r_idx)].vals.push_back(val);
                }
            }
            
        } else if (current_section == Section::RHS) {
            if (tokens.size() < 3) {
                throw std::runtime_error("MPS parser error: Malformed RHS record at line " + std::to_string(line_number));
            }
            std::string rhs_name = tokens[0];
            if (active_rhs_name.empty()) {
                active_rhs_name = rhs_name;
            } else if (rhs_name != active_rhs_name) {
                // Ignore other RHS vectors safely
                continue;
            }
            
            for (size_t i = 1; i < tokens.size(); i += 2) {
                if (i + 1 >= tokens.size()) {
                    throw std::runtime_error("MPS parser error: Missing value for row in RHS at line " + std::to_string(line_number));
                }
                std::string r_name = tokens[i];
                Float val;
                try {
                    val = std::stod(tokens[i+1]);
                } catch (...) {
                    throw std::runtime_error("MPS parser error: Invalid numeric value in RHS at line " + std::to_string(line_number));
                }
                
                if (row_map.find(r_name) == row_map.end()) {
                    throw std::runtime_error("MPS parser error: Reference to unknown row '" + r_name + "' in RHS");
                }
                Index r_idx = row_map[r_name];
                rows[static_cast<std::size_t>(r_idx)].rhs = val;
                rows[static_cast<std::size_t>(r_idx)].has_rhs = true;
            }
            
        } else if (current_section == Section::BOUNDS) {
            if (tokens.size() < 3) {
                throw std::runtime_error("MPS parser error: Malformed BOUNDS record at line " + std::to_string(line_number));
            }
            std::string bound_type = tokens[0];
            std::string bound_name = tokens[1];
            std::string col_name = tokens[2];
            
            if (active_bound_name.empty()) {
                active_bound_name = bound_name;
            } else if (bound_name != active_bound_name) {
                // Ignore other BOUNDS vectors safely
                continue;
            }
            
            if (col_map.find(col_name) == col_map.end()) {
                throw std::runtime_error("MPS parser error: Reference to unknown column '" + col_name + "' in BOUNDS");
            }
            Index c_idx = col_map[col_name];
            ColInfo& col = cols[static_cast<std::size_t>(c_idx)];
            
            if (bound_type == "LO") {
                if (tokens.size() < 4) throw std::runtime_error("MPS parser error: LO bound missing value");
                col.lb = std::stod(tokens[3]);
                col.has_lb = true;
            } else if (bound_type == "UP") {
                if (tokens.size() < 4) throw std::runtime_error("MPS parser error: UP bound missing value");
                col.ub = std::stod(tokens[3]);
                col.has_ub = true;
            } else if (bound_type == "FX") {
                if (tokens.size() < 4) throw std::runtime_error("MPS parser error: FX bound missing value");
                col.lb = col.ub = std::stod(tokens[3]);
                col.has_lb = col.has_ub = true;
            } else if (bound_type == "FR") {
                col.lb = -math::kInfinity;
                col.ub = math::kInfinity;
                col.has_lb = col.has_ub = true;
            } else if (bound_type == "MI") {
                col.lb = -math::kInfinity;
                col.has_lb = true;
            } else if (bound_type == "PL") {
                col.ub = math::kInfinity;
                col.has_ub = true;
            } else if (bound_type == "BV") {
                col.vtype = VariableType::Binary;
                col.lb = 0.0;
                col.ub = 1.0;
                col.has_lb = col.has_ub = true;
            } else if (bound_type == "LI") {
                if (tokens.size() < 4) throw std::runtime_error("MPS parser error: LI bound missing value");
                col.vtype = VariableType::Integer;
                col.lb = std::stod(tokens[3]);
                col.has_lb = true;
            } else if (bound_type == "UI") {
                if (tokens.size() < 4) throw std::runtime_error("MPS parser error: UI bound missing value");
                col.vtype = VariableType::Integer;
                col.ub = std::stod(tokens[3]);
                col.has_ub = true;
            } else {
                throw std::runtime_error("MPS parser error: Unsupported bound type '" + bound_type + "'");
            }
        }
    }

    // Assembly into core::Model
    core::Model model;
    
    // 1. Add all structural variables
    for (const auto& c : cols) {
        model.add_variable(c.obj_coeff, c.lb, c.ub, c.vtype);
    }
    
    // 2. Process rows and determine slack variables
    Index current_var_count = static_cast<Index>(cols.size());
    for (auto& r : rows) {
        if (r.type == 'N') continue; // Skip objective / free rows
        
        if (r.type == 'L') {
            // Ax + s = b  => s >= 0
            r.slack_var_index = current_var_count++;
            model.add_variable(0.0, 0.0, math::kInfinity, VariableType::Continuous);
        } else if (r.type == 'G') {
            // Ax - s = b  => s >= 0, so coeff is -1
            r.slack_var_index = current_var_count++;
            model.add_variable(0.0, 0.0, math::kInfinity, VariableType::Continuous);
        } else if (r.type == 'E') {
            // No slack needed
        }
    }
    
    // 3. Add constraints
    for (const auto& r : rows) {
        if (r.type == 'N') {
            if (r.has_rhs && r.rhs != 0.0) {
                throw std::runtime_error("MPS parser error: Objective row RHS (constant offset) is not supported by the current Model API.");
            }
            continue; // Handled as objective
        }
        
        std::vector<Index> constraint_cols = r.cols;
        std::vector<Float> constraint_vals = r.vals;
        
        if (r.type == 'L') {
            constraint_cols.push_back(r.slack_var_index);
            constraint_vals.push_back(1.0);
        } else if (r.type == 'G') {
            constraint_cols.push_back(r.slack_var_index);
            constraint_vals.push_back(-1.0);
        }
        
        model.add_constraint(constraint_cols, constraint_vals, r.rhs);
    }
    
    model.sense = OptimizationSense::Minimize; // MPS standard is usually Minimize
    model.finalize();
    
    return model;
}

} // namespace parsers
} // namespace sankhya
