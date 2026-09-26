#include "sankhya/sankhya.h"
#include "core/problem.hpp"
#include "simplex/primal.hpp"
#include "simplex/basis.hpp"
#include "numerics/sparse_lu.hpp"
#include <string>
#include <vector>
#include <memory>

struct sankhya_env_s {
    // Empty for now
};

struct sankhya_model_s {
    sankhya::core::Model model;
    sankhya::simplex::Basis basis;
    std::vector<sankhya::Float> x;
    sankhya::simplex::SimplexStatus status;
    bool has_solution = false;
};

extern "C" {

int sankhya_create_env(sankhya_env_t** env) {
    if (!env) return SANKHYA_ERROR_INVALID_ARGUMENT;
    try {
        *env = new sankhya_env_t();
        return SANKHYA_SUCCESS;
    } catch (...) {
        return SANKHYA_ERROR_INTERNAL;
    }
}

int sankhya_load_model(sankhya_env_t* env, const char* filename, sankhya_model_t** model) {
    if (!env || !filename || !model) return SANKHYA_ERROR_INVALID_ARGUMENT;
    
    try {
        auto* m = new sankhya_model_s();
        std::string fname(filename);
        
        if (fname == "fixture") {
            // Trivial LP: min -x1 - x2 s.t. x1 + s1 = 1, x2 + s2 = 1, x, s >= 0
            m->model.sense = sankhya::OptimizationSense::Minimize;
            
            m->model.add_variable(-1.0, 0.0, sankhya::math::kInfinity, sankhya::VariableType::Continuous);
            m->model.add_variable(-1.0, 0.0, sankhya::math::kInfinity, sankhya::VariableType::Continuous);
            m->model.add_variable(0.0, 0.0, sankhya::math::kInfinity, sankhya::VariableType::Continuous);
            m->model.add_variable(0.0, 0.0, sankhya::math::kInfinity, sankhya::VariableType::Continuous);
            
            m->model.add_constraint({0, 2}, {1.0, 1.0}, 1.0);
            m->model.add_constraint({1, 3}, {1.0, 1.0}, 1.0);
            
            m->model.finalize();
            
            m->basis.col_status.resize(4, sankhya::simplex::BasisStatus::AtLower);
            m->basis.basic_indices = {2, 3}; // slacks are basic
            m->basis.col_status[2] = sankhya::simplex::BasisStatus::Basic;
            m->basis.col_status[3] = sankhya::simplex::BasisStatus::Basic;
            
            m->x.resize(4, 0.0);
            m->x[2] = 1.0;
            m->x[3] = 1.0;
            
            *model = m;
            return SANKHYA_SUCCESS;
        }
        
        delete m;
        return SANKHYA_ERROR_LOAD_FAILED;
        
    } catch (...) {
        return SANKHYA_ERROR_INTERNAL;
    }
}

int sankhya_solve(sankhya_model_t* model) {
    if (!model) return SANKHYA_ERROR_INVALID_ARGUMENT;
    try {
        sankhya::numerics::SparseLUFactorization factorizer;
        model->status = sankhya::simplex::primal_simplex_phase2(
            model->model, model->basis, model->x, factorizer
        );
        
        if (model->status == sankhya::simplex::SimplexStatus::Optimal) {
            model->has_solution = true;
            return SANKHYA_SUCCESS;
        } else {
            return SANKHYA_ERROR_SOLVE_FAILED;
        }
    } catch (...) {
        return SANKHYA_ERROR_INTERNAL;
    }
}

int sankhya_get_objval(sankhya_model_t* model, double* objval) {
    if (!model || !objval) return SANKHYA_ERROR_INVALID_ARGUMENT;
    if (!model->has_solution) return SANKHYA_ERROR_NOT_AVAILABLE;
    
    try {
        double val = 0.0;
        for (size_t i = 0; i < model->model.obj.size(); ++i) {
            val += static_cast<double>(model->model.obj[i] * model->x[i]);
        }
        *objval = val;
        return SANKHYA_SUCCESS;
    } catch (...) {
        return SANKHYA_ERROR_INTERNAL;
    }
}

void sankhya_free_model(sankhya_model_t* model) {
    if (model) delete model;
}

void sankhya_free_env(sankhya_env_t* env) {
    if (env) delete env;
}

} // extern "C"
