#include "gpu/device_dual_simplex.cuh"
#include "gpu/device_ftran.cuh"
#include <cmath>

namespace sankhya {
namespace gpu {

// Engineering Decision: using repository standard tolerances.
__device__ const Float kDefaultFeasibilityTol = 1e-7;
__device__ const Float kDefaultPivotTol = 1e-7;

__global__ void dual_simplex_kernel(
    DeviceModel model,
    DeviceSparseLU lu,
    WorkingBasisState ws,
    Float obj_sign,
    Index max_iterations,
    DeviceSimplexStatus* status_out,
    Index* iter_count_out
) {
    if (blockIdx.x != 0) return;

    __shared__ Index p_shared;
    __shared__ Index q_shared;
    __shared__ Float theta_shared;
    __shared__ DeviceSimplexStatus status_shared;

    if (threadIdx.x == 0) {
        status_shared = DeviceSimplexStatus::IterationLimit;
    }
    __syncthreads();

    for (Index iter = 0; iter < max_iterations; ++iter) {
        
        // -------------------------------------------------------------
        // 1. Leaving variable selection (Primal infeasibility)
        // -------------------------------------------------------------
        if (threadIdx.x == 0) {
            Index p = -1;
            Float min_xb = 0.0;

            for (Index i = 0; i < ws.m; ++i) {
                Float xb_i = ws.x[ws.basis_indices[i]];
                if (xb_i < -kDefaultFeasibilityTol) {
                    if (p == -1 || xb_i < min_xb) {
                        p = i;
                        min_xb = xb_i;
                    }
                }
            }
            p_shared = p;
            
            if (p == -1) {
                status_shared = DeviceSimplexStatus::Optimal;
            }
        }
        __syncthreads();

        if (status_shared == DeviceSimplexStatus::Optimal) {
            break;
        }

        // -------------------------------------------------------------
        // 2. BTRAN: solve B^T d_pi = e_p
        // -------------------------------------------------------------
        for (Index i = threadIdx.x; i < ws.m; i += blockDim.x) {
            ws.d_pi[i] = (i == p_shared) ? 1.0 : 0.0;
        }
        __syncthreads();

        device_btran(lu, ws, ws.d_pi);

        // Also compute pi = B^-T c_B for reduced costs
        for (Index i = threadIdx.x; i < ws.m; i += blockDim.x) {
            Index bi = ws.basis_indices[i];
            ws.pi[i] = obj_sign * model.obj[bi];
        }
        __syncthreads();

        device_btran(lu, ws, ws.pi);

        // -------------------------------------------------------------
        // 3. Dual ratio pricing (Entering variable selection)
        // -------------------------------------------------------------
        if (threadIdx.x == 0) {
            Index q = -1;
            Float min_theta = 1e30; 

            // [ENGINEERING DECISION] Single-threaded pricing loop
            // Why: Avoids complex block reduction and spinlocks for deterministic tie-breaking. 
            // Ensures mathematical exactness matching CPU dual_simplex_phase2.
            for (Index j = 0; j < model.cols; ++j) {
                if (ws.is_basic[j]) {
                    continue;
                }

                // d_{pi, j} = d_pi^T A_j
                Float d_pi_j = 0.0;
                Index col_start = model.col_ptrs[j];
                Index col_end = model.col_ptrs[j+1];
                for (Index k = col_start; k < col_end; ++k) {
                    d_pi_j += ws.d_pi[model.row_indices[k]] * model.values[k];
                }

                if (d_pi_j < -kDefaultPivotTol) {
                    Float r_j = obj_sign * model.obj[j];
                    for (Index k = col_start; k < col_end; ++k) {
                        r_j -= ws.pi[model.row_indices[k]] * model.values[k];
                    }
                    // Matches CPU exact numerical safety rule from dual.cpp:91
                    if (r_j < 0.0) r_j = 0.0;

                    Float theta_j = r_j / std::abs(d_pi_j);
                    if (theta_j < min_theta) {
                        min_theta = theta_j;
                        q = j;
                    }
                }
            }
            
            q_shared = q;
            if (q == -1) {
                status_shared = DeviceSimplexStatus::Infeasible;
            }
        }
        __syncthreads();

        if (status_shared == DeviceSimplexStatus::Infeasible) {
            break;
        }

        // -------------------------------------------------------------
        // 4. FTRAN: solve B d = A_q
        // -------------------------------------------------------------
        for (Index i = threadIdx.x; i < ws.m; i += blockDim.x) {
            ws.d[i] = 0.0;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            Index col_start = model.col_ptrs[q_shared];
            Index col_end = model.col_ptrs[q_shared+1];
            for (Index k = col_start; k < col_end; ++k) {
                ws.d[model.row_indices[k]] = model.values[k];
            }
        }
        __syncthreads();

        device_ftran(lu, ws, ws.d);

        // -------------------------------------------------------------
        // 5. Update primal variables & Basis state
        // -------------------------------------------------------------
        if (threadIdx.x == 0) {
            Float d_p = ws.d[p_shared];
            
            if (std::abs(d_p) < kDefaultPivotTol) {
                status_shared = DeviceSimplexStatus::NumericalFailure;
            } else {
                theta_shared = ws.x[ws.basis_indices[p_shared]] / d_p;
            }
        }
        __syncthreads();

        if (status_shared == DeviceSimplexStatus::NumericalFailure) {
            break;
        }

        Float theta_primal = theta_shared;
        for (Index i = threadIdx.x; i < ws.m; i += blockDim.x) {
            ws.x[ws.basis_indices[i]] -= theta_primal * ws.d[i];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            Index leaving_var = ws.basis_indices[p_shared];
            Index entering_var = q_shared;

            ws.x[leaving_var] = 0.0;
            ws.is_basic[leaving_var] = false;

            ws.is_basic[entering_var] = true;
            ws.x[entering_var] = theta_primal;
        }
        __syncthreads();

        // 6. Factorization update
        bool update_ok = device_update_basis(ws, p_shared, q_shared, ws.d, kDefaultPivotTol);
        
        if (threadIdx.x == 0) {
            if (!update_ok) {
                status_shared = DeviceSimplexStatus::NumericalFailure;
            }
        }
        __syncthreads();

        if (status_shared == DeviceSimplexStatus::NumericalFailure) {
            break;
        }
    }

    if (threadIdx.x == 0) {
        *status_out = status_shared;
        if (iter_count_out != nullptr) {
            *iter_count_out = iter;
        }
    }
}

} // namespace gpu
} // namespace sankhya

