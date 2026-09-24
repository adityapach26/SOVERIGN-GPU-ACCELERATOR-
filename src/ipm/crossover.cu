/**
 * @file crossover.cu
 * @brief Phase 16.1: In-VRAM Megiddo Crossover
 */

#include "ipm/crossover.cuh"
#include "gpu/device_ftran.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CHECK_CUDA_CROSSOVER(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in crossover: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

namespace sankhya {
namespace ipm {

// Initialize Identity LU
__global__ void init_lu_kernel(gpu::DeviceSparseLU lu) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < lu.m) {
        lu.L_col_ptrs[i] = 0;
        lu.U_vals[i] = 1.0;
        lu.U_cols[i] = i;
        lu.U_row_ptrs[i] = i;
        lu.perm_row[i] = i;
        lu.perm_col[i] = i;
        lu.inv_perm_row[i] = i;
        lu.inv_perm_col[i] = i;
    }
    if (i == 0) {
        lu.L_col_ptrs[lu.m] = 0;
        lu.U_row_ptrs[lu.m] = lu.m;
    }
}

// Initialize Artificial Basis
__global__ void init_ws_kernel(gpu::WorkingBasisState ws, Index n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < ws.m) {
        ws.basis_indices[i] = n + i; // Artificial variables
    }
    if (i < n) {
        ws.is_basic[i] = false;
    }
}

// Single step of purification for an interior variable
__global__ void purify_step_kernel(
    gpu::DeviceModel model,
    gpu::DeviceSparseLU lu,
    gpu::WorkingBasisState ws,
    Index q,
    Float* d_Aq,
    bool* d_pivoted
) {
    __shared__ bool skip;
    if (threadIdx.x == 0) {
        skip = false;
        *d_pivoted = false;
        Float lb = model.lb[q];
        Float ub = model.ub[q];
        Float xq = ws.x[q];
        if (xq <= lb + 1e-6 || xq >= ub - 1e-6) skip = true;
        if (ws.is_basic[q]) skip = true;
    }
    __syncthreads();
    if (skip) return;

    for (int i = threadIdx.x; i < ws.m; i += blockDim.x) {
        ws.d[i] = 0.0;
        d_Aq[i] = 0.0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        Index start = model.col_ptrs[q];
        Index end = model.col_ptrs[q+1];
        for (Index p = start; p < end; ++p) {
            ws.d[model.row_indices[p]] = model.values[p];
            d_Aq[model.row_indices[p]] = model.values[p];
        }
    }
    __syncthreads();

    gpu::device_ftran(lu, ws, ws.d);

    __shared__ bool pivoted;
    __shared__ Index shared_best_leave;
    __shared__ bool shared_ok;

    if (threadIdx.x == 0) {
        pivoted = false;
        shared_best_leave = -1;
        shared_ok = false;
        Float max_abs_d = 1e-7;

        for (Index i = 0; i < ws.m; ++i) {
            if (ws.basis_indices[i] >= model.cols) {
                Float abs_d = std::abs(ws.d[i]);
                if (abs_d > max_abs_d) {
                    max_abs_d = abs_d;
                    shared_best_leave = i;
                }
            }
        }
    }
    __syncthreads();

    if (shared_best_leave != -1) {
        bool ok = gpu::device_update_basis(ws, shared_best_leave, q, ws.d, 1e-7);
        if (threadIdx.x == 0) {
            shared_ok = ok;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && shared_ok) {
        ws.is_basic[q] = true;
        pivoted = true;
        *d_pivoted = true;
    }
    __syncthreads();

    __shared__ Index try_leave_row_1;
    __shared__ Index try_leave_row_2;
    __shared__ bool ok_1;
    __shared__ bool ok_2;

    // local to thread 0
    Float max_theta_pos = 1e30;
    Index pos_leave = -1;
    Float max_theta_neg = 1e30;
    Index neg_leave = -1;

    if (threadIdx.x == 0 && !pivoted) {
        ok_1 = false;
        ok_2 = false;
        try_leave_row_1 = -1;
        try_leave_row_2 = -1;

        Float dist_ub_q = model.ub[q] - ws.x[q];
        Float dist_lb_q = ws.x[q] - model.lb[q];
        
        if (dist_ub_q < max_theta_pos) { max_theta_pos = dist_ub_q; pos_leave = q; }
        if (dist_lb_q < max_theta_neg) { max_theta_neg = dist_lb_q; neg_leave = q; }
        
        for (Index i = 0; i < ws.m; ++i) {
            Index basic_var = ws.basis_indices[i];
            if (basic_var >= model.cols) continue;
            
            Float v_i = -ws.d[i];
            Float dist_ub = model.ub[basic_var] - ws.x[basic_var];
            Float dist_lb = ws.x[basic_var] - model.lb[basic_var];
            
            if (v_i > 1e-9) {
                Float theta = dist_ub / v_i;
                if (theta < max_theta_pos) { max_theta_pos = theta; pos_leave = basic_var; }
                Float theta_n = dist_lb / v_i;
                if (theta_n < max_theta_neg) { max_theta_neg = theta_n; neg_leave = basic_var; }
            } else if (v_i < -1e-9) {
                Float theta = dist_lb / (-v_i);
                if (theta < max_theta_pos) { max_theta_pos = theta; pos_leave = basic_var; }
                Float theta_n = dist_ub / (-v_i);
                if (theta_n < max_theta_neg) { max_theta_neg = theta_n; neg_leave = basic_var; }
            }
        }
        
        if (pos_leave != -1 && pos_leave != q) {
            Index leave_row = -1;
            for (Index i = 0; i < ws.m; ++i) {
                if (ws.basis_indices[i] == pos_leave) leave_row = i;
            }
            if (leave_row != -1 && std::abs(ws.d[leave_row]) > 1e-7) {
                try_leave_row_1 = leave_row;
            }
        }

        if (neg_leave != -1 && neg_leave != q) {
            Index leave_row = -1;
            for (Index i = 0; i < ws.m; ++i) {
                if (ws.basis_indices[i] == neg_leave) leave_row = i;
            }
            if (leave_row != -1 && std::abs(ws.d[leave_row]) > 1e-7) {
                try_leave_row_2 = leave_row;
            }
        }
    }
    __syncthreads();

    if (!pivoted) {
        bool local_ok_1 = false;
        if (try_leave_row_1 != -1) {
            local_ok_1 = gpu::device_update_basis(ws, try_leave_row_1, q, ws.d, 1e-7);
        }
        if (threadIdx.x == 0) ok_1 = local_ok_1;
        __syncthreads();

        bool local_ok_2 = false;
        if (try_leave_row_2 != -1 && !ok_1) {
            local_ok_2 = gpu::device_update_basis(ws, try_leave_row_2, q, ws.d, 1e-7);
        }
        if (threadIdx.x == 0) ok_2 = local_ok_2;
        __syncthreads();
    }

    if (threadIdx.x == 0 && !pivoted) {
        Float chosen_theta = 0.0;
        Index chosen_leave = -1;
        bool pivot_success = false;

        if (ok_1) {
            chosen_theta = max_theta_pos;
            chosen_leave = pos_leave;
            pivot_success = true;
        } else if (ok_2) {
            chosen_theta = -max_theta_neg;
            chosen_leave = neg_leave;
            pivot_success = true;
        } else if (pos_leave == q || neg_leave == q) {
            if (pos_leave == q) {
                chosen_theta = max_theta_pos;
            } else {
                chosen_theta = -max_theta_neg;
            }
            chosen_leave = q;
            pivot_success = true;
        }

        if (pivot_success) {
            ws.x[q] += chosen_theta;
            for (Index i = 0; i < ws.m; ++i) {
                Index basic_var = ws.basis_indices[i];
                if (basic_var < model.cols) {
                    ws.x[basic_var] += chosen_theta * (-ws.d[i]);
                }
            }
            
            if (chosen_leave != q) {
                ws.is_basic[q] = true;
                ws.is_basic[chosen_leave] = false;
            }
            *d_pivoted = true;
        } else {
            *ws.error_code = 1;
        }
    }
    __syncthreads();
}

__global__ void purify_bound_step_kernel(
    gpu::DeviceModel model,
    gpu::DeviceSparseLU lu,
    gpu::WorkingBasisState ws,
    Index q,
    Float* d_Aq
) {
    __shared__ bool skip;
    if (threadIdx.x == 0) {
        skip = false;
        if (ws.is_basic[q]) skip = true;
    }
    __syncthreads();
    if (skip) return;

    for (int i = threadIdx.x; i < ws.m; i += blockDim.x) {
        ws.d[i] = 0.0;
        d_Aq[i] = 0.0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        Index start = model.col_ptrs[q];
        Index end = model.col_ptrs[q+1];
        for (Index p = start; p < end; ++p) {
            ws.d[model.row_indices[p]] = model.values[p];
            d_Aq[model.row_indices[p]] = model.values[p];
        }
    }
    __syncthreads();

    gpu::device_ftran(lu, ws, ws.d);

    __shared__ Index shared_best_leave;
    __shared__ bool shared_ok;

    if (threadIdx.x == 0) {
        shared_best_leave = -1;
        shared_ok = false;
        Float max_abs_d = 1e-7;
        for (Index i = 0; i < ws.m; ++i) {
            if (ws.basis_indices[i] >= model.cols) {
                Float abs_d = std::abs(ws.d[i]);
                if (abs_d > max_abs_d) {
                    max_abs_d = abs_d;
                    shared_best_leave = i;
                }
            }
        }
    }
    __syncthreads();

    if (shared_best_leave != -1) {
        bool ok = gpu::device_update_basis(ws, shared_best_leave, q, ws.d, 1e-7);
        if (threadIdx.x == 0) {
            shared_ok = ok;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && shared_ok) {
        ws.is_basic[q] = true;
    }
    __syncthreads();
}

simplex::Basis generate_crossover_basis(
    const gpu::DeviceModel& model,
    const std::vector<Float>& x_ipm
) {
    Index m = model.rows;
    Index n = model.cols;

    // Allocate GPU states
    gpu::DeviceSparseLU lu;
    lu.m = m;
    lu.L_nnz = 0; lu.U_nnz = m;
    CHECK_CUDA_CROSSOVER(cudaMalloc(&lu.L_vals, 1));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&lu.L_rows, 1));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&lu.L_col_ptrs, (m + 1) * sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&lu.U_vals, m * sizeof(Float)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&lu.U_cols, m * sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&lu.U_row_ptrs, (m + 1) * sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&lu.perm_row, m * sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&lu.perm_col, m * sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&lu.inv_perm_row, m * sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&lu.inv_perm_col, m * sizeof(Index)));

    gpu::WorkingBasisState ws;
    ws.m = m; ws.num_cols = n;
    ws.eta_col_capacity = 2 * n;
    ws.eta_capacity = 2 * n * m + 10000;
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.lb, n * sizeof(Float)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.ub, n * sizeof(Float)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.error_code, sizeof(int)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.basis_indices, m * sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.eta_vals, ws.eta_capacity * sizeof(Float)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.eta_rows, ws.eta_capacity * sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.eta_col_starts, (ws.eta_col_capacity + 1) * sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.eta_pivot_row, ws.eta_col_capacity * sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.num_eta_cols, sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.eta_nnz, sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.work_vec, m * sizeof(Float)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.aux_vec, m * sizeof(Float)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.x, n * sizeof(Float)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.is_basic, n * sizeof(bool)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.pi, m * sizeof(Float)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.d, m * sizeof(Float)));
    CHECK_CUDA_CROSSOVER(cudaMalloc(&ws.d_pi, m * sizeof(Float)));
    
    CHECK_CUDA_CROSSOVER(cudaMemset(ws.error_code, 0, sizeof(int)));
    CHECK_CUDA_CROSSOVER(cudaMemset(ws.num_eta_cols, 0, sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMemset(ws.eta_nnz, 0, sizeof(Index)));
    CHECK_CUDA_CROSSOVER(cudaMemset(ws.eta_col_starts, 0, sizeof(Index)));

    Float* d_Aq;
    CHECK_CUDA_CROSSOVER(cudaMalloc(&d_Aq, m * sizeof(Float)));
    bool* d_pivoted;
    CHECK_CUDA_CROSSOVER(cudaMalloc(&d_pivoted, sizeof(bool)));

    // Upload x_ipm
    CHECK_CUDA_CROSSOVER(cudaMemcpy(ws.x, x_ipm.data(), n * sizeof(Float), cudaMemcpyHostToDevice));

    int blocks = (std::max(m, n) + 255) / 256;
    init_lu_kernel<<<blocks, 256>>>(lu);
    init_ws_kernel<<<blocks, 256>>>(ws, n);
    cudaDeviceSynchronize();

    int pivot_failed = 0;
    for (Index q = 0; q < n; ++q) {
        bool pivoted_host = false;
        do {
            purify_step_kernel<<<1, 256>>>(model, lu, ws, q, d_Aq, d_pivoted);
            cudaDeviceSynchronize();
            cudaMemcpy(&pivoted_host, d_pivoted, sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pivot_failed, ws.error_code, sizeof(int), cudaMemcpyDeviceToHost);
            if (pivot_failed != 0) break;
        } while (pivoted_host);
        if (pivot_failed != 0) break;
    }

    if (pivot_failed == 0) {
        for (Index q = 0; q < n; ++q) {
            purify_bound_step_kernel<<<1, 256>>>(model, lu, ws, q, d_Aq);
            cudaDeviceSynchronize();
        }
    }

    std::vector<Index> h_basis_indices(m);
    cudaMemcpy(h_basis_indices.data(), ws.basis_indices, m * sizeof(Index), cudaMemcpyDeviceToHost);
    std::vector<Float> h_x(n);
    cudaMemcpy(h_x.data(), ws.x, n * sizeof(Float), cudaMemcpyDeviceToHost);
    std::vector<Float> h_lb(n);
    cudaMemcpy(h_lb.data(), model.lb, n * sizeof(Float), cudaMemcpyDeviceToHost);
    std::vector<Float> h_ub(n);
    cudaMemcpy(h_ub.data(), model.ub, n * sizeof(Float), cudaMemcpyDeviceToHost);

    simplex::Basis basis;
    basis.basic_indices = h_basis_indices;
    basis.col_status.resize(n, simplex::BasisStatus::AtLower);
    
    bool rank_deficient = false;
    for (Index i = 0; i < m; ++i) {
        if (basis.basic_indices[i] < n) {
            basis.col_status[basis.basic_indices[i]] = simplex::BasisStatus::Basic;
        } else {
            rank_deficient = true;
        }
    }
    
    for (Index q = 0; q < n; ++q) {
        if (basis.col_status[q] != simplex::BasisStatus::Basic) {
            if (std::abs(h_x[q] - h_ub[q]) < 1e-6) {
                basis.col_status[q] = simplex::BasisStatus::AtUpper;
            } else {
                basis.col_status[q] = simplex::BasisStatus::AtLower;
            }
        }
    }

    // Free all temporary device memory
    cudaFree(lu.L_vals); cudaFree(lu.L_rows); cudaFree(lu.L_col_ptrs);
    cudaFree(lu.U_vals); cudaFree(lu.U_cols); cudaFree(lu.U_row_ptrs);
    cudaFree(lu.perm_row); cudaFree(lu.perm_col);
    cudaFree(lu.inv_perm_row); cudaFree(lu.inv_perm_col);

    cudaFree(ws.lb); cudaFree(ws.ub); cudaFree(ws.error_code);
    cudaFree(ws.basis_indices); cudaFree(ws.eta_vals); cudaFree(ws.eta_rows);
    cudaFree(ws.eta_col_starts); cudaFree(ws.eta_pivot_row);
    cudaFree(ws.num_eta_cols); cudaFree(ws.eta_nnz);
    cudaFree(ws.work_vec); cudaFree(ws.aux_vec); cudaFree(ws.x);
    cudaFree(ws.is_basic); cudaFree(ws.pi); cudaFree(ws.d); cudaFree(ws.d_pi);

    cudaFree(d_Aq); cudaFree(d_pivoted);

    if (pivot_failed != 0) {
        throw std::runtime_error("Crossover failed: Invalid pivot encountered");
    }
    if (rank_deficient) {
        throw std::runtime_error("Crossover failed: Matrix is rank deficient");
    }

    return basis;
}

} // namespace ipm
} // namespace sankhya

