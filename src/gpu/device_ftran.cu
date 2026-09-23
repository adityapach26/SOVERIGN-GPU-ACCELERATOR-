#include "gpu/device_ftran.cuh"
#include <cmath>

namespace sankhya {
namespace gpu {

__device__ void device_ftran(const DeviceSparseLU& lu, const WorkingBasisState& ws, Float* vec) {
    Float* w = ws.aux_vec;
    
    // 1. Permute RHS: w = P * vec
    for (Index i = threadIdx.x; i < lu.m; i += blockDim.x) {
        w[i] = vec[lu.perm_row[i]];
    }
    __syncthreads();

    // 2. L solve (forward substitution)
    for (Index k = 0; k < lu.m; ++k) {
        Float w_k = w[k];
        Index start = lu.L_col_ptrs[k];
        Index end = lu.L_col_ptrs[k+1];
        for (Index p = start + threadIdx.x; p < end; p += blockDim.x) {
            w[lu.L_rows[p]] -= lu.L_vals[p] * w_k;
        }
        __syncthreads();
    }

    // 3. U solve (backward substitution)
    for (Index k = lu.m - 1; k >= 0; --k) {
        if (threadIdx.x == 0) {
            Index start = lu.U_row_ptrs[k];
            Index end = lu.U_row_ptrs[k+1];
            Float diag = lu.U_vals[start];
            Float dot = 0.0;
            for (Index p = start + 1; p < end; ++p) {
                dot += lu.U_vals[p] * w[lu.U_cols[p]];
            }
            w[k] = (w[k] - dot) / diag;
        }
        __syncthreads();
    }

    // 4. Inverse Permute: vec = Q * w
    for (Index i = threadIdx.x; i < lu.m; i += blockDim.x) {
        vec[lu.perm_col[i]] = w[i];
    }
    __syncthreads();

    // 5. Apply FT updates: vec = E_k^-1 ... E_1^-1 vec
    // Order: forward, 0 to num_eta_cols - 1
    Index num_eta = *(ws.num_eta_cols);
    for (Index t = 0; t < num_eta; ++t) {
        Index p = ws.eta_pivot_row[t];
        Float x_p = vec[p];
        Index start = ws.eta_col_starts[t];
        Index end = ws.eta_col_starts[t+1];
        
        for (Index j = start + threadIdx.x; j < end; j += blockDim.x) {
            Index row = ws.eta_rows[j];
            Float val = ws.eta_vals[j];
            if (row == p) {
                vec[row] = val * x_p;
            } else {
                vec[row] += val * x_p;
            }
        }
        __syncthreads();
    }
}

__device__ void device_btran(const DeviceSparseLU& lu, const WorkingBasisState& ws, Float* vec) {
    Float* w = ws.aux_vec;

    // 1. Apply transposed FT updates: vec = E_1^-T ... E_k^-T vec
    // Order: reverse, num_eta_cols - 1 down to 0
    Index num_eta = *(ws.num_eta_cols);
    for (int t = static_cast<int>(num_eta) - 1; t >= 0; --t) {
        if (threadIdx.x == 0) {
            Index p = ws.eta_pivot_row[t];
            Index start = ws.eta_col_starts[t];
            Index end = ws.eta_col_starts[t+1];
            
            Float dot = 0.0;
            for (Index j = start; j < end; ++j) {
                dot += ws.eta_vals[j] * vec[ws.eta_rows[j]];
            }
            vec[p] = dot;
        }
        __syncthreads();
    }

    // 2. Permute: w = Q^T * vec
    for (Index i = threadIdx.x; i < lu.m; i += blockDim.x) {
        w[i] = vec[lu.perm_col[i]];
    }
    __syncthreads();

    // 3. U^T solve (forward substitution)
    for (Index k = 0; k < lu.m; ++k) {
        if (threadIdx.x == 0) {
            Index start = lu.U_row_ptrs[k];
            Float diag = lu.U_vals[start];
            w[k] /= diag;
        }
        __syncthreads();
        Float w_k = w[k];
        Index start = lu.U_row_ptrs[k];
        Index end = lu.U_row_ptrs[k+1];
        for (Index p = start + 1 + threadIdx.x; p < end; p += blockDim.x) {
            w[lu.U_cols[p]] -= lu.U_vals[p] * w_k;
        }
        __syncthreads();
    }

    // 4. L^T solve (backward substitution)
    for (Index k = lu.m - 1; k >= 0; --k) {
        if (threadIdx.x == 0) {
            Index start = lu.L_col_ptrs[k];
            Index end = lu.L_col_ptrs[k+1];
            Float dot = 0.0;
            for (Index p = start; p < end; ++p) {
                dot += lu.L_vals[p] * w[lu.L_rows[p]];
            }
            w[k] -= dot;
        }
        __syncthreads();
    }

    // 5. Permute: vec = P^T * w
    for (Index i = threadIdx.x; i < lu.m; i += blockDim.x) {
        vec[lu.perm_row[i]] = w[i];
    }
    __syncthreads();
}

__device__ bool device_update_basis(
    WorkingBasisState& ws,
    Index leaving_row,
    Index entering_col,
    const Float* Aq,
    Float pivot_tolerance
) {
    Float d_p = Aq[leaving_row];
    if (std::abs(d_p) < pivot_tolerance) {
        return false;
    }

    __shared__ bool success_flag;

    if (threadIdx.x == 0) {
        success_flag = false;
        Index cur_eta = *(ws.num_eta_cols);
        
        if (cur_eta < ws.eta_col_capacity) {
            Index start = ws.eta_col_starts[cur_eta];
            
            // Pass 1: Count exact nnz to avoid writing past capacity
            Index required_nnz = 0;
            for (Index i = 0; i < ws.m; ++i) {
                Float val = Aq[i];
                val = (i == leaving_row) ? (1.0 / d_p) : (-val / d_p);
                if (std::abs(val) > 1e-15) {
                    required_nnz++;
                }
            }

            if (start + required_nnz <= ws.eta_capacity) {
                // Pass 2: Actually write the eta entries
                Index nnz = 0;
                for (Index i = 0; i < ws.m; ++i) {
                    Float val = Aq[i];
                    val = (i == leaving_row) ? (1.0 / d_p) : (-val / d_p);
                    if (std::abs(val) > 1e-15) {
                        ws.eta_vals[start + nnz] = val;
                        ws.eta_rows[start + nnz] = i;
                        nnz++;
                    }
                }
                
                ws.eta_pivot_row[cur_eta] = leaving_row;
                ws.eta_col_starts[cur_eta + 1] = start + required_nnz;
                *(ws.num_eta_cols) = cur_eta + 1;
                *(ws.eta_nnz) = start + required_nnz;
                ws.basis_indices[leaving_row] = entering_col;
                success_flag = true;
            } else {
                *(ws.error_code) = 3; // overflow
            }
        } else {
            *(ws.error_code) = 3; // overflow
        }
    }
    __syncthreads();
    
    return success_flag; 
}

} // namespace gpu
} // namespace sankhya

