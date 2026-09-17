#include <algorithm>
#include "cuda/kkt.cuh"
#include <stdexcept>
#include <iostream>

namespace sankhya {
namespace gpu {

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        throw std::runtime_error("cuSPARSE API failed at line " +              \
                                 std::to_string(__LINE__));                    \
    }                                                                          \
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        throw std::runtime_error("CUDA API failed at line " +                  \
                                 std::to_string(__LINE__));                    \
    }                                                                          \
}

__global__ void permute_vector_kernel(Index n, const Float* in, Float* out, const Index* P) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[P[i]];
}

__global__ void inv_permute_vector_kernel(Index n, const Float* in, Float* out, const Index* P) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[P[i]] = in[i];
}

__global__ void compute_M_numerics_kernel(
    Index m, Index n,
    const Index* M_row_ptrs, const Index* M_col_indices, Float* M_values,
    const Index* A_row_ptrs, const Index* A_col_indices, const Float* A_values,
    const Float* Theta,
    const Index* P
) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    Index orig_i = P[i];
    
    // For each non-zero in row i of M:
    for (Index p = M_row_ptrs[i]; p < M_row_ptrs[i+1]; ++p) {
        Index j = M_col_indices[p];
        Index orig_j = P[j];
        
        // compute dot product of row orig_i and row orig_j of A, weighted by Theta
        Float sum = 0.0;
        Index ptr_i = A_row_ptrs[orig_i];
        Index end_i = A_row_ptrs[orig_i+1];
        Index ptr_j = A_row_ptrs[orig_j];
        Index end_j = A_row_ptrs[orig_j+1];
        
        while (ptr_i < end_i && ptr_j < end_j) {
            Index col_i = A_col_indices[ptr_i];
            Index col_j = A_col_indices[ptr_j];
            if (col_i == col_j) {
                sum += A_values[ptr_i] * Theta[col_i] * A_values[ptr_j];
                ptr_i++;
                ptr_j++;
            } else if (col_i < col_j) {
                ptr_i++;
            } else {
                ptr_j++;
            }
        }
        M_values[p] = sum;
    }
}

__global__ void cholesky_level_kernel(
    Index num_nodes, const Index* level_nodes,
    const Index* M_row_ptrs, const Index* M_col_indices, const Float* M_vals,
    const Index* L_row_ptrs, const Index* L_col_indices, Float* L_vals,
    int* error_flag
) {
    if (*error_flag > 0) return; // Propagate failure immediately
    
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    Index i = level_nodes[idx];
    
    Index L_start = L_row_ptrs[i];
    Index L_end = L_row_ptrs[i + 1];
    Index M_ptr = M_row_ptrs[i];
    Index M_end = M_row_ptrs[i + 1];
    
    for (Index p = L_start; p < L_end; ++p) {
        Index j = L_col_indices[p];
        
        Float m_val = 0.0;
        while (M_ptr < M_end && M_col_indices[M_ptr] < j) M_ptr++;
        if (M_ptr < M_end && M_col_indices[M_ptr] == j) {
            m_val = M_vals[M_ptr];
        }
        
        if (j == i) {
            Float sum = m_val;
            for (Index k_ptr = L_start; k_ptr < p; ++k_ptr) {
                Float lik = L_vals[k_ptr];
                sum -= lik * lik;
            }
            if (sum <= 0.0) {
                atomicAdd(error_flag, 1);
                return; // Mathematical failure (not SPD). Do not fake L.
            } else {
                L_vals[p] = sqrt(sum);
            }
        } else {
            Float sum = m_val;
            
            // Sparse dot product
            Index ptr_i = L_start;
            Index ptr_j = L_row_ptrs[j];
            Index end_j = L_row_ptrs[j + 1];
            
            while (ptr_i < p && ptr_j < end_j) {
                Index col_i = L_col_indices[ptr_i];
                Index col_j = L_col_indices[ptr_j];
                if (col_i >= j || col_j >= j) break;
                
                if (col_i == col_j) {
                    sum -= L_vals[ptr_i] * L_vals[ptr_j];
                    ptr_i++;
                    ptr_j++;
                } else if (col_i < col_j) {
                    ptr_i++;
                } else {
                    ptr_j++;
                }
            }
            
            Index diag_ptr_j = end_j - 1;
            Float l_jj = L_vals[diag_ptr_j];
            L_vals[p] = sum / l_jj;
        }
    }
}

GPUKKTCholeskySolver::GPUKKTCholeskySolver(
    VRAMArena& arena, 
    const core::CSRMatrix& A, 
    const ipm::SymbolicFactorization& sym,
    const core::CSRMatrix& M_pattern
) : arena_(arena), m_(A.rows), n_(A.cols) {
    
    // 1. Allocate device memory for A
    size_t A_row_bytes = (m_ + 1) * sizeof(Index);
    size_t A_col_bytes = A.col_indices.size() * sizeof(Index);
    size_t A_val_bytes = A.values.size() * sizeof(Float);
    
    d_A_row_ptrs_ = static_cast<Index*>(arena_.allocate(A_row_bytes));
    d_A_col_indices_ = static_cast<Index*>(arena_.allocate(A_col_bytes));
    d_A_vals_ = static_cast<Float*>(arena_.allocate(A_val_bytes));

    CHECK_CUDA(cudaMemcpy(d_A_row_ptrs_, A.row_ptrs.data(), A_row_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_col_indices_, A.col_indices.data(), A_col_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_vals_, A.values.data(), A_val_bytes, cudaMemcpyHostToDevice));

    // 2. Allocate device memory for M pattern
    size_t M_row_bytes = (m_ + 1) * sizeof(Index);
    size_t M_col_bytes = M_pattern.col_indices.size() * sizeof(Index);
    size_t M_val_bytes = M_pattern.values.size() * sizeof(Float);
    
    d_M_row_ptrs_ = static_cast<Index*>(arena_.allocate(M_row_bytes));
    d_M_col_indices_ = static_cast<Index*>(arena_.allocate(M_col_bytes));
    d_M_vals_ = static_cast<Float*>(arena_.allocate(M_val_bytes));
    
    CHECK_CUDA(cudaMemcpy(d_M_row_ptrs_, M_pattern.row_ptrs.data(), M_row_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_M_col_indices_, M_pattern.col_indices.data(), M_col_bytes, cudaMemcpyHostToDevice));
    
        // 3. Allocate device memory for L pattern (including diagonal for cuSPARSE)
    // [B] Engineering Decision: Phase 14.1 provides strictly lower L_pattern. 
    // cuSPARSE SpSV requires explicit diagonal entries. We expand the CSR structure 
    // on the host before uploading to VRAM to maintain GPU numerical compatibility.
    Index strictly_lower_nnz = sym.L_pattern.values.size();
    Index full_nnz = strictly_lower_nnz + m_;
    
    std::vector<Index> full_L_row_ptrs(m_ + 1, 0);
    std::vector<Index> full_L_col_indices(full_nnz);
    
    Index p = 0;
    for(Index i = 0; i < m_; ++i) {
        full_L_row_ptrs[i] = p;
        for(Index orig_p = sym.L_pattern.row_ptrs[i]; orig_p < sym.L_pattern.row_ptrs[i+1]; ++orig_p) {
            full_L_col_indices[p++] = sym.L_pattern.col_indices[orig_p];
        }
        full_L_col_indices[p++] = i; // Diagonal is the largest column index in lower triangular row
    }
    full_L_row_ptrs[m_] = p;

    size_t L_row_bytes = (m_ + 1) * sizeof(Index);
    size_t L_col_bytes = full_nnz * sizeof(Index);
    size_t L_val_bytes = full_nnz * sizeof(Float);
    
    d_L_row_ptrs_ = static_cast<Index*>(arena_.allocate(L_row_bytes));
    d_L_col_indices_ = static_cast<Index*>(arena_.allocate(L_col_bytes));
    d_L_vals_ = static_cast<Float*>(arena_.allocate(L_val_bytes));

    CHECK_CUDA(cudaMemcpy(d_L_row_ptrs_, full_L_row_ptrs.data(), L_row_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_L_col_indices_, full_L_col_indices.data(), L_col_bytes, cudaMemcpyHostToDevice));

    // 4. Allocate and upload permutation P
    size_t P_bytes = m_ * sizeof(Index);
    d_P_ = static_cast<Index*>(arena_.allocate(P_bytes));
    CHECK_CUDA(cudaMemcpy(d_P_, sym.P.data(), P_bytes, cudaMemcpyHostToDevice));

    // 5. Allocate intermediate vectors
    d_z_ = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
    
        // 6. Compute elimination tree levels for parallel GPU scheduling
    // Topological depth calculation handling chains, stars, and branching trees correctly.
    std::vector<Index> level(m_, 0);
    std::vector<Index> in_degree(m_, 0);
    for (Index i = 0; i < m_; ++i) {
        if (sym.parent[i] != -1) {
            in_degree[sym.parent[i]]++;
        }
    }
    
    std::vector<Index> queue;
    for (Index i = 0; i < m_; ++i) {
        if (in_degree[i] == 0) {
            queue.push_back(i);
        }
    }
    
    Index head = 0;
    Index max_level = 0;
    while (head < queue.size()) {
        Index u = queue[head++];
        Index p = sym.parent[u];
        if (p != -1) {
            level[p] = std::max(level[p], level[u] + 1);
            max_level = std::max(max_level, level[p]);
            if (--in_degree[p] == 0) {
                queue.push_back(p);
            }
        }
    }
    num_levels_ = max_level + 1;
    
    std::vector<std::vector<Index>> level_nodes(num_levels_);
    for (Index i = 0; i < m_; ++i) {
        level_nodes[level[i]].push_back(i);
    }
    
    level_ptrs_.assign(num_levels_ + 1, 0);
    std::vector<Index> host_level_nodes;
    for (Index lvl = 0; lvl < num_levels_; ++lvl) {
        level_ptrs_[lvl] = host_level_nodes.size();
        for (Index node : level_nodes[lvl]) {
            host_level_nodes.push_back(node);
        }
    }
    level_ptrs_[num_levels_] = host_level_nodes.size();
    
    size_t level_nodes_bytes = host_level_nodes.size() * sizeof(Index);
    d_level_nodes_ = static_cast<Index*>(arena_.allocate(level_nodes_bytes));
    CHECK_CUDA(cudaMemcpy(d_level_nodes_, host_level_nodes.data(), level_nodes_bytes, cudaMemcpyHostToDevice));
    
    d_factorization_error_ = static_cast<int*>(arena_.allocate(sizeof(int)));

    // Initialize cuSPARSE and descriptors
    initialize_cusparse();
}

void GPUKKTCholeskySolver::initialize_cusparse() {
    CHECK_CUSPARSE(cusparseCreate(&handle_));

    // The GPU numerical factorization natively generates the true L factor.
    // The L matrix is lower triangular and has a non-unit diagonal.
    // We explicitly configure the cuSPARSE descriptor for these properties.
    Index nnz_L;
    CHECK_CUDA(cudaMemcpy(&nnz_L, d_L_row_ptrs_ + m_, sizeof(Index), cudaMemcpyDeviceToHost));

    CHECK_CUSPARSE(cusparseCreateCsr(&descr_L_, m_, m_, nnz_L,
                                     d_L_row_ptrs_, d_L_col_indices_, d_L_vals_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseFillMode_t fill_mode = CUSPARSE_FILL_MODE_LOWER;
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr_L_, CUSPARSE_SPMAT_FILL_MODE, &fill_mode, sizeof(fill_mode)));

    cusparseDiagType_t diag_type = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr_L_, CUSPARSE_SPMAT_DIAG_TYPE, &diag_type, sizeof(diag_type)));

    // cuSPARSE requires SpSV descriptor initialization
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsv_descr_L_));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsv_descr_LT_));
    
    // Dummy dense vectors for buffer size computation
    Float alpha = 1.0;
    Float* dummy_x;
    Float* dummy_y;
    CHECK_CUDA(cudaMalloc(&dummy_x, m_ * sizeof(Float)));
    CHECK_CUDA(cudaMalloc(&dummy_y, m_ * sizeof(Float)));
    
    cusparseDnVecDescr_t dummy_vec_x, dummy_vec_y;
    CHECK_CUSPARSE(cusparseCreateDnVec(&dummy_vec_x, m_, dummy_x, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&dummy_vec_y, m_, dummy_y, CUDA_R_64F));

    size_t bufferSize1 = 0, bufferSize2 = 0;
    
    // Forward solve: L z = r (Operation: NON_TRANSPOSE)
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descr_L_, dummy_vec_x, dummy_vec_y, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_L_, &bufferSize1));

    // Backward solve: L^T \Delta y = z (Operation: TRANSPOSE)
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(
        handle_, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, descr_L_, dummy_vec_x, dummy_vec_y, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_LT_, &bufferSize2));

    size_t bufferSize = std::max(bufferSize1, bufferSize2);
    if (bufferSize > 0) {
        d_spsv_buffer_ = arena_.allocate(bufferSize);
    } else {
        d_spsv_buffer_ = nullptr;
    }

    CHECK_CUSPARSE(cusparseDestroyDnVec(dummy_vec_x));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dummy_vec_y));
    CHECK_CUDA(cudaFree(dummy_x));
    CHECK_CUDA(cudaFree(dummy_y));
}

GPUKKTCholeskySolver::~GPUKKTCholeskySolver() {
    if (handle_) {
        cusparseSpSV_destroyDescr(spsv_descr_L_);
        cusparseSpSV_destroyDescr(spsv_descr_LT_);
        cusparseDestroySpMat(descr_L_);
        cusparseDestroy(handle_);
    }
}

void GPUKKTCholeskySolver::gpu_cholesky_factorize(
    const core::CSRMatrix& A, 
    const std::vector<Float>& Theta
) {
    if (static_cast<Index>(Theta.size()) != n_) {
        throw std::invalid_argument("Theta dimension mismatch");
    }

    Float* d_Theta = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
    CHECK_CUDA(cudaMemcpy(d_Theta, Theta.data(), n_ * sizeof(Float), cudaMemcpyHostToDevice));

    // 1. Construct numerical normal matrix M = A Theta A^T on GPU
    int threads = 256;
    int blocks = (m_ + threads - 1) / threads;
    
    compute_M_numerics_kernel<<<blocks, threads>>>(
        m_, n_,
        d_M_row_ptrs_, d_M_col_indices_, d_M_vals_,
        d_A_row_ptrs_, d_A_col_indices_, d_A_vals_,
        d_Theta,
        d_P_
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    arena_.free(d_Theta);

    // 2. Cholesky numerical factorization (M = LL^T)
    // [B] Engineering Decision: We implement a custom, exact, up-looking sparse Cholesky factorization
    // executed entirely on the GPU. The symbolic elimination tree computes topological levels.
    // Rows at the same topological level are processed in parallel via batched kernels, natively avoiding CPU fallback.
    int h_error = 0;
    CHECK_CUDA(cudaMemcpy(d_factorization_error_, &h_error, sizeof(int), cudaMemcpyHostToDevice));

    for (Index lvl = 0; lvl < num_levels_; ++lvl) {
        Index num_nodes = level_ptrs_[lvl + 1] - level_ptrs_[lvl];
        if (num_nodes == 0) continue;
        
        int lvl_threads = 256;
        int lvl_blocks = (num_nodes + lvl_threads - 1) / lvl_threads;
        
        cholesky_level_kernel<<<lvl_blocks, lvl_threads>>>(
            num_nodes, d_level_nodes_ + level_ptrs_[lvl],
            d_M_row_ptrs_, d_M_col_indices_, d_M_vals_,
            d_L_row_ptrs_, d_L_col_indices_, d_L_vals_,
            d_factorization_error_
        );
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_error, d_factorization_error_, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_error > 0) {
        throw std::runtime_error("Numerical factorization failed: matrix is not positive definite.");
    }
}

void GPUKKTCholeskySolver::gpu_cholesky_solve(std::vector<Float>& rhs) {
    if (static_cast<Index>(rhs.size()) != m_) {
        throw std::invalid_argument("rhs dimension mismatch");
    }

    Float* d_rhs_orig = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
    Float* d_rhs_perm = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));

    CHECK_CUDA(cudaMemcpy(d_rhs_orig, rhs.data(), m_ * sizeof(Float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (m_ + threads - 1) / threads;

    // Permute RHS: P[i] = orig_i, so permuted[i] = orig[P[i]]
    permute_vector_kernel<<<blocks, threads>>>(m_, d_rhs_orig, d_rhs_perm, d_P_);
    CHECK_CUDA(cudaGetLastError());

    // Create dense vector descriptors
    cusparseDnVecDescr_t vec_r, vec_z, vec_dy;
    Float alpha = 1.0;

    // 1. Forward solve: L z = r (working on permuted RHS)
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_r, m_, d_rhs_perm, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_z, m_, d_z_, CUDA_R_64F));

    CHECK_CUSPARSE(cusparseSpSV_analysis(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descr_L_, vec_r, vec_z, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_L_, d_spsv_buffer_));

    CHECK_CUSPARSE(cusparseSpSV_solve(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descr_L_, vec_r, vec_z, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_L_));

    // 2. Backward solve: L^T \Delta y = z
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_dy, m_, d_rhs_perm, CUDA_R_64F));

    CHECK_CUSPARSE(cusparseSpSV_analysis(
        handle_, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, descr_L_, vec_z, vec_dy, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_LT_, d_spsv_buffer_));

    CHECK_CUSPARSE(cusparseSpSV_solve(
        handle_, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, descr_L_, vec_z, vec_dy, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_LT_));

    CHECK_CUDA(cudaDeviceSynchronize());

    // Inverse permute \Delta y back to original ordering
    inv_permute_vector_kernel<<<blocks, threads>>>(m_, d_rhs_perm, d_rhs_orig, d_P_);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(rhs.data(), d_rhs_orig, m_ * sizeof(Float), cudaMemcpyDeviceToHost));
    arena_.free(d_rhs_perm);
    arena_.free(d_rhs_orig);

    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_r));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_z));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_dy));
}

} // namespace gpu
} // namespace sankhya