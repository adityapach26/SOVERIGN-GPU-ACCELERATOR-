import re

with open('src/cuda/kkt.cu', 'r') as f:
    content = f.read()

# Replace solve_device
match_str = r"void GPUKKTCholeskySolver::gpu_cholesky_solve_device\(Float\* d_rhs_orig\) \{.*?\n\}"
replace_str = r'''void GPUKKTCholeskySolver::gpu_cholesky_solve_device(Float* d_rhs_orig) {
    int threads = 256;
    int blocks = (m_ + threads - 1) / threads;

    Float* d_rhs_perm_orig = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
    permute_vector_kernel<<<blocks, threads>>>(m_, d_rhs_orig, d_rhs_perm_orig, d_P_);
    
    Float* d_dy_perm = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
    CHECK_CUDA(cudaMemset(d_dy_perm, 0, m_ * sizeof(Float)));
    
    Float* d_r_perm = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
    Float* d_correction = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
    
    cusparseDnVecDescr_t vec_r, vec_z, vec_dy;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_r, m_, d_r_perm, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_z, m_, d_z_, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_dy, m_, d_correction, CUDA_R_64F));
    
    Float alpha = 1.0;
    Float rhs_norm = compute_norm_kkt(m_, d_rhs_perm_orig);
    Float stop_tol = 1e-10 * std::max(Float(1.0), rhs_norm);
    
    for (int iter = 0; iter < 5; ++iter) {
        compute_M_residual_kernel<<<blocks, threads>>>(m_, d_M_row_ptrs_, d_M_col_indices_, d_M_orig_vals_, d_dy_perm, d_rhs_perm_orig, d_r_perm);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        if (iter > 0) {
            Float r_norm = compute_norm_kkt(m_, d_r_perm);
            if (r_norm < stop_tol) break;
        }
        
        if (iter == 0) {
            CHECK_CUSPARSE(cusparseSpSV_analysis(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descr_L_, vec_r, vec_z, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_L_, d_spsv_buffer_));
            CHECK_CUSPARSE(cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descr_L_, vec_r, vec_z, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_L_));
            CHECK_CUSPARSE(cusparseSpSV_analysis(handle_, CUSPARSE_OPERATION_TRANSPOSE, &alpha, descr_L_, vec_z, vec_dy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_LT_, d_spsv_buffer_));
            CHECK_CUSPARSE(cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_TRANSPOSE, &alpha, descr_L_, vec_z, vec_dy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_LT_));
        } else {
            CHECK_CUSPARSE(cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descr_L_, vec_r, vec_z, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_L_));
            CHECK_CUSPARSE(cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_TRANSPOSE, &alpha, descr_L_, vec_z, vec_dy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_LT_));
        }
        
        add_correction_kernel<<<blocks, threads>>>(m_, d_dy_perm, d_correction);
    }
    
    inv_permute_vector_kernel<<<blocks, threads>>>(m_, d_dy_perm, d_rhs_orig, d_P_);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    arena_.free(d_correction);
    arena_.free(d_r_perm);
    arena_.free(d_dy_perm);
    arena_.free(d_rhs_perm_orig);
    
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_r));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_z));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_dy));
}'''

new_content = re.sub(match_str, replace_str, content, flags=re.DOTALL)
with open('src/cuda/kkt.cu', 'w') as f:
    f.write(new_content)
