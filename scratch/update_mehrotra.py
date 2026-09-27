import re

with open('src/ipm/mehrotra_device.cu', 'r') as f:
    content = f.read()

# 1. Update compute_step_length
old_step = """inline Float compute_step_length(Index n, const Float* d_v, const Float* d_dv) {
    thrust::device_ptr<const Float> pv(d_v), pdv(d_dv);
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(pv, pdv));
    Float min_ratio = thrust::transform_reduce(
        thrust::device, begin, begin + n,
        step_length_functor{}, Float(1.0e30), thrust::minimum<Float>());
    return std::min(1.0, min_ratio);
}"""

new_step = """inline Float compute_step_length(Index n, const Float* d_v, const Float* d_dv) {
    thrust::device_ptr<const Float> pv(d_v), pdv(d_dv);
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(pv, pdv));
    Float min_ratio = thrust::transform_reduce(
        thrust::device, begin, begin + n,
        step_length_functor{}, Float(1.0e30), thrust::minimum<Float>());
    return min_ratio;
}"""
content = content.replace(old_step, new_step)

# 2. Update affine step lengths
old_aff = """        // Affine step lengths
        Float alpha_p_aff = kernels::compute_step_length(n_, d_x_, d_dx_aff_);
        Float alpha_d_aff = kernels::compute_step_length(n_, d_s_, d_ds_aff_);"""
new_aff = """        // Affine step lengths (capped at 1.0)
        Float alpha_p_aff = std::min(Float(1.0), kernels::compute_step_length(n_, d_x_, d_dx_aff_));
        Float alpha_d_aff = std::min(Float(1.0), kernels::compute_step_length(n_, d_s_, d_ds_aff_));"""
content = content.replace(old_aff, new_aff)

# 3. Update corrector step sizes and backtracking
bt_old_pattern = r"        // Corrector step lengths with fraction-to-boundary \(eta = 0.995\).*?        alpha_p = current_alpha_p;\n        alpha_d = current_alpha_d;"

bt_new = """        // Corrector step lengths with fraction-to-boundary (eta = 0.995)
        Float alpha_p_max = kernels::compute_step_length(n_, d_x_, d_dx_);
        Float alpha_d_max = kernels::compute_step_length(n_, d_s_, d_ds_);
        constexpr Float eta = 0.995;
        Float alpha_p = std::min(Float(1.0), eta * alpha_p_max);
        Float alpha_d = std::min(Float(1.0), eta * alpha_d_max);

        // Backtracking globalization
        Float orig_Phi = std::max({norm_rp / rp_scale0_, norm_rd / rd_scale0_, mu / mu_scale0_});
        
        Float current_alpha_p = alpha_p;
        Float current_alpha_d = alpha_d;
        
        Float* d_bt_x = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        Float* d_bt_y = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
        Float* d_bt_s = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        Float* d_bt_rp = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
        Float* d_bt_rd = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        
        cusparseDnVecDescr_t vec_bt_x, vec_bt_y, vec_bt_rp, vec_bt_rd;
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_bt_x, n_, d_bt_x, CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_bt_y, m_, d_bt_y, CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_bt_rp, m_, d_bt_rp, CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_bt_rd, n_, d_bt_rd, CUDA_R_64F));
        
        int backtrack_iters = 0;
        bool step_accepted = false;
        
        while (backtrack_iters < 15) {
            CHECK_CUDA_IPM(cudaMemcpy(d_bt_x, d_x_, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA_IPM(cudaMemcpy(d_bt_y, d_y_, m_ * sizeof(Float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA_IPM(cudaMemcpy(d_bt_s, d_s_, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
            
            kernels::update_variables(n_, current_alpha_p, d_dx_, d_bt_x);
            kernels::update_variables(n_, current_alpha_d, d_ds_, d_bt_s);
            kernels::update_variables(m_, current_alpha_d, d_r_kkt_, d_bt_y); // d_r_kkt_ contains dy
            
            CHECK_CUDA_IPM(cudaMemcpy(d_bt_rp, d_b_, m_ * sizeof(Float), cudaMemcpyDeviceToDevice));
            {
                int blocks_m = (m_ + 255) / 256;
                kernels::vector_add_kernel<<<blocks_m, 256>>>(m_, -2.0, d_b_, d_bt_rp);
            }
            Float a_spmv = 1.0, b_spmv = 1.0;
            size_t buf1 = 0;
            CHECK_CUSPARSE_IPM(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &a_spmv, descr_A_, vec_bt_x, &b_spmv, vec_bt_rp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf1));
            void* dbuf1 = arena_.allocate(buf1);
            CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &a_spmv, descr_A_, vec_bt_x, &b_spmv, vec_bt_rp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dbuf1));
            arena_.free(dbuf1);
            
            CHECK_CUDA_IPM(cudaMemcpy(d_bt_rd, d_bt_s, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
            {
                int blocks_n = (n_ + 255) / 256;
                kernels::vector_add_kernel<<<blocks_n, 256>>>(n_, -1.0, d_c_, d_bt_rd);
            }
            size_t buf2 = 0;
            CHECK_CUSPARSE_IPM(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_TRANSPOSE,
                &a_spmv, descr_A_, vec_bt_y, &b_spmv, vec_bt_rd, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf2));
            void* dbuf2 = arena_.allocate(buf2);
            CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_TRANSPOSE,
                &a_spmv, descr_A_, vec_bt_y, &b_spmv, vec_bt_rd, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dbuf2));
            arena_.free(dbuf2);
            
            Float rp_new_norm = kernels::compute_norm(m_, d_bt_rp);
            Float rd_new_norm = kernels::compute_norm(n_, d_bt_rd);
            Float mu_new = kernels::compute_mu(n_, d_bt_x, d_bt_s);
            
            Float Phi_new = std::max({rp_new_norm / rp_scale0_, rd_new_norm / rd_scale0_, mu_new / mu_scale0_});
            
            thrust::device_ptr<Float> ptr_x(d_bt_x);
            thrust::device_ptr<Float> ptr_s(d_bt_s);
            Float min_x = thrust::reduce(thrust::device, ptr_x, ptr_x + n_, Float(1.0e30), thrust::minimum<Float>());
            Float min_s = thrust::reduce(thrust::device, ptr_s, ptr_s + n_, Float(1.0e30), thrust::minimum<Float>());
            
            if (n_ < 200) {
                // AFIRO / ADLITTLE diagnostics
                std::cout << "    [Diag] iter " << iter << " bt " << backtrack_iters << ":" << std::endl
                          << "      a_p: " << current_alpha_p << " a_d: " << current_alpha_d << std::endl
                          << "      min(x+adx): " << min_x << " min(s+ads): " << min_s << std::endl
                          << "      mu_before: " << mu << " mu_after: " << mu_new << std::endl
                          << "      Phi_before: " << orig_Phi << " Phi_after: " << Phi_new << std::endl;
            }
            
            // Accept step if strictly positive and merit function does not grow significantly
            if (min_x > 0.0 && min_s > 0.0 && Phi_new <= (1.0 + 1e-4) * orig_Phi) {
                step_accepted = true;
                break;
            }
            
            current_alpha_p *= 0.5;
            current_alpha_d *= 0.5;
            backtrack_iters++;
        }
        
        cusparseDestroyDnVec(vec_bt_x);
        cusparseDestroyDnVec(vec_bt_y);
        cusparseDestroyDnVec(vec_bt_rp);
        cusparseDestroyDnVec(vec_bt_rd);
        
        arena_.free(d_bt_rd);
        arena_.free(d_bt_rp);
        arena_.free(d_bt_s);
        arena_.free(d_bt_y);
        arena_.free(d_bt_x);
        
        if (!step_accepted) {
            result.status = simplex::SimplexStatus::IterationLimit;
            result.iterations = iter;
            break;
        }
        
        alpha_p = current_alpha_p;
        alpha_d = current_alpha_d;"""
content = re.sub(bt_old_pattern, bt_new, content, flags=re.DOTALL)

with open('src/ipm/mehrotra_device.cu', 'w') as f:
    f.write(content)
print("Updated mehrotra_device.cu")

