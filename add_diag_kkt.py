import re

with open('src/cuda/kkt.cu', 'r') as f:
    content = f.read()

# Add diagnostics to iterative refinement loop
# Find the for loop
match_str = r"    for \(int iter = 0; iter < 5; \+\+iter\) \{\n        compute_M_residual_kernel"
replace_str = r'''    for (int iter = 0; iter < 5; ++iter) {
        compute_M_residual_kernel<<<blocks, threads>>>(m_, d_M_row_ptrs_, d_M_col_indices_, d_M_orig_vals_, d_dy_perm, d_rhs_perm_orig, d_r_perm);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        Float r_norm_before = compute_norm_kkt(m_, d_r_perm);
        
        if (iter > 0) {
            if (r_norm_before < stop_tol) break;
        }
'''

content = re.sub(match_str, replace_str, content)

match_str2 = r"        add_correction_kernel<<<blocks, threads>>>\(m_, d_dy_perm, d_correction\);\n    \}"
replace_str2 = r'''        add_correction_kernel<<<blocks, threads>>>(m_, d_dy_perm, d_correction);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Compute residual after correction
        compute_M_residual_kernel<<<blocks, threads>>>(m_, d_M_row_ptrs_, d_M_col_indices_, d_M_orig_vals_, d_dy_perm, d_rhs_perm_orig, d_r_perm);
        CHECK_CUDA(cudaDeviceSynchronize());
        Float r_norm_after = compute_norm_kkt(m_, d_r_perm);
        
        std::cout << "    [Refinement] iter " << iter 
                  << " | res_before: " << r_norm_before 
                  << " | res_after: " << r_norm_after << std::endl;
    }
'''

content = re.sub(match_str2, replace_str2, content)

with open('src/cuda/kkt.cu', 'w') as f:
    f.write(content)
