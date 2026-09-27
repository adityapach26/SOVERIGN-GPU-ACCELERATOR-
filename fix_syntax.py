import re

with open('src/cuda/kkt.cu', 'r') as f:
    content = f.read()

# Let's find the exact corrupted block and replace it
bad_block = r'''        if \(iter > 0\) \{
            if \(r_norm_before < stop_tol\) break;
        \}
<<<blocks, threads>>>\(m_, d_M_row_ptrs_, d_M_col_indices_, d_M_vals_, d_dy_perm, d_rhs_perm_orig, d_r_perm\);
        CHECK_CUDA\(cudaDeviceSynchronize\(\)\);
        
        if \(iter > 0\) \{
            Float r_norm = compute_norm_kkt\(m_, d_r_perm\);
            if \(r_norm < stop_tol\) break;
        \}
        
        if \(iter == 0\) \{'''

good_block = r'''        if (iter > 0) {
            if (r_norm_before < stop_tol) break;
        }
        
        if (iter == 0) {'''

new_content = re.sub(bad_block, good_block, content)
with open('src/cuda/kkt.cu', 'w') as f:
    f.write(new_content)
