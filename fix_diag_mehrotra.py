import re

with open('src/ipm/mehrotra_device.cu', 'r') as f:
    content = f.read()

# First we need to undo the previous replace for rhs_norm_p because we need dy_norm
content = content.replace('Float rhs_norm_p = kernels::compute_norm(m_, d_r_kkt_);\n', 'Float dy_norm_p = kernels::compute_norm(m_, d_r_kkt_);\n')
content = content.replace('<< "  rhs_norm: " << rhs_norm_p << "\\n"', '<< "  dy_norm: " << dy_norm_p << "\\n"')

content = content.replace('Float rhs_norm_c = kernels::compute_norm(m_, d_r_kkt_);\n', 'Float dy_norm_c = kernels::compute_norm(m_, d_r_kkt_);\n')
content = content.replace('<< "  rhs_norm: " << rhs_norm_c << "\\n"', '<< "  dy_norm: " << dy_norm_c << "\\n"')

# Now insert rhs_norm BEFORE the solve
match_rkkt = r'''        compute_rkkt\(\);
        cudaDeviceSynchronize\(\);

        // Solve for dy_aff \(result in d_r_kkt_ in-place\)
        kkt_->gpu_cholesky_solve_device\(d_r_kkt_\);'''

replace_rkkt = r'''        compute_rkkt();
        cudaDeviceSynchronize();

        Float true_rhs_norm_p = kernels::compute_norm(m_, d_r_kkt_);
        std::cout << "[PREDICTOR] Pre-solve RHS norm: " << true_rhs_norm_p << std::endl;

        // Solve for dy_aff (result in d_r_kkt_ in-place)
        kkt_->gpu_cholesky_solve_device(d_r_kkt_);'''

content = re.sub(match_rkkt, replace_rkkt, content)

match_rkkt_c = r'''        compute_rkkt\(\);
        cudaDeviceSynchronize\(\);

        kkt_->gpu_cholesky_solve_device\(d_r_kkt_\);'''

replace_rkkt_c = r'''        compute_rkkt();
        cudaDeviceSynchronize();

        Float true_rhs_norm_c = kernels::compute_norm(m_, d_r_kkt_);
        std::cout << "[CORRECTOR] Pre-solve RHS norm: " << true_rhs_norm_c << std::endl;

        kkt_->gpu_cholesky_solve_device(d_r_kkt_);'''

content = re.sub(match_rkkt_c, replace_rkkt_c, content)

with open('src/ipm/mehrotra_device.cu', 'w') as f:
    f.write(content)
