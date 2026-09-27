import re

with open('src/ipm/mehrotra_device.cu', 'r') as f:
    content = f.read()

# For predictor:
match_p = r'''        kernels::compute_dx\(n_, d_Theta_, d_ds_aff_, d_r_xs_, d_s_, d_dx_aff_\);
        cudaDeviceSynchronize\(\);

        if \(!verify_newton_direction\(d_dx_aff_, d_r_kkt_, d_ds_aff_, d_r_xs_, norm_rp, norm_rd\)\) \{'''

replace_p = r'''        kernels::compute_dx(n_, d_Theta_, d_ds_aff_, d_r_xs_, d_s_, d_dx_aff_);
        cudaDeviceSynchronize();

        Float rhs_norm_p = kernels::compute_norm(m_, d_r_kkt_);
        Float dx_norm_p  = kernels::compute_norm(n_, d_dx_aff_);
        Float ds_norm_p  = kernels::compute_norm(n_, d_ds_aff_);
        
        std::cout << "[PREDICTOR]\n"
                  << "  rhs_norm: " << rhs_norm_p << "\n"
                  << "  dx_norm: " << dx_norm_p << "\n"
                  << "  ds_norm: " << ds_norm_p << "\n";

        if (!verify_newton_direction(d_dx_aff_, d_r_kkt_, d_ds_aff_, d_r_xs_, norm_rp, norm_rd)) {'''

content = re.sub(match_p, replace_p, content)

# For corrector:
match_c = r'''        kernels::compute_dx\(n_, d_Theta_, d_ds_, d_r_xs_, d_s_, d_dx_\);
        cudaDeviceSynchronize\(\);

        if \(!verify_newton_direction\(d_dx_, d_r_kkt_, d_ds_, d_r_xs_, norm_rp, norm_rd\)\) \{'''

replace_c = r'''        kernels::compute_dx(n_, d_Theta_, d_ds_, d_r_xs_, d_s_, d_dx_);
        cudaDeviceSynchronize();

        Float rhs_norm_c = kernels::compute_norm(m_, d_r_kkt_);
        Float dx_norm_c  = kernels::compute_norm(n_, d_dx_);
        Float ds_norm_c  = kernels::compute_norm(n_, d_ds_);
        
        std::cout << "[CORRECTOR]\n"
                  << "  rhs_norm: " << rhs_norm_c << "\n"
                  << "  dx_norm: " << dx_norm_c << "\n"
                  << "  ds_norm: " << ds_norm_c << "\n";

        if (!verify_newton_direction(d_dx_, d_r_kkt_, d_ds_, d_r_xs_, norm_rp, norm_rd)) {'''

content = re.sub(match_c, replace_c, content)

with open('src/ipm/mehrotra_device.cu', 'w') as f:
    f.write(content)
