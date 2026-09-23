#include <cstddef>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "ipm/mehrotra.hpp"
#include "cuda/kkt.cuh"
#include "gpu/vram_arena.cuh"
#include "ipm/symbolic.hpp"
#include "core/sparse_matrix.hpp"
#include "simplex/primal.hpp"
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <cusparse.h>
#include <cmath>
#include <algorithm>
#include <iostream>

#define CHECK_CUDA_IPM(func)                                                   \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        throw std::runtime_error(std::string("CUDA API failed: ") +            \
                                 cudaGetErrorString(status) +                  \
                                 " at line " + std::to_string(__LINE__));      \
    }                                                                          \
}

#define CHECK_CUSPARSE_IPM(func)                                               \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        throw std::runtime_error("cuSPARSE API failed at line " +              \
                                 std::to_string(__LINE__));                    \
    }                                                                          \
}

namespace sankhya {
namespace ipm {

// ============================================================
// CUDA kernels (internal to this TU)
// ============================================================
namespace kernels {

__global__ void init_vars_kernel(Index m, Index n, Float* x, Float* y, Float* s) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 1.0;
        s[i] = 1.0;
    }
    if (i < m) {
        y[i] = 0.0;
    }
}

__global__ void compute_theta_kernel(Index n, const Float* x, const Float* s, Float* Theta) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Theta[i] = x[i] / s[i];
    }
}

__global__ void compute_r_xs_kernel(Index n, const Float* x, const Float* s,
                                    const Float* dx_aff, const Float* ds_aff,
                                    Float sigma_mu, Float* r_xs) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Float aff_term = (dx_aff != nullptr && ds_aff != nullptr) ? (dx_aff[i] * ds_aff[i]) : 0.0;
        r_xs[i] = -x[i] * s[i] + sigma_mu - aff_term;
    }
}

__global__ void compute_v_kernel(Index n, const Float* Theta, const Float* rd, const Float* r_xs, const Float* s, Float* v) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        v[i] = -Theta[i] * rd[i] - r_xs[i] / s[i];
    }
}

__global__ void compute_dx_kernel(Index n, const Float* Theta, const Float* ds, const Float* r_xs, const Float* s, Float* dx) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] = -Theta[i] * ds[i] + r_xs[i] / s[i];
    }
}

__global__ void vector_add_kernel(Index n, Float a, const Float* x, Float* y) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void update_vars_kernel(Index n, Float alpha, const Float* dx, Float* x) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += alpha * dx[i];
    }
}

// Thrust functors for host-callable reductions over device memory
struct mu_functor {
    __host__ __device__
    Float operator()(const thrust::tuple<Float, Float>& t) const {
        return thrust::get<0>(t) * thrust::get<1>(t);
    }
};

struct norm_sq_functor {
    __host__ __device__
    Float operator()(const Float& x) const { return x * x; }
};

struct step_length_functor {
    __host__ __device__
    Float operator()(const thrust::tuple<Float, Float>& t) const {
        Float var  = thrust::get<0>(t);
        Float dvar = thrust::get<1>(t);
        if (dvar >= 0.0) return 1.0e30; // infinity proxy
        return -var / dvar;
    }
};

// ============================================================
// Host-callable wrappers over kernels/Thrust
// ============================================================

// Launch init kernel
inline void initialize_variables(Index m, Index n, Float* d_x, Float* d_y, Float* d_s) {
    int blocks = (std::max(m, n) + 255) / 256;
    init_vars_kernel<<<blocks, 256>>>(m, n, d_x, d_y, d_s);
}

// 2-norm via Thrust
inline Float compute_norm(Index n, const Float* d_v) {
    thrust::device_ptr<const Float> ptr(d_v);
    Float sum_sq = thrust::transform_reduce(
        thrust::device, ptr, ptr + n,
        norm_sq_functor{}, Float(0.0), thrust::plus<Float>());
    return std::sqrt(sum_sq);
}

// Complementarity measure mu = x^T s / n
inline Float compute_mu(Index n, const Float* d_x, const Float* d_s) {
    thrust::device_ptr<const Float> px(d_x), ps(d_s);
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(px, ps));
    Float dot = thrust::transform_reduce(
        thrust::device, begin, begin + n,
        mu_functor{}, Float(0.0), thrust::plus<Float>());
    return dot / static_cast<Float>(n);
}

// c^T x
inline Float compute_objective(Index n, const Float* d_c, const Float* d_x) {
    thrust::device_ptr<const Float> pc(d_c), px(d_x);
    return thrust::inner_product(thrust::device, pc, pc + n, px, Float(0.0));
}

// Theta[i] = x[i] / s[i]
inline void compute_theta(Index n, const Float* d_x, const Float* d_s, Float* d_Theta) {
    int blocks = (n + 255) / 256;
    compute_theta_kernel<<<blocks, 256>>>(n, d_x, d_s, d_Theta);
}

// r_xs[i] = -x[i]*s[i] + sigma_mu - dx_aff[i]*ds_aff[i]
inline void compute_r_xs(Index n, const Float* d_x, const Float* d_s,
                         const Float* d_dx_aff, const Float* d_ds_aff,
                         Float sigma_mu, Float* d_r_xs) {
    int blocks = (n + 255) / 256;
    compute_r_xs_kernel<<<blocks, 256>>>(n, d_x, d_s, d_dx_aff, d_ds_aff, sigma_mu, d_r_xs);
}

// v[i] = -Theta[i]*rd[i] - r_xs[i]/s[i]
inline void compute_v(Index n, const Float* d_Theta, const Float* d_rd,
                      const Float* d_r_xs, const Float* d_s, Float* d_v) {
    int blocks = (n + 255) / 256;
    compute_v_kernel<<<blocks, 256>>>(n, d_Theta, d_rd, d_r_xs, d_s, d_v);
}

// dx[i] = -Theta[i]*ds[i] + r_xs[i]/s[i]
inline void compute_dx(Index n, const Float* d_Theta, const Float* d_ds,
                       const Float* d_r_xs, const Float* d_s, Float* d_dx) {
    int blocks = (n + 255) / 256;
    compute_dx_kernel<<<blocks, 256>>>(n, d_Theta, d_ds, d_r_xs, d_s, d_dx);
}

// Fraction-to-boundary step length: min over i where d[i] < 0 of -v[i]/d[i]
inline Float compute_step_length(Index n, const Float* d_v, const Float* d_dv) {
    thrust::device_ptr<const Float> pv(d_v), pdv(d_dv);
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(pv, pdv));
    Float min_ratio = thrust::transform_reduce(
        thrust::device, begin, begin + n,
        step_length_functor{}, Float(1.0e30), thrust::minimum<Float>());
    return std::min(1.0, min_ratio);
}

// x[i] += alpha * dx[i]
inline void update_variables(Index n, Float alpha, const Float* d_dx, Float* d_x) {
    int blocks = (n + 255) / 256;
    update_vars_kernel<<<blocks, 256>>>(n, alpha, d_dx, d_x);
}

} // namespace kernels

// ============================================================
// MehrotraSolver::Impl
// ============================================================
class MehrotraSolver::Impl {
public:
    Impl(const core::Model& model) : model_(model), arena_(1024 * 1024 * 64) {
        m_ = model.A.rows;
        n_ = model.A.cols;

        // Convert CSC model matrix to CSR for AMD/symbolic/KKT APIs
        A_csr_ = core::to_csr(model.A);

        std::vector<Index> P = compute_amd_ordering(A_csr_);
        compute_symbolic_factorization(A_csr_, P, sym_);

        // Construct M_pattern = P (A A^T) P^T structurally using CSR A
        core::CSRMatrix M_pattern;
        M_pattern.rows = m_;
        M_pattern.cols = m_;
        M_pattern.row_ptrs.push_back(0);

        std::vector<std::vector<Index>> M_adj(m_);
        for (Index i = 0; i < m_; ++i) {
            Index orig_i = P[i];
            for (Index j = 0; j < m_; ++j) {
                Index orig_j = P[j];
                // Check if row orig_i and row orig_j share any column in A (CSR)
                bool intersect = false;
                Index ptr1 = A_csr_.row_ptrs[orig_i];
                Index ptr2 = A_csr_.row_ptrs[orig_j];
                while (ptr1 < A_csr_.row_ptrs[orig_i + 1] &&
                       ptr2 < A_csr_.row_ptrs[orig_j + 1]) {
                    if (A_csr_.col_indices[ptr1] == A_csr_.col_indices[ptr2]) {
                        intersect = true;
                        break;
                    } else if (A_csr_.col_indices[ptr1] < A_csr_.col_indices[ptr2]) {
                        ++ptr1;
                    } else {
                        ++ptr2;
                    }
                }
                if (intersect) M_adj[i].push_back(j);
            }
            M_pattern.col_indices.insert(M_pattern.col_indices.end(),
                                         M_adj[i].begin(), M_adj[i].end());
            M_pattern.row_ptrs.push_back(static_cast<Index>(M_pattern.col_indices.size()));
        }
        M_pattern.values.resize(M_pattern.col_indices.size(), 0.0);

        // GPUKKTCholeskySolver expects CSRMatrix for A
        kkt_ = new gpu::GPUKKTCholeskySolver(arena_, A_csr_, sym_, M_pattern);

        CHECK_CUSPARSE_IPM(cusparseCreate(&handle_));

        // Allocate persistent device vectors for IPM state
        d_x_  = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        d_y_  = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
        d_s_  = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));

        d_rp_ = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
        d_rd_ = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));

        d_Theta_  = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        d_r_xs_   = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        d_v_      = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        d_r_kkt_  = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));

        d_dx_aff_ = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        d_ds_aff_ = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));

        d_dx_ = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        d_ds_ = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));

        d_b_ = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
        d_c_ = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        CHECK_CUDA_IPM(cudaMemcpy(d_b_, model.rhs.data(), m_ * sizeof(Float), cudaMemcpyHostToDevice));
        CHECK_CUDA_IPM(cudaMemcpy(d_c_, model.obj.data(), n_ * sizeof(Float), cudaMemcpyHostToDevice));

        // Upload CSR A for cuSPARSE SpMV
        d_A_row_ptrs_   = static_cast<Index*>(arena_.allocate((m_ + 1) * sizeof(Index)));
        d_A_col_indices_ = static_cast<Index*>(arena_.allocate(A_csr_.col_indices.size() * sizeof(Index)));
        d_A_vals_        = static_cast<Float*>(arena_.allocate(A_csr_.values.size() * sizeof(Float)));

        CHECK_CUDA_IPM(cudaMemcpy(d_A_row_ptrs_,    A_csr_.row_ptrs.data(),    (m_ + 1) * sizeof(Index),                     cudaMemcpyHostToDevice));
        CHECK_CUDA_IPM(cudaMemcpy(d_A_col_indices_,  A_csr_.col_indices.data(), A_csr_.col_indices.size() * sizeof(Index),    cudaMemcpyHostToDevice));
        CHECK_CUDA_IPM(cudaMemcpy(d_A_vals_,          A_csr_.values.data(),     A_csr_.values.size()     * sizeof(Float),    cudaMemcpyHostToDevice));

        CHECK_CUSPARSE_IPM(cusparseCreateCsr(&descr_A_, m_, n_,
            static_cast<int64_t>(A_csr_.values.size()),
            d_A_row_ptrs_, d_A_col_indices_, d_A_vals_,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_x_,    n_, d_x_,     CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_y_,    m_, d_y_,     CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_s_,    n_, d_s_,     CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_rp_,   m_, d_rp_,    CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_rd_,   n_, d_rd_,    CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_b_,    m_, d_b_,     CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_c_,    n_, d_c_,     CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_v_,    n_, d_v_,     CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_rkkt_, m_, d_r_kkt_, CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_dy_,   m_, d_r_kkt_, CUDA_R_64F)); // dy reuses d_r_kkt_ in-place
    }

    ~Impl() {
        delete kkt_;
        cusparseDestroyDnVec(vec_x_);
        cusparseDestroyDnVec(vec_y_);
        cusparseDestroyDnVec(vec_s_);
        cusparseDestroyDnVec(vec_rp_);
        cusparseDestroyDnVec(vec_rd_);
        cusparseDestroyDnVec(vec_b_);
        cusparseDestroyDnVec(vec_c_);
        cusparseDestroyDnVec(vec_v_);
        cusparseDestroyDnVec(vec_rkkt_);
        cusparseDestroyDnVec(vec_dy_);
        cusparseDestroySpMat(descr_A_);
        cusparseDestroy(handle_);
        arena_.free(d_A_row_ptrs_);
        arena_.free(d_A_col_indices_);
        arena_.free(d_A_vals_);
    }

    MehrotraResult solve();

private:
    // rp = Ax - b,   rd = A^T y + s - c
    void compute_residuals() {
        // rp = -b first
        CHECK_CUDA_IPM(cudaMemcpy(d_rp_, d_b_, m_ * sizeof(Float), cudaMemcpyDeviceToDevice));
        {
            int blocks_m = (m_ + 255) / 256;
            kernels::vector_add_kernel<<<blocks_m, 256>>>(m_, -2.0, d_b_, d_rp_); // rp = -b
        }
        // rp += A x
        Float alpha = 1.0, beta = 1.0;
        size_t bufferSize = 0;
        CHECK_CUSPARSE_IPM(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A_, vec_x_, &beta, vec_rp_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
        void* dBuffer = arena_.allocate(bufferSize);
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A_, vec_x_, &beta, vec_rp_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        arena_.free(dBuffer);

        // rd = s - c
        CHECK_CUDA_IPM(cudaMemcpy(d_rd_, d_s_, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
        {
            int blocks_n = (n_ + 255) / 256;
            kernels::vector_add_kernel<<<blocks_n, 256>>>(n_, -1.0, d_c_, d_rd_);
        }
        // rd += A^T y
        alpha = 1.0; beta = 1.0;
        CHECK_CUSPARSE_IPM(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, descr_A_, vec_y_, &beta, vec_rd_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
        dBuffer = arena_.allocate(bufferSize);
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, descr_A_, vec_y_, &beta, vec_rd_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        arena_.free(dBuffer);
    }

    // rkkt = -rp + A v
    void compute_rkkt() {
        CHECK_CUDA_IPM(cudaMemcpy(d_r_kkt_, d_rp_, m_ * sizeof(Float), cudaMemcpyDeviceToDevice));
        {
            int blocks_m = (m_ + 255) / 256;
            kernels::vector_add_kernel<<<blocks_m, 256>>>(m_, -2.0, d_rp_, d_r_kkt_); // rkkt = -rp
        }
        Float alpha = 1.0, beta = 1.0;
        size_t bufferSize = 0;
        CHECK_CUSPARSE_IPM(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A_, vec_v_, &beta, vec_rkkt_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
        void* dBuffer = arena_.allocate(bufferSize);
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A_, vec_v_, &beta, vec_rkkt_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        arena_.free(dBuffer);
    }

    // ds = -rd - A^T dy   (dy is d_r_kkt_ in-place after the KKT solve)
    void compute_ds(Float* d_ds_out) {
        CHECK_CUDA_IPM(cudaMemcpy(d_ds_out, d_rd_, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
        {
            int blocks_n = (n_ + 255) / 256;
            kernels::vector_add_kernel<<<blocks_n, 256>>>(n_, -2.0, d_rd_, d_ds_out); // ds = -rd
        }
        Float alpha = -1.0, beta = 1.0;
        cusparseDnVecDescr_t vec_ds;
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&vec_ds, n_, d_ds_out, CUDA_R_64F));
        size_t bufferSize = 0;
        CHECK_CUSPARSE_IPM(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, descr_A_, vec_dy_, &beta, vec_ds, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
        void* dBuffer = arena_.allocate(bufferSize);
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, descr_A_, vec_dy_, &beta, vec_ds, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        arena_.free(dBuffer);
        cusparseDestroyDnVec(vec_ds);
    }

    const core::Model& model_;
    core::CSRMatrix    A_csr_;       // CSR view of model.A for all GPU/symbolic APIs
    gpu::VRAMArena     arena_;
    Index m_, n_;
    SymbolicFactorization sym_;
    gpu::GPUKKTCholeskySolver* kkt_;
    cusparseHandle_t handle_;

    Float *d_x_, *d_y_, *d_s_;
    Float *d_rp_, *d_rd_;
    Float *d_Theta_, *d_r_xs_, *d_v_, *d_r_kkt_;
    Float *d_dx_aff_, *d_ds_aff_;
    Float *d_dx_, *d_ds_;
    Float *d_b_, *d_c_;

    Index *d_A_row_ptrs_, *d_A_col_indices_;
    Float *d_A_vals_;

    cusparseSpMatDescr_t descr_A_;
    cusparseDnVecDescr_t vec_x_, vec_y_, vec_s_, vec_rp_, vec_rd_,
                          vec_b_, vec_c_, vec_v_, vec_rkkt_, vec_dy_;
};

MehrotraResult MehrotraSolver::Impl::solve() {
    MehrotraResult result;
    result.status = simplex::SimplexStatus::Optimal;
    result.iterations = 0;

    // Initialize: x = 1, s = 1, y = 0
    kernels::initialize_variables(m_, n_, d_x_, d_y_, d_s_);
    cudaDeviceSynchronize();

    Index max_iter = 200;
    for (Index iter = 0; iter < max_iter; ++iter) {
        compute_residuals();
        cudaDeviceSynchronize();

        Float norm_rp = kernels::compute_norm(m_, d_rp_);
        Float norm_rd = kernels::compute_norm(n_, d_rd_);
        Float mu      = kernels::compute_mu(n_, d_x_, d_s_);

        if (norm_rp < math::kDefaultFeasibilityTol &&
            norm_rd < math::kDefaultFeasibilityTol &&
            mu      < math::kDefaultFeasibilityTol) {
            result.status          = simplex::SimplexStatus::Optimal;
            result.iterations      = iter;
            result.primal_residual = norm_rp;
            result.dual_residual   = norm_rd;
            result.duality_gap     = mu;
            result.objective_value = kernels::compute_objective(n_, d_c_, d_x_);
            return result;
        }

        // ---- Predictor (affine scaling) step ----
        kernels::compute_theta(n_, d_x_, d_s_, d_Theta_);
        kkt_->gpu_cholesky_factorize_device(d_Theta_);

        // r_xs_aff = -x * s  (sigma_mu = 0, no affine terms yet)
        kernels::compute_r_xs(n_, d_x_, d_s_, nullptr, nullptr, 0.0, d_r_xs_);
        kernels::compute_v(n_, d_Theta_, d_rd_, d_r_xs_, d_s_, d_v_);
        cudaDeviceSynchronize();

        compute_rkkt();
        cudaDeviceSynchronize();

        // Solve for dy_aff (result in d_r_kkt_ in-place)
        kkt_->gpu_cholesky_solve_device(d_r_kkt_);

        // ds_aff = -rd - A^T dy_aff
        compute_ds(d_ds_aff_);
        // dx_aff = -Theta * ds_aff + r_xs / s
        kernels::compute_dx(n_, d_Theta_, d_ds_aff_, d_r_xs_, d_s_, d_dx_aff_);
        cudaDeviceSynchronize();

        // Affine step lengths
        Float alpha_p_aff = kernels::compute_step_length(n_, d_x_, d_dx_aff_);
        Float alpha_d_aff = kernels::compute_step_length(n_, d_s_, d_ds_aff_);

        // mu_aff = (x + alpha_p_aff * dx_aff)^T (s + alpha_d_aff * ds_aff) / n
        Float* d_tmp_x = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        Float* d_tmp_s = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        CHECK_CUDA_IPM(cudaMemcpy(d_tmp_x, d_x_, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA_IPM(cudaMemcpy(d_tmp_s, d_s_, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
        kernels::update_variables(n_, alpha_p_aff, d_dx_aff_, d_tmp_x);
        kernels::update_variables(n_, alpha_d_aff, d_ds_aff_, d_tmp_s);
        cudaDeviceSynchronize();
        Float mu_aff = kernels::compute_mu(n_, d_tmp_x, d_tmp_s);
        arena_.free(d_tmp_s);
        arena_.free(d_tmp_x);

        Float sigma = std::pow(std::max(0.0, mu_aff) / std::max(1e-16, mu), 3.0);

        // ---- Corrector step ----
        // r_xs = -x*s + sigma*mu - dx_aff * ds_aff
        kernels::compute_r_xs(n_, d_x_, d_s_, d_dx_aff_, d_ds_aff_, sigma * mu, d_r_xs_);
        kernels::compute_v(n_, d_Theta_, d_rd_, d_r_xs_, d_s_, d_v_);
        cudaDeviceSynchronize();

        compute_rkkt();
        cudaDeviceSynchronize();

        kkt_->gpu_cholesky_solve_device(d_r_kkt_);

        compute_ds(d_ds_);
        kernels::compute_dx(n_, d_Theta_, d_ds_, d_r_xs_, d_s_, d_dx_);
        cudaDeviceSynchronize();

        // Corrector step lengths with fraction-to-boundary (eta = 0.995)
        Float alpha_p = kernels::compute_step_length(n_, d_x_, d_dx_);
        Float alpha_d = kernels::compute_step_length(n_, d_s_, d_ds_);
        constexpr Float eta = 0.995;
        alpha_p = std::min(1.0, eta * alpha_p);
        alpha_d = std::min(1.0, eta * alpha_d);

        // Update primal, dual, slack
        kernels::update_variables(n_, alpha_p, d_dx_, d_x_);
        kernels::update_variables(n_, alpha_d, d_ds_, d_s_);
        kernels::update_variables(m_, alpha_d, d_r_kkt_, d_y_); // d_r_kkt_ contains dy
        cudaDeviceSynchronize();
    }

    // Iteration limit
    compute_residuals();
    cudaDeviceSynchronize();
    result.status          = simplex::SimplexStatus::IterationLimit;
    result.iterations      = max_iter;
    result.primal_residual = kernels::compute_norm(m_, d_rp_);
    result.dual_residual   = kernels::compute_norm(n_, d_rd_);
    result.duality_gap     = kernels::compute_mu(n_, d_x_, d_s_);
    result.objective_value = kernels::compute_objective(n_, d_c_, d_x_);
    return result;
}

MehrotraSolver::MehrotraSolver(const core::Model& model) {
    impl_ = new Impl(model);
}

MehrotraSolver::~MehrotraSolver() {
    delete impl_;
}

MehrotraResult MehrotraSolver::solve() {
    return impl_->solve();
}

} // namespace ipm
} // namespace sankhya
