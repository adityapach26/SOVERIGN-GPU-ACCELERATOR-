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
#include <thrust/reduce.h>
#include <cusparse.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cstdlib>

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

#include "profiler.hpp"
#include <map>




#include <vector>
namespace sankhya {
namespace ipm {

struct AsyncEvent {
    std::string name;
    cudaEvent_t start;
    cudaEvent_t stop;
};
static std::vector<AsyncEvent> g_async_events;

inline void record_gpu_start(const std::string& name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    g_async_events.push_back({name, start, stop});
}

inline void record_gpu_stop() {
    if (!g_async_events.empty()) {
        cudaEventRecord(g_async_events.back().stop);
    }
}

inline void flush_gpu_events() {
    for (auto& ev : g_async_events) {
        cudaEventSynchronize(ev.stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev.start, ev.stop);
        sankhya::profile::add_time(ev.name, ms);
        cudaEventDestroy(ev.start);
        cudaEventDestroy(ev.stop);
    }
    g_async_events.clear();
}

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

__global__ void compute_e3_kernel(
    Index n, const Float* S, const Float* dx, const Float* X, const Float* ds, const Float* r_xs, Float* out
) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = S[i]*dx[i] + X[i]*ds[i] - r_xs[i];
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
    record_gpu_start("kernel_init_vars");
    init_vars_kernel<<<blocks, 256>>>(m, n, d_x, d_y, d_s);
    record_gpu_stop();
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
    record_gpu_start("kernel_compute_theta");
    compute_theta_kernel<<<blocks, 256>>>(n, d_x, d_s, d_Theta);
    record_gpu_stop();
}

// r_xs[i] = -x[i]*s[i] + sigma_mu - dx_aff[i]*ds_aff[i]
inline void compute_r_xs(Index n, const Float* d_x, const Float* d_s,
                         const Float* d_dx_aff, const Float* d_ds_aff,
                         Float sigma_mu, Float* d_r_xs) {
    int blocks = (n + 255) / 256;
    record_gpu_start("kernel_compute_r_xs");
    compute_r_xs_kernel<<<blocks, 256>>>(n, d_x, d_s, d_dx_aff, d_ds_aff, sigma_mu, d_r_xs);
    record_gpu_stop();
}

// v[i] = -Theta[i]*rd[i] - r_xs[i]/s[i]
inline void compute_v(Index n, const Float* d_Theta, const Float* d_rd,
                      const Float* d_r_xs, const Float* d_s, Float* d_v) {
    int blocks = (n + 255) / 256;
    record_gpu_start("kernel_compute_v");
    compute_v_kernel<<<blocks, 256>>>(n, d_Theta, d_rd, d_r_xs, d_s, d_v);
    record_gpu_stop();
}

// dx[i] = -Theta[i]*ds[i] + r_xs[i]/s[i]
inline void compute_dx(Index n, const Float* d_Theta, const Float* d_ds,
                       const Float* d_r_xs, const Float* d_s, Float* d_dx) {
    int blocks = (n + 255) / 256;
    record_gpu_start("kernel_compute_dx");
    compute_dx_kernel<<<blocks, 256>>>(n, d_Theta, d_ds, d_r_xs, d_s, d_dx);
    record_gpu_stop();
}

// Fraction-to-boundary step length: min over i where d[i] < 0 of -v[i]/d[i]
inline Float compute_step_length(Index n, const Float* d_v, const Float* d_dv) {
    thrust::device_ptr<const Float> pv(d_v), pdv(d_dv);
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(pv, pdv));
    Float min_ratio = thrust::transform_reduce(
        thrust::device, begin, begin + n,
        step_length_functor{}, Float(1.0e30), thrust::minimum<Float>());
    return min_ratio;
}

// x[i] += alpha * dx[i]
inline void update_variables(Index n, Float alpha, const Float* d_dx, Float* d_x) {
    int blocks = (n + 255) / 256;
    record_gpu_start("kernel_update_vars");
    update_vars_kernel<<<blocks, 256>>>(n, alpha, d_dx, d_x);
    record_gpu_stop();
}

} // namespace kernels

// ============================================================
// MehrotraSolver::Impl
// ============================================================
class MehrotraSolver::Impl {
public:
    Impl(const core::Model& model) : model_(model), arena_(1024 * 1024 * 64) {
        sankhya::profile::start_cpu("Presolve");
        m_ = model.A.rows;
        n_ = model.A.cols;

        // Convert CSC model matrix to CSR for AMD/symbolic/KKT APIs
        sankhya::profile::start_cpu("Matrix conversion");
        A_csr_ = core::to_csr(model.A);
        sankhya::profile::stop_cpu("Matrix conversion");
        
        b_scaled_ = model.rhs;
        c_scaled_ = model.obj;

        R_.assign(m_, 1.0);
        C_.assign(n_, 1.0);

        sankhya::profile::start_cpu("CPU model preparation");
        int num_ruiz_iters = 10;
        for (int iter = 0; iter < num_ruiz_iters; ++iter) {
            std::vector<Float> r_max(m_, 0.0);
            std::vector<Float> c_max(n_, 0.0);
            for (Index i = 0; i < m_; ++i) {
                for (Index p = A_csr_.row_ptrs[i]; p < A_csr_.row_ptrs[i+1]; ++p) {
                    Index j = A_csr_.col_indices[p];
                    Float val = std::abs(A_csr_.values[p]);
                    if (val > r_max[i]) r_max[i] = val;
                    if (val > c_max[j]) c_max[j] = val;
                }
            }
            for (Index i = 0; i < m_; ++i) {
                Float factor = (r_max[i] > 1e-8) ? 1.0 / std::sqrt(r_max[i]) : 1.0;
                R_[i] *= factor;
                b_scaled_[i] *= factor;
            }
            for (Index j = 0; j < n_; ++j) {
                Float factor = (c_max[j] > 1e-8) ? 1.0 / std::sqrt(c_max[j]) : 1.0;
                C_[j] *= factor;
                c_scaled_[j] *= factor;
            }
            for (Index i = 0; i < m_; ++i) {
                for (Index p = A_csr_.row_ptrs[i]; p < A_csr_.row_ptrs[i+1]; ++p) {
                    Index j = A_csr_.col_indices[p];
                    Float factor_r = (r_max[i] > 1e-8) ? 1.0 / std::sqrt(r_max[i]) : 1.0;
                    Float factor_c = (c_max[j] > 1e-8) ? 1.0 / std::sqrt(c_max[j]) : 1.0;
                    A_csr_.values[p] *= factor_r * factor_c;
                }
            }
        }

        sankhya::profile::stop_cpu("CPU model preparation");

        sankhya::profile::start_cpu("CSR construction");
        std::vector<Index> P = compute_amd_ordering(A_csr_);
        compute_symbolic_factorization(A_csr_, P, sym_);
        sankhya::profile::stop_cpu("Presolve");

        sankhya::profile::start_cpu("GPU initialization");
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
        sankhya::profile::stop_cpu("CSR construction");

        sankhya::profile::start_cpu("Basis factorization setup");
        kkt_ = new gpu::GPUKKTCholeskySolver(arena_, A_csr_, sym_, M_pattern);
        sankhya::profile::stop_cpu("Basis factorization setup");

        sankhya::profile::start_cpu("CUDA library initialization");
        CHECK_CUSPARSE_IPM(cusparseCreate(&handle_));
        sankhya::profile::stop_cpu("CUDA library initialization");
        sankhya::profile::stop_cpu("GPU initialization");

        sankhya::profile::start_cpu("GPU allocation");
        // Allocate persistent device vectors for IPM state
        sankhya::profile::start_cpu("GPU buffer preparation");
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
        
        // Upload CSR A for cuSPARSE SpMV
        d_A_row_ptrs_   = static_cast<Index*>(arena_.allocate((m_ + 1) * sizeof(Index)));
        d_A_col_indices_ = static_cast<Index*>(arena_.allocate(A_csr_.col_indices.size() * sizeof(Index)));
        d_A_vals_        = static_cast<Float*>(arena_.allocate(A_csr_.values.size() * sizeof(Float)));
        sankhya::profile::stop_cpu("GPU allocation");
        
        sankhya::profile::start_cpu("H->D transfer");
        sankhya::profile::stop_cpu("GPU buffer preparation");

        sankhya::profile::start_cpu("H->D transfer");
        CHECK_CUDA_IPM(cudaMemcpy(d_b_, b_scaled_.data(), m_ * sizeof(Float), cudaMemcpyHostToDevice));
        CHECK_CUDA_IPM(cudaMemcpy(d_c_, c_scaled_.data(), n_ * sizeof(Float), cudaMemcpyHostToDevice));

        CHECK_CUDA_IPM(cudaMemcpy(d_A_row_ptrs_,    A_csr_.row_ptrs.data(),    (m_ + 1) * sizeof(Index),                     cudaMemcpyHostToDevice));
        CHECK_CUDA_IPM(cudaMemcpy(d_A_col_indices_,  A_csr_.col_indices.data(), A_csr_.col_indices.size() * sizeof(Index),    cudaMemcpyHostToDevice));
        CHECK_CUDA_IPM(cudaMemcpy(d_A_vals_,          A_csr_.values.data(),     A_csr_.values.size()     * sizeof(Float),    cudaMemcpyHostToDevice));
        sankhya::profile::stop_cpu("H->D transfer");
        sankhya::profile::stop_cpu("H->D transfer");

        sankhya::profile::start_cpu("GPU initialization");
        sankhya::profile::start_cpu("GPU data structure setup");
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
        sankhya::profile::stop_cpu("GPU data structure setup");
        sankhya::profile::stop_cpu("GPU initialization");
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
    bool verify_newton_direction(const Float* d_dx, const Float* d_dy, const Float* d_ds, const Float* d_r_xs, Float norm_rp, Float norm_rd) {
        sankhya::profile::start_cpu("KKT verification");
        sankhya::profile::start_cpu("KKT vector/matrix operations");
        Float* d_r1 = static_cast<Float*>(arena_.allocate(m_ * sizeof(Float)));
        Float* d_r2 = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        Float* d_r3 = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));

        CHECK_CUDA_IPM(cudaMemcpy(d_r1, d_rp_, m_ * sizeof(Float), cudaMemcpyDeviceToDevice));
        Float a_spmv = 1.0, b_spmv = 1.0;
        cusparseDnVecDescr_t v_dx, v_r1, v_dy, v_r2;
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&v_dx, n_, (void*)d_dx, CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&v_r1, m_, d_r1, CUDA_R_64F));
        
        size_t buf1 = 0;
        CHECK_CUSPARSE_IPM(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &a_spmv, descr_A_, v_dx, &b_spmv, v_r1, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf1));
        void* dbuf = arena_.allocate(buf1);
        record_gpu_start("kernel_cusparseSpMV");
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &a_spmv, descr_A_, v_dx, &b_spmv, v_r1, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dbuf));
        record_gpu_stop();
        arena_.free(dbuf);

        CHECK_CUDA_IPM(cudaMemcpy(d_r2, d_rd_, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
        int blocks_n = (n_ + 255) / 256;
        kernels::vector_add_kernel<<<blocks_n, 256>>>(n_, 1.0, d_ds, d_r2);
        
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&v_dy, m_, (void*)d_dy, CUDA_R_64F));
        CHECK_CUSPARSE_IPM(cusparseCreateDnVec(&v_r2, n_, d_r2, CUDA_R_64F));
        
        size_t buf2 = 0;
        CHECK_CUSPARSE_IPM(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_TRANSPOSE,
            &a_spmv, descr_A_, v_dy, &b_spmv, v_r2, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf2));
        dbuf = arena_.allocate(buf2);
        record_gpu_start("kernel_cusparseSpMV");
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_TRANSPOSE,
            &a_spmv, descr_A_, v_dy, &b_spmv, v_r2, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dbuf));
        record_gpu_stop();
        arena_.free(dbuf);

        record_gpu_start("kernel_compute_e3");
    kernels::compute_e3_kernel<<<blocks_n, 256>>>(n_, d_s_, d_dx, d_x_, d_ds, d_r_xs, d_r3);
    record_gpu_stop();
        
        sankhya::profile::stop_cpu("KKT vector/matrix operations");

        sankhya::profile::start_cpu("KKT synchronization and reduction");
        Float norm_r1 = kernels::compute_norm(m_, d_r1);
        Float norm_r2 = kernels::compute_norm(n_, d_r2);
        Float norm_r3 = kernels::compute_norm(n_, d_r3);
        Float norm_rxs = kernels::compute_norm(n_, d_r_xs);

        Float norm_dx = kernels::compute_norm(n_, d_dx);
        Float norm_ds = kernels::compute_norm(n_, d_ds);

        // Measure FP accumulation error in late iterations
        sankhya::profile::stop_cpu("KKT synchronization and reduction");

        sankhya::profile::start_cpu("KKT residual calculation");
        Float e1 = norm_r1 / std::max(Float(1.0), norm_rp);
        Float e2 = norm_r2 / std::max(Float(1.0), norm_rd);
        Float e3 = norm_r3 / std::max(Float(1.0), norm_rxs);
        
        arena_.free(d_r3);
        arena_.free(d_r2);
        arena_.free(d_r1);
        cusparseDestroyDnVec(v_r2);
        cusparseDestroyDnVec(v_dy);
        cusparseDestroyDnVec(v_r1);
        cusparseDestroyDnVec(v_dx);

        std::cout << "[KKT Verify]"
                  << "  physical_M0_residual: " << norm_r1
                  << "  e1: " << e1
                  << "  e2: " << e2
                  << "  e3: " << e3
                  << std::endl;
        sankhya::profile::stop_cpu("KKT residual calculation");
        if (e1 > 1e-8 || e2 > 1e-8 || e3 > 1e-8) {
            sankhya::profile::add_time("KKT verification failures", 1.0);
            std::cout << "    [KKT Verify] FAILED!" << std::endl;
            if (model_.obj.size() == 138 || model_.obj.size() == 51) {
                Float norm_dy = kernels::compute_norm(m_, d_dy);
                Float norm_dx = kernels::compute_norm(n_, d_dx);
                Float norm_ds = kernels::compute_norm(n_, d_ds);
                std::cout << "      [Diag] norm_rp: " << norm_rp << "  norm_rd: " << norm_rd << std::endl
                          << "      [Diag] norm_dy: " << norm_dy << "  norm_dx: " << norm_dx << "  norm_ds: " << norm_ds << std::endl;
            }
            sankhya::profile::stop_cpu("KKT verification");
            return false;
        }
        sankhya::profile::stop_cpu("KKT verification");
        return true;
    }

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
        record_gpu_start("kernel_cusparseSpMV");
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A_, vec_x_, &beta, vec_rp_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        record_gpu_stop();
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
        record_gpu_start("kernel_cusparseSpMV");
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, descr_A_, vec_y_, &beta, vec_rd_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        record_gpu_stop();
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
        record_gpu_start("kernel_cusparseSpMV");
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A_, vec_v_, &beta, vec_rkkt_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        record_gpu_stop();
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
        record_gpu_start("kernel_cusparseSpMV");
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, descr_A_, vec_dy_, &beta, vec_ds, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        record_gpu_stop();
        arena_.free(dBuffer);
        cusparseDestroyDnVec(vec_ds);
    }

    const core::Model& model_;
    core::CSRMatrix    A_csr_;
    std::vector<Float> R_, C_;
    std::vector<Float> b_scaled_, c_scaled_;
    Float rp_scale0_, rd_scale0_, mu_scale0_;
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
    sankhya::profile::start_cpu("Sync: Init"); cudaDeviceSynchronize(); sankhya::profile::stop_cpu("Sync: Init"); flush_gpu_events();

    Index max_iter = 200;
    // Opt-in, bounded diagnostics for the small-LP globalization validation.
    const bool trace_globalization = n_ < 200 && m_ < 200 &&
                                     std::getenv("SANKHYA_IPM_TRACE") != nullptr;
    int trace_trials_left = 96;
    for (Index iter = 0; iter < max_iter; ++iter) {
        compute_residuals();
        sankhya::profile::start_cpu("Sync: Residuals"); cudaDeviceSynchronize(); sankhya::profile::stop_cpu("Sync: Residuals"); flush_gpu_events();

        Float norm_rp = kernels::compute_norm(m_, d_rp_);
        Float norm_rd = kernels::compute_norm(n_, d_rd_);
        Float mu      = kernels::compute_mu(n_, d_x_, d_s_);

        if (iter == 0) {
            rp_scale0_ = std::max(Float(1.0), norm_rp);
            rd_scale0_ = std::max(Float(1.0), norm_rd);
            mu_scale0_ = std::max(Float(1.0), mu);
        }

        if (model_.obj.size() == 138 && (iter % 10 == 0 || iter >= 195)) { // DIAGNOSTIC: only for adlittle
            Float theta_min = 1e30, theta_max = 0.0;
            Float x_min = 1e30, x_max = -1e30;
            Float s_min = 1e30, s_max = -1e30;
            if (iter > 0) { // d_Theta_ is populated after iter 0 starts
                thrust::device_ptr<const Float> pt(d_Theta_);
                theta_min = thrust::reduce(thrust::device, pt, pt + n_, Float(1e30), thrust::minimum<Float>());
                theta_max = thrust::reduce(thrust::device, pt, pt + n_, Float(-1e30), thrust::maximum<Float>());
            }
            thrust::device_ptr<const Float> px(d_x_), ps(d_s_);
            x_min = thrust::reduce(thrust::device, px, px + n_, Float(1e30), thrust::minimum<Float>());
            x_max = thrust::reduce(thrust::device, px, px + n_, Float(-1e30), thrust::maximum<Float>());
            s_min = thrust::reduce(thrust::device, ps, ps + n_, Float(1e30), thrust::minimum<Float>());
            s_max = thrust::reduce(thrust::device, ps, ps + n_, Float(-1e30), thrust::maximum<Float>());
            
            std::cout << "  Iter " << iter 
                      << " | rp: " << norm_rp 
                      << " | rd: " << norm_rd 
                      << " | mu: " << mu 
                      << " | th_min: " << theta_min 
                      << " | th_max: " << theta_max 
                      << " | x_min: " << x_min << " | x_max: " << x_max
                      << " | s_min: " << s_min << " | s_max: " << s_max << std::endl;
        }

        if (norm_rp < math::kDefaultFeasibilityTol &&
            norm_rd < math::kDefaultFeasibilityTol &&
            mu      < math::kDefaultFeasibilityTol) {
            result.status          = simplex::SimplexStatus::Optimal;
            result.iterations      = iter;
            result.primal_residual = norm_rp;
            result.dual_residual   = norm_rd;
            result.duality_gap     = mu;
            result.objective_value = kernels::compute_objective(n_, d_c_, d_x_);
            
            result.x.resize(n_);
            result.pi.resize(m_);
            sankhya::profile::start_cpu("D->H transfer");
            CHECK_CUDA_IPM(cudaMemcpy(result.x.data(), d_x_, n_ * sizeof(Float), cudaMemcpyDeviceToHost));
            CHECK_CUDA_IPM(cudaMemcpy(result.pi.data(), d_y_, m_ * sizeof(Float), cudaMemcpyDeviceToHost));
            sankhya::profile::stop_cpu("D->H transfer");
            for (Index i = 0; i < n_; ++i) result.x[i] *= C_[i];
            for (Index i = 0; i < m_; ++i) result.pi[i] *= R_[i];
            
            return result;
        }

        // ---- Predictor (affine scaling) step ----
        kernels::compute_theta(n_, d_x_, d_s_, d_Theta_);
        
        try {
            record_gpu_start("kernel_cholesky_factorize");
        kkt_->gpu_cholesky_factorize_device(d_Theta_);
        record_gpu_stop();
        } catch (const std::exception& e) {
            if (model_.obj.size() == 138) {
                std::cout << "  [ADLITTLE DIAGNOSTIC] Cholesky failed at iter " << iter << " with: " << e.what() << std::endl;
            }
            throw; // Rethrow to maintain existing behavior
        }

        // r_xs_aff = -x * s  (sigma_mu = 0, no affine terms yet)
        kernels::compute_r_xs(n_, d_x_, d_s_, nullptr, nullptr, 0.0, d_r_xs_);
        kernels::compute_v(n_, d_Theta_, d_rd_, d_r_xs_, d_s_, d_v_);
        sankhya::profile::start_cpu("Sync: Predictor Setup"); cudaDeviceSynchronize(); sankhya::profile::stop_cpu("Sync: Predictor Setup"); flush_gpu_events();

        compute_rkkt();
        sankhya::profile::start_cpu("Sync: Predictor RHS"); cudaDeviceSynchronize(); sankhya::profile::stop_cpu("Sync: Predictor RHS"); flush_gpu_events();

        Float true_rhs_norm_p = kernels::compute_norm(m_, d_r_kkt_);
        std::cout << "[PREDICTOR] Pre-solve RHS norm: " << true_rhs_norm_p << std::endl;

        // Solve for dy_aff (result in d_r_kkt_ in-place)
        if (!kkt_->gpu_cholesky_solve_device(d_r_kkt_)) {
            result.status = simplex::SimplexStatus::IterationLimit; // Numerical failure
            result.iterations = iter;
            break;
        }

        // ds_aff = -rd - A^T dy_aff
        compute_ds(d_ds_aff_);
        // dx_aff = -Theta * ds_aff + r_xs / s
        kernels::compute_dx(n_, d_Theta_, d_ds_aff_, d_r_xs_, d_s_, d_dx_aff_);
        sankhya::profile::start_cpu("Sync: Predictor Solve"); cudaDeviceSynchronize(); sankhya::profile::stop_cpu("Sync: Predictor Solve"); flush_gpu_events();

        Float dy_norm_p = kernels::compute_norm(m_, d_r_kkt_);
        Float dx_norm_p  = kernels::compute_norm(n_, d_dx_aff_);
        Float ds_norm_p  = kernels::compute_norm(n_, d_ds_aff_);
        
        std::cout << "[PREDICTOR]"
                  << "  dy_norm: " << dy_norm_p
                  << "  dx_norm: " << dx_norm_p
                  << "  ds_norm: " << ds_norm_p
                  << std::endl;

        if (!verify_newton_direction(d_dx_aff_, d_r_kkt_, d_ds_aff_, d_r_xs_, norm_rp, norm_rd)) {
            // verify_newton_direction rejected the predictor direction.
            // This is a NUMERICAL FAILURE (ill-conditioned KKT system, residual too large),
            // NOT a mathematically certified infeasibility result.
            // Use IterationLimit so the Phase-I recovery path can be invoked by the caller.
            result.status = simplex::SimplexStatus::IterationLimit;
            result.iterations = iter;
            break;
        }

        // Affine step lengths (capped at 1.0)
        Float alpha_p_aff = std::min(Float(1.0), kernels::compute_step_length(n_, d_x_, d_dx_aff_));
        Float alpha_d_aff = std::min(Float(1.0), kernels::compute_step_length(n_, d_s_, d_ds_aff_));

        // mu_aff = (x + alpha_p_aff * dx_aff)^T (s + alpha_d_aff * ds_aff) / n
        Float* d_tmp_x = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        Float* d_tmp_s = static_cast<Float*>(arena_.allocate(n_ * sizeof(Float)));
        CHECK_CUDA_IPM(cudaMemcpy(d_tmp_x, d_x_, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA_IPM(cudaMemcpy(d_tmp_s, d_s_, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
        kernels::update_variables(n_, alpha_p_aff, d_dx_aff_, d_tmp_x);
        kernels::update_variables(n_, alpha_d_aff, d_ds_aff_, d_tmp_s);
        sankhya::profile::start_cpu("Sync: Affine Step"); cudaDeviceSynchronize(); sankhya::profile::stop_cpu("Sync: Affine Step"); flush_gpu_events();
        Float mu_aff = kernels::compute_mu(n_, d_tmp_x, d_tmp_s);
        arena_.free(d_tmp_s);
        arena_.free(d_tmp_x);

        Float sigma = std::min(1.0, std::pow(std::max(0.0, mu_aff) / std::max(1e-16, mu), 3.0));

        // ---- Corrector step ----
        // r_xs = -x*s + sigma*mu - dx_aff * ds_aff
        kernels::compute_r_xs(n_, d_x_, d_s_, d_dx_aff_, d_ds_aff_, sigma * mu, d_r_xs_);
        kernels::compute_v(n_, d_Theta_, d_rd_, d_r_xs_, d_s_, d_v_);
        sankhya::profile::start_cpu("Sync: Corrector Setup"); cudaDeviceSynchronize(); sankhya::profile::stop_cpu("Sync: Corrector Setup"); flush_gpu_events();

        compute_rkkt();
        sankhya::profile::start_cpu("Sync: Corrector RHS"); cudaDeviceSynchronize(); sankhya::profile::stop_cpu("Sync: Corrector RHS"); flush_gpu_events();

        Float true_rhs_norm_c = kernels::compute_norm(m_, d_r_kkt_);
        std::cout << "[CORRECTOR] Pre-solve RHS norm: " << true_rhs_norm_c << std::endl;

        if (!kkt_->gpu_cholesky_solve_device(d_r_kkt_)) {
            result.status = simplex::SimplexStatus::IterationLimit; // Numerical failure
            result.iterations = iter;
            break;
        }

        compute_ds(d_ds_);
        kernels::compute_dx(n_, d_Theta_, d_ds_, d_r_xs_, d_s_, d_dx_);
        sankhya::profile::start_cpu("Sync: Corrector Solve"); cudaDeviceSynchronize(); sankhya::profile::stop_cpu("Sync: Corrector Solve"); flush_gpu_events();

        Float dy_norm_c = kernels::compute_norm(m_, d_r_kkt_);
        Float dx_norm_c  = kernels::compute_norm(n_, d_dx_);
        Float ds_norm_c  = kernels::compute_norm(n_, d_ds_);
        
        std::cout << "[CORRECTOR]"
                  << "  dy_norm: " << dy_norm_c
                  << "  dx_norm: " << dx_norm_c
                  << "  ds_norm: " << ds_norm_c
                  << std::endl;

        bool corrector_valid = verify_newton_direction(d_dx_, d_r_kkt_, d_ds_, d_r_xs_, norm_rp, norm_rd);

        // Corrector step lengths with fraction-to-boundary (eta = 0.995)
        Float alpha_p_max = kernels::compute_step_length(n_, d_x_, d_dx_);
        Float alpha_d_max = kernels::compute_step_length(n_, d_s_, d_ds_);
        constexpr Float eta = 0.995;
        Float alpha_p = std::min(Float(1.0), eta * alpha_p_max);
        Float alpha_d = std::min(Float(1.0), eta * alpha_d_max);

        // The max merit need not descend along a Mehrotra direction with
        // separate primal/dual steps. Use a sum so residual reduction can
        // compensate for a temporary increase in complementarity. Armijo
        // decrease of this sum still bounds every normalized component.
        Float orig_Phi = std::max({norm_rp / rp_scale0_, norm_rd / rd_scale0_, mu / mu_scale0_});
        const Float orig_merit = norm_rp / rp_scale0_ + norm_rd / rd_scale0_ + mu / mu_scale0_;
        constexpr Float armijo = 1e-4;
        
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
        
        for (int direction = (corrector_valid ? 0 : 1); direction < 2 && !step_accepted; ++direction) {
            if (direction == 1) {
                // Centered safeguard, using exactly the existing elimination:
                // S dx + X ds = -X s + sigma_c * mu * e,
                // ds = -rd - A^T dy, dx = -Theta ds + r_xs / s,
                // (A Theta A^T) dy = -rp + A(-Theta rd - r_xs / s).
                // sigma_c = 1/2 retains half the average complementarity as
                // a positive centering target and removes half per unit step.
                // With a COMMON step: rp'=-rp, rd'=-rd, mu'=-mu/2, hence
                // merit' <= -merit/2. No affine cross term belongs in this RHS.
                constexpr Float sigma_c = 0.5;
                kernels::compute_r_xs(n_, d_x_, d_s_, nullptr, nullptr, sigma_c * mu, d_r_xs_);
                kernels::compute_v(n_, d_Theta_, d_rd_, d_r_xs_, d_s_, d_v_);
                compute_rkkt();
                if (!kkt_->gpu_cholesky_solve_device(d_r_kkt_)) break;
                compute_ds(d_ds_);
                kernels::compute_dx(n_, d_Theta_, d_ds_, d_r_xs_, d_s_, d_dx_);
                if (!verify_newton_direction(d_dx_, d_r_kkt_, d_ds_, d_r_xs_, norm_rp, norm_rd)) break;

                const Float boundary = std::min(kernels::compute_step_length(n_, d_x_, d_dx_),
                                                kernels::compute_step_length(n_, d_s_, d_ds_));
                current_alpha_p = current_alpha_d = std::min(Float(1.0), eta * boundary);
            }
            backtrack_iters = 0;
            const int max_backtracks = direction == 0 ? 15 : 30;
            while (backtrack_iters < max_backtracks) {
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
                record_gpu_start("kernel_cusparseSpMV");
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &a_spmv, descr_A_, vec_bt_x, &b_spmv, vec_bt_rp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dbuf1));
        record_gpu_stop();
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
                record_gpu_start("kernel_cusparseSpMV");
        CHECK_CUSPARSE_IPM(cusparseSpMV(handle_, CUSPARSE_OPERATION_TRANSPOSE,
                    &a_spmv, descr_A_, vec_bt_y, &b_spmv, vec_bt_rd, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dbuf2));
        record_gpu_stop();
                arena_.free(dbuf2);

                Float rp_new_norm = kernels::compute_norm(m_, d_bt_rp);
                Float rd_new_norm = kernels::compute_norm(n_, d_bt_rd);
                Float mu_new = kernels::compute_mu(n_, d_bt_x, d_bt_s);

                Float Phi_new = std::max({rp_new_norm / rp_scale0_, rd_new_norm / rd_scale0_, mu_new / mu_scale0_});

                thrust::device_ptr<Float> ptr_x(d_bt_x);
                thrust::device_ptr<Float> ptr_s(d_bt_s);
                Float min_x = thrust::reduce(thrust::device, ptr_x, ptr_x + n_, Float(1.0e30), thrust::minimum<Float>());
                Float min_s = thrust::reduce(thrust::device, ptr_s, ptr_s + n_, Float(1.0e30), thrust::minimum<Float>());

                const Float merit_new = rp_new_norm / rp_scale0_ + rd_new_norm / rd_scale0_ + mu_new / mu_scale0_;
                const Float step = std::min(current_alpha_p, current_alpha_d);
                const bool finite = std::isfinite(rp_new_norm) && std::isfinite(rd_new_norm) &&
                                    std::isfinite(mu_new) && std::isfinite(merit_new) &&
                                    std::isfinite(orig_merit) && std::isfinite(min_x) && std::isfinite(min_s) &&
                                    std::isfinite(current_alpha_p) && std::isfinite(current_alpha_d) &&
                                    std::isfinite(kernels::compute_norm(m_, d_bt_y));
                const bool positive = min_x > 0.0 && min_s > 0.0 && mu_new > 0.0 && step > 0.0;
                const bool progress = merit_new < orig_merit &&
                                      merit_new <= (1.0 - armijo * step) * orig_merit;
                const char* reason = !finite ? "reject: non-finite trial" :
                                     !positive ? "reject: positivity/step" :
                                     !progress ? "reject: insufficient merit decrease" : "accept";
                if (trace_globalization && trace_trials_left > 0) {
                    --trace_trials_left;
                    std::cout << "    [Globalization] " << (direction == 0 ? "Mehrotra" : "Centered")
                              << " iter " << iter << " bt " << backtrack_iters << ": " << reason << std::endl
                              << "      a_p: " << current_alpha_p << " a_d: " << current_alpha_d << std::endl
                              << "      min(x+adx): " << min_x << " min(s+ads): " << min_s << std::endl
                              << "      rp_before: " << norm_rp << " rp_after: " << rp_new_norm << std::endl
                              << "      rd_before: " << norm_rd << " rd_after: " << rd_new_norm << std::endl
                              << "      mu_before: " << mu << " mu_after: " << mu_new << std::endl
                              << "      Phi_before: " << orig_Phi << " Phi_after: " << Phi_new << std::endl
                              << "      merit_before: " << orig_merit << " merit_after: " << merit_new << std::endl;
                }

                if (finite && positive && progress) {
                    step_accepted = true;
                    break;
                }

                current_alpha_p *= 0.5;
                current_alpha_d *= 0.5;
                backtrack_iters++;
            }
        }

        // Commit the exact trial that passed all checks; failed searches leave
        // the current iterate untouched (including a failed safeguard solve).
        if (step_accepted) {
            CHECK_CUDA_IPM(cudaMemcpy(d_x_, d_bt_x, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA_IPM(cudaMemcpy(d_y_, d_bt_y, m_ * sizeof(Float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA_IPM(cudaMemcpy(d_s_, d_bt_s, n_ * sizeof(Float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA_IPM(cudaDeviceSynchronize());
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
        alpha_d = current_alpha_d;

        if (model_.obj.size() == 138 && (iter % 10 == 0 || iter >= 195)) {
            std::cout << "    sigma: " << sigma 
                      << " | a_p_aff: " << alpha_p_aff << " | a_d_aff: " << alpha_d_aff
                      << " | a_p: " << alpha_p << " | a_d: " << alpha_d 
                      << " | bt_iters: " << backtrack_iters << std::endl;
        }
    }

    // End of loop
    compute_residuals();
    sankhya::profile::start_cpu("Sync: Final Residuals"); cudaDeviceSynchronize(); sankhya::profile::stop_cpu("Sync: Final Residuals"); flush_gpu_events();
    if (result.status == simplex::SimplexStatus::Optimal) {
        result.status          = simplex::SimplexStatus::IterationLimit;
        result.iterations      = max_iter;
    }
    result.primal_residual = kernels::compute_norm(m_, d_rp_);
    result.dual_residual   = kernels::compute_norm(n_, d_rd_);
    result.duality_gap     = kernels::compute_mu(n_, d_x_, d_s_);
    result.objective_value = kernels::compute_objective(n_, d_c_, d_x_);
    
    result.x.resize(n_);
    result.pi.resize(m_);
    CHECK_CUDA_IPM(cudaMemcpy(result.x.data(), d_x_, n_ * sizeof(Float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_IPM(cudaMemcpy(result.pi.data(), d_y_, m_ * sizeof(Float), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < n_; ++i) result.x[i] *= C_[i];
    for (Index i = 0; i < m_; ++i) result.pi[i] *= R_[i];
    
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
















