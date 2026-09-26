#include "cuda/feasibility_pump.cuh"
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <string>

using namespace nvcuda;
using namespace sankhya::math;

namespace sankhya {
namespace gpu {

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

/**
 * @brief Computes r = A * x - b using Tensor Cores
 *
 * Each warp computes a 16x16 tile of the output. 
 * Since the output is a vector, we only need 1 warp per 16 rows,
 * and we only use the first column of the resulting MMA tile.
 */
__global__ void wmma_compute_residual(
    Index rows, Index cols,
    const Index* A_col_ptrs, const Index* A_row_indices, const Float* A_values,
    const Float* b,
    const Float* x,
    Float* r
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int row_start = warp_id * WMMA_M;
    if (row_start >= rows) return;

    int lane_id = threadIdx.x % 32;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    int warp_in_block = threadIdx.x / 32;
    __shared__ half A_tile[4][WMMA_M * WMMA_K];
    __shared__ half X_tile[4][WMMA_K * WMMA_N];

    // Iterate over columns in blocks of WMMA_K
    for (int k_start = 0; k_start < cols; k_start += WMMA_K) {
        // Initialize shared memory tiles with 0.0 (padding for non-multiples of 16)
        if (lane_id < WMMA_M * WMMA_K) {
            A_tile[warp_in_block][lane_id] = __float2half(0.0f);
        }
        if (lane_id < WMMA_K * WMMA_N) {
            X_tile[warp_in_block][lane_id] = __float2half(0.0f);
        }
        __syncwarp();

        // Threads cooperatively load non-zeros from the CSC matrix into A_tile
        if (lane_id < WMMA_K) {
            int col = k_start + lane_id;
            if (col < cols) {
                Index start = A_col_ptrs[col];
                Index end = A_col_ptrs[col + 1];
                for (Index p = start; p < end; ++p) {
                    Index row = A_row_indices[p];
                    if (row >= row_start && row < row_start + WMMA_M) {
                        A_tile[warp_in_block][(row - row_start) * WMMA_K + lane_id] = __float2half(static_cast<float>(A_values[p]));
                    }
                }
                // X is a vector, treated as a matrix with non-zeros only in the 0th column
                X_tile[warp_in_block][lane_id * WMMA_N + 0] = __float2half(static_cast<float>(x[col]));
            }
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> x_frag;

        wmma::load_matrix_sync(a_frag, A_tile[warp_in_block], WMMA_K);
        wmma::load_matrix_sync(x_frag, X_tile[warp_in_block], WMMA_N);

        wmma::mma_sync(acc_frag, a_frag, x_frag, acc_frag);
        __syncwarp();
    }

    __shared__ float R_tile[4][WMMA_M * WMMA_N];
    wmma::store_matrix_sync(R_tile[warp_in_block], acc_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    if (lane_id < WMMA_M) {
        int row = row_start + lane_id;
        if (row < rows) {
            float ax = R_tile[warp_in_block][lane_id * WMMA_N + 0];
            r[row] = static_cast<Float>(ax) - b[row];
        }
    }
}

/**
 * @brief Computes d = - A^T * r using Tensor Cores
 */
__global__ void wmma_compute_direction(
    Index rows, Index cols,
    const Index* A_col_ptrs, const Index* A_row_indices, const Float* A_values,
    const Float* r,
    Float* d
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int col_start = warp_id * WMMA_M;
    if (col_start >= cols) return;

    int lane_id = threadIdx.x % 32;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    int warp_in_block = threadIdx.x / 32;
    __shared__ half AT_tile[4][WMMA_M * WMMA_K];
    __shared__ half R_tile[4][WMMA_K * WMMA_N];

    for (int j_start = 0; j_start < rows; j_start += WMMA_K) {
        if (lane_id < WMMA_M * WMMA_K) {
            AT_tile[warp_in_block][lane_id] = __float2half(0.0f);
        }
        if (lane_id < WMMA_K * WMMA_N) {
            R_tile[warp_in_block][lane_id] = __float2half(0.0f);
        }
        __syncwarp();

        // Fill A^T tile of size WMMA_M x WMMA_K. 
        // A^T has 'cols' rows and 'rows' columns.
        if (lane_id < WMMA_M) {
            int col = col_start + lane_id;
            if (col < cols) {
                Index start = A_col_ptrs[col];
                Index end = A_col_ptrs[col + 1];
                for (Index p = start; p < end; ++p) {
                    Index row = A_row_indices[p];
                    if (row >= j_start && row < j_start + WMMA_K) {
                        AT_tile[warp_in_block][lane_id * WMMA_K + (row - j_start)] = __float2half(static_cast<float>(A_values[p]));
                    }
                }
            }
        }
        
        // R is a vector, non-zeros only in the 0th column
        if (lane_id < WMMA_K) {
            int row = j_start + lane_id;
            if (row < rows) {
                R_tile[warp_in_block][lane_id * WMMA_N + 0] = __float2half(static_cast<float>(r[row]));
            }
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> r_frag;

        wmma::load_matrix_sync(a_frag, AT_tile[warp_in_block], WMMA_K);
        wmma::load_matrix_sync(r_frag, R_tile[warp_in_block], WMMA_N);

        wmma::mma_sync(acc_frag, a_frag, r_frag, acc_frag);
        __syncwarp();
    }

    __shared__ float D_tile[4][WMMA_M * WMMA_N];
    wmma::store_matrix_sync(D_tile[warp_in_block], acc_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    if (lane_id < WMMA_M) {
        int col = col_start + lane_id;
        if (col < cols) {
            float at_r = D_tile[warp_in_block][lane_id * WMMA_N + 0];
            d[col] = -static_cast<Float>(at_r);
        }
    }
}

std::vector<Float> compute_fp_projection_direction(
    const DeviceModel& model,
    const std::vector<Float>& x_tilde,
    VRAMArena& arena
) {
    if (model.cols == 0 || model.rows == 0) {
        return std::vector<Float>(model.cols, 0.0);
    }

    Float* d_x_tilde = static_cast<Float*>(arena.allocate(model.cols * sizeof(Float)));
    Float* d_r = static_cast<Float*>(arena.allocate(model.rows * sizeof(Float)));
    Float* d_d = static_cast<Float*>(arena.allocate(model.cols * sizeof(Float)));

    cudaError_t err = cudaMemcpy(d_x_tilde, x_tilde.data(), model.cols * sizeof(Float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy for d_x_tilde failed: ") + cudaGetErrorString(err));
    }

    // Compute grid dimensions for 1 warp per 16 rows/cols
    int num_warps_r = (model.rows + WMMA_M - 1) / WMMA_M;
    int block_size_r = 128; // 4 warps per block
    int grid_size_r = (num_warps_r * 32 + block_size_r - 1) / block_size_r;
    if (grid_size_r == 0) grid_size_r = 1;

    wmma_compute_residual<<<grid_size_r, block_size_r>>>(
        model.rows, model.cols,
        model.col_ptrs, model.row_indices, model.values,
        model.rhs, d_x_tilde, d_r
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("wmma_compute_residual kernel launch failed: ") + cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("wmma_compute_residual kernel execution failed: ") + cudaGetErrorString(err));
    }

    int num_warps_d = (model.cols + WMMA_M - 1) / WMMA_M;
    int block_size_d = 128;
    int grid_size_d = (num_warps_d * 32 + block_size_d - 1) / block_size_d;
    if (grid_size_d == 0) grid_size_d = 1;

    wmma_compute_direction<<<grid_size_d, block_size_d>>>(
        model.rows, model.cols,
        model.col_ptrs, model.row_indices, model.values,
        d_r, d_d
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("wmma_compute_direction kernel launch failed: ") + cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("wmma_compute_direction kernel execution failed: ") + cudaGetErrorString(err));
    }

    std::vector<Float> direction(model.cols);
    err = cudaMemcpy(direction.data(), d_d, model.cols * sizeof(Float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy for direction failed: ") + cudaGetErrorString(err));
    }

    arena.free(d_d);
    arena.free(d_r);
    arena.free(d_x_tilde);

    return direction;
}

// ---------------------------------------------------------------------------
// Phase 23.2 — Strict FP64 CUDA-Core Incumbent Auditor
// ---------------------------------------------------------------------------

/**
 * @brief FP64 CUDA-core kernel: evaluates |Ax - b| for every row and writes
 *        the per-row squared residual into d_row_residual.
 *
 * Each thread block is responsible for one or more rows. We use a simple
 * parallel reduction over columns within each row to compute (Ax)_i, then
 * subtract b_i. Arithmetic is entirely in double precision (no WMMA, no half).
 *
 * The CSC layout means column j contributes A[j][row] * x[j] to row `row`.
 * We iterate over all columns and accumulate directly into a per-row double
 * accumulator stored in shared memory (one accumulator per row in the block).
 *
 * Grid: one block per row (grid_size = rows).
 * Block: up to 256 threads cooperatively reduce over columns.
 */
__global__ void fp64_audit_kernel(
    Index rows, Index cols,
    const Index* A_col_ptrs,
    const Index* A_row_indices,
    const double* A_values_d,
    const double* b_d,
    const double* x_d,
    double* d_row_residual   // output: |residual_i| per row (size = rows)
) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;

    // Shared accumulator for this row (one double per block)
    __shared__ double s_ax;
    if (threadIdx.x == 0) {
        s_ax = 0.0;
    }
    __syncthreads();

    // Iterate over columns in chunks of blockDim.x.
    // Each thread accumulates its own partial sum, then we atomicAdd into s_ax.
    double local_sum = 0.0;
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += static_cast<int>(blockDim.x)) {
        // Find A[row][col] in CSC format
        Index start = A_col_ptrs[col];
        Index end   = A_col_ptrs[col + 1];
        for (Index p = start; p < end; ++p) {
            if (A_row_indices[p] == row) {
                local_sum += A_values_d[p] * x_d[col];
                break; // Each (row, col) pair appears at most once in CSC
            }
        }
    }

    // Reduce across threads via atomicAdd (double atomics require SM60+; T4 is SM75)
    atomicAdd(&s_ax, local_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        double residual = s_ax - b_d[row];
        d_row_residual[row] = (residual < 0.0 ? -residual : residual); // |residual|
    }
}

/**
 * @brief GPU reduce kernel: finds max over d_row_residual[0..rows-1].
 *        Result is written to d_max_residual[0].
 *        Uses a simple single-block reduction; rows <= typical LP row count.
 */
__global__ void fp64_max_reduce_kernel(
    const double* d_row_residual,
    Index rows,
    double* d_max_residual
) {
    extern __shared__ double s_data[];

    int tid = static_cast<int>(threadIdx.x);
    int stride = static_cast<int>(blockDim.x);

    double local_max = 0.0;
    for (int i = tid; i < rows; i += stride) {
        double v = d_row_residual[i];
        if (v > local_max) local_max = v;
    }
    s_data[tid] = local_max;
    __syncthreads();

    // Standard tree reduction
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_data[tid + s] > s_data[tid]) {
                s_data[tid] = s_data[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_max_residual[0] = s_data[0];
    }
}

bool verify_incumbent_fp64(
    const DeviceModel& model,
    const std::vector<Float>& candidate
) {
    if (model.rows == 0 || model.cols == 0) {
        return true; // Vacuously feasible
    }

    // -----------------------------------------------------------------------
    // 1. Upload candidate x and reinterpret model data as double on device.
    //    Float == double in this project (see types.hpp), so pointers alias safely.
    //    We still promote explicitly through typed device allocations to make
    //    the FP64 intent clear and to guard against any future Float == float.
    // -----------------------------------------------------------------------
    const std::size_t sz_cols = static_cast<std::size_t>(model.cols);
    const std::size_t sz_rows = static_cast<std::size_t>(model.rows);
    const std::size_t sz_nnz  = static_cast<std::size_t>(model.nnz);

    // Allocate device buffers via cudaMalloc (VRAMArena is not injected here
    // to keep the API simple; the audit is a light, infrequent operation).
    double* d_x   = nullptr;
    double* d_b   = nullptr;
    double* d_Av  = nullptr; // A values as double
    double* d_row_res = nullptr;
    double* d_max_res = nullptr;

    auto check = [](cudaError_t e, const char* msg) {
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
        }
    };

    check(cudaMalloc(&d_x,       sz_cols * sizeof(double)), "fp64 audit: alloc d_x");
    check(cudaMalloc(&d_b,       sz_rows * sizeof(double)), "fp64 audit: alloc d_b");
    check(cudaMalloc(&d_Av,      sz_nnz  * sizeof(double)), "fp64 audit: alloc d_Av");
    check(cudaMalloc(&d_row_res, sz_rows * sizeof(double)), "fp64 audit: alloc d_row_res");
    check(cudaMalloc(&d_max_res, sizeof(double)),            "fp64 audit: alloc d_max_res");

    // Promote candidate to double on host (no-op when Float==double, safe otherwise)
    std::vector<double> x_d(sz_cols);
    for (std::size_t j = 0; j < sz_cols; ++j) {
        x_d[j] = static_cast<double>(candidate[j]);
    }

    check(cudaMemcpy(d_x, x_d.data(), sz_cols * sizeof(double), cudaMemcpyHostToDevice),
          "fp64 audit: copy d_x");

    // b is stored in device memory via model.rhs (Float* which is double* here)
    // Copy device→device at double precision
    check(cudaMemcpy(d_b, model.rhs, sz_rows * sizeof(double), cudaMemcpyDeviceToDevice),
          "fp64 audit: copy d_b");

    // A values: already double in VRAM via model.values
    check(cudaMemcpy(d_Av, model.values, sz_nnz * sizeof(double), cudaMemcpyDeviceToDevice),
          "fp64 audit: copy d_Av");

    // -----------------------------------------------------------------------
    // 2. Launch FP64 audit kernel: one block per row, 256 threads per block.
    // -----------------------------------------------------------------------
    {
        int block_threads = 256;
        int grid_rows = static_cast<int>(model.rows);
        fp64_audit_kernel<<<grid_rows, block_threads>>>(
            model.rows, model.cols,
            model.col_ptrs, model.row_indices,
            d_Av, d_b, d_x,
            d_row_res
        );
        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            cudaFree(d_x); cudaFree(d_b); cudaFree(d_Av);
            cudaFree(d_row_res); cudaFree(d_max_res);
            throw std::runtime_error(std::string("fp64_audit_kernel launch failed: ") + cudaGetErrorString(kerr));
        }
        kerr = cudaDeviceSynchronize();
        if (kerr != cudaSuccess) {
            cudaFree(d_x); cudaFree(d_b); cudaFree(d_Av);
            cudaFree(d_row_res); cudaFree(d_max_res);
            throw std::runtime_error(std::string("fp64_audit_kernel execution failed: ") + cudaGetErrorString(kerr));
        }
    }

    // -----------------------------------------------------------------------
    // 3. Find max |residual| via a second single-block reduction kernel.
    // -----------------------------------------------------------------------
    {
        int reduce_threads = 256;
        std::size_t smem = static_cast<std::size_t>(reduce_threads) * sizeof(double);
        fp64_max_reduce_kernel<<<1, reduce_threads, smem>>>(d_row_res, model.rows, d_max_res);
        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            cudaFree(d_x); cudaFree(d_b); cudaFree(d_Av);
            cudaFree(d_row_res); cudaFree(d_max_res);
            throw std::runtime_error(std::string("fp64_max_reduce_kernel launch failed: ") + cudaGetErrorString(kerr));
        }
        kerr = cudaDeviceSynchronize();
        if (kerr != cudaSuccess) {
            cudaFree(d_x); cudaFree(d_b); cudaFree(d_Av);
            cudaFree(d_row_res); cudaFree(d_max_res);
            throw std::runtime_error(std::string("fp64_max_reduce_kernel execution failed: ") + cudaGetErrorString(kerr));
        }
    }

    // -----------------------------------------------------------------------
    // 4. Read result back to host.
    // -----------------------------------------------------------------------
    double max_res = 0.0;
    check(cudaMemcpy(&max_res, d_max_res, sizeof(double), cudaMemcpyDeviceToHost),
          "fp64 audit: copy d_max_res");

    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_Av);
    cudaFree(d_row_res);
    cudaFree(d_max_res);

    // Strict FP64 feasibility check: accept iff max |Ax - b| <= kDefaultFeasibilityTol
    return max_res <= math::kDefaultFeasibilityTol;
}

} // namespace gpu
} // namespace sankhya
