#include "cuda/feasibility_pump.cuh"
#include <mma.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <iostream>

using namespace nvcuda;

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

} // namespace gpu
} // namespace sankhya
