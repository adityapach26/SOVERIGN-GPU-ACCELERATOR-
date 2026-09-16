#include "gpu/device_sparse_lu.cuh"
#include <stdexcept>
#include <string>

namespace sankhya {
namespace gpu {

static void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

template <typename T>
T* DeviceSparseLUManager::allocate_and_upload(const std::vector<T>& host_vec) {
    if (host_vec.empty()) return nullptr;
    size_t bytes = host_vec.size() * sizeof(T);
    T* d_ptr = static_cast<T*>(arena_.allocate(bytes));
    check_cuda_error(
        cudaMemcpy(d_ptr, host_vec.data(), bytes, cudaMemcpyHostToDevice),
        "Failed to copy Sparse LU data to device"
    );
    tracked_allocations_.push_back(d_ptr);
    return d_ptr;
}

DeviceSparseLUManager::DeviceSparseLUManager(VRAMArena& arena)
    : arena_(arena) {
    d_struct_ = DeviceSparseLU{};
}

DeviceSparseLUManager::~DeviceSparseLUManager() {
    free_all();
}

void DeviceSparseLUManager::free_all() {
    for (void* ptr : tracked_allocations_) {
        if (ptr) {
            arena_.free(ptr);
        }
    }
    tracked_allocations_.clear();
    d_struct_ = DeviceSparseLU{};
    is_uploaded_ = false;
}

void DeviceSparseLUManager::upload(const numerics::SparseLUFactorization& cpu_lu) {
    if (!cpu_lu.is_factorized()) {
        throw std::runtime_error("DeviceSparseLUManager::upload: CPU factorization is not ready.");
    }
    
    free_all(); // Clear any previous state

    d_struct_.m = cpu_lu.get_m();
    
    const auto& L_vals = cpu_lu.get_L_vals();
    const auto& U_vals = cpu_lu.get_U_vals();

    d_struct_.L_nnz = static_cast<Index>(L_vals.size());
    d_struct_.U_nnz = static_cast<Index>(U_vals.size());

    // Upload L arrays
    d_struct_.L_vals = allocate_and_upload(L_vals);
    d_struct_.L_rows = allocate_and_upload(cpu_lu.get_L_rows());
    d_struct_.L_col_ptrs = allocate_and_upload(cpu_lu.get_L_col_ptrs());

    // Upload U arrays
    d_struct_.U_vals = allocate_and_upload(U_vals);
    d_struct_.U_cols = allocate_and_upload(cpu_lu.get_U_cols());
    d_struct_.U_row_ptrs = allocate_and_upload(cpu_lu.get_U_row_ptrs());

    // Upload permutations
    d_struct_.perm_row = allocate_and_upload(cpu_lu.get_perm_row());
    d_struct_.perm_col = allocate_and_upload(cpu_lu.get_perm_col());
    d_struct_.inv_perm_row = allocate_and_upload(cpu_lu.get_inv_perm_row());
    d_struct_.inv_perm_col = allocate_and_upload(cpu_lu.get_inv_perm_col());

    is_uploaded_ = true;
}

DeviceSparseLU DeviceSparseLUManager::get_device_struct() const {
    if (!is_uploaded_) {
        throw std::runtime_error("DeviceSparseLUManager::get_device_struct: factorization not uploaded yet.");
    }
    return d_struct_;
}

} // namespace gpu
} // namespace sankhya

