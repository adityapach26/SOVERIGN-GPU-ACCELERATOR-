#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vector>
#include <iostream>
#include <iomanip>
#include <functional>
#include <cuda_runtime.h>

#include "core/problem.hpp"
#include "numerics/sparse_lu.hpp"
#include "gpu/device_model.cuh"
#include "gpu/device_sparse_lu.cuh"
#include "gpu/hbf.cuh"
#include "gpu/node_dispatch.cuh"
#include "gpu/vram_arena.cuh"

using namespace sankhya;

// A simple deterministic hash over a vector of floats
static uint64_t hash_solution(const std::vector<Float>& x) {
    uint64_t hash = 0x811c9dc5; // FNV-1a offset basis
    for (Float val : x) {
        // We hash the actual raw byte representation of the double/float
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&val);
        for (size_t i = 0; i < sizeof(Float); ++i) {
            hash ^= bytes[i];
            hash *= 0x01000193; // FNV-1a prime
        }
    }
    return hash;
}

TEST_CASE("Phase 31.1: Determinism Validation Framework", "[integration][determinism]") {
    // 5. Verify CUDA is actually available
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        FAIL("CUDA GPU execution is required for Phase 31.1 determinism validation.");
    }

    const Index num_structural = 250;
    const Index num_slacks = 250;
    const Index total_vars = num_structural + num_slacks;
    const Index num_constraints = 250;

    std::cout << "\n=== Phase 31.1 Determinism Validation ===\n\n";
    std::cout << "Problem:\n";
    std::cout << "  Variables: " << total_vars << "\n";
    std::cout << "  Runs: 10\n";
    std::cout << "  Execution: CUDA GPU\n\n";

    bool status_identical = true;
    bool iterations_identical = true;
    bool obj_within_tolerance = true;
    bool res_within_tolerance = true;
    bool hash_identical = true;

    gpu::DeviceSimplexStatus first_status;
    Index first_iterations = -1;
    Float first_obj = 0.0;
    Float first_res = 0.0;
    uint64_t first_hash = 0;

    for (int run = 1; run <= 10; ++run) {
        // 2. Construct deterministic 500-variable LP
        core::Model host_model;
        host_model.sense = OptimizationSense::Minimize;

        // Structural variables: x_0 .. x_249
        for (Index i = 0; i < num_structural; ++i) {
            host_model.add_variable(1.0, 0.0, math::kInfinity, VariableType::Continuous);
        }
        // Slack variables: s_0 .. s_249
        for (Index i = 0; i < num_slacks; ++i) {
            host_model.add_variable(0.0, 0.0, math::kInfinity, VariableType::Continuous);
        }

        // Constraints: -x_i + s_i = -1
        for (Index i = 0; i < num_constraints; ++i) {
            host_model.add_constraint({i, num_structural + i}, {-1.0, 1.0}, -1.0);
        }

        host_model.finalize();

        // 3. Setup fresh solver state
        simplex::Basis host_basis;
        host_basis.col_status.resize(total_vars, simplex::BasisStatus::AtLower);
        for (Index i = 0; i < num_slacks; ++i) {
            host_basis.basic_indices.push_back(num_structural + i);
            host_basis.col_status[static_cast<std::size_t>(num_structural + i)] = simplex::BasisStatus::Basic;
        }

        gpu::VRAMArena arena(1024 * 1024 * 16);
        
        numerics::SparseLUFactorization root_lu;
        root_lu.factorize(host_model.A, host_basis);

        gpu::DeviceModel d_model = gpu::upload_to_device(host_model, arena);
        gpu::DeviceSparseLUManager manager(arena);
        manager.upload(root_lu);
        gpu::DeviceSparseLU d_lu = manager.get_device_struct();

        gpu::HBFManager hbf_manager(arena);
        std::vector<gpu::BoundDelta> root_deltas;
        hbf_manager.create_node(100, 100, root_deltas);
        hbf_manager.set_root_basis(host_basis.basic_indices);

        gpu::WorkingBasisState ws = hbf_manager.allocate_working_state(num_constraints, total_vars);
        hbf_manager.inherit_basis(100, d_model, ws);

        std::vector<Float> init_x(total_vars, 0.0);
        std::vector<uint8_t> init_is_basic(total_vars, 0);
        for (Index i = 0; i < num_slacks; ++i) {
            init_x[num_structural + i] = -1.0;
            init_is_basic[num_structural + i] = 1;
        }
        cudaMemcpy(ws.x, init_x.data(), total_vars * sizeof(Float), cudaMemcpyHostToDevice);
        cudaMemcpy(ws.is_basic, init_is_basic.data(), total_vars * sizeof(uint8_t), cudaMemcpyHostToDevice);

        Float obj_sign = (host_model.sense == OptimizationSense::Minimize) ? 1.0 : -1.0;
        Index max_iterations = 10000;

        // Actual GPU solver path
        gpu::NodeDispatchResult result = gpu::dispatch_single_node(
            d_model, d_lu, ws, obj_sign, max_iterations, arena, total_vars
        );

        // Compute diagnostic quantities
        Float obj_val = 0.0;
        for (Index i = 0; i < total_vars; ++i) {
            obj_val += host_model.obj[static_cast<std::size_t>(i)] * result.x[static_cast<std::size_t>(i)];
        }

        // Calculate primal residual: ||Ax - b||_inf
        Float primal_residual = 0.0;
        std::vector<Float> Ax(num_constraints, 0.0);
        for (Index j = 0; j < total_vars; ++j) {
            Float xj = result.x[static_cast<std::size_t>(j)];
            Index start = host_model.A.col_ptrs[static_cast<std::size_t>(j)];
            Index end = host_model.A.col_ptrs[static_cast<std::size_t>(j) + 1];
            for (Index k = start; k < end; ++k) {
                Index row = host_model.A.row_indices[static_cast<std::size_t>(k)];
                Float val = host_model.A.values[static_cast<std::size_t>(k)];
                Ax[static_cast<std::size_t>(row)] += val * xj;
            }
        }
        for (Index i = 0; i < num_constraints; ++i) {
            Float res = std::abs(Ax[static_cast<std::size_t>(i)] - host_model.rhs[static_cast<std::size_t>(i)]);
            if (res > primal_residual) {
                primal_residual = res;
            }
        }

        uint64_t hash = hash_solution(result.x);
        
        // We use ws to get iterations since it's not in NodeDispatchResult directly.
        // Wait, gpu::dispatch_single_node actually launches the kernel but we need to 
        // pull the iteration count from device if it was returned.
        // Let's check how iter count is fetched. If it isn't in NodeDispatchResult,
        // we might not have it. The prompt: "If the current solver does not expose a required diagnostic... NOT EXPOSED"
        // Let's just print NOT EXPOSED for iterations unless we can easily read it.
        // Actually, we can read it from `d_iter_count_array` which dispatch_single_node hides.
        // So we will say NOT EXPOSED for iterations.
        
        std::cout << "Run " << run << ":\n";
        std::cout << "  Status: " << static_cast<int>(result.status) << "\n";
        std::cout << "  Iterations: NOT EXPOSED\n";
        std::cout << "  Objective: " << obj_val << "\n";
        std::cout << "  Primal Residual: " << primal_residual << "\n";
        std::cout << "  Solution Hash: " << std::hex << std::setw(16) << std::setfill('0') << hash << std::dec << "\n\n";

        if (run == 1) {
            first_status = result.status;
            first_obj = obj_val;
            first_res = primal_residual;
            first_hash = hash;
        } else {
            if (result.status != first_status) status_identical = false;
            if (std::abs(obj_val - first_obj) > 1e-6) obj_within_tolerance = false;
            if (std::abs(primal_residual - first_res) > 1e-6) res_within_tolerance = false;
            if (hash != first_hash) hash_identical = false;
            
            // Fail fast on non-determinism for contractually identical quantities
            if (result.status != first_status) {
                FAIL("Nondeterminism diagnosis: Solver status varied between runs. Likely CUDA scheduling dependency or race condition.");
            }
        }

        hbf_manager.free_working_state(ws);
        manager.free_all();
    }

    std::cout << "Determinism:\n";
    std::cout << "  Status: " << (status_identical ? "PASS" : "FAIL") << "\n";
    std::cout << "  Iteration Count: NOT EXPOSED\n";
    std::cout << "  Pricing Decisions: NOT EXPOSED\n";
    std::cout << "  Branch Decisions: N/A\n";
    std::cout << "  Objective: " << (obj_within_tolerance ? "WITHIN TOLERANCE" : "VARIED") << "\n";
    std::cout << "  Primal Residual: " << (res_within_tolerance ? "WITHIN TOLERANCE" : "VARIED") << "\n";
    std::cout << "  Solution Hash: " << (hash_identical ? "IDENTICAL" : "VARIED") << "\n";
    std::cout << "  Bitwise Reproducibility: " << (hash_identical ? "DEMONSTRATED" : "NOT DEMONSTRATED") << "\n";
    std::cout << "=============================================\n";

    REQUIRE(status_identical);
    REQUIRE(obj_within_tolerance);
    REQUIRE(res_within_tolerance);
}
