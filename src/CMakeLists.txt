# SANKHYA Core Library Target
add_library(sankhya_core
    sankhya_core.cpp
    core/sparse_matrix.cpp
    core/problem.cpp
    numerics/sparse_lu.cpp
    numerics/perturbation.cpp
    mrpl/mccormick.cpp
    simplex/primal.cpp
    simplex/dual.cpp
    ipm/symbolic.cpp
    presolve/scaling.cpp
    qp/qp_solver.cpp
    milp/tree.cpp
    milp/branching.cpp
    cuts/cut_pool.cpp
)

target_include_directories(sankhya_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

if(SANKHYA_ENABLE_CUDA)
    target_sources(sankhya_core PRIVATE
        gpu/vram_arena.cu
        gpu/device_sparse_lu.cu
        gpu/device_ftran.cu
        gpu/device_dual_simplex.cu
        gpu/batched_dual_simplex.cu
        gpu/device_model.cu
        gpu/hbf.cu
        cuda/pricing.cu
        cuda/ratio_test.cu
        cuda/kkt.cu
        cuda/gomory.cu
        ipm/mehrotra_device.cu
        ipm/crossover.cu
        gpu/node_dispatch.cu
        milp/node_solver.cpp
        heuristics/rounding.cpp
        heuristics/diving.cpp
        cuda/feasibility_pump.cu
    )

    # Enable CUDA Separable Compilation (relocatable device code, -rdc=true).
    # This is required so that __device__ functions defined in one .cu translation
    # unit (e.g. device_ftran.cu, device_dual_simplex.cu) can be called from
    # other .cu translation units (e.g. batched_dual_simplex.cu, mehrotra_device.cu).
    # Without this, the CUDA device linker cannot resolve cross-TU device symbols.
    set_target_properties(sankhya_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75
    )
endif()

if(SANKHYA_ENABLE_CUDA)
    target_link_libraries(sankhya_core PRIVATE cusparse)
endif()
