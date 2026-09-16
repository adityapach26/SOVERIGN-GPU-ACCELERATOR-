# SANKHYA Core Library Target
add_library(sankhya_core
    sankhya_core.cpp
    core/sparse_matrix.cpp
    core/problem.cpp
    numerics/sparse_lu.cpp
    simplex/primal.cpp
    simplex/dual.cpp
    ipm/symbolic.cpp
    presolve/scaling.cpp
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
    )
endif()
