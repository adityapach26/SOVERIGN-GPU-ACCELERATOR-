include(FetchContent)

# Disable Catch2's own self-tests and docs to optimize build time
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CATCH_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)

# Integrate Catch2 v3 test framework
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# Unit test executable
add_executable(sankhya_unit_tests
    unit/test_main.cpp
    unit/test_sparse_matrix.cpp
    unit/test_problem.cpp
    unit/test_factorization.cpp
    unit/test_sparse_lu.cpp
    unit/test_primal_simplex.cpp
    unit/test_dual_simplex.cpp
    unit/test_scaling.cpp
    unit/test_ipm_symbolic.cpp
    unit/test_qp_contract.cpp
    unit/test_qp_solver.cpp
    unit/test_milp_tree.cpp
    unit/test_milp_branching.cpp
    unit/test_cut_pool.cpp
    unit/test_perturbation.cpp
    unit/test_mccormick.cpp
    unit/test_spatial_branching.cpp
    unit/test_heuristics.cpp
    unit/test_mps_reader.cpp
    unit/test_certificate.cpp
)

# Link sankhya_core and Catch2 main entrypoint
target_link_libraries(sankhya_unit_tests PRIVATE
    sankhya_core
    Catch2::Catch2WithMain
)

target_include_directories(sankhya_unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Register test with CTest
add_test(NAME sankhya_unit_tests COMMAND sankhya_unit_tests)

if(SANKHYA_ENABLE_CUDA)
    target_sources(sankhya_unit_tests PRIVATE
        unit/test_vram_arena.cu
        unit/test_device_sparse_lu.cu
        unit/test_device_ftran.cu
        unit/test_device_dual_simplex.cu
        unit/test_batched_dual_simplex.cu
        unit/test_device_model.cu
        unit/test_hbf.cu
        unit/test_pricing.cu
        unit/test_ratio_test.cu
        unit/test_gpu_kkt.cu
        unit/test_mehrotra.cu
        unit/test_crossover.cu
        unit/test_gomory.cu
        unit/test_diving.cu
        unit/test_feasibility_pump.cu
        unit/test_feasibility_pump_audit.cu
        unit/test_milp_node_solver.cpp
    )

    set_target_properties(sankhya_unit_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75
    )
endif()

# Pure C API Test
add_executable(sankhya_c_api_test unit/test_c_api.c)
target_link_libraries(sankhya_c_api_test PRIVATE sankhya_core)
add_test(NAME sankhya_c_api_test COMMAND sankhya_c_api_test)
