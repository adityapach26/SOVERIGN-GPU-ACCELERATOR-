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
        unit/test_device_model.cu
        unit/test_hbf.cu
        unit/test_pricing.cu
        unit/test_ratio_test.cu
    )
endif()
