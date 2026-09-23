cmake_minimum_required(VERSION 3.18)

project(SANKHYA
    VERSION 0.1.0
    DESCRIPTION "SANKHYA / VAJRA-OPT Core Optimization Engine"
    LANGUAGES CXX
)

# Require C++17 standard
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define strict compiler warnings: -Wall -Wextra -Wpedantic -Werror
# Explicitly disable fast-math to ensure IEEE-754 compliance
if(MSVC)
    add_compile_options(
        /W4
        /WX
        /permissive-
        /fp:precise
    )
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Werror>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic,-Werror,-fno-fast-math>
    )
endif()

# CMake options
option(SANKHYA_ENABLE_CUDA "Enable CUDA acceleration" ON)
option(SANKHYA_CPU_ONLY "Build CPU-only version without CUDA" OFF)

# If SANKHYA_CPU_ONLY is requested, force-disable CUDA support
if(SANKHYA_CPU_ONLY)
    message(STATUS "SANKHYA_CPU_ONLY is ON. Disabling CUDA support.")
    set(SANKHYA_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA acceleration" FORCE)
endif()

# If SANKHYA_ENABLE_CUDA is ON, find the CUDA toolkit using enable_language(CUDA)
if(SANKHYA_ENABLE_CUDA)
    message(STATUS "CUDA support enabled. Finding CUDA toolkit...")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# Add core library subdirectory
add_subdirectory(src)

# Enable testing and add test suite subdirectory
enable_testing()
add_subdirectory(tests)
