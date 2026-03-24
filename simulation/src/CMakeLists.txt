cmake_minimum_required(VERSION 3.24)
project(lbm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)   # RTX 4070 = SM 89
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── zlib (fetched if not found) ─────────────────────────────────────────────
include(FetchContent)
find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
    message(STATUS "System zlib not found — fetching via FetchContent")
    FetchContent_Declare(
        zlib
        URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
        URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
    )
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(zlib)
    # zlib's CMakeLists creates zlibstatic target
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
    target_include_directories(zlibstatic PUBLIC ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
endif()

# ── cnpy (vendored, MIT) ────────────────────────────────────────────────────
add_library(cnpy STATIC third_party/cnpy/cnpy.cpp)
target_include_directories(cnpy PUBLIC third_party/cnpy)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

# ── lbm_cuda executable ────────────────────────────────────────────────────
add_executable(lbm_cuda main.cu lbm_kernels.cu)
target_link_libraries(lbm_cuda PRIVATE cnpy)
target_include_directories(lbm_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
