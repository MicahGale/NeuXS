cmake_minimum_required(VERSION 3.29)

project(neuxs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")



include(cmake/CPM.cmake)

CPMAddPackage(
        NAME cuco
        GITHUB_REPOSITORY NVIDIA/cuCollections
        GIT_TAG dev
        OPTIONS
        "BUILD_TESTS OFF"
        "BUILD_BENCHMARKS OFF"
        "BUILD_EXAMPLES OFF"
)

find_package(HDF5 REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)