cmake_minimum_required(VERSION 3.18)

project(neuxs LANGUAGES CXX CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(HDF5 REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
