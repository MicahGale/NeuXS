cmake_minimum_required(VERSION 3.18)

project(neuxs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)


add_subdirectory(openmc)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)