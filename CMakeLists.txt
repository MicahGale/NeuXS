cmake_minimum_required(VERSION 3.18)

project(neuxs LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OPENMC_INCLUDE_DIRS})


find_package(OpenMC REQUIRED)



add_subdirectory(src)
add_subdirectory(tests)