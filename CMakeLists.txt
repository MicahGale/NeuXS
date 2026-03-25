cmake_minimum_required(VERSION 3.18)

project(neuxs LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#dealing with openmc includes and it's dependencies
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/openmc/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/build/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/openmc/vendor/fmt/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/openmc/vendor/xtensor/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/openmc/vendor/xtl/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/openmc/vendor/fmt/include)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/openmc)
add_subdirectory(examples)