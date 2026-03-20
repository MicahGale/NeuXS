cmake_minimum_required(VERSION 3.11)
project(neuxs)

set(CMAKE_CXX_COMPILER nvcc)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_BUILD_TYPE Release)

find_package(OpenMC REQUIRED)


include_directories(${OPENMC_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB NEXUS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()


add_subdirectory(src)

