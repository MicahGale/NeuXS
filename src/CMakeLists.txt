file(GLOB NEUXS_SOURCES CONFIGURE_DEPENDS *.cpp *.cxx *.c *.C *.cu)


add_library(neuxslib SHARED ${NEUXS_SOURCES})


target_include_directories(neuxslib PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(neuxslib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)