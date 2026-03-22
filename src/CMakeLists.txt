file(GLOB NEUXS_SOURCES CONFIGURE_DEPENDS *.cpp *.cxx *.c *.C *.cu)


add_library(neuxslib SHARED ${NEUXS_SOURCES}
        input_parser.cu)


target_include_directories(neuxslib PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(neuxslib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(neuxslib PRIVATE libopenmc)