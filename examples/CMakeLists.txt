add_executable(geometry build_geometry.cu)
target_link_libraries(geometry PRIVATE neuxslib)

add_executable(cross_section  read_cross_section.cpp)
target_link_libraries(cross_section PRIVATE neuxslib)

add_executable(setup set_cross_section_data_structure.cu)
target_link_libraries(setup PRIVATE neuxslib)