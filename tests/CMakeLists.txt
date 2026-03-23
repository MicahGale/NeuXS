enable_testing()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cpp *.cxx *.c *.C *.cu)


# executable for every test

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME}  PRIVATE neuxslib libopenmc)

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

endforeach()