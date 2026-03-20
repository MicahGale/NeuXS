set(OPENMC_BUILD_DIR "$ENV{OPENMC_INSTALL_DIR}")
set(OPENMC_SOURCE_DIR "$ENV{OPENMC_INSTALL_DIR}")

find_package(HDF5 REQUIRED)

if(EXISTS "${OPENMC_BUILD_DIR}")
    set(OPENMC_LIB_FOUND TRUE)
    set(OPENMC_LIBRARIES "${OPENMC_BUILD_DIR}/lib/libopenmc.so")

    message(STATUS "OpenMC library: ${OPENMC_LIBRARIES}")

else()
    set(OPENMC_LIB_FOUND FALSE)
    message( "OpenMC lib not found at ${OPENMC_INSTALL_DIR}")
endif()


if(EXISTS "${OPENMC_SOURCE_DIR}")
    set(OPENMC_SRC_FOUND TRUE)
    set(OPENMC_INCLUDE_DIRS "${OPENMC_SOURCE_DIR}/include/")

    message(STATUS "OpenMC library: ${OPENMC_LIBRARIES}")

else()
    set(OPENMC_SRC_FOUND FALSE)
    message( "OpenMC source directory not found at ${OPENMC_SOURCE_DIR}")
endif()


list(APPEND OPENMC_LIBRARIES ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
list(APPEND OPENMC_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenMC REQUIRED_VARS OPENMC_LIBRARIES OPENMC_INCLUDE_DIRS)

mark_as_advanced(OPENMC_INCLUDE_DIRS OPENMC_LIBRARIES)
