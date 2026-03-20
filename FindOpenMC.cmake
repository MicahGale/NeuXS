set(OPENMC_INSTALL_DIR "$ENV{OPENMC_INSTALL_DIR}")

find_package(HDF5 REQUIRED)

if(EXISTS "${OPENMC_INSTALL_DIR}")
    set(OPENMC_FOUND TRUE)

    set(OPENMC_INCLUDE_DIRS "${OPENMC_INSTALL_DIR}/include/openmc/")
    set(OPENMC_LIBRARIES "${OPENMC_INSTALL_DIR}/lib/libopenmc.so")

    message(STATUS "OpenMC includes: ${OPENMC_INCLUDE_DIRS}")
    message(STATUS "OpenMC library: ${OPENMC_LIBRARIES}")

else()
    set(OPENMC_FOUND FALSE)
    message( "OpenMC not found at ${OPENMC_INSTALL_DIR}")
endif()


list(APPEND OPENMC_LIBRARIES ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
list(APPEND OPENMC_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenMC REQUIRED_VARS OPENMC_LIBRARIES OPENMC_INCLUDE_DIRS)

mark_as_advanced(OPENMC_INCLUDE_DIRS OPENMC_LIBRARIES)