# include guard
include_guard(DIRECTORY)

# ########################################################################################
#
# timemory Package installation
#
# ########################################################################################

include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
foreach(_LANG C CXX CUDA)
    foreach(_TYPE COMPILE LINK)
        set(PROJECT_${_LANG}_${_TYPE}_OPTIONS
            ${${PROJECT_NAME}_${_LANG}_${_TYPE}_OPTIONS})
    endforeach()
endforeach()

set(_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
if(TIMEMORY_USE_PYTHON)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print('{}'.format(sys.prefix))"
        OUTPUT_VARIABLE _INSTALL_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

set(_PATH_VARS INCLUDE_INSTALL_DIR)
if(NOT TIMEMORY_SKIP_BUILD)
    list(APPEND _PATH_VARS LIB_INSTALL_DIR)
endif()

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}-config.cmake.in
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_CONFIGDIR}
    INSTALL_PREFIX ${_INSTALL_PREFIX}
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}-config-build.cmake.in
               ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-build.cmake @ONLY)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}-config-components.cmake.in
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-components.cmake @ONLY)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}-target-extract.cmake.in
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-target-extract.cmake @ONLY)

if(TIMEMORY_INSTALL_CONFIG)
    install(
        FILES ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
              ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
              ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-components.cmake
              ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-target-extract.cmake
        DESTINATION ${CMAKE_INSTALL_CONFIGDIR}
        OPTIONAL)
endif()

# only if master project
if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

    # documentation
    set_property(
        GLOBAL APPEND
        PROPERTY TIMEMORY_DOCUMENTATION_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/source
                 ${PROJECT_SOURCE_DIR}/timemory ${PROJECT_SOURCE_DIR}/examples)

    set(EXCLUDE_LIST
        ${PROJECT_SOURCE_DIR}/external ${PROJECT_SOURCE_DIR}/external/cereal
        ${PROJECT_SOURCE_DIR}/external/pybind11 ${PROJECT_SOURCE_DIR}/external/gotcha
        ${PROJECT_SOURCE_DIR}/external/google-test)

    timemory_add_feature(TIMEMORY_COMPILED_LIBRARIES "Compiled libraries")
    # timemory_add_feature(TIMEMORY_INTERFACE_LIBRARIES "Interface libraries")

endif()

if(TIMEMORY_USE_GPERFTOOLS)
    foreach(_TYPE cpu heap)
        configure_file(${PROJECT_SOURCE_DIR}/scripts/gperf-${_TYPE}-profile.sh
                       ${PROJECT_BINARY_DIR}/gperf-${_TYPE}-profile.sh COPYONLY)
    endforeach()
    configure_file(${PROJECT_SOURCE_DIR}/scripts/gprof2dot.py
                   ${PROJECT_BINARY_DIR}/gprof2dot.py COPYONLY)
endif()

export(PACKAGE ${PROJECT_NAME})
