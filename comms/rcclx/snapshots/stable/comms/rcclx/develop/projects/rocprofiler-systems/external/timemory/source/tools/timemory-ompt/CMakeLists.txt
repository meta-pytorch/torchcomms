cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# this is for internal use
if(NOT TIMEMORY_USE_OMPT)
    return()
endif()

if(NOT TIMEMORY_BUILD_OMPT_LIBRARY
   OR TIMEMORY_SKIP_BUILD
   OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
    set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

project(timemory-ompt-tool)

foreach(_TYPE SHARED STATIC)

    if(NOT BUILD_${_TYPE}_LIBS)
        continue()
    endif()

    string(TOLOWER "${_TYPE}" _LC_TYPE)
    if(NOT TARGET timemory-cxx-${_LC_TYPE})
        continue()
    endif()

    set(TARGET_NAME timemory-ompt-${_LC_TYPE})

    add_library(${TARGET_NAME} ${_TYPE} ${_EXCLUDE}
                               ${PROJECT_SOURCE_DIR}/timemory-ompt.cpp)
    add_library(timemory::${TARGET_NAME} ALIAS ${TARGET_NAME})

    # public link targets
    target_link_libraries(
        ${TARGET_NAME} PUBLIC timemory::timemory-headers timemory::timemory-ompt
                              timemory::timemory-cxx-${_LC_TYPE})

    # private link targets
    target_link_libraries(${TARGET_NAME} PRIVATE timemory::timemory-default-visibility
                                                 timemory::timemory-compile-options)

    # use rpath
    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON
                   OUTPUT_NAME timemory-ompt
                   VERSION ${timemory_VERSION}
                   SOVERSION ${timemory_VERSION_MAJOR}.${timemory_VERSION_MINOR})

    # installation
    install(
        TARGETS ${TARGET_NAME}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        OPTIONAL)

endforeach()
