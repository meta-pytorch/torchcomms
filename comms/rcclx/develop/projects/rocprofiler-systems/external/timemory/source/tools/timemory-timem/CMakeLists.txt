# no windows support
if(WIN32)
    return()
endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# ----------------------------------------------------------------------------------------#
# Build and install timem tool
#
if(NOT TIMEMORY_BUILD_TIMEM OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
    # if not enabled as option, make it available to be built and installed
    set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

add_library(timem-libexplain INTERFACE)

timemory_add_option(TIMEMORY_BUILD_TOOLS_LIBEXPLAIN
                    "Enable using libexplain to diagnose errors" OFF NO_FEATURE)
if(TIMEMORY_BUILD_TOOLS_LIBEXPLAIN)

    find_path(
        LIBEXPLAIN_INCLUDE_DIR
        NAMES libexplain/execvp.h
        PATH_SUFFIXES include)

    find_library(
        LIBEXPLAIN_LIBRARY
        NAMES explain
        PATH_SUFFIXES lib lib64 lib32)

    if(LIBEXPLAIN_INCLUDE_DIR AND LIBEXPLAIN_LIBRARY)
        timemory_target_compile_definitions(timem-libexplain INTERFACE
                                            TIMEMORY_USE_LIBEXPLAIN)
        target_include_directories(timem-libexplain INTERFACE ${LIBEXPLAIN_INCLUDE_DIR})
        target_link_libraries(timem-libexplain INTERFACE ${LIBEXPLAIN_LIBRARY})
    else()
        set(TIMEMORY_BUILD_TOOLS_LIBEXPLAIN OFF)
    endif()
endif()

# tests whether the static papi library can be linked (errors typically occur when
# libpapi.a is not PIC and PIE gets enabled)
function(TEST_PAPI_STATIC_LINKAGE _VAR)
    set(EXE_INCLUDE "#include <papi.h>")
    set(EXE_BODY "return (PAPI_is_initialized() == 0) ? 0 : 1;")
    configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/exe.cpp.in
                   ${PROJECT_BINARY_DIR}/compile-tests/papi_link.cpp @ONLY)
    try_compile(
        _RET ${PROJECT_BINARY_DIR}/compile-tests
        ${PROJECT_BINARY_DIR}/compile-tests/papi_link.cpp
        LINK_LIBRARIES ${ARGN})
    set(${_VAR}
        ${_RET}
        PARENT_SCOPE)
endfunction()

set(_RPATH ON)
find_package(PAPI QUIET)
if(NOT DEFINED TIMEM_PAPI_TARGET OR NOT TIMEM_PAPI_TARGET)
    if(PAPI_STATIC_LIBRARIES OR TARGET PAPI::papi-static)
        test_papi_static_linkage(_USE_PAPI_STATIC PAPI::papi-static)
        if(_USE_PAPI_STATIC)
            set(TIMEM_PAPI_TARGET PAPI::papi-static)
            set(_RPATH OFF)
            timemory_target_compile_definitions(${TIMEM_PAPI_TARGET} INTERFACE
                                                TIMEMORY_USE_PAPI)
        else()
            set(TIMEM_PAPI_TARGET timemory::timemory-papi)
        endif()
    else()
        set(TIMEM_PAPI_TARGET timemory::timemory-papi)
    endif()
endif()

add_library(timem-md5sum EXCLUDE_FROM_ALL OBJECT md5.cpp md5.hpp)

target_link_libraries(timem-md5sum PRIVATE timemory::timemory-compile-options
                                           timemory::timemory-develop-options)

# non-MPI version
add_executable(timem ${_EXCLUDE} timem.cpp timem.hpp $<TARGET_OBJECTS:timem-md5sum>)

add_dependencies(timem timem-md5sum)

target_link_libraries(
    timem
    PRIVATE timemory::timemory-compile-options timemory::timemory-develop-options
            timemory::timemory-threading timemory::timemory-headers ${TIMEM_PAPI_TARGET}
            timem-libexplain)

add_target_flag_if_avail(timem "-g")

set_target_properties(timem PROPERTIES INSTALL_RPATH_USE_LINK_PATH ${_RPATH})

# timem installation
install(
    TARGETS timem
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tools
    OPTIONAL)

# do not build timem-mpi if not using mpi
if(NOT TIMEMORY_USE_MPI)
    return()
endif()

add_executable(timem-mpi ${_EXCLUDE} timem.cpp timem.hpp $<TARGET_OBJECTS:timem-md5sum>)

add_dependencies(timem-mpi timem-md5sum)

target_link_libraries(
    timem-mpi
    PRIVATE timemory::timemory-compile-options timemory::timemory-develop-options
            timemory::timemory-threading timemory::timemory-headers
            timemory::timemory-mpi ${TIMEM_PAPI_TARGET} timem-libexplain)

add_target_flag_if_avail(timem-mpi "-g")

set_target_properties(timem-mpi PROPERTIES INSTALL_RPATH_USE_LINK_PATH ${_RPATH})

# ensure timemory-pid is built
if(TARGET timemory-pid)
    add_dependencies(timem-mpi timemory-pid)
endif()

# timem installation
install(
    TARGETS timem-mpi
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tools
    OPTIONAL)
