cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(timemory-Minimal-Example LANGUAGES C CXX)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-minimal-example)
set(COMPONENTS headers compile-options analysis-tools OPTIONAL_COMPONENTS cxx c)

option(USE_PAPI "Enable PAPI support" OFF)
option(USE_CALIPER "Enable Caliper support" OFF)
if(USE_PAPI)
    list(APPEND COMPONENTS papi)
endif()
if(USE_CALIPER)
    list(APPEND COMPONENTS caliper)
endif()
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

function(GET_LIBRARY _VAR _LANG)
    string(TOLOWER ${_LANG} _LOW)
    set(_PREFIX timemory::timemory-${_LOW})
    if(TARGET ${_PREFIX}-shared)
        set(${_VAR}
            ${_PREFIX}-shared
            PARENT_SCOPE)
    elseif(TARGET ${_PREFIX}-static)
        set(${_VAR}
            ${_PREFIX}-static
            PARENT_SCOPE)
    endif()
endfunction()

get_library(TIMEMORY_C_LIBRARY C)
get_library(TIMEMORY_CXX_LIBRARY CXX)

# headers are always available
add_executable(ex_cxx_minimal ex_minimal.cpp)
target_link_libraries(ex_cxx_minimal PRIVATE timemory-minimal-example)
install(
    TARGETS ex_cxx_minimal
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)

# if C stubs are not available via CXX library
if(NOT TIMEMORY_CXX_LIBRARY)
    return()
endif()

add_executable(ex_cxx_minimal_library ex_minimal_library.cpp)
target_link_libraries(ex_cxx_minimal_library timemory-minimal-example
                      ${TIMEMORY_CXX_LIBRARY})
install(
    TARGETS ex_cxx_minimal_library
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)

add_executable(ex_cxx_minimal_library_overload ex_minimal_library_overload.cpp)
target_link_libraries(ex_cxx_minimal_library_overload timemory-minimal-example
                      ${TIMEMORY_CXX_LIBRARY})
install(
    TARGETS ex_cxx_minimal_library_overload
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python")
endif()

configure_file(${PROJECT_SOURCE_DIR}/ex_python_minimal.py
               ${CMAKE_BINARY_DIR}/ex_python_minimal @ONLY)

install(
    FILES ${CMAKE_BINARY_DIR}/ex_python_minimal
    DESTINATION bin
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
    COMPONENT examples
    OPTIONAL)

# if C library is not available
if(NOT TIMEMORY_C_LIBRARY)
    return()
endif()

add_executable(ex_c_minimal ex_minimal.c)
target_link_libraries(ex_c_minimal PRIVATE timemory-minimal-example ${TIMEMORY_C_LIBRARY})
install(
    TARGETS ex_c_minimal
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)

# if C stubs via CXX library are available
add_executable(ex_c_minimal_library ex_minimal_library.c)
target_link_libraries(
    ex_c_minimal_library PRIVATE timemory-minimal-example ${TIMEMORY_CXX_LIBRARY}
                                 ${TIMEMORY_C_LIBRARY})
install(
    TARGETS ex_c_minimal_library
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)

add_executable(ex_c_minimal_library_overload ex_minimal_library_overload.c)
target_link_libraries(
    ex_c_minimal_library_overload PRIVATE timemory-minimal-example
                                          ${TIMEMORY_CXX_LIBRARY} ${TIMEMORY_C_LIBRARY})
install(
    TARGETS ex_c_minimal_library_overload
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)
