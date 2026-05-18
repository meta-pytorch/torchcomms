cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_USE_DYNINST)
    return()
endif()

project(timemory-custom-dynamic-instr-example LANGUAGES CXX)

include(GNUInstallDirs)

set(LIB_NAME ex_custom_dynamic_instr)
set(COMPONENTS cxx compile-options analysis-tools)

set(timemory_FIND_COMPONENTS_INTERFACE timemory::custom-dynamic-instr-example)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

foreach(_TYPE SHARED STATIC)
    if(NOT BUILD_${_TYPE}_LIBS)
        continue()
    endif()
    string(TOLOWER "${_TYPE}" _LC_TYPE)
    set(_TARG ${LIB_NAME}-${_LC_TYPE})
    add_library(${_TARG} ${_TYPE} ${LIB_NAME}.cpp)
    target_link_libraries(${_TARG} PUBLIC timemory::custom-dynamic-instr-example)
    set_target_properties(${_TARG} PROPERTIES OUTPUT_NAME ${LIB_NAME})
    install(
        TARGETS ${_TARG}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT examples
        OPTIONAL)
endforeach()
