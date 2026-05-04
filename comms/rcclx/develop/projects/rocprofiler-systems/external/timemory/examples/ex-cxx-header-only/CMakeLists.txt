cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(timemory-Component-Example LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_UNITY_BUILD OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# this example demonstrates building a component within the app (header-only) and using a
# pre-compiled (extern) component
#
set(timemory_FIND_COMPONENTS_INTERFACE timemory-component-example)
set(COMPONENTS headers compile-options analysis-tools OPTIONAL_COMPONENTS core common)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_custom_target(ex_header_component)
add_custom_target(ex_extern_component)

function(EX_COMPONENT_BUILD COMPONENT _EXTERN_TARGET)
    if(NOT TARGET ${_EXTERN_TARGET})
        return()
    endif()

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ex_component.hpp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/ex_component_${COMPONENT}.hpp @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ex_component_impl.cpp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/ex_component_impl_${COMPONENT}.cpp @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ex_component_main.cpp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/ex_component_main_${COMPONENT}.cpp @ONLY)

    set(_sources
        ${CMAKE_CURRENT_BINARY_DIR}/ex_component_impl_${COMPONENT}.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ex_component_main_${COMPONENT}.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ex_component_${COMPONENT}.hpp)

    # build the component from headers
    add_executable(ex_header_component_${COMPONENT})
    target_sources(ex_header_component_${COMPONENT} PRIVATE ${_sources})
    target_link_libraries(ex_header_component_${COMPONENT}
                          PRIVATE timemory-component-example)

    # build the component from library
    add_executable(ex_extern_component_${COMPONENT})
    target_sources(ex_extern_component_${COMPONENT} PRIVATE ${_sources})
    target_link_libraries(ex_extern_component_${COMPONENT}
                          PRIVATE timemory-component-example ${_EXTERN_TARGET})

    add_dependencies(ex_header_component ex_header_component_${COMPONENT})
    add_dependencies(ex_extern_component ex_extern_component_${COMPONENT})

    install(
        TARGETS ex_header_component_${COMPONENT} ex_extern_component_${COMPONENT}
        DESTINATION bin
        OPTIONAL)
endfunction()

# foreach(_COMPONENT wall_clock cpu_clock) ex_component_build(${_COMPONENT}
# timemory::timemory-timing-component) endforeach()

# foreach(_COMPONENT peak_rss page_rss) ex_component_build(${_COMPONENT}
# timemory::timemory-rusage-component) endforeach()

foreach(_COMPONENT read_char written_char read_bytes written_bytes)
    ex_component_build(${_COMPONENT} timemory::timemory-io-component)
endforeach()
