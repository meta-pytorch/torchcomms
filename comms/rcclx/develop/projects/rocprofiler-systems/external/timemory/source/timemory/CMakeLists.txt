set_property(GLOBAL APPEND PROPERTY BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})

get_property(_CMAKE_DEFINES GLOBAL PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES)
list(SORT _CMAKE_DEFINES)

set(${PROJECT_NAME}_CMAKE_DEFINES)
foreach(_DEF ${_CMAKE_DEFINES})
    set(${PROJECT_NAME}_CMAKE_DEFINES
        "${${PROJECT_NAME}_CMAKE_DEFINES}\n#cmakedefine ${_DEF}")
endforeach()
set(${PROJECT_NAME}_CMAKE_DEFINES "${${PROJECT_NAME}_CMAKE_DEFINES}\n")

get_property(_DEFAULT_CMAKE_DEFINES GLOBAL PROPERTY ${PROJECT_NAME}_DEFAULT_CMAKE_DEFINES)
set(${PROJECT_NAME}_DEFAULT_CMAKE_DEFINES)
foreach(_DEF ${_DEFAULT_CMAKE_DEFINES})
    set(${PROJECT_NAME}_DEFAULT_CMAKE_DEFINES
        "${${PROJECT_NAME}_DEFAULT_CMAKE_DEFINES}\n${_DEF}")
endforeach()
set(${PROJECT_NAME}_DEFAULT_CMAKE_DEFINES "${${PROJECT_NAME}_DEFAULT_CMAKE_DEFINES}\n")

# remove old way of doing this
foreach(_FILE defines.h version.h)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_FILE})
        file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/${_FILE})
    endif()
endforeach()

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/defines.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/defines.h.in @ONLY)

configure_file(${CMAKE_CURRENT_BINARY_DIR}/defines.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/defines.h @ONLY)

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

if(TIMEMORY_INSTALL_HEADERS)
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/defines.h ${CMAKE_CURRENT_BINARY_DIR}/version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/timemory
        OPTIONAL)
endif()

if(NOT timemory_MAIN_PROJECT AND TIMEMORY_SOURCE_TREE_HEADERS)
    configure_file(${CMAKE_CURRENT_BINARY_DIR}/defines.h
                   ${CMAKE_CURRENT_LIST_DIR}/defines.h COPYONLY)
    configure_file(${CMAKE_CURRENT_BINARY_DIR}/version.h
                   ${CMAKE_CURRENT_LIST_DIR}/version.h COPYONLY)
endif()

timemory_build_intermediate_library(
    USE_INTERFACE
    NAME core
    TARGET core
    CATEGORY GLOBAL
    FOLDER ${CMAKE_INSTALL_LIBDIR}
    DEPENDS timemory::timemory-libunwind $<TARGET_EXISTS:timemory::timemory-shlwapi>)

# core
add_subdirectory(tpls)
add_subdirectory(environment)
add_subdirectory(hash)
add_subdirectory(utility)
add_subdirectory(settings)
add_subdirectory(plotting)
add_subdirectory(signals)
add_subdirectory(unwind)
add_subdirectory(data)
add_subdirectory(process)

if(TIMEMORY_USE_COVERAGE)
    if(TARGET timemory-core-shared)
        target_link_libraries(timemory-core-shared PUBLIC timemory-analysis-tools)
    endif()
    if(TARGET timemory-core-static)
        target_link_libraries(timemory-core-static PUBLIC timemory-analysis-tools)
    endif()
endif()

# global
add_subdirectory(ert)

# do this here for IDE benefit
timemory_build_intermediate_library(
    USE_INTERFACE
    NAME common
    TARGET common
    CATEGORY GLOBAL
    FOLDER ${CMAKE_INSTALL_LIBDIR}
    DEPENDS timemory::timemory-core
    PROPERTY_DEPENDS GLOBAL)

add_subdirectory(manager)
add_subdirectory(storage)
add_subdirectory(operations)
add_subdirectory(variadic)
add_subdirectory(config)
add_subdirectory(backends)
add_subdirectory(sampling)

# deprecated
add_subdirectory(deprecated)

# fake libraries add_subdirectory(mpl) add_subdirectory(data) add_subdirectory(general)
# add_subdirectory(compat) add_subdirectory(tpls)

# components
add_subdirectory(components)

add_subdirectory(runtime)
add_subdirectory(containers)

# add_subdirectory(compat) add_subdirectory(data) add_subdirectory(general)
# add_subdirectory(mpl) add_subdirectory(utility)

# tools
add_subdirectory(tools)
