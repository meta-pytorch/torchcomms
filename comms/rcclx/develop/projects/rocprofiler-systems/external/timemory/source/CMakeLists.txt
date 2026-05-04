set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
include_directories(BEFORE ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME library)

if(MSVC)
    # set debug postfix so debug library and executable artifact names do not conflict
    # with release artifacts
    set(CMAKE_DEBUG_POSTFIX "d")
endif()

# ----------------------------------------------------------------------------------------#
# Locate sources and headers for this project - headers are included so they will show up
# in IDEs
#
file(GLOB c_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.h)
file(GLOB cxx_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)
file(GLOB cxx_headers_variadic ${CMAKE_CURRENT_LIST_DIR}/timemory/variadic/*.hpp)
file(GLOB_RECURSE cxx_headers_backends ${CMAKE_CURRENT_LIST_DIR}/timemory/backends/*.hpp)
file(GLOB_RECURSE cxx_headers_components
     ${CMAKE_CURRENT_LIST_DIR}/timemory/components/*.hpp)
file(GLOB cxx_headers_sampling ${CMAKE_CURRENT_LIST_DIR}/timemory/sampling/*.hpp)
file(GLOB cxx_headers_macros ${CMAKE_CURRENT_LIST_DIR}/timemory/macros/*.hpp)
file(GLOB cxx_headers_api ${CMAKE_CURRENT_LIST_DIR}/timemory/api/*.hpp)
file(GLOB cxx_headers_ert ${CMAKE_CURRENT_LIST_DIR}/timemory/ert/*.hpp)
file(GLOB cxx_headers_log ${CMAKE_CURRENT_LIST_DIR}/timemory/log/*.hpp)
file(GLOB cxx_headers_mpl ${CMAKE_CURRENT_LIST_DIR}/timemory/mpl/*.hpp)
file(GLOB cxx_headers_math ${CMAKE_CURRENT_LIST_DIR}/timemory/math/*.hpp)
file(GLOB cxx_headers_general ${CMAKE_CURRENT_LIST_DIR}/timemory/general/*.hpp)
file(GLOB cxx_headers_data ${CMAKE_CURRENT_LIST_DIR}/timemory/data/*.hpp)
file(GLOB cxx_headers_utility ${CMAKE_CURRENT_LIST_DIR}/timemory/utility/*.hpp)
file(GLOB cxx_headers_bits ${CMAKE_CURRENT_LIST_DIR}/timemory/bits/*.hpp
     ${CMAKE_CURRENT_LIST_DIR}/timemory/bits/*.h
     ${CMAKE_CURRENT_LIST_DIR}/timemory/bits/*.icpp)
file(GLOB compat_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/compat/*.h
     ${CMAKE_CURRENT_LIST_DIR}/timemory/compat/*.hpp)
file(GLOB c_sources ${CMAKE_CURRENT_LIST_DIR}/*.c)
file(GLOB cxx_sources ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB fortran_sources ${CMAKE_CURRENT_LIST_DIR}/*.f90)
file(GLOB_RECURSE tpl_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/tpls/*.hpp
     ${CMAKE_CURRENT_LIST_DIR}/timemory/tpls/*.h)
file(GLOB_RECURSE tpl_licenses ${CMAKE_CURRENT_LIST_DIR}/timemory/tpls/*license*)

set(_TIMEMORY_CXX_LIBRARY_EXCLUDE)
foreach(_SRC ${TIMEMORY_CXX_LIBRARY_EXCLUDE})
    list(APPEND _TIMEMORY_CXX_LIBRARY_EXCLUDE ${CMAKE_CURRENT_LIST_DIR}/${_SRC} ${_SRC})
endforeach()
if(_TIMEMORY_CXX_LIBRARY_EXCLUDE)
    list(REMOVE_ITEM cxx_sources ${_TIMEMORY_CXX_LIBRARY_EXCLUDE})
endif()

set(_CATEGORIES backends mpl utility variadic)
# bits folder
set(cxx_headers_category_bits)
foreach(_CATEGORY ${_CATEGORIES})
    file(
        GLOB
        cxx_headers_${_CATEGORY}_bits
        ${CMAKE_CURRENT_LIST_DIR}/timemory/${_CATEGORY}/bits/*.h
        ${CMAKE_CURRENT_LIST_DIR}/timemory/${_CATEGORY}/bits/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/timemory/${_CATEGORY}/bits/*.icpp)
    list(APPEND cxx_headers_category_bits ${cxx_headers_${_CATEGORY}_bits})
endforeach()

# all the headers
set(timemory_glob_headers
    ${c_headers}
    ${cxx_headers}
    ${cxx_headers_variadic}
    ${cxx_headers_api}
    ${cxx_headers_ert}
    ${cxx_headers_log}
    ${cxx_headers_backends}
    ${cxx_headers_components}
    ${cxx_headers_sampling}
    ${cxx_headers_mpl}
    ${cxx_headers_math}
    ${cxx_headers_utility}
    ${cxx_headers_bits}
    ${cxx_headers_data}
    ${cxx_headers_category_bits}
    ${cxx_headers_general}
    ${cxx_headers_macros}
    ${tpl_headers}
    ${compat_headers})

if(TIMEMORY_SOURCE_GROUP)
    source_group(TREE ${CMAKE_CURRENT_LIST_DIR} FILES ${timemory_glob_headers})
endif()

# this is only needed for windows
file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/python/*.hpp)
file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/python/*.cpp)

# ----------------------------------------------------------------------------------------#
# build library setup
#
# directly compile sources
set(C_LIBRARY_SOURCES ${c_sources} ${c_headers})
set(CXX_LIBRARY_SOURCES ${cxx_sources} ${timemory_glob_headers})
set(Fortran_LIBRARY_SOURCES ${fortran_sources} ${fortran_headers})

# ----------------------------------------------------------------------------------------#
# build the C++ libraries
#
set(_USE_EXTERN TIMEMORY_USE_EXTERN)

set(_ARCH_LIBRARY)
if(TIMEMORY_USE_ARCH)
    set(_ARCH_LIBRARY timemory-arch)
endif()

set(_ANALYSIS_TOOLS timemory-analysis-tools)
if(NOT _GPERF_IN_LIBRARY)
    set(_ANALYSIS_TOOLS)
endif()

set(_LINKER_LANGUAGE CXX)
if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(_LINKER_LANGUAGE CUDA)
endif()

# ----------------------------------------------------------------------------------------#
#
# Intermediate libraries
#
# ----------------------------------------------------------------------------------------#

timemory_message(STATUS "Adding source/timemory...")
add_subdirectory(timemory)

# ----------------------------------------------------------------------------------------#
#
# MASTER CXX libraries
#
# ----------------------------------------------------------------------------------------#

get_property(CXX_EXTRA_SOURCES GLOBAL PROPERTY TIMEMORY_CXX_LIBRARY_SOURCES)

timemory_get_property_depends(TIMEMORY_OBJECT_GLOBAL_LIBRARIES OBJECT GLOBAL)
timemory_get_property_depends(TIMEMORY_OBJECT_EXTERN_LIBRARIES OBJECT EXTERN)
timemory_get_property_depends(TIMEMORY_OBJECT_COMPONENT_LIBRARIES OBJECT COMPONENT)
timemory_get_property_depends(TIMEMORY_OBJECT_USER_BUNDLE_LIBRARIES OBJECT USER_BUNDLE)

# if(NOT TIMEMORY_BUILD_LTO)
timemory_get_property_depends(TIMEMORY_SHARED_GLOBAL_LIBRARIES SHARED GLOBAL)
timemory_get_property_depends(TIMEMORY_STATIC_GLOBAL_LIBRARIES STATIC GLOBAL)
timemory_get_property_depends(TIMEMORY_SHARED_EXTERN_LIBRARIES SHARED EXTERN)
timemory_get_property_depends(TIMEMORY_STATIC_EXTERN_LIBRARIES STATIC EXTERN)
timemory_get_property_depends(TIMEMORY_SHARED_COMPONENT_LIBRARIES SHARED COMPONENT)
timemory_get_property_depends(TIMEMORY_STATIC_COMPONENT_LIBRARIES STATIC COMPONENT)
timemory_get_property_depends(TIMEMORY_SHARED_USER_BUNDLE_LIBRARIES SHARED USER_BUNDLE)
timemory_get_property_depends(TIMEMORY_STATIC_USER_BUNDLE_LIBRARIES STATIC USER_BUNDLE)
# endif()

if(CMAKE_CUDA_COMPILER)
    set_source_files_properties(${cxx_sources} PROPERTIES # LANGUAGE CUDA
                                                          LINKER_LANGUAGE CUDA)
endif()

if(_BUILD_SHARED_CXX)
    timemory_build_library(
        PIC
        TYPE SHARED
        TARGET_NAME timemory-cxx-shared
        OUTPUT_NAME timemory
        LANGUAGE CXX
        LINKER_LANGUAGE ${_LINKER_LANGUAGE}
        OUTPUT_DIR ${PROJECT_BINARY_DIR}
        SOURCES ${CXX_LIBRARY_SOURCES}
                ${CXX_EXTERN_TARGETS}
                ${CXX_EXTRA_SOURCES}
                ${TIMEMORY_OBJECT_GLOBAL_LIBRARIES}
                ${TIMEMORY_OBJECT_EXTERN_LIBRARIES}
                ${TIMEMORY_OBJECT_COMPONENT_LIBRARIES}
                ${TIMEMORY_OBJECT_USER_BUNDLE_LIBRARIES}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        COMPILE_DEFINITIONS ${_USE_EXTERN})

    target_link_libraries(
        timemory-cxx-shared
        PUBLIC timemory-headers
               timemory-extern
               timemory-vector
               ${TIMEMORY_EXTERNAL_SHARED_INTERFACES}
               ${TIMEMORY_SHARED_GLOBAL_LIBRARIES}
               ${TIMEMORY_SHARED_EXTERN_LIBRARIES}
               ${TIMEMORY_SHARED_COMPONENT_LIBRARIES}
               ${TIMEMORY_SHARED_USER_BUNDLE_LIBRARIES})

    target_link_libraries(timemory-cxx-shared PRIVATE timemory-compile-options
                                                      timemory-develop-options)

    if(TIMEMORY_BUILD_C)
        target_compile_definitions(timemory-cxx-shared PRIVATE TIMEMORY_BUILD_C)
    endif()

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        target_compile_definitions(timemory-cxx-shared PRIVATE DEBUG)
    endif()

    if(WIN32)
        target_compile_definitions(
            timemory-cxx-shared
            PRIVATE TIMEMORY_DLL_EXPORT
            INTERFACE TIMEMORY_DLL_IMPORT)
    endif()

endif()

if(_BUILD_STATIC_CXX)

    timemory_build_library(
        TYPE STATIC
        TARGET_NAME timemory-cxx-static
        OUTPUT_NAME timemory
        LANGUAGE CXX
        LINKER_LANGUAGE ${_LINKER_LANGUAGE}
        OUTPUT_DIR ${PROJECT_BINARY_DIR}
        SOURCES ${CXX_LIBRARY_SOURCES}
                ${CXX_EXTERN_TARGETS}
                ${CXX_EXTRA_SOURCES}
                ${TIMEMORY_OBJECT_GLOBAL_LIBRARIES}
                ${TIMEMORY_OBJECT_EXTERN_LIBRARIES}
                ${TIMEMORY_OBJECT_COMPONENT_LIBRARIES}
                ${TIMEMORY_OBJECT_USER_BUNDLE_LIBRARIES}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        COMPILE_DEFINITIONS ${_USE_EXTERN})

    target_link_libraries(
        timemory-cxx-static
        PUBLIC timemory-headers
               timemory-extern
               timemory-vector
               ${TIMEMORY_EXTERNAL_STATIC_INTERFACES}
               ${TIMEMORY_STATIC_GLOBAL_LIBRARIES}
               ${TIMEMORY_STATIC_EXTERN_LIBRARIES}
               ${TIMEMORY_STATIC_COMPONENT_LIBRARIES}
               ${TIMEMORY_STATIC_USER_BUNDLE_LIBRARIES})

    target_link_libraries(timemory-cxx-static PRIVATE timemory-compile-options
                                                      timemory-develop-options)

    if(TIMEMORY_BUILD_C)
        target_compile_definitions(timemory-cxx-static PRIVATE TIMEMORY_BUILD_C)
    endif()

    target_include_directories(
        timemory-cxx-static
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
               $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>)

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        target_compile_definitions(timemory-cxx-static PRIVATE DEBUG)
    endif()
endif()

if(WIN32
   AND _BUILD_SHARED_CXX
   AND _BUILD_STATIC_CXX)
    # make sure on Windows, static is built first. There is something weird happening here
    # when built in parallel...
    add_dependencies(timemory-cxx-shared timemory-cxx-static)
endif()

# ----------------------------------------------------------------------------------------#
#
# EXTERN C libraries
#
# ----------------------------------------------------------------------------------------#

if(TIMEMORY_BUILD_C)

    if(_BUILD_SHARED_CXX)
        timemory_build_library(
            PIC
            TYPE SHARED
            TARGET_NAME timemory-c-shared
            OUTPUT_NAME ctimemory
            LANGUAGE C
            LINKER_LANGUAGE ${_LINKER_LANGUAGE}
            OUTPUT_DIR ${PROJECT_BINARY_DIR}
            SOURCES ${C_LIBRARY_SOURCES}
            C_COMPILE_OPTIONS ${${PROJECT_NAME}_C_COMPILE_OPTIONS}
            LINK_LIBRARIES
                timemory-external-shared timemory-cxx-shared PRIVATE
                timemory-compile-options timemory-develop-options timemory-analysis-tools)

        target_include_directories(
            timemory-c-shared
            PUBLIC
                $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source>
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>)

        if(WIN32)
            target_compile_definitions(
                timemory-c-shared
                PRIVATE TIMEMORY_CDLL_EXPORT
                INTERFACE TIMEMORY_CDLL_IMPORT)
        endif()
    endif()

    if(_BUILD_STATIC_CXX)

        timemory_build_library(
            TYPE STATIC
            TARGET_NAME timemory-c-static
            OUTPUT_NAME ctimemory
            LANGUAGE C
            LINKER_LANGUAGE ${_LINKER_LANGUAGE}
            OUTPUT_DIR ${PROJECT_BINARY_DIR}
            SOURCES ${C_LIBRARY_SOURCES}
            C_COMPILE_OPTIONS ${${PROJECT_NAME}_C_COMPILE_OPTIONS}
            LINK_LIBRARIES
                timemory-external-static timemory-cxx-static PRIVATE
                timemory-compile-options timemory-develop-options timemory-analysis-tools)

        target_include_directories(
            timemory-c-static
            PUBLIC
                $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source>
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>)
    endif()

    if(WIN32
       AND _BUILD_SHARED_CXX
       AND _BUILD_STATIC_CXX)
        add_dependencies(timemory-c-shared timemory-c-static)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Fortran libraries
#
# ----------------------------------------------------------------------------------------#

if(TIMEMORY_BUILD_FORTRAN)

    if(_BUILD_SHARED_CXX)
        timemory_build_library(
            PIC
            TYPE SHARED
            TARGET_NAME timemory-fortran-shared
            OUTPUT_NAME ftimemory
            LANGUAGE Fortran
            LINKER_LANGUAGE ${_LINKER_LANGUAGE}
            OUTPUT_DIR ${PROJECT_BINARY_DIR}
            SOURCES ${Fortran_LIBRARY_SOURCES}
            LINK_LIBRARIES
                timemory-external-shared PRIVATE timemory-compile-options
                timemory-develop-options timemory-analysis-tools timemory-cxx-shared)

        set_target_properties(timemory-fortran-shared
                              PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

        if(WIN32)
            target_compile_definitions(
                timemory-fortran-shared
                PRIVATE TIMEMORY_CDLL_EXPORT
                INTERFACE TIMEMORY_CDLL_IMPORT)
        endif()
    endif()

    if(_BUILD_STATIC_CXX)

        timemory_build_library(
            TYPE STATIC
            TARGET_NAME timemory-fortran-static
            OUTPUT_NAME ftimemory
            LANGUAGE Fortran
            LINKER_LANGUAGE ${_LINKER_LANGUAGE}
            OUTPUT_DIR ${PROJECT_BINARY_DIR}
            SOURCES ${Fortran_LIBRARY_SOURCES}
            LINK_LIBRARIES
                timemory-external-static PRIVATE timemory-compile-options
                timemory-develop-options timemory-analysis-tools timemory-cxx-static)

        set_target_properties(timemory-fortran-static
                              PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()

    if(WIN32
       AND _BUILD_SHARED_CXX
       AND _BUILD_STATIC_CXX)
        add_dependencies(timemory-fortran-shared timemory-fortran-static)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Language aliases libraries
#
# ----------------------------------------------------------------------------------------#

foreach(_LANG c cxx fortran)
    if(TARGET timemory-${_LANG}-shared)
        add_library(timemory::timemory-${_LANG} ALIAS timemory-${_LANG}-shared)
    elseif(TARGET timemory-${_LANG}-static)
        add_library(timemory::timemory-${_LANG} ALIAS timemory-${_LANG}-static)
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#
#
# TOOL executables and libraries
#
# ----------------------------------------------------------------------------------------#
#
# timem wrapper tool, mpip, etc.
#
timemory_message(STATUS "Adding source/tools...")
add_subdirectory(tools)

set(_OMPT_LIB)
if(TARGET timemory-ompt-static)
    set(_OMPT_LIB timemory-ompt-static)
    list(APPEND TIMEMORY_TOOL_LIBRARIES timemory-ompt-static)
endif()

if(TARGET timemory-ompt-shared)
    set(_OMPT_LIB timemory-ompt-shared)
    list(APPEND TIMEMORY_TOOL_LIBRARIES timemory-ompt-shared)
endif()

if(TARGET timemory-mpip-shared)
    timemory_target_compile_definitions(timemory-mpip-library INTERFACE
                                        TIMEMORY_USE_MPIP_LIBRARY)
    target_link_libraries(timemory-mpip-library INTERFACE timemory-mpip-shared)
    list(APPEND TIMEMORY_TOOL_LIBRARIES timemory-mpip-shared)
else()
    timemory_inform_empty_interface(timemory-mpip-library
                                    "MPIP standalone instrumentation library")
    set(TIMEMORY_BUILD_MPIP_LIBRARY
        OFF
        PARENT_SCOPE)
endif()

if(_OMPT_LIB)
    timemory_target_compile_definitions(timemory-ompt-library INTERFACE
                                        TIMEMORY_USE_OMPT_LIBRARY)
    target_link_libraries(timemory-ompt-library INTERFACE timemory-ompt ${_OMPT_LIB})
else()
    timemory_inform_empty_interface(timemory-ompt-library
                                    "OpenMP stand-alone instrumentation library")
    set(TIMEMORY_BUILD_OMPT_LIBRARY
        OFF
        PARENT_SCOPE)
endif()

if(TARGET timemory-ncclp-shared)
    timemory_target_compile_definitions(timemory-ncclp-library INTERFACE
                                        TIMEMORY_USE_NCCLP_LIBRARY)
    target_link_libraries(timemory-ncclp-library INTERFACE timemory-ncclp-shared)
    list(APPEND TIMEMORY_TOOL_LIBRARIES timemory-ncclp-shared)
else()
    timemory_inform_empty_interface(timemory-ncclp-library
                                    "NCCLP standalone instrumentation library")
    set(TIMEMORY_BUILD_NCCLP_LIBRARY
        OFF
        PARENT_SCOPE)
endif()

if(TARGET timemory-mallocp-shared)
    timemory_target_compile_definitions(timemory-mallocp-library INTERFACE
                                        TIMEMORY_USE_MALLOCP_LIBRARY)
    target_link_libraries(timemory-mallocp-library INTERFACE timemory-mallocp-shared)
    list(APPEND TIMEMORY_TOOL_LIBRARIES timemory-mallocp-shared)
else()
    timemory_inform_empty_interface(timemory-mallocp-library
                                    "MALLOCP standalone instrumentation library")
    set(TIMEMORY_BUILD_MALLOCP_LIBRARY
        OFF
        PARENT_SCOPE)
endif()

if(TARGET timemory-compiler-instrument-base)
    list(APPEND TIMEMORY_TOOL_LIBRARIES timemory-compiler-instrument-base)
endif()

if(TARGET timemory-compiler-instrument-shared)
    target_link_libraries(timemory-compiler-instrument
                          INTERFACE timemory-compiler-instrument-shared)
    list(APPEND TIMEMORY_TOOL_LIBRARIES timemory-compiler-instrument-shared)
    if(TARGET timemory-compiler-instrument-static)
        list(APPEND TIMEMORY_TOOL_LIBRARIES timemory-compiler-instrument-static)
    endif()
elseif(TARGET timemory-compiler-instrument-static)
    target_link_libraries(timemory-compiler-instrument
                          INTERFACE timemory-compiler-instrument-static)
    list(APPEND TIMEMORY_TOOL_LIBRARIES timemory-compiler-instrument-static)
else()
    timemory_inform_empty_interface(timemory-compiler-instrument
                                    "Compiler instrumentation library")
endif()

# ----------------------------------------------------------------------------------------#
#
# INSTALL and EXPORT
#
# ----------------------------------------------------------------------------------------#

list(REMOVE_DUPLICATES TIMEMORY_COMPILED_LIBRARIES)

# C/C++ compiled libraries
timemory_install_libraries(
    TARGETS ${TIMEMORY_COMPILED_LIBRARIES} ${TIMEMORY_TOOL_LIBRARIES}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    EXPORT ${PROJECT_NAME}-library-depends)

get_property(timemory_headers GLOBAL PROPERTY TIMEMORY_HEADERS)
get_property(timemory_sources GLOBAL PROPERTY TIMEMORY_SOURCES)

list(APPEND timemory_headers ${timemory_sources} ${timemory_glob_headers})
list(REMOVE_DUPLICATES timemory_headers)

set(_installed_headers ${timemory_headers} ${tpl_headers} ${tpl_licenses})
list(REMOVE_DUPLICATES _installed_headers)
timemory_install_header_files(${_installed_headers})

# Install the export set for use with the install-tree
if(TIMEMORY_INSTALL_CONFIG)
    install(
        EXPORT ${PROJECT_NAME}-library-depends
        DESTINATION ${CMAKE_INSTALL_CONFIGDIR}
        NAMESPACE ${PROJECT_NAME}::
        OPTIONAL)
endif()

get_property(TIMEMORY_INTERMEDIATE_LIBRARIES GLOBAL
             PROPERTY TIMEMORY_INTERMEDIATE_TARGETS)

set(LIBRARIES
    ${TIMEMORY_COMPILED_LIBRARIES} ${TIMEMORY_INTERFACE_LIBRARIES}
    ${TIMEMORY_INTERMEDIATE_LIBRARIES} ${TIMEMORY_PACKAGE_LIBRARIES}
    ${TIMEMORY_TOOL_LIBRARIES})

if(NOT "${LIBRARIES}" STREQUAL "")
    list(REMOVE_DUPLICATES LIBRARIES)
endif()

export(
    TARGETS ${LIBRARIES}
    NAMESPACE ${PROJECT_NAME}::
    FILE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-build.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES)

# ----------------------------------------------------------------------------------------#
#
# PYTHON Bindings
#
# ----------------------------------------------------------------------------------------#

timemory_message(STATUS "Adding source/python...")
add_subdirectory(python)

# ----------------------------------------------------------------------------------------#
#
# Package setup
#
# ----------------------------------------------------------------------------------------#

timemory_message(STATUS "Configuring packaging...")
include(PackageConfigure)

# ----------------------------------------------------------------------------------------#
#
# TESTING -- Unit tests
#
# ----------------------------------------------------------------------------------------#

timemory_message(STATUS "Adding source/tests...")
add_subdirectory(tests)

set(TIMEMORY_MAKEFILE_COMPILED_LIBRARIES
    ${TIMEMORY_COMPILED_LIBRARIES}
    PARENT_SCOPE)

set(TIMEMORY_MAKEFILE_INTERFACE_LIBRARIES
    ${TIMEMORY_INTERFACE_LIBRARIES} ${${PROJECT_NAME_UC}_INTERFACE_LIBRARIES}
    PARENT_SCOPE)
