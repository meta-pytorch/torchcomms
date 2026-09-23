# ########################################################################################
#
# timemory (Python)
#
# ########################################################################################

include(ProcessorCount)
processorcount(NCPU)

# if set, will screw up loading library
unset(CMAKE_DEBUG_POSTFIX)
set(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME python)

# handle unity build for python
option(TIMEMORY_BUILD_PYTHON_UNITY "Build python interface with unity build support"
       ${TIMEMORY_UNITY_BUILD})
mark_as_advanced(TIMEMORY_BUILD_PYTHON_UNITY)
set(CMAKE_UNITY_BUILD ${TIMEMORY_BUILD_PYTHON_UNITY})
if(NOT NCPU GREATER 2 OR "$ENV{CI}" STREQUAL "true")
    set(CMAKE_UNITY_BUILD OFF)
else()
    set(CMAKE_UNITY_BUILD_BATCH_SIZE 4)
endif()

option(TIMEMORY_BUILD_PYTHON_HIDDEN "Build python interface with hidden symbols" OFF)
mark_as_advanced(TIMEMORY_BUILD_PYTHON_HIDDEN)
set(_visibility timemory::timemory-default-visibility)
if(TIMEMORY_BUILD_PYTHON_HIDDEN)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
    set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
    set(_visibility timemory::timemory-hidden-visibility)
endif()

# ########################################################################################

function(TIMEMORY_CONFIGURE_PYTARGET _TARGET)

    add_library(timemory::${_TARGET} ALIAS ${_TARGET})
    target_link_libraries(${_TARGET} PRIVATE libpytimemory-interface)

    set(_SUBDIR ${ARGN})
    if(_SUBDIR)
        set(_SUBDIR "/${_SUBDIR}")
    endif()

    if(TIMEMORY_USE_CUDA)
        set(EXTRA_PROPERTIES LINKER_LANGUAGE CUDA)
    endif()

    set_target_properties(
        ${_TARGET}
        PROPERTIES PREFIX ""
                   SUFFIX "${PYTHON_MODULE_EXTENSION}"
                   LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory${_SUBDIR}
                   ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory${_SUBDIR}
                   RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory${_SUBDIR}
                   PDB_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory${_SUBDIR}
                   INSTALL_RPATH_USE_LINK_PATH ON
                   ${EXTRA_PROPERTIES})

    set(_PYLIB ${CMAKE_INSTALL_PYTHONDIR}/timemory${_SUBDIR})
    if(NOT IS_ABSOLUTE "${_PYLIB}")
        set(_PYLIB ${CMAKE_INSTALL_PREFIX}/${_PYLIB})
    endif()

    if(SKBUILD)
        set(LIB_RELPATH ../../../..)
    else()
        file(RELATIVE_PATH LIB_RELPATH "${_PYLIB}"
             "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    endif()

    if(UNIX AND NOT APPLE)
        set_target_properties(
            ${_TARGET}
            PROPERTIES
                INSTALL_RPATH
                "\$ORIGIN:\$ORIGIN/${LIB_RELPATH}:\$ORIGIN/../../../..:${CMAKE_INSTALL_RPATH}"
            )
    endif()

    install(
        TARGETS ${_TARGET}
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/timemory${_SUBDIR}
        OPTIONAL)

    if(NOT "${_TARGET}" STREQUAL "libpytimemory")
        add_dependencies(libpytimemory ${_TARGET})
    endif()
endfunction()

# ########################################################################################

if(TIMEMORY_USE_PYTHON)

    add_library(timemory-python-compile-options INTERFACE)
    add_library(timemory::timemory-python-compile-options ALIAS
                timemory-python-compile-options)
    add_cxx_flag_if_avail("-frtti" timemory-python-compile-options)
    add_cxx_flag_if_avail("-Wno-unused-value" timemory-python-compile-options)
    add_cxx_flag_if_avail("-Wno-range-loop-analysis" timemory-python-compile-options)
    add_cxx_flag_if_avail("-ftls-model=global-dynamic" timemory-python-compile-options)
    add_cxx_flag_if_avail("-Wno-deprecated-declarations" timemory-python-compile-options)
    add_cxx_flag_if_avail("-Wno-unused-but-set-parameter" timemory-python-compile-options)

    file(GLOB_RECURSE pybind_headers ${PROJECT_SOURCE_DIR}/external/pybind11/include/*.h)
    file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/libpytimemory*.hpp)
    # file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/libpytimemory*.cpp)
    set(pysources
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-api.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-auto-timer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-component-bundle.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-component-list.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-components.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-enumeration.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-hardware-counters.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-profile.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-rss-usage.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-settings.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-signals.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-statistics.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-storage.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-trace.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-units.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory.cpp)

    if(TIMEMORY_USE_CUDA
       AND CMAKE_CUDA_COMPILER
       AND TIMEMORY_BUILD_PYTHON_CUDA)
        set_source_files_properties(${pysources} PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE
                                                                          CUDA)
    endif()

    set(pybind_libs pybind11::module)
    if(TIMEMORY_BUILD_LTO)
        list(APPEND pybind_libs timemory::timemory-lto)
    endif()

    set(_LINK shared)
    if((APPLE AND BUILD_STATIC_LIBS) OR (BUILD_STATIC_LIBS AND NOT BUILD_SHARED_LIBS))
        set(_LINK static)
    endif()

    add_library(libpytimemory-interface INTERFACE)
    target_link_libraries(
        libpytimemory-interface
        INTERFACE ${pybind_libs}
                  timemory::timemory-python
                  timemory::timemory-headers
                  timemory::timemory-cxx-${_LINK}
                  timemory::timemory-compile-options
                  timemory::timemory-external-${_LINK}
                  timemory::timemory-mpip-library
                  timemory::timemory-ompt-library
                  timemory::timemory-ncclp-library
                  timemory::timemory-mallocp-library
                  timemory::timemory-python-compile-options
                  ${_visibility})

    target_compile_definitions(libpytimemory-interface INTERFACE TIMEMORY_PYBIND11_SOURCE)

    target_include_directories(
        libpytimemory-interface SYSTEM
        INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/pybind11/include>)

    add_library(libpytimemory MODULE ${TIMEMORY_EXCLUDE_FROM_ALL} ${pysources}
                                     ${pyheaders} ${pybind_headers})
    timemory_configure_pytarget(libpytimemory libs)

    timemory_message(STATUS "Adding source/python/tools...")
    add_subdirectory(tools)

    # if(APPLE)
    if(APPLE AND "$ENV{TRAVIS}" STREQUAL "true")
        #
        file(
            WRITE ${PROJECT_BINARY_DIR}/.pdbrc
            "r
bt
q
")

        file(
            WRITE ${CMAKE_CURRENT_BINARY_DIR}/deadlock-trace.py
            "#!/usr/bin/env python

print('importing timemory')
import timemory
print('timemory imported')
")

        set(_IGNORE
            # --ignore-module=_bootstrap
            # --ignore-module=_bootstrap_external
            # --ignore-module=sre_parse
            # --ignore-module=sre_compile
            )

        set(_FILE ${CMAKE_CURRENT_BINARY_DIR}/deadlock-trace.py)

        add_test(
            NAME python-deadlock-pdb
            COMMAND ${PYTHON_EXECUTABLE} -m pdb ${_FILE}
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

        add_test(
            NAME python-deadlock-trace
            COMMAND ${PYTHON_EXECUTABLE} -m trace -t ${_IGNORE} ${_FILE}
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

        add_test(
            NAME python-deadlock-listfuncs
            COMMAND ${PYTHON_EXECUTABLE} -m trace -l ${_IGNORE} ${_FILE}
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

        add_test(
            NAME python-deadlock-trackcalls
            COMMAND ${PYTHON_EXECUTABLE} -m trace -T ${_IGNORE} ${_FILE}
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

        set_tests_properties(python-deadlock-pdb PROPERTIES DEPENDS libpytimemory)
        set_tests_properties(python-deadlock-trace PROPERTIES DEPENDS libpytimemory)
        set_tests_properties(python-deadlock-listfuncs PROPERTIES DEPENDS libpytimemory)
        set_tests_properties(python-deadlock-trackcalls PROPERTIES DEPENDS libpytimemory)

        find_program(LLDB_EXE NAMES lldb)
        if(LLDB_EXE)
            file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/pylldb-run "run\nrun\n")
            file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/pylldb-crash "bt")
            add_test(
                NAME python-deadlock-lldb
                COMMAND
                    ${LLDB_EXE} -b -s ${CMAKE_CURRENT_BINARY_DIR}/pylldb-run
                    --source-on-crash ${CMAKE_CURRENT_BINARY_DIR}/pylldb-crash --
                    ${PYTHON_EXECUTABLE} ${_FILE}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
            set_tests_properties(python-deadlock-lldb PROPERTIES DEPENDS libpytimemory)
        endif()
    endif()

endif()
