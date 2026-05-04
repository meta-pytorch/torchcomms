# ----------------------------------------------------------------------------------------#
# recommend out-of-source builds
# ----------------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND CMAKE_CURRENT_SOURCE_DIR STREQUAL
                                                  CMAKE_SOURCE_DIR)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(AUTHOR_WARNING "In-source build")
endif()

# if(APPLE AND NOT "$ENV{CONDA_PYTHON_EXE}" STREQUAL "") # disable by default bc conda
# will set these and cause problem with python bindings set(CMAKE_C_FLAGS "" CACHE STRING
# "") set(CMAKE_CXX_FLAGS "" CACHE STRING "") endif()

# ----------------------------------------------------------------------------------------#
# project
# ----------------------------------------------------------------------------------------#

# sets version info, policies, some options, and other miscellaneous stuff
include("${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/ProjectSetup.cmake")

project(
    timemory
    LANGUAGES C CXX
    VERSION ${TIMEMORY_VERSION})

cmake_policy(SET CMP0010 NEW)
cmake_policy(SET CMP0022 NEW)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0053 NEW)
cmake_policy(SET CMP0063 NEW)
cmake_policy(SET CMP0069 NEW)
if(NOT CMAKE_VERSION VERSION_LESS 3.13)
    cmake_policy(SET CMP0077 NEW)
    cmake_policy(SET CMP0079 NEW)
endif()
if(NOT CMAKE_VERSION VERSION_LESS 3.14)
    cmake_policy(SET CMP0082 NEW)
endif()
if(NOT CMAKE_VERSION VERSION_LESS 3.18)
    if(CMAKE_VERSION VERSION_LESS 3.27.6)
        cmake_policy(SET CMP0104 OLD)
    else()
        set(CMAKE_CUDA_ARCHITECTURES OFF)
    endif()
endif()
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # DOWNLOAD_EXTRACT_TIMESTAMP
endif()

# Check if project is being used directly or via add_subdirectory
set(${PROJECT_NAME}_MAIN_PROJECT ON)
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR OR NOT "${CMAKE_PROJECT_NAME}"
                                                             STREQUAL "${PROJECT_NAME}")
    set(${PROJECT_NAME}_MAIN_PROJECT OFF)
endif()

# set these as the defaults
if(timemory_MAIN_PROJECT)
    set(CMAKE_ENABLE_EXPORTS
        ON
        CACHE BOOL "Executable exports symbols for loadable modules")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH
        ON
        CACHE BOOL "Append directories in the linker search path")
    # set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build position independent code")
    mark_as_advanced(CMAKE_ENABLE_EXPORTS)
    foreach(_FILE version.h defines.h)
        if(EXISTS ${PROJECT_SOURCE_DIR}/source/timemory/${_FILE})
            file(REMOVE ${PROJECT_SOURCE_DIR}/source/timemory/${_FILE})
        endif()
    endforeach()
else()
    set(CMAKE_ENABLE_EXPORTS ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
    # set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# install directories
include(GNUInstallDirs)
# cmake installation folder -- change CMAKE_INSTALL_DATAROOTDIR to tweak this
set(CMAKE_INSTALL_CONFIGDIR ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME})

if(UNIX AND NOT APPLE)
    if(timemory_MAIN_PROJECT)
        set(CMAKE_INSTALL_RPATH
            "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/timemory:\$ORIGIN")
        if(NOT SKBUILD AND NOT SPACK_BUILD)
            set(CMAKE_INSTALL_RPATH
                "${CMAKE_INSTALL_RPATH}:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/timemory"
                )
        endif()
    else()
        set(CMAKE_INSTALL_RPATH "\$ORIGIN")
    endif()
endif()

# create the full path version and generic path versions
foreach(_TYPE DATAROOT CMAKE INCLUDE LIB BIN MAN DOC)
    # generic "PROJECT_INSTALL_" variables (used by documentation)"
    set(PROJECT_INSTALL_${_TYPE}DIR ${CMAKE_INSTALL_${TYPE}DIR})
endforeach()

set(CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
    ${CMAKE_CURRENT_LIST_DIR}/external/pybind11/tools ${CMAKE_MODULE_PATH})
set(CMAKE_DIRECTORY_LABELS "${PROJECT_NAME}")
set(CMAKE_INSTALL_MESSAGE LAZY)

foreach(_TYPE MAJOR MINOR PATCH)
    set(TIMEMORY_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}})
    set(${PROJECT_NAME}_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}})
endforeach()
set(timemory_VERSION ${PROJECT_VERSION})
set(LIBNAME timemory)

# ----------------------------------------------------------------------------------------#
# set the output directory (critical for Windows and Xcode)
#
foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    set(_BIN_DIR ${CMAKE_BINARY_DIR})
    if(NOT ${PROJECT_NAME}_MAIN_PROJECT)
        set(_BIN_DIR ${PROJECT_BINARY_DIR})
    endif()
    if(WIN32)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${_BIN_DIR}/outputs/runtime)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${_BIN_DIR}/outputs/runtime)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_BIN_DIR}/outputs/archive)
    else()
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${_BIN_DIR})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${_BIN_DIR})
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_BIN_DIR})
    endif()
endforeach()

if(WIN32)
    # use folders for cmake predefined projects
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()
# ----------------------------------------------------------------------------------------#
# configuration
# ----------------------------------------------------------------------------------------#

include(MacroUtilities)
include(Options)

# ----------------------------------------------------------------------------------------#
# compile-timing-perf
# ----------------------------------------------------------------------------------------#

option(TIMEMORY_USE_CTP "Enable compile-time-perf" OFF)
mark_as_advanced(TIMEMORY_USE_CTP)
if(TIMEMORY_BUILD_DEVELOPER OR TIMEMORY_USE_CTP)
    find_package(compile-time-perf)
    if(compile-time-perf_FOUND)
        enable_compile_time_perf(
            timemory-compile-time
            ${TIMEMORY_CTP_OPTIONS}
            LINK
            ANALYZER_OPTIONS
            -s
            "${PROJECT_BINARY_DIR}/"
            "${CMAKE_BINARY_DIR}/"
            "${PROJECT_SOURCE_DIR}/"
            "${CMAKE_SOURCE_DIR}/"
            -f
            "lang-all"
            "so"
            "a"
            "dylib"
            "dll"
            -i
            "^-"
            -e
            "^(@rpath).*")
        set(TIMEMORY_USE_CTP ON)
    else()
        set(TIMEMORY_USE_CTP OFF)
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------------------------------------#
# handle some additional configuration before other includes
# ----------------------------------------------------------------------------------------#

set(_BUILD_SHARED_CXX ${BUILD_SHARED_LIBS})
set(_BUILD_STATIC_CXX ${BUILD_STATIC_LIBS})

option(TIMEMORY_BUILD_QUIET "Suppress author warnings without -Wno-dev" OFF)
mark_as_advanced(TIMEMORY_BUILD_QUIET)

option(TIMEMORY_QUIET_CONFIG "Make timemory configuration quieter" OFF)
mark_as_advanced(TIMEMORY_QUIET_CONFIG)

if(SKBUILD)
    if(BUILD_STATIC_LIBS AND TIMEMORY_USE_PYTHON)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    endif()
else()
    if(BUILD_STATIC_LIBS
       AND NOT BUILD_SHARED_LIBS
       AND TIMEMORY_USE_PYTHON
       AND NOT CMAKE_POSITION_INDEPENDENT_CODE)
        message(
            FATAL_ERROR
                "Error! Python compilation with static libraries requires CMAKE_POSITION_INDEPENDENT_CODE to be ON"
            )
    endif()
endif()

# ----------------------------------------------------------------------------------------#
# thread-local static settings. NOTE: if building Python, the thread-local storage MUST be
# global-dynamic
# ----------------------------------------------------------------------------------------#

set(_TLS_DESCRIPT
    "Thread-local static model: 'global-dynamic', 'local-dynamic', 'initial-exec', 'local-exec'"
    )
set(_TLS_OPTIONS "global-dynamic" "local-dynamic" "initial-exec" "local-exec")

if(SKBUILD
   OR TIMEMORY_USE_PYTHON
   OR TIMEMORY_USE_DYNINST)
    set(TIMEMORY_TLS_MODEL
        "global-dynamic"
        CACHE STRING "${_TLS_DESCRIPT}")
    # ensure local override
    set(TIMEMORY_TLS_MODEL "global-dynamic")
else()
    set(TIMEMORY_TLS_MODEL
        "initial-exec"
        CACHE STRING "${_TLS_DESCRIPT}")
endif()

set_property(CACHE TIMEMORY_TLS_MODEL PROPERTY STRINGS "${_TLS_OPTIONS}")
if(NOT "${TIMEMORY_TLS_MODEL}" IN_LIST _TLS_OPTIONS)
    message(FATAL_ERROR "TIMEMORY_TLS_MODEL must be one of: \"${_TLS_OPTIONS}\"")
endif()

timemory_add_feature(TIMEMORY_TLS_MODEL "${_TLS_DESCRIPT}")
unset(_TLS_DESCRIPT)
unset(_TLS_OPTIONS)

# ----------------------------------------------------------------------------------------#
# rest of configuration
# ----------------------------------------------------------------------------------------#

include(CMakeParseArguments)

timemory_message(STATUS "Adding clang-format...")
include(ClangFormat)

timemory_message(STATUS "Configuring compilers...")
include(Compilers)

timemory_message(STATUS "Configuring build settings...")
include(BuildSettings)

timemory_message(STATUS "Configuring external packages and interface libraries...")
include(Packages)

# only after submodules have been added
if(TIMEMORY_USE_COVERAGE)
    set(CMAKE_UNITY_BUILD OFF)
    set(TIMEMORY_UNITY_BUILD
        OFF
        CACHE BOOL "" FORCE)
else()
    set(CMAKE_UNITY_BUILD ${TIMEMORY_UNITY_BUILD})
endif()

# set these as the defaults
if(timemory_MAIN_PROJECT)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN
        OFF
        CACHE BOOL "Add compile flag to hide symbols of inline functions")
    set(CMAKE_C_VISIBILITY_PRESET
        "default"
        CACHE STRING "Default visibility")
    set(CMAKE_CXX_VISIBILITY_PRESET
        "default"
        CACHE STRING "Default visibility")
elseif(TIMEMORY_BUILD_HIDDEN_VISIBILITY)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
    set(CMAKE_C_VISIBILITY_PRESET "hidden")
    set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
else()
    set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
    set(CMAKE_C_VISIBILITY_PRESET "default")
    set(CMAKE_CXX_VISIBILITY_PRESET "default")
endif()

# ----------------------------------------------------------------------------------------#
# timemory source
# ----------------------------------------------------------------------------------------#

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    _timemory_activate_clang_tidy() # activate clang-tidy if enabled
endif()

set(timemory_DIR
    ${PROJECT_BINARY_DIR}
    CACHE PATH "Path to timemory" FORCE)

timemory_message(STATUS "Adding timemory (python)...")
add_subdirectory(timemory)

timemory_message(STATUS "Adding source...")
add_subdirectory(source)

timemory_message(STATUS "Adding external...")
add_subdirectory(external)

# Install the export set for use with the install-tree
if(TIMEMORY_INSTALL_CONFIG)
    timemory_message(STATUS "Exporting library dependencies...")
    install(
        EXPORT ${PROJECT_NAME}-library-depends
        DESTINATION ${CMAKE_INSTALL_CONFIGDIR}
        NAMESPACE ${PROJECT_NAME}::
        OPTIONAL)
endif()

# ----------------------------------------------------------------------------------------#
# Examples and Testing
# ----------------------------------------------------------------------------------------#

if(TIMEMORY_BUILD_EXAMPLES)
    timemory_message(STATUS "Adding examples...")
    add_subdirectory(examples)
elseif(TIMEMORY_USE_PYTHON AND TIMEMORY_CI)
    timemory_message(STATUS "Adding examples/ex-python...")
    # copies over some python scripts
    function(CONFIGURE_PYTHON_SCRIPT)
        foreach(_TYPE ${ARGN})
            set(FILENAME ex_python_${_TYPE})
            configure_file(${PROJECT_SOURCE_DIR}/examples/ex-python/ex_${_TYPE}.py
                           ${PROJECT_BINARY_DIR}/ex_python_${_TYPE} @ONLY)
        endforeach()
    endfunction()
    configure_python_script(sample tracer profiler general builtin external)
endif()

# ----------------------------------------------------------------------------------------#
# Documentation
# ----------------------------------------------------------------------------------------#

if(timemory_MAIN_PROJECT)

    include(Documentation)

    if(TIMEMORY_BUILD_DOCS)
        timemory_message(STATUS "Adding documentation...")
        generate_documentation(Doxyfile.${PROJECT_NAME})
    endif()

    option(TIMEMORY_PRINT_FEATURES
           "Print the list of enabled/disabled options and interface libraries" ON)
    mark_as_advanced(TIMEMORY_PRINT_FEATURES)

    if(TIMEMORY_PRINT_FEATURES AND NOT TIMEMORY_QUIET_CONFIG)
        timemory_print_features()
    endif()

endif()

# ----------------------------------------------------------------------------------------#
# Makefile.timemory.inc
# ----------------------------------------------------------------------------------------#

option(TIMEMORY_GENERATE_MAKEFILE_INCLUDE
       "Generate a Makefile.timemory.inc for Makefile build systems"
       ${TIMEMORY_INSTALL_CONFIG})
mark_as_advanced(TIMEMORY_GENERATE_MAKEFILE_INCLUDE)

if(TIMEMORY_GENERATE_MAKEFILE_INCLUDE)
    include(GenerateMakefile)
endif()

# remove testing configure warnings: "Manually-specified variables were not used by the
# project: ... etc. ..."
#
foreach(_EXAMPLE_OPTION MPI TAU CUDA PAPI CUPTI KOKKOS CALIPER)
    set(_IGNORE ${USE_${_EXAMPLE_OPTION}})
endforeach()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(ConfigCPack)
endif()
