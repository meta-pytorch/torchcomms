# include guard
include_guard(DIRECTORY)

# ########################################################################################
#
# timemory Options
#
# ########################################################################################

include(MacroUtilities)
include(CheckLanguage)

set(TIMEMORY_REQUIRE_PACKAGES
    ON
    CACHE BOOL "Disable auto-detection and explicitly require packages")

# advanced options not using the "timemory_add_option" macro
option(SPACK_BUILD "Tweak some installation directories when building via spack" OFF)
option(TIMEMORY_CI "Timemory continuous integration" OFF)
option(TIMEMORY_SOURCE_GROUP "Enable source_group" OFF)
mark_as_advanced(SPACK_BUILD)
mark_as_advanced(TIMEMORY_CI)
mark_as_advanced(TIMEMORY_SOURCE_GROUP)

function(DEFINE_DEFAULT_OPTION VAR VAL)
    if(TIMEMORY_REQUIRE_PACKAGES)
        set(${VAR}
            OFF
            PARENT_SCOPE)
    else()
        foreach(_ARG ${ARGN})
            if(NOT ${_ARG})
                set(VAL OFF)
                break()
            endif()
        endforeach()
        set(${VAR}
            ${VAL}
            PARENT_SCOPE)
    endif()
    set_property(GLOBAL APPEND PROPERTY DEFAULT_OPTION_VARIABLES ${VAR})
endfunction()

set(_FEATURE)
set(_USE_PAPI OFF)
set(_USE_COVERAGE OFF)
set(_BUILD_OPT OFF)
set(_BUILD_GOTCHA OFF)
set(_BUILD_CALIPER ON)
set(_BUILD_LIBUNWIND ON)
set(_BUILD_FORTRAN ON)
set(_NON_APPLE_UNIX OFF)
set(_UNIX_OS ${UNIX})
set(_DEFAULT_BUILD_SHARED ON)
set(_DEFAULT_BUILD_STATIC OFF)
set(_USE_XML ON)
set(_USE_LIBUNWIND OFF)

set(TIMEMORY_SANITIZER_TYPE
    leak
    CACHE STRING "Sanitizer type")
set(TIMEMORY_gperftools_COMPONENTS
    "profiler"
    CACHE STRING "gperftools components")
set(TIMEMORY_gperftools_COMPONENTS_OPTIONS
    "profiler;tcmalloc;tcmalloc_and_profiler;tcmalloc_debug;tcmalloc_minimal;tcmalloc_minimal_debug"
    )
set_property(CACHE TIMEMORY_gperftools_COMPONENTS
             PROPERTY STRINGS "${TIMEMORY_gperftools_COMPONENTS_OPTIONS}")

# use generic if defined
if(DEFINED SANITIZER_TYPE AND NOT "${SANITIZER_TYPE}" STREQUAL "")
    set(TIMEMORY_SANITIZER_TYPE
        "${SANITIZER_TYPE}"
        CACHE STRING "Sanitizer type" FORCE)
endif()

if(TIMEMORY_USE_SANITIZER)
    set(PTL_USE_SANITIZER
        ${TIMEMORY_USE_SANITIZER}
        CACHE BOOL "Enable sanitizer" FORCE)
    set(PTL_SANITIZER_TYPE
        ${TIMEMORY_SANITIZER_TYPE}
        CACHE STRING "Sanitizer type" FORCE)
    mark_as_advanced(PTL_SANITIZER_TYPE)
endif()

if(NOT ${PROJECT_NAME}_MAIN_PROJECT)
    set(_FEATURE NO_FEATURE)
endif()

if(UNIX AND NOT APPLE)
    set(_NON_APPLE_UNIX ON)
    set(_USE_PAPI ON)
    set(_BUILD_GOTCHA ON)
endif()

if(NOT UNIX)
    set(_UNIX_OS OFF)
    set(_USE_LIBUNWIND OFF)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(_BUILD_OPT ON)
endif()

if(TIMEMORY_BUILD_TESTING)
    set(TIMEMORY_BUILD_EXAMPLES ON)
endif()

if(TIMEMORY_BUILD_MINIMAL_TESTING)
    set(TIMEMORY_BUILD_TESTING ON)
    set(TIMEMORY_BUILD_EXAMPLES OFF)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND TIMEMORY_CI)
    set(_USE_COVERAGE ON)
endif()

if(WIN32)
    set(_BUILD_CALIPER OFF)
    set(_BUILD_LIBUNWIND OFF)
endif()

# intel compiler has had issues with rapidxml library in past so default to off
if(CMAKE_CXX_COMPILER_IS_INTEL)
    set(_USE_XML OFF)
endif()

if(NOT TIMEMORY_BUILD_LIBUNWIND)
    timemory_test_find_package(libunwind _USE_LIBUNWIND)
else()
    if(UNIX AND NOT APPLE)
        set(_USE_LIBUNWIND ON)
    else()
        set(_USE_LIBUNWIND OFF)
    endif()
endif()

set(_REQUIRE_LIBUNWIND OFF)
set(_HATCHET ${_UNIX_OS})

# once the cuda language is enabled, the CMAKE_CUDA_HOST_COMPILER variable is read-only
# and change to it are undefined
timemory_get_enabled_languages(TIMEMORY_ENABLED_LANGUAGES)
if(NOT "CUDA" IN_LIST TIMEMORY_ENABLED_LANGUAGES
   AND "$ENV{CUDAHOSTCXX}" STREQUAL ""
   AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()

# skip check_language if supposed to fail
if(DEFINED TIMEMORY_USE_CUDA
   AND TIMEMORY_USE_CUDA
   AND TIMEMORY_REQUIRE_PACKAGES)
    enable_language(CUDA)
endif()

# Check if CUDA can be enabled if CUDA is enabled or in auto-detect mode
if(TIMEMORY_USE_CUDA OR (NOT DEFINED TIMEMORY_USE_CUDA AND NOT TIMEMORY_REQUIRE_PACKAGES))
    set(_USE_CUDA ON)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        timemory_message(STATUS "CUDA support not detected")
        set(_USE_CUDA OFF)
    endif()
else()
    set(_USE_CUDA OFF)
endif()

set(_USE_HIP OFF)
if(NOT _USE_CUDA OR (NOT DEFINED TIMEMORY_USE_HIP AND NOT TIMEMORY_REQUIRE_PACKAGES))
    set(_USE_HIP ON)
endif()

# On Windows do not try run check_language because it tends to hang indefinitely
if(WIN32 AND NOT DEFINED TIMEMORY_BUILD_FORTRAN)
    set(_BUILD_FORTRAN OFF)
elseif(TIMEMORY_BUILD_FORTRAN OR NOT DEFINED TIMEMORY_BUILD_FORTRAN)
    check_language(Fortran)
    if(CMAKE_Fortran_COMPILER)
        enable_language(Fortran)
    else()
        timemory_message(STATUS "Fortran support not detected")
        set(_BUILD_FORTRAN OFF)
    endif()
endif()

# if already defined, set default for shared to OFF
if(DEFINED BUILD_STATIC_LIBS AND BUILD_STATIC_LIBS)
    set(_DEFAULT_BUILD_STATIC ON)
    set(_DEFAULT_BUILD_SHARED OFF)
endif()

# if already defined, set default for shared to OFF
if(DEFINED BUILD_SHARED_LIBS AND BUILD_SHARED_LIBS)
    set(_DEFAULT_BUILD_STATIC OFF)
    set(_DEFAULT_BUILD_SHARED ON)
endif()

# something got messed up, so reset
if(NOT _DEFAULT_BUILD_SHARED AND NOT _DEFAULT_BUILD_STATIC)
    set(_DEFAULT_BUILD_SHARED ON)
    set(_DEFAULT_BUILD_STATIC ON)
endif()

# except is setup.py, always default static to off
if(SKBUILD)
    set(_DEFAULT_BUILD_SHARED ON)
    set(_DEFAULT_BUILD_STATIC OFF)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _CONFIG)

# CMake options
timemory_add_feature(CMAKE_BUILD_TYPE
                     "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" DOC)
timemory_add_feature(CMAKE_INSTALL_PREFIX "Installation prefix" DOC)
timemory_add_feature(CMAKE_C_STANDARD "C language standard" DOC)
timemory_add_feature(CMAKE_CXX_STANDARD "C++ language standard" DOC)
timemory_add_feature(CMAKE_CUDA_STANDARD "CUDA language standard" DOC)
timemory_add_feature(CMAKE_C_FLAGS "C build flags")
timemory_add_feature(CMAKE_CXX_FLAGS "C++ build flags")
timemory_add_feature(CMAKE_CUDA_FLAGS "CUDA build flags")
timemory_add_feature(CMAKE_C_FLAGS_${_CONFIG} "C optimization type build flags")
timemory_add_feature(CMAKE_CXX_FLAGS_${_CONFIG} "C++ optimization type build flags")
timemory_add_feature(CMAKE_CUDA_FLAGS_${_CONFIG} "CUDA optimization type build flags")

timemory_add_option(BUILD_SHARED_LIBS "Build shared libraries" ${_DEFAULT_BUILD_SHARED}
                    CMAKE_DEFINE)
timemory_add_option(BUILD_STATIC_LIBS "Build static libraries" ${_DEFAULT_BUILD_STATIC}
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_SKIP_BUILD "Disable building any libraries" OFF)

if(TIMEMORY_SKIP_BUILD)
    # local override
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_STATIC_LIBS OFF)
endif()

if(NOT BUILD_SHARED_LIBS
   AND NOT BUILD_STATIC_LIBS
   AND NOT TIMEMORY_SKIP_BUILD)
    message(STATUS "")
    message(
        STATUS
            "Set TIMEMORY_SKIP_BUILD=ON instead of BUILD_SHARED_LIBS=OFF and BUILD_STATIC_LIBS=OFF"
        )
    message(STATUS "")
    message(FATAL_ERROR "Confusing settings")
endif()

if(NOT BUILD_SHARED_LIBS AND NOT BUILD_STATIC_LIBS)
    # local override
    set(TIMEMORY_BUILD_C OFF)
    set(TIMEMORY_BUILD_PYTHON OFF)
    set(TIMEMORY_BUILD_FORTRAN OFF)
    set(TIMEMORY_USE_PYTHON OFF)
    set(TIMEMORY_USE_NCCL OFF)
    set(TIMEMORY_BUILD_KOKKOS_TOOLS OFF)
    set(TIMEMORY_BUILD_DYNINST_TOOLS OFF)
    set(TIMEMORY_BUILD_MPIP_LIBRARY OFF)
    set(TIMEMORY_BUILD_OMPT_LIBRARY OFF)
    set(TIMEMORY_BUILD_NCCLP_LIBRARY OFF)
endif()

if(${PROJECT_NAME}_MAIN_PROJECT OR TIMEMORY_LANGUAGE_STANDARDS)
    if("${CMAKE_CXX_STANDARD}" LESS 17)
        unset(CMAKE_CXX_STANDARD CACHE)
    endif()

    if("${CMAKE_CUDA_STANDARD}" LESS 14)
        unset(CMAKE_CUDA_STANDARD CACHE)
    endif()
endif()

if(${PROJECT_NAME}_MAIN_PROJECT OR TIMEMORY_LANGUAGE_STANDARDS)
    # standard
    set(CMAKE_C_STANDARD
        11
        CACHE STRING "C language standard")
    set(CMAKE_CXX_STANDARD
        17
        CACHE STRING "CXX language standard")
    if(CMAKE_VERSION VERSION_LESS 3.18.0)
        set(CMAKE_CUDA_STANDARD
            14
            CACHE STRING "CUDA language standard")
    else()
        set(CMAKE_CUDA_STANDARD
            ${CMAKE_CXX_STANDARD}
            CACHE STRING "CUDA language standard")
    endif()

    # standard required
    timemory_add_option(CMAKE_C_STANDARD_REQUIRED "Require C language standard" ON)
    timemory_add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard" ON)
    timemory_add_option(CMAKE_CUDA_STANDARD_REQUIRED "Require C++ language standard" ON)

    # compiling with Clang + NVCC will produce many warnings w/o GCC extensions
    set(_CXX_EXT OFF)
    if(CMAKE_CXX_COMPILER_IS_CLANG AND (TIMEMORY_USE_CUDA OR _USE_CUDA))
        set(_CXX_EXT ON)
    endif()

    # extensions
    timemory_add_option(CMAKE_C_EXTENSIONS "C language standard extensions (e.g. gnu11)"
                        OFF)
    timemory_add_option(CMAKE_CXX_EXTENSIONS "C++ language standard (e.g. gnu++17)"
                        ${_CXX_EXT})
    timemory_add_option(CMAKE_CUDA_EXTENSIONS "CUDA language standard (e.g. gnu++14)" OFF)
else()
    timemory_add_feature(CMAKE_C_STANDARD_REQUIRED "Require C language standard")
    timemory_add_feature(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard")
    timemory_add_feature(CMAKE_CUDA_STANDARD_REQUIRED "Require C++ language standard")
    timemory_add_feature(CMAKE_C_EXTENSIONS "C language standard extensions (e.g. gnu11)")
    timemory_add_feature(CMAKE_CXX_EXTENSIONS
                         "C++ language standard extensions (e.g. gnu++17)")
    timemory_add_feature(CMAKE_CUDA_EXTENSIONS "CUDA language standard (e.g. gnu++14)")

    if(NOT DEFINED CMAKE_CXX_STANDARD)
        timemory_message(
            AUTHOR_WARNING
            "timemory requires settings CMAKE_CXX_STANDARD. Defaulting CMAKE_CXX_STANDARD to 17..."
            )
        set(CMAKE_CXX_STANDARD 17)
    endif()
endif()

timemory_add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Embed RPATH using link path" ON)

set(_INSTALL_PYTHON auto)
if(SKBUILD)
    set(_INSTALL_PYTHON prefix)
elseif(SPACK_BUILD)
    set(_INSTALL_PYTHON lib)
endif()

# Install settings
timemory_add_option(TIMEMORY_INSTALL_HEADERS "Install the header files" ON)
timemory_add_option(
    TIMEMORY_INSTALL_CONFIG
    "Install the cmake package config files, i.e. timemory-config.cmake, etc." ON)
timemory_add_option(TIMEMORY_INSTALL_LIBRARIES "Install the libraries" ON)
timemory_add_option(
    TIMEMORY_INSTALL_ALL
    "'install' target depends on 'all' target. Set to OFF to only install artifacts which were explicitly built"
    ON)
set(TIMEMORY_INSTALL_PYTHON
    "${_INSTALL_PYTHON}"
    CACHE STRING "Installation mode for Python")
set(TIMEMORY_INSTALL_PYTHON_OPTIONS auto global lib prefix)
set_property(CACHE TIMEMORY_INSTALL_PYTHON PROPERTY STRINGS
                                                    "${TIMEMORY_INSTALL_PYTHON_OPTIONS}")
if(NOT "${TIMEMORY_INSTALL_PYTHON}" IN_LIST TIMEMORY_INSTALL_PYTHON_OPTIONS)
    message("")
    message(STATUS "TIMEMORY_INSTALL_PYTHON options:")
    message("    global = Python3_SITEARCH")
    message("    lib    = ${CMAKE_INSTALL_PREFIX}/lib/python<VERSION>/site-packages")
    message("    prefix = ${CMAKE_INSTALL_PREFIX} (generally only used by scikit-build)")
    message("    auto   = global (if writable) otherwise lib")
    message("")
    message(FATAL_ERROR "TIMEMORY_INSTALL_PYTHON set to invalid option. See guide above")
endif()
timemory_add_feature(TIMEMORY_INSTALL_PYTHON
                     "Installation mode for python (${TIMEMORY_INSTALL_PYTHON_OPTIONS})")

if(NOT TIMEMORY_INSTALL_ALL)
    set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)
    set(_BUILD_CALIPER OFF)
    set(_BUILD_GOTCHA OFF)
    set(_BUILD_LIBUNWIND OFF)
endif()

# Build settings
timemory_add_option(TIMEMORY_BUILD_DOCS "Make a `doc` make target" OFF)
timemory_add_option(TIMEMORY_BUILD_TESTING "Enable testing" OFF)
timemory_add_option(TIMEMORY_BUILD_GOOGLE_TEST "Enable GoogleTest"
                    ${TIMEMORY_BUILD_TESTING})
timemory_add_option(TIMEMORY_BUILD_EXAMPLES "Build the examples"
                    ${TIMEMORY_BUILD_TESTING})
timemory_add_option(TIMEMORY_BUILD_C "Build the C compatible library" ON)
timemory_add_option(TIMEMORY_BUILD_FORTRAN "Build the Fortran compatible library"
                    ${_BUILD_FORTRAN})
timemory_add_option(
    TIMEMORY_BUILD_PORTABLE
    "Disable arch flags which may cause portability issues (e.g. AVX-512)" OFF)
timemory_add_option(TIMEMORY_BUILD_PYTHON "Build Python bindings with internal pybind11"
                    ON)
timemory_add_option(TIMEMORY_BUILD_PYTHON_HATCHET "Build internal Hatchet distribution"
                    ON)
timemory_add_option(TIMEMORY_BUILD_LTO "Enable link-time optimizations in build" OFF)
timemory_add_option(TIMEMORY_BUILD_TOOLS "Enable building tools" ON)
timemory_add_option(
    TIMEMORY_BUILD_COMPILER_INSTRUMENTATION
    "Enable building compiler instrumentation libraries" ${TIMEMORY_BUILD_TOOLS})
timemory_add_option(
    TIMEMORY_INLINE_COMPILER_INSTRUMENTATION
    "Insert compiler instrumentation around inlined function calls" OFF NO_FEATURE)
timemory_add_option(TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS "Add extra optimization flags"
                    ${_BUILD_OPT})
timemory_add_option(
    TIMEMORY_BUILD_CALIPER "Enable building Caliper submodule (set to OFF for external)"
    ${_BUILD_CALIPER})
timemory_add_option(TIMEMORY_BUILD_OMPT "Enable building OpenMP-Tools from submodule" OFF)
timemory_add_option(TIMEMORY_BUILD_YAML "Enable building yaml-cpp from submodule" ON)
timemory_add_option(TIMEMORY_BUILD_DYNINST
                    "Enable building Dyninst submodule (set to OFF for external)" OFF)
timemory_add_option(
    TIMEMORY_BUILD_DYNINST_TPLS
    "Enable building Dyninst third-party library dependencies (TBB, Boost, elfutils, libiberty). See also: DYNINST_BUILD_<TPL> options"
    OFF)
timemory_add_option(
    TIMEMORY_BUILD_LIBUNWIND
    "Enable building libunwind submodule (set to OFF for external)" ${_BUILD_LIBUNWIND})
timemory_add_option(TIMEMORY_BUILD_DEVELOPER "Enable building with developer flags" OFF)
timemory_add_option(TIMEMORY_FORCE_GPERFTOOLS_PYTHON
                    "Enable gperftools + Python (may cause termination errors)" ON)
timemory_add_option(TIMEMORY_BUILD_QUIET "Disable verbose messages" OFF NO_FEATURE)
timemory_add_option(TIMEMORY_REQUIRE_PACKAGES "All find_package(...) use REQUIRED" ON)
timemory_add_option(TIMEMORY_BUILD_GOTCHA
                    "Enable building GOTCHA (set to OFF for external)" ${_BUILD_GOTCHA})
timemory_add_option(TIMEMORY_UNITY_BUILD
                    "Same as CMAKE_UNITY_BUILD but is not propagated to submodules" ON)
timemory_add_option(
    TIMEMORY_BUILD_EXCLUDE_FROM_ALL
    "When timemory is a subproject, ensure only your timemory target dependencies are built"
    OFF)
if(NOT CMAKE_VERSION VERSION_LESS 3.16)
    timemory_add_option(TIMEMORY_PRECOMPILE_HEADERS "Pre-compile headers where possible"
                        OFF)
else()
    set(TIMEMORY_PRECOMPILE_HEADERS OFF)
endif()

if(TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
    set(TIMEMORY_EXCLUDE_FROM_ALL EXCLUDE_FROM_ALL)
else()
    set(TIMEMORY_EXCLUDE_FROM_ALL)
endif()

if(NOT _NON_APPLE_UNIX)
    set(TIMEMORY_BUILD_GOTCHA OFF)
endif()

if(TIMEMORY_BUILD_QUIET)
    set(TIMEMORY_FIND_QUIETLY QUIET)
endif()

if(TIMEMORY_REQUIRE_PACKAGES)
    set(TIMEMORY_FIND_REQUIREMENT REQUIRED)
endif()

if(NOT CMAKE_CXX_COMPILER_IS_CLANG OR CMAKE_CXX_COMPILER_IS_APPLE_CLANG)
    set(TIMEMORY_INLINE_COMPILER_INSTRUMENTATION
        ON
        CACHE BOOL "Only the Clang compiler supports instrumentation after inlining"
              FORCE)
endif()

if(TIMEMORY_BUILD_FORTRAN)
    enable_language(Fortran)
endif()

# Features
if(${PROJECT_NAME}_MAIN_PROJECT)
    timemory_add_feature(${PROJECT_NAME}_C_FLAGS "C compiler flags")
    timemory_add_feature(${PROJECT_NAME}_CXX_FLAGS "C++ compiler flags")
    timemory_add_feature(${PROJECT_NAME}_CUDA_FLAGS "CUDA compiler flags")
endif()

define_default_option(_MPI ON)
define_default_option(_UPCXX ON)
define_default_option(_TAU ON)
define_default_option(_PAPI ${_USE_PAPI})
define_default_option(_GPERFTOOLS ON)
define_default_option(_VTUNE ON)
define_default_option(_CUDA ${_USE_CUDA})
define_default_option(_HIP ${_USE_HIP})
define_default_option(_CALIPER ${_BUILD_CALIPER})
define_default_option(_PYTHON OFF)
define_default_option(_DYNINST ON)
define_default_option(_ALLINEA_MAP ON)
define_default_option(_CRAYPAT ON)
define_default_option(_OMPT OFF)
define_default_option(_LIKWID ${_NON_APPLE_UNIX})
define_default_option(_GOTCHA ${_NON_APPLE_UNIX})
define_default_option(_PERFETTO ${_NON_APPLE_UNIX})
define_default_option(_NCCL ${_USE_CUDA})
define_default_option(_LIKWID_NVMON ${_LIKWID} ${_NON_APPLE_UNIX} ${_CUDA})
define_default_option(_BFD ${_NON_APPLE_UNIX})

# timemory options
timemory_add_option(TIMEMORY_USE_VISIBILITY "Enable visibility attributes" ON
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_DEPRECATED "Enable deprecated code" OFF CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_STATISTICS "Enable statistics by default" ON
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_MPI "Enable MPI usage" ${_MPI} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_MPI_INIT "Enable MPI_Init and MPI_Init_thread wrappers"
                    OFF CMAKE_DEFINE)
timemory_add_option(
    TIMEMORY_USE_UPCXX "Enable UPCXX usage (MPI support takes precedence)" ${_UPCXX}
    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_SANITIZER
                    "Enable -fsanitize flag (=${TIMEMORY_SANITIZER_TYPE})" OFF)
timemory_add_option(TIMEMORY_USE_TAU "Enable TAU marking API" ${_TAU} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_PAPI "Enable PAPI" ${_PAPI} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_CLANG_TIDY "Enable running clang-tidy" OFF)
timemory_add_option(TIMEMORY_USE_COVERAGE "Enable code-coverage" ${_USE_COVERAGE})
timemory_add_option(TIMEMORY_USE_GPERFTOOLS "Enable gperftools" ${_GPERFTOOLS}
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_ARCH "Enable architecture flags" OFF CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_VTUNE "Enable VTune marking API" ${_VTUNE} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_CUDA "Enable CUDA option for GPU measurements" ${_CUDA}
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_NVTX "Enable NVTX marking API" ${TIMEMORY_USE_CUDA}
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_CUPTI "Enable CUPTI profiling for NVIDIA GPUs"
                    ${TIMEMORY_USE_CUDA} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_NVML "Enable support for NVIDIA Management Library"
                    ${TIMEMORY_USE_CUDA} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_NCCL "Enable NCCL support for NVIDIA GPUs" ${_NCCL}
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_HIP "Enable HIP option for GPU roofline" ${_HIP}
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_CALIPER "Enable Caliper" ${_CALIPER} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_PERFETTO "Enable Perfetto" ${_PERFETTO} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_PYTHON "Enable Python" ${_PYTHON} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_COMPILE_TIMING
                    "Enable -ftime-report for compilation times" OFF)
timemory_add_option(TIMEMORY_USE_DYNINST "Enable dynamic instrumentation" ${_DYNINST})
timemory_add_option(TIMEMORY_USE_ALLINEA_MAP "Enable control for AllineaMAP sampler"
                    ${_ALLINEA_MAP} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_CRAYPAT "Enable CrayPAT support" ${_CRAYPAT}
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_OMPT "Enable OpenMP tooling" ${_OMPT} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_LIKWID "Enable LIKWID marker forwarding" ${_LIKWID}
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_LIKWID_PERFMON "Enable LIKWID support for perf (CPU)"
                    ${TIMEMORY_USE_LIKWID} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_LIKWID_NVMON "Enable LIKWID support for nvidia (GPU)"
                    ${_LIKWID_NVMON} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_GOTCHA "Enable GOTCHA" ${_GOTCHA} CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_XML "Enable XML serialization support" ${_USE_XML}
                    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_YAML "Enable YAML serialization support" ON CMAKE_DEFINE)
timemory_add_option(TIMEMORY_USE_LIBUNWIND "Enable libunwind" ${_USE_LIBUNWIND}
                    CMAKE_DEFINE)
timemory_add_option(
    TIMEMORY_USE_BFD "Enable BFD support for mapping instruction pointers to LOC" ${_BFD}
    CMAKE_DEFINE)
timemory_add_option(TIMEMORY_BUILD_ERT "Build ERT library" ON)
if(CMAKE_CXX_COMPILER_IS_CLANG OR TIMEMORY_BUILD_DOCS)
    timemory_add_option(TIMEMORY_USE_XRAY "Enable XRay instrumentation" OFF CMAKE_DEFINE)
endif()

set(TIMEMORY_MAX_THREADS
    2048
    CACHE STRING "Maximum number of statically allocated thread-local statics")
timemory_add_feature(TIMEMORY_MAX_THREADS
                     "Maximum number of statically allocated thread-local statics")
timemory_add_cmake_defines(TIMEMORY_MAX_THREADS VALUE DEFAULT)

if(TIMEMORY_BUILD_EXAMPLES
   AND TIMEMORY_USE_COVERAGE
   AND "$ENV{CONTINUOUS_INTEGRATION}" STREQUAL "true")
    set(BUILD_ERT
        OFF
        CACHE BOOL "Disable ERT example")
endif()

if(NOT TIMEMORY_USE_GPERFTOOLS)
    set(TIMEMORY_FORCE_GPERFTOOLS_PYTHON OFF)
endif()

# disable these for Debug builds
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(TIMEMORY_BUILD_LTO OFF)
    set(TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS OFF)
endif()

if(${PROJECT_NAME}_MAIN_PROJECT)
    timemory_add_feature(TIMEMORY_gperftools_COMPONENTS "gperftool components" DOC)
endif()

if(TIMEMORY_USE_CUDA OR TIMEMORY_BUILD_DOCS)
    timemory_add_option(TIMEMORY_USE_CUDA_HALF "Enable half/half2 if CUDA_ARCH >= 60" OFF)
endif()

set(_DYNINST OFF)
if(TIMEMORY_BUILD_TOOLS AND TIMEMORY_USE_DYNINST)
    set(_DYNINST ON)
endif()

set(_MPIP ${TIMEMORY_USE_MPI})
if(_MPIP AND NOT TIMEMORY_USE_GOTCHA)
    set(_MPIP OFF)
endif()

set(_OMPT ${TIMEMORY_USE_OMPT})

set(_NCCLP ${TIMEMORY_USE_NCCL})
if(_NCCLP AND NOT TIMEMORY_USE_GOTCHA)
    set(_NCCLP OFF)
endif()

set(_MALLOCP ${TIMEMORY_USE_GOTCHA})
if(_MALLOCP AND NOT TIMEMORY_USE_GOTCHA)
    set(_MALLOCP OFF)
endif()

set(_TIMEM ${TIMEMORY_BUILD_TOOLS})
if(_TIMEM AND (WIN32 OR APPLE))
    set(_TIMEM OFF)
endif()

timemory_add_option(TIMEMORY_BUILD_AVAIL "Build the timemory-avail tool"
                    ${TIMEMORY_BUILD_TOOLS})
timemory_add_option(TIMEMORY_BUILD_TIMEM "Build the timem tool" ${_TIMEM})
timemory_add_option(TIMEMORY_BUILD_KOKKOS_TOOLS "Build the kokkos-tools libraries" OFF)
timemory_add_option(TIMEMORY_BUILD_KOKKOS_CONFIG "Build various connector configurations"
                    OFF)
timemory_add_option(TIMEMORY_BUILD_DYNINST_TOOLS
                    "Build the timemory-run dynamic instrumentation tool" ${_DYNINST})
timemory_add_option(TIMEMORY_BUILD_MPIP_LIBRARY "Build the mpiP library" ${_MPIP})
timemory_add_option(TIMEMORY_BUILD_OMPT_LIBRARY "Build the OMPT library" ${_OMPT})
timemory_add_option(TIMEMORY_BUILD_NCCLP_LIBRARY "Build the ncclP library" ${_NCCLP})
timemory_add_option(TIMEMORY_BUILD_MALLOCP_LIBRARY "Build the mallocP library"
                    ${_MALLOCP})

unset(_MPIP)
unset(_OMPT)
unset(_NCCLP)
unset(_MALLOCP)
unset(_DYNINST)

if(TIMEMORY_BUILD_MPIP_LIBRARY AND (NOT TIMEMORY_USE_MPI OR NOT TIMEMORY_USE_GOTCHA))
    timemory_message(
        AUTHOR_WARNING
        "TIMEMORY_BUILD_MPIP_LIBRARY requires TIMEMORY_USE_MPI=ON and TIMEMORY_USE_GOTCHA=ON..."
        )
    set(TIMEMORY_BUILD_MPIP_LIBRARY
        OFF
        CACHE BOOL "Build the mpiP library" FORCE)
endif()

if(TIMEMORY_BUILD_OMPT_LIBRARY AND NOT TIMEMORY_USE_OMPT)
    timemory_message(AUTHOR_WARNING
                     "TIMEMORY_BUILD_OMPT_LIBRARY requires TIMEMORY_USE_OMPT=ON...")
    set(TIMEMORY_BUILD_OMPT_LIBRARY
        OFF
        CACHE BOOL "Build the OMPT library" FORCE)
endif()

if(TIMEMORY_BUILD_NCCLP_LIBRARY AND (NOT TIMEMORY_USE_NCCL OR NOT TIMEMORY_USE_GOTCHA))
    timemory_message(
        AUTHOR_WARNING
        "TIMEMORY_BUILD_NCCLP_LIBRARY requires TIMEMORY_USE_NCCL=ON, and TIMEMORY_USE_GOTCHA=ON..."
        )
    set(TIMEMORY_BUILD_NCCLP_LIBRARY
        OFF
        CACHE BOOL "Build the ncclP library" FORCE)
endif()

if(TIMEMORY_BUILD_MALLOCP_LIBRARY AND NOT TIMEMORY_USE_GOTCHA)
    timemory_message(AUTHOR_WARNING
                     "TIMEMORY_BUILD_MALLOCP_LIBRARY requires TIMEMORY_USE_GOTCHA=ON...")
    set(TIMEMORY_BUILD_MALLOCP_LIBRARY
        OFF
        CACHE BOOL "Build the ncclP library" FORCE)
endif()

if(NOT BUILD_SHARED_LIBS
   AND TIMEMORY_BUILD_KOKKOS_TOOLS
   AND NOT CMAKE_POSITION_INDEPENDENT_CODE)
    timemory_message(
        AUTHOR_WARNING
        "TIMEMORY_BUILD_KOKKOS_TOOLS requires CMAKE_POSITION_INDEPENDENT_CODE=ON if shared libraries are not enabled..."
        )
    set(TIMEMORY_BUILD_KOKKOS_TOOLS
        OFF
        CACHE BOOL "Build the kokkos-tools libraries" FORCE)
endif()

timemory_set_if_empty(TIMEMORY_SETTINGS_PREFIX "TIMEMORY_")
timemory_set_if_empty(TIMEMORY_PROJECT_NAME "timemory")

timemory_add_cmake_defines(TIMEMORY_SETTINGS_PREFIX VALUE QUOTE DEFAULT)
timemory_add_cmake_defines(TIMEMORY_PROJECT_NAME VALUE QUOTE DEFAULT)

if(NOT timemory_MAIN_PROJECT)
    timemory_add_feature(
        TIMEMORY_SETTINGS_PREFIX
        "timemory settings will be prefixed with \"${TIMEMORY_SETTINGS_PREFIX}\"")
    timemory_add_feature(TIMEMORY_PROJECT_NAME
                         "timemory settings config will use \"${TIMEMORY_PROJECT_NAME}\"")
endif()

# cereal options
timemory_add_option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" OFF NO_FEATURE)
timemory_add_option(THREAD_SAFE "Compile Cereal with THREAD_SAFE option" ON NO_FEATURE)
timemory_add_option(JUST_INSTALL_CEREAL "Skip testing of Cereal" ON NO_FEATURE)
timemory_add_option(SKIP_PORTABILITY_TEST "Skip Cereal portability test" ON NO_FEATURE)

if(TIMEMORY_BUILD_DOCS)
    timemory_add_option(TIMEMORY_BUILD_DOXYGEN "Include `doc` make target in all" OFF
                        NO_FEATURE)
    mark_as_advanced(TIMEMORY_BUILD_DOXYGEN)
endif()

set(PYBIND11_INSTALL
    OFF
    CACHE BOOL "Install Pybind11")
if(PYBIND11_INSTALL AND (SKBUILD OR SPACK_BUILD))
    timemory_message(
        WARNING
        "Pybind11 will be installed. This may overwrite an existing pip/conda/spack PyBind11 installation..."
        )
endif()

# clang-tidy
macro(_TIMEMORY_ACTIVATE_CLANG_TIDY)
    if(TIMEMORY_USE_CLANG_TIDY)
        find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
        timemory_add_feature(CLANG_TIDY_COMMAND "Path to clang-tidy command")
        if(NOT CLANG_TIDY_COMMAND)
            timemory_message(WARNING
                             "TIMEMORY_USE_CLANG_TIDY is ON but clang-tidy is not found!")
            set(TIMEMORY_USE_CLANG_TIDY OFF)
        else()
            set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_COMMAND})

            # Create a preprocessor definition that depends on .clang-tidy content so the
            # compile command will change when .clang-tidy changes.  This ensures that a
            # subsequent build re-runs clang-tidy on all sources even if they do not
            # otherwise need to be recompiled.  Nothing actually uses this definition.  We
            # add it to targets on which we run clang-tidy just to get the build
            # dependency on the .clang-tidy file.
            file(SHA1 ${PROJECT_SOURCE_DIR}/.clang-tidy clang_tidy_sha1)
            set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
            unset(clang_tidy_sha1)
        endif()
    endif()
endmacro()

# these variables conflict with variables in examples, leading to things like: -lON flags
get_property(DEFAULT_OPTION_VARIABLES GLOBAL PROPERTY DEFAULT_OPTION_VARIABLES)
foreach(_VAR ${DEFAULT_OPTION_VARIABLES})
    unset(${_VAR})
endforeach()

# some logic depends on this not being set
if(NOT TIMEMORY_USE_CUDA)
    unset(CMAKE_CUDA_COMPILER CACHE)
endif()

if(WIN32)
    option(TIMEMORY_USE_WINSOCK "Include winsock.h with the windows build" OFF)
endif()
