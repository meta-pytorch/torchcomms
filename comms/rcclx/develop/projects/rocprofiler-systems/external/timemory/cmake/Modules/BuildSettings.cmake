# include guard
include_guard(DIRECTORY)

# ########################################################################################
#
# Handles the build settings
#
# ########################################################################################

include(GNUInstallDirs)
include(Compilers)
include(FindPackageHandleStandardArgs)

target_compile_definitions(timemory-compile-options INTERFACE $<$<CONFIG:DEBUG>:DEBUG>)

# ----------------------------------------------------------------------------------------#
# dynamic linking and runtime libraries
#
if(CMAKE_DL_LIBS AND NOT "${CMAKE_DL_LIBS}" STREQUAL "dl")
    # if cmake provides dl library, use that
    set(dl_LIBRARY
        ${CMAKE_DL_LIBS}
        CACHE FILEPATH "dynamic linking system library")
    mark_as_advanced(dl_LIBRARY)
endif()

foreach(_TYPE dl rt dw)
    if(NOT ${_TYPE}_LIBRARY)
        find_library(${_TYPE}_LIBRARY NAMES ${_TYPE})
        mark_as_advanced(${_TYPE}_LIBRARY)
    endif()
endforeach()

find_package_handle_standard_args(dl-library REQUIRED_VARS dl_LIBRARY)
find_package_handle_standard_args(rt-library REQUIRED_VARS rt_LIBRARY)
# find_package_handle_standard_args(dw-library REQUIRED_VARS dw_LIBRARY)

if(TIMEMORY_BUILD_PORTABLE)
    if(dl_LIBRARY)
        set(dl_LIBRARY dl)
    else()
        set(dl_LIBRARY)
    endif()
elseif(dl_LIBRARY)
    timemory_add_rpath(${dl_LIBRARY})
endif()

if(dl_LIBRARY)
    target_link_libraries(timemory-compile-options INTERFACE ${dl_LIBRARY})
endif()

if(TIMEMORY_BUILD_PORTABLE)
    if(rt_LIBRARY)
        set(rt_LIBRARY rt)
    else()
        set(rt_LIBRARY)
    endif()
elseif(rt_LIBRARY)
    timemory_add_rpath(${rt_LIBRARY})
endif()

# ----------------------------------------------------------------------------------------#

if(WIN32)
    set(OS_FLAG "/bigobj")
else()
    set(OS_FLAG "-Wall")
endif()

# ----------------------------------------------------------------------------------------#
# set the compiler flags
#
add_flag_if_avail("-W" "${OS_FLAG}" "-Wno-unknown-pragmas" "-Wno-ignored-attributes"
                  "-Wno-attributes" "-Wno-missing-field-initializers")

if(WIN32)
    # suggested by MSVC for spectre mitigation in rapidjson implementation
    add_cxx_flag_if_avail("/Qspectre")
endif()

if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_cxx_flag_if_avail("-Wno-mismatched-tags")
endif()

if(CMAKE_CXX_COMPILER_IS_GNU)
    add_target_cxx_flag_if_avail(timemory-compile-options "-Wno-class-memaccess")
endif()

if(TIMEMORY_BUILD_QUIET)
    add_flag_if_avail(
        "-Wno-unused-value" "-Wno-unused-function" "-Wno-unknown-pragmas"
        "-Wno-deprecated-declarations" "-Wno-implicit-fallthrough"
        "-Wno-unused-command-line-argument")
endif()

# ----------------------------------------------------------------------------------------#
# extra flags for debug information in debug or optimized binaries
#
timemory_add_interface_library(
    timemory-compile-debuginfo
    "Attempts to set best flags for more expressive profiling information in debug or optimized binaries"
    )

add_target_flag_if_avail(timemory-compile-debuginfo "-g" "-fno-omit-frame-pointer"
                         "-fno-optimize-sibling-calls")

if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
    add_target_cuda_flag(timemory-compile-debuginfo "-lineinfo")
endif()

target_compile_options(
    timemory-compile-debuginfo
    INTERFACE $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-rdynamic>>
              $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-rdynamic>>)

if(NOT APPLE)
    target_link_options(timemory-compile-debuginfo INTERFACE
                        $<$<CXX_COMPILER_ID:GNU>:-rdynamic>)
endif()

if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
    target_compile_options(
        timemory-compile-debuginfo
        INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CXX_COMPILER_ID:GNU>:-Xcompiler=-rdynamic>>)
endif()

if(dl_LIBRARY)
    target_link_libraries(timemory-compile-debuginfo INTERFACE ${dl_LIBRARY})
endif()

if(rt_LIBRARY)
    target_link_libraries(timemory-compile-debuginfo INTERFACE ${rt_LIBRARY})
endif()

# ----------------------------------------------------------------------------------------#
# non-debug optimizations
#
timemory_add_interface_library(timemory-compile-extra "Extra optimization flags")
if(NOT TIMEMORY_USE_COVERAGE AND TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS)
    add_target_flag_if_avail(
        timemory-compile-extra "-finline-functions" "-funroll-loops" "-ftree-vectorize"
        "-ftree-loop-optimize" "-ftree-loop-vectorize")
endif()

if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS)
    target_link_libraries(timemory-compile-options
                          INTERFACE $<BUILD_INTERFACE:timemory-compile-extra>)
    add_flag_if_avail(
        "-fno-signaling-nans" "-fno-trapping-math" "-fno-signed-zeros"
        "-ffinite-math-only" "-fno-math-errno" "-fpredictive-commoning"
        "-fvariable-expansion-in-unroller")
    # add_flag_if_avail("-freciprocal-math" "-fno-signed-zeros" "-mfast-fp")
endif()

# ----------------------------------------------------------------------------------------#
# debug-safe optimizations
#
add_cxx_flag_if_avail("-faligned-new")
if(NOT TIMEMORY_USE_SANITIZER)
    add_cxx_flag_if_avail("-ftls-model=${TIMEMORY_TLS_MODEL}")
endif()

if(TIMEMORY_BUILD_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_CHECK_OUTPUT)
    if(NOT IPO_SUPPORTED)
        timemory_message(
            STATUS
            "TIMEMORY_BUILD_LTO must be disabled due to check_ipo_supported() failure")
        timemory_message(AUTHOR_WARNING "${IPO_CHECK_OUTPUT}")
        message(FATAL_ERROR "TIMEMORY_BUILD_LTO not supported")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ${IPO_SUPPORTED})
    set(TIMEMORY_BUILD_LTO ${IPO_SUPPORTED})
endif()

timemory_save_variables(FLTO VARIABLES CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
set(CMAKE_C_FLAGS "-flto=thin ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-flto=thin ${CMAKE_CXX_FLAGS}")

timemory_add_interface_library(timemory-lto "Adds link-time-optimization flags")
add_target_flag_if_avail(timemory-lto "-flto=thin")
if(NOT cxx_timemory_lto_flto_thin)
    set(CMAKE_C_FLAGS "-flto ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-flto ${CMAKE_CXX_FLAGS}")
    add_target_flag_if_avail(timemory-lto "-flto")
    if(NOT cxx_timemory_lto_flto)
        timemory_add_disabled_interface(timemory-lto)
        set(TIMEMORY_BUILD_LTO OFF)
    else()
        target_link_options(timemory-lto INTERFACE -flto)
    endif()
else()
    target_link_options(timemory-lto INTERFACE -flto=thin)
endif()

if(TIMEMORY_BUILD_LTO)
    target_link_libraries(timemory-compile-options INTERFACE timemory::timemory-lto)
endif()

timemory_restore_variables(FLTO VARIABLES CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

# ----------------------------------------------------------------------------------------#
# print compilation timing reports (Clang compiler)
#
timemory_add_interface_library(
    timemory-compile-timing "Adds compiler flags which report compilation timing metrics")
if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_target_flag_if_avail(timemory-compile-timing "-ftime-trace")
    if(NOT cxx_timemory_compile_timing_ftime_trace)
        add_target_flag_if_avail(timemory-compile-timing "-ftime-report")
    endif()
else()
    add_target_flag_if_avail(timemory-compile-timing "-ftime-report")
endif()

if(TIMEMORY_USE_COMPILE_TIMING)
    target_link_libraries(timemory-compile-options INTERFACE timemory-compile-timing)
endif()

if(NOT cxx_timemory_compile_timing_ftime_report
   AND NOT cxx_timemory_compile_timing_ftime_trace)
    timemory_add_disabled_interface(timemory-compile-timing)
endif()

# ----------------------------------------------------------------------------------------#
# use xray instrumentation
#
timemory_add_interface_library(
    timemory-xray "Adds compiler flags to enable xray-instrumentation (Clang only)")
if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_target_flag_if_avail(timemory-xray "-fxray-instrument"
                             "-fxray-instruction-threshold=1")
    if(NOT cxx_timemory_xray_fxray_instrument)
        timemory_add_disabled_interface(timemory-xray)
    endif()
else()
    timemory_add_disabled_interface(timemory-xray)
endif()

# ----------------------------------------------------------------------------------------#
# use compiler instrumentation
#
timemory_add_interface_library(
    timemory-instrument-functions
    "Adds compiler flags to enable compile-time instrumentation")

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/compiler-instr.cpp.in
               ${PROJECT_BINARY_DIR}/compile-tests/compiler-instr.c COPYONLY)

if(NOT TIMEMORY_INLINE_COMPILER_INSTRUMENTATION)
    try_compile(
        c_timemory_instrument_finstrument_functions_after_inlining
        ${PROJECT_BINARY_DIR}/compile-tests
        SOURCES ${PROJECT_BINARY_DIR}/compile-tests/compiler-instr.c
        CMAKE_FLAGS -finstrument-functions-after-inlining)
else()
    set(c_timemory_instrument_finstrument_functions_after_inlining FALSE)
endif()

try_compile(
    c_timemory_instrument_finstrument_functions ${PROJECT_BINARY_DIR}/compile-tests
    SOURCES ${PROJECT_BINARY_DIR}/compile-tests/compiler-instr.c
    CMAKE_FLAGS -finstrument-functions)

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/compiler-instr.cpp.in
               ${PROJECT_BINARY_DIR}/compile-tests/compiler-instr.cpp COPYONLY)

if(NOT TIMEMORY_INLINE_COMPILER_INSTRUMENTATION)
    try_compile(
        cxx_timemory_instrument_finstrument_functions_after_inlining
        ${PROJECT_BINARY_DIR}/compile-tests
        SOURCES ${PROJECT_BINARY_DIR}/compile-tests/compiler-instr.cpp
        CMAKE_FLAGS -finstrument-functions-after-inlining)
else()
    set(cxx_timemory_instrument_finstrument_functions_after_inlining FALSE)
endif()

try_compile(
    cxx_timemory_instrument_finstrument_functions ${PROJECT_BINARY_DIR}/compile-tests
    SOURCES ${PROJECT_BINARY_DIR}/compile-tests/compiler-instr.cpp
    CMAKE_FLAGS -finstrument-functions)

if(c_timemory_instrument_finstrument_functions_after_inlining
   AND cxx_timemory_instrument_finstrument_functions_after_inlining)
    set(_OPT "$<NOT:$<BOOL:TIMEMORY_INLINE_COMPILER_INSTRUMENTATION>>")
    set(_C_CHK "$<C_COMPILER_ID:Clang>")
    set(_C_AND "$<AND:${_OPT},${_C_CHK}>")
    set(_C_NOT "$<NOT:${_C_AND}>")
    set(_C_FLG "$<${_C_AND}:-finstrument-functions-after-inlining>"
               "$<${_C_NOT}:-finstrument-functions>")
    set(_CXX_CHK "$<CXX_COMPILER_ID:Clang>")
    set(_CXX_AND "$<AND:${_OPT},${_CXX_CHK}>")
    set(_CXX_NOT "$<NOT:${_CXX_AND}>")
    set(_CXX_FLG "$<${_CXX_AND}:-finstrument-functions-after-inlining>"
                 "$<${_CXX_NOT}:-finstrument-functions>")
    target_compile_options(
        timemory-instrument-functions INTERFACE $<$<COMPILE_LANGUAGE:C>:${_C_FLG}>
                                                $<$<COMPILE_LANGUAGE:CXX>:${_CXX_FLG}>)
elseif(c_timemory_instrument_finstrument_functions
       AND cxx_timemory_instrument_finstrument_functions)
    set(TIMEMORY_INLINE_COMPILER_INSTRUMENTATION
        ON
        CACHE BOOL "Enable compiler instrumentation for inlined function calls" FORCE)
    set(TIMEMORY_INLINE_COMPILER_INSTRUMENTATION ON)
    target_compile_options(
        timemory-instrument-functions
        INTERFACE $<$<COMPILE_LANGUAGE:C>:-finstrument-functions>
                  $<$<COMPILE_LANGUAGE:CXX>:-finstrument-functions>)
else()
    set(TIMEMORY_BUILD_COMPILER_INSTRUMENTATION OFF)
    timemory_add_disabled_interface(timemory-instrument-functions)
endif()

if(TIMEMORY_BUILD_COMPILER_INSTRUMENTATION)
    target_link_libraries(timemory-instrument-functions
                          INTERFACE timemory-compile-debuginfo)
endif()

# ----------------------------------------------------------------------------------------#
# developer build flags
#
timemory_add_interface_library(timemory-develop-options "Adds developer compiler flags")
if(TIMEMORY_BUILD_DEVELOPER)
    add_target_flag_if_avail(
        timemory-develop-options
        # "-Wabi"
        "-Wdouble-promotion" "-Wshadow" "-Wextra" "-Wpedantic" "-Werror" "/showIncludes")
endif()

# ----------------------------------------------------------------------------------------#
# visibility build flags
#
timemory_add_interface_library(timemory-use-visibility
                               "Defines whether to use visibility attributes")
timemory_add_interface_library(timemory-default-visibility
                               "Adds -fvisibility=default compiler flag")
timemory_add_interface_library(timemory-hidden-visibility
                               "Adds -fvisibility=hidden compiler flag")

timemory_target_compile_definitions(
    timemory-use-visibility INTERFACE
    TIMEMORY_USE_VISIBILITY=$<BOOL:${TIMEMORY_USE_VISIBILITY}>)
add_target_flag_if_avail(timemory-default-visibility "-fvisibility=default")
add_target_flag_if_avail(timemory-hidden-visibility "-fvisibility=hidden"
                         "-fvisibility-inlines-hidden")

foreach(_TYPE default hidden)
    if(NOT cxx_timemory_${_TYPE}_visibility_fvisibility_${_TYPE})
        timemory_add_disabled_interface(timemory-${_TYPE}-visibility)
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#
# developer build flags
#
if(dl_LIBRARY)
    # This instructs the linker to add all symbols, not only used ones, to the dynamic
    # symbol table. This option is needed for some uses of dlopen or to allow obtaining
    # backtraces from within a program.
    add_flag_if_avail("-rdynamic")
endif()

# ----------------------------------------------------------------------------------------#
# architecture optimizations
#
timemory_add_interface_library(
    timemory-vector
    "Adds pre-processor definition of the max vectorization width in bytes")
timemory_add_interface_library(timemory-arch "Adds architecture-specific compiler flags")
target_link_libraries(timemory-compile-options INTERFACE timemory-vector)

set(VECTOR_DEFINITION TIMEMORY_VEC)
set(VECTOR_INTERFACE_TARGET timemory-vector)
set(ARCH_INTERFACE_TARGET timemory-arch)

include(ConfigCpuArch)

timemory_add_cmake_defines(TIMEMORY_VEC VALUE)

# ----------------------------------------------------------------------------------------#
# sanitizer
#
set(TIMEMORY_SANITIZER_TYPES
    address
    memory
    thread
    leak
    undefined
    unreachable
    null
    bounds
    alignment)
set_property(CACHE TIMEMORY_SANITIZER_TYPE PROPERTY STRINGS "${TIMEMORY_SANITIZER_TYPES}")
timemory_add_interface_library(timemory-sanitizer-compile-options
                               "Adds compiler flags for sanitizers")
timemory_add_interface_library(
    timemory-sanitizer
    "Adds compiler flags to enable ${TIMEMORY_SANITIZER_TYPE} sanitizer (-fsanitizer=${TIMEMORY_SANITIZER_TYPE})"
    )

set(COMMON_SANITIZER_FLAGS "-fno-optimize-sibling-calls" "-fno-omit-frame-pointer"
                           "-fno-inline-functions")
add_target_flag(timemory-sanitizer-compile-options ${COMMON_SANITIZER_FLAGS})

foreach(_TYPE ${TIMEMORY_SANITIZER_TYPES})
    set(_FLAG "-fsanitize=${_TYPE}")
    timemory_add_interface_library(
        timemory-${_TYPE}-sanitizer
        "Adds compiler flags to enable ${_TYPE} sanitizer (${_FLAG})")
    add_target_flag(timemory-${_TYPE}-sanitizer ${_FLAG})
    target_link_libraries(timemory-${_TYPE}-sanitizer
                          INTERFACE timemory-sanitizer-compile-options)
    set_property(TARGET timemory-${_TYPE}-sanitizer
                 PROPERTY INTERFACE_LINK_OPTIONS ${_FLAG} ${COMMON_SANITIZER_FLAGS})
endforeach()

unset(_FLAG)
unset(COMMON_SANITIZER_FLAGS)

if(TIMEMORY_USE_SANITIZER)
    foreach(_TYPE ${TIMEMORY_SANITIZER_TYPE})
        if(TARGET timemory-${_TYPE}-sanitizer)
            target_link_libraries(timemory-sanitizer
                                  INTERFACE timemory-${_TYPE}-sanitizer)
        else()
            message(
                FATAL_ERROR "Error! Target 'timemory-${_TYPE}-sanitizer' does not exist!")
        endif()
    endforeach()
else()
    set(TIMEMORY_USE_SANITIZER OFF)
    timemory_inform_empty_interface(timemory-sanitizer
                                    "${TIMEMORY_SANITIZER_TYPE} sanitizer")
endif()

if(MSVC)
    # VTune is much more helpful when debug information is included in the generated
    # release code.
    add_flag_if_avail("/Zi")
    add_flag_if_avail("/DEBUG")
endif()

# ----------------------------------------------------------------------------------------#
# user customization
#
get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

if(NOT APPLE OR "$ENV{CONDA_PYTHON_EXE}" STREQUAL "")
    add_user_flags(timemory-compile-options "C")
    add_user_flags(timemory-compile-options "CXX")
    if(CMAKE_CUDA_COMPILER AND "CUDA" IN_LIST ENABLED_LANGUAGES)
        add_user_flags(timemory-compile-options "CUDA")
    endif()
endif()
