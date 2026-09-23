# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying file
# Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

# ----------------------------------------------------------------------------------------#

set(_CUDA_PATHS $ENV{CUDA_HOME} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR})
set(_CUDA_INC ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# ----------------------------------------------------------------------------------------#

find_path(
    CUPTI_ROOT_DIR
    NAMES include/cupti.h
    HINTS ${_CUDA_PATHS}
    PATHS ${_CUDA_PATHS}
    PATH_SUFFIXES extras/CUPTI)

find_file(
    CUPTI_nvperf_host_HEADER
    NAMES nvperf_host.h
    HINTS ${_CUDA_PATHS}
    PATHS ${_CUDA_PATHS}
    PATH_SUFFIXES include include/extras/CUPTI)

find_file(
    CUPTI_nvperf_target_HEADER
    NAMES nvperf_target.h
    HINTS ${_CUDA_PATHS}
    PATHS ${_CUDA_PATHS}
    PATH_SUFFIXES include include/extras/CUPTI)

find_file(
    CUPTI_pcsampling_HEADER
    NAMES cupti_pcsampling.h
    HINTS ${_CUDA_PATHS}
    PATHS ${_CUDA_PATHS}
    PATH_SUFFIXES include include/extras/CUPTI)

find_file(
    CUPTI_pcsampling_util_HEADER
    NAMES cupti_pcsampling_util.h
    HINTS ${_CUDA_PATHS}
    PATHS ${_CUDA_PATHS}
    PATH_SUFFIXES include include/extras/CUPTI)

mark_as_advanced(CUPTI_ROOT_DIR CUPTI_nvperf_host_HEADER CUPTI_nvperf_target_HEADER
                 CUPTI_pcsampling_HEADER)

# ----------------------------------------------------------------------------------------#

# try to find cupti header
find_path(
    CUPTI_INCLUDE_DIR
    NAMES cupti.h
    HINTS ${CUPTI_ROOT_DIR} ${_CUDA_INC} ${_CUDA_PATHS}
    PATHS ${CUPTI_ROOT_DIR} ${_CUDA_INC} ${_CUDA_PATHS}
    PATH_SUFFIXES extras/CUPTI/include extras/CUPTI extras/include CUTPI/include include)

mark_as_advanced(CUPTI_INCLUDE_DIR)

# ----------------------------------------------------------------------------------------#

# try to find cuda driver library
find_library(
    CUPTI_cupti_LIBRARY
    NAMES cupti
    HINTS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES lib lib64 lib/nvidia lib64/nvidia nvidia)

if(CUPTI_cupti_LIBRARY)
    get_filename_component(CUPTI_cupti_LIBRARY_DIR "${CUPTI_cupti_LIBRARY}" PATH CACHE)
endif()

mark_as_advanced(CUPTI_cupti_LIBRARY)

# ----------------------------------------------------------------------------------------#

# try to find nvperf_host library
find_library(
    CUPTI_nvperf_host_LIBRARY
    NAMES nvperf_host
    HINTS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES lib lib64 lib/nvidia lib64/nvidia nvidia)

mark_as_advanced(CUPTI_nvperf_host_LIBRARY)

# ----------------------------------------------------------------------------------------#

# try to find nvperf_host library
find_library(
    CUPTI_nvperf_host_STATIC_LIBRARY
    NAMES nvperf_host_static
    HINTS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES lib lib64 lib/nvidia lib64/nvidia nvidia)

mark_as_advanced(CUPTI_nvperf_host_STATIC_LIBRARY)

# ----------------------------------------------------------------------------------------#

# try to find nvperf_target library
find_library(
    CUPTI_nvperf_target_LIBRARY
    NAMES nvperf_target
    HINTS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES lib lib64 lib/nvidia lib64/nvidia nvidia)

mark_as_advanced(CUPTI_nvperf_target_LIBRARY)

# ----------------------------------------------------------------------------------------#

# try to find pc sampling utility library
find_library(
    CUPTI_pcsampling_util_LIBRARY
    NAMES pcsamplingutil
    HINTS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES lib lib64 lib/nvidia lib64/nvidia nvidia)

mark_as_advanced(CUPTI_pcsamplingutil_LIBRARY)

# ----------------------------------------------------------------------------------------#

# try to find cuda driver library
find_library(
    CUPTI_cuda_LIBRARY
    NAMES cuda
    HINTS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES lib lib64 lib/nvidia lib64/nvidia nvidia)

mark_as_advanced(CUPTI_cuda_LIBRARY)

# ----------------------------------------------------------------------------------------#

# try to find cuda driver stubs library if no real driver
if(NOT CUPTI_cuda_LIBRARY)
    find_library(
        CUPTI_cuda_LIBRARY
        NAMES cuda
        HINTS ${_CUDA_PATHS}
        PATHS ${_CUDA_PATHS}
        PATH_SUFFIXES lib/stubs lib64/stubs stubs)
    set(HAS_CUDA_DRIVER_LIBRARY
        OFF
        CACHE BOOL "Using stubs library")
else()
    set(HAS_CUDA_DRIVER_LIBRARY
        ON
        CACHE BOOL "Using stubs library")
endif()

# ------------------------------------------------------------------------------#

find_package_handle_standard_args(CUPTI DEFAULT_MSG CUPTI_INCLUDE_DIR CUPTI_cupti_LIBRARY
                                  CUPTI_cuda_LIBRARY CUPTI_cupti_LIBRARY_DIR)

# ------------------------------------------------------------------------------#

if(CUPTI_FOUND)
    set(CUPTI_INCLUDE_DIRS ${CUPTI_INCLUDE_DIR})
    set(CUPTI_LIBRARIES ${CUPTI_cupti_LIBRARY} ${CUPTI_cuda_LIBRARY})
    set(CUPTI_LIBRARY_DIRS ${CUPTI_cupti_LIBRARY_DIR})

    foreach(_COMP nvperf_host nvperf_target pcsampling pcsampling_util)
        set(CUPTI_${_COMP}_FOUND OFF)
        if(NOT DEFINED CUPTI_${_COMP}_LIBRARY AND CUPTI_${_COMP}_HEADER)
            set(CUPTI_${_COMP}_FOUND ON)
        elseif(CUPTI_${_COMP}_LIBRARY AND CUPTI_${_COMP}_HEADER)
            set(CUPTI_${_COMP}_FOUND ON)
            list(APPEND CUPTI_LIBRARIES ${CUPTI_${_COMP}_LIBRARY})
        endif()
    endforeach()
endif()

# ------------------------------------------------------------------------------#

unset(_CUDA_PATHS)
unset(_CUDA_INC)

# ------------------------------------------------------------------------------#
