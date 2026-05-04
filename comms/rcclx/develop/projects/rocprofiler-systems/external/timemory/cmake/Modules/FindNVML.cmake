# Try to find the NVML library and headers Usage of this module is as follows
#
# find_package( NVML ) if(NVML_FOUND) include_directories(${NVML_INCLUDE_DIRS})
# add_executable(foo foo.cc) target_link_libraries(foo ${NVML_LIBRARIES}) endif()
#
# ===========================================================================
# Variables used by this module which can be used to change the default behaviour, and
# hence need to be set before calling find_package:
#
# NVML_ROOT_DIR The preferred installation prefix for searching for NVML Set this if the
# module has problems finding the proper NVML installation.
#
# If you don't supply NVML_ROOT_DIR, the module will search on the standard system paths.
#
# ============================================================================
# Variables set by this module:
#
# NVML_FOUND           System has NVML.
#
# NVML_INCLUDE_DIRS    NVML include directories: not cached.
#
# NVML_LIBRARIES       Link to these to use the NVML library: not cached.
#
# ===========================================================================
# If NVML is installed in a non-standard way, e.g. a non GNU-style install of
# <prefix>/{lib,include}, then this module may fail to locate the headers and libraries as
# needed. In this case, the following cached variables can be editted to point to the
# correct locations.
#
# NVML_INCLUDE_DIR    The path to the NVML include directory: cached
#
# NVML_LIBRARY        The path to the NVML library: cached
#
# You should not need to set these in the vast majority of cases
#

# ----------------------------------------------------------------------------------------#

find_path(
    NVML_ROOT_DIR
    NAMES include/nvml.h
    HINTS ENV NVML_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR} /usr/local/cuda
    PATHS ENV NVML_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR} /usr/local/cuda
    PATH_SUFFIXES targets/x86_64-linux targets/x86_64-darwin targets/x86_64-win32
                  targets/x86_64-win64
    DOC "NVML root installation directory")

# ----------------------------------------------------------------------------------------#

find_path(
    NVML_INCLUDE_DIR
    NAMES nvml.h
    HINTS ${NVML_ROOT_DIR} ENV NVML_ROOT_DIR ENV CPATH
    PATH_SUFFIXES include
    DOC "Path to the NVML headers")

# ----------------------------------------------------------------------------------------#

set(_NVML_PATHS ${NVML_ROOT_DIR} $ENV{NVML_ROOT_DIR} $ENV{CUDA_HOME}
                ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR})

find_library(
    NVML_LIBRARY
    NAMES nvidia-ml
    PATHS ${_NVML_PATHS}
    HINTS ${_NVML_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "Path to the NVML library")

# ----------------------------------------------------------------------------------------#

# try to find cuda driver stubs library if no real driver
if(NOT NVML_LIBRARY)
    find_library(
        NVML_LIBRARY
        NAMES nvidia-ml
        PATHS ${_NVML_PATHS}
        HINTS ${_NVML_PATHS}
        PATH_SUFFIXES lib/stubs lib64/stubs stubs)
    set(NVML_LIBRARY_HAS_SYMBOLS
        OFF
        CACHE BOOL "Using stubs library" FORCE)
else()
    set(NVML_LIBRARY_HAS_SYMBOLS
        ON
        CACHE BOOL "Using stubs library" FORCE)
endif()

# ----------------------------------------------------------------------------------------#

if(NVML_ROOT_DIR)
    set(_NVML_ROOT_DIR NVML_ROOT_DIR)
endif()

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set NVML_FOUND to TRUE if all listed
# variables are TRUE
find_package_handle_standard_args(NVML DEFAULT_MSG ${_NVML_ROOT_DIR} NVML_INCLUDE_DIR
                                  NVML_LIBRARY)

# ----------------------------------------------------------------------------------------#

if(NVML_FOUND)
    if(NOT TARGET nvml)
        add_library(nvml INTERFACE)
        add_library(NVML::nvml ALIAS nvml)
    endif()
    target_link_libraries(nvml INTERFACE ${NVML_LIBRARY})
    target_include_directories(nvml INTERFACE ${NVML_INCLUDE_DIR})
    get_filename_component(NVML_INCLUDE_DIRS ${NVML_INCLUDE_DIR} REALPATH)
    get_filename_component(NVML_LIBRARIES ${NVML_LIBRARY} REALPATH)
endif()

mark_as_advanced(NVML_INCLUDE_DIR NVML_LIBRARY)

unset(_NVML_ROOT_DIR)
