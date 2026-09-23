# Try to find the NVTX library and headers Usage of this module is as follows
#
# find_package( NVTX ) if(NVTX_FOUND) include_directories(${NVTX_INCLUDE_DIRS})
# add_executable(foo foo.cc) target_link_libraries(foo ${NVTX_LIBRARIES}) endif()
#
# ===========================================================================
# Variables used by this module which can be used to change the default behaviour, and
# hence need to be set before calling find_package:
#
# NVTX_ROOT_DIR The preferred installation prefix for searching for NVTX Set this if the
# module has problems finding the proper NVTX installation.
#
# If you don't supply NVTX_ROOT_DIR, the module will search on the standard system paths.
#
# ============================================================================
# Variables set by this module:
#
# NVTX_FOUND           System has NVTX.
#
# NVTX_INCLUDE_DIRS    NVTX include directories: not cached.
#
# NVTX_LIBRARIES       Link to these to use the NVTX library: not cached.
#
# ===========================================================================
# If NVTX is installed in a non-standard way, e.g. a non GNU-style install of
# <prefix>/{lib,include}, then this module may fail to locate the headers and libraries as
# needed. In this case, the following cached variables can be editted to point to the
# correct locations.
#
# NVTX_INCLUDE_DIR    The path to the NVTX include directory: cached
#
# NVTX_LIBRARY        The path to the NVTX library: cached
#
# You should not need to set these in the vast majority of cases
#

# ----------------------------------------------------------------------------------------#

find_path(
    NVTX_ROOT_DIR
    NAMES include/nvtx3/nvToolsExt.h
    HINTS ENV NVTX_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR} /usr/local/cuda
    PATHS ENV NVTX_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR} /usr/local/cuda
    PATH_SUFFIXES targets/x86_64-linux targets/x86_64-darwin targets/x86_64-win32
                  targets/x86_64-win64
    DOC "NVTX root installation directory")

# ----------------------------------------------------------------------------------------#

find_path(
    NVTX_INCLUDE_DIR
    NAMES nvtx3/nvToolsExt.h
    HINTS ${NVTX_ROOT_DIR} ENV NVTX_ROOT_DIR ENV CPATH
    PATH_SUFFIXES include
    DOC "Path to the NVTX headers")

# ----------------------------------------------------------------------------------------#

find_library(
    NVTX_LIBRARY
    NAMES nvToolsExt
    PATHS ${NVTX_ROOT_DIR}
          ENV
          NVTX_ROOT_DIR
          ${CUDA_TOOLKIT_ROOT_DIR}
          ${CUDA_SDK_ROOT_DIR}
          /usr/local/cuda
          ENV
          LIBRARY_PATH
          ENV
          LD_LIBRARY_PATH
          ENV
          DYLD_LIBRARY_PATH
    HINTS ${NVTX_ROOT_DIR}
          ENV
          NVTX_ROOT_DIR
          ${CUDA_TOOLKIT_ROOT_DIR}
          ${CUDA_SDK_ROOT_DIR}
          /usr/local/cuda
          ENV
          LIBRARY_PATH
          ENV
          LD_LIBRARY_PATH
          ENV
          DYLD_LIBRARY_PATH
    PATH_SUFFIXES lib lib64
    DOC "Path to the NVTX library")

# ----------------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set NVTX_FOUND to TRUE if all listed
# variables are TRUE
find_package_handle_standard_args(NVTX DEFAULT_MSG NVTX_INCLUDE_DIR)

# ----------------------------------------------------------------------------------------#

if(NVTX_FOUND)
    add_library(nvtx INTERFACE)
    target_include_directories(nvtx INTERFACE ${NVTX_INCLUDE_DIR})
    get_filename_component(NVTX_INCLUDE_DIRS ${NVTX_INCLUDE_DIR} REALPATH)
    if(NVTX_LIBRARY)
        target_link_libraries(nvtx INTERFACE ${NVTX_LIBRARY})
        get_filename_component(NVTX_LIBRARIES ${NVTX_LIBRARY} REALPATH)
    endif()
endif()

mark_as_advanced(NVTX_INCLUDE_DIR NVTX_LIBRARY)
