# Try to find the LIKWID library and headers Usage of this module is as follows
#
# find_package( LIKWID ) if(LIKWID_FOUND) include_directories(${LIKWID_INCLUDE_DIRS})
# add_executable(foo foo.cc) target_link_libraries(foo ${LIKWID_LIBRARIES}) endif()
#
# You can provide a minimum version number that should be used. If you provide this
# version number and specify the REQUIRED attribute, this module will fail if it can't
# find a LIKWID of the specified version or higher. If you further specify the EXACT
# attribute, then this module will fail if it can't find a LIKWID with a version eaxctly
# as specified.
#
# ===========================================================================
# Variables used by this module which can be used to change the default behaviour, and
# hence need to be set before calling find_package:
#
# LIKWID_ROOT_DIR The preferred installation prefix for searching for LIKWID Set this if
# the module has problems finding the proper LIKWID installation.
#
# If you don't supply LIKWID_ROOT_DIR, the module will search on the standard system
# paths.
#
# ============================================================================
# Variables set by this module:
#
# LIKWID_FOUND           System has LIKWID.
#
# LIKWID_INCLUDE_DIRS    LIKWID include directories: not cached.
#
# LIKWID_LIBRARIES       Link to these to use the LIKWID library: not cached.
#
# ===========================================================================
# If LIKWID is installed in a non-standard way, e.g. a non GNU-style install of
# <prefix>/{lib,include}, then this module may fail to locate the headers and libraries as
# needed. In this case, the following cached variables can be editted to point to the
# correct locations.
#
# LIKWID_INCLUDE_DIR    The path to the LIKWID include directory: cached
#
# LIKWID_LIBRARY        The path to the LIKWID library: cached
#
# You should not need to set these in the vast majority of cases
#

# ----------------------------------------------------------------------------------------#

include(CMakeParseArguments)
include(TimemoryFindUtilities)

# ----------------------------------------------------------------------------------------#
# search for likwid marker because if this file does not exist, we cannot forward to
# Likwid
find_path(
    LIKWID_ROOT_DIR
    NAMES include/likwid.h
    HINTS ENV LIKWID_ROOT_DIR
    DOC "LIKWID root installation directory")

# ----------------------------------------------------------------------------------------#

find_path(
    LIKWID_INCLUDE_DIR
    NAMES likwid-marker.h
    HINTS ${LIKWID_ROOT_DIR} ENV LIKWID_ROOT_DIR ENV CPATH
    PATH_SUFFIXES include
    DOC "Path to the LIKWID headers")

# ----------------------------------------------------------------------------------------#

find_library(
    LIKWID_LIBRARY
    NAMES likwid
    HINTS ${LIKWID_ROOT_DIR}
          ENV
          LIKWID_ROOT_DIR
          ENV
          LIBRARY_PATH
          ENV
          LD_LIBRARY_PATH
          ENV
          DYLD_LIBRARY_PATH
    PATH_SUFFIXES lib lib64
    DOC "Path to the LIKWID library")

# ----------------------------------------------------------------------------------------#

find_library(
    LIKWID_hwloc_LIBRARY
    NAMES likwid-hwloc
    HINTS ${LIKWID_ROOT_DIR}
          ENV
          LIKWID_ROOT_DIR
          ENV
          LIBRARY_PATH
          ENV
          LD_LIBRARY_PATH
          ENV
          DYLD_LIBRARY_PATH
    PATH_SUFFIXES lib lib64
    DOC "Path to the LIKWID HWLOC library")

# ----------------------------------------------------------------------------------------#

find_library(
    LIKWID_lua_LIBRARY
    NAMES likwid-lua
    HINTS ${LIKWID_ROOT_DIR}
          ENV
          LIKWID_ROOT_DIR
          ENV
          LD_LIBRARY_PATH
          ENV
          LIBRARY_PATH
          ENV
          DYLD_LIBRARY_PATH
    PATH_SUFFIXES lib lib64
    DOC "Path to the LIKWID lua library")

# ----------------------------------------------------------------------------------------#

timemory_find_static_library(
    LIKWID_lua_STATIC_LIBRARY
    NAMES likwid-lua
    HINTS ${LIKWID_ROOT_DIR}
          ENV
          LIKWID_ROOT_DIR
          ENV
          LD_LIBRARY_PATH
          ENV
          LIBRARY_PATH
          ENV
          DYLD_LIBRARY_PATH
    PATH_SUFFIXES lib lib64
    DOC "Path to the LIKWID lua library")

# ----------------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIKWID_FOUND to TRUE if all listed
# variables are TRUE
find_package_handle_standard_args(LIKWID DEFAULT_MSG LIKWID_INCLUDE_DIR LIKWID_LIBRARY)

# ----------------------------------------------------------------------------------------#

if(LIKWID_FOUND)
    add_library(LIKWID::likwid INTERFACE IMPORTED)
    add_library(LIKWID::likwid-hwloc INTERFACE IMPORTED)
    target_link_libraries(LIKWID::likwid INTERFACE ${LIKWID_LIBRARY})
    target_include_directories(LIKWID::likwid INTERFACE ${LIKWID_INCLUDE_DIR})
    get_filename_component(LIKWID_INCLUDE_DIRS ${LIKWID_INCLUDE_DIR} REALPATH)
    get_filename_component(LIKWID_LIBRARIES ${LIKWID_LIBRARY} REALPATH)
    if(LIKWID_hwloc_LIBRARY)
        target_link_libraries(LIKWID::likwid-hwloc INTERFACE ${LIKWID_hwloc_LIBRARY})
        target_link_libraries(LIKWID::likwid INTERFACE ${LIKWID_hwloc_LIBRARY})
        list(APPEND LIKWID_LIBRARIES ${LIKWID_hwloc_LIBRARY})
    endif()
endif()

mark_as_advanced(LIKWID_INCLUDE_DIR LIKWID_LIBRARY)
