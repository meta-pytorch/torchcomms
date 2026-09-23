# Try to find the PAPI library and headers Usage of this module is as follows
#
# find_package( PAPI ) if(PAPI_FOUND) include_directories(${PAPI_INCLUDE_DIRS})
# add_executable(foo foo.cc) target_link_libraries(foo ${PAPI_LIBRARIES}) endif()
#
# You can provide a minimum version number that should be used. If you provide this
# version number and specify the REQUIRED attribute, this module will fail if it can't
# find a PAPI of the specified version or higher. If you further specify the EXACT
# attribute, then this module will fail if it can't find a PAPI with a version eaxctly as
# specified.
#
# ===========================================================================
# Variables used by this module which can be used to change the default behaviour, and
# hence need to be set before calling find_package:
#
# PAPI_ROOT_DIR The preferred installation prefix for searching for PAPI Set this if the
# module has problems finding the proper PAPI installation.
#
# If you don't supply PAPI_ROOT_DIR, the module will search on the standard system paths.
#
# ============================================================================
# Variables set by this module:
#
# PAPI_FOUND           System has PAPI.
#
# PAPI_INCLUDE_DIRS    PAPI include directories: not cached.
#
# PAPI_LIBRARIES       Link to these to use the PAPI library: not cached.
#
# ===========================================================================
# If PAPI is installed in a non-standard way, e.g. a non GNU-style install of
# <prefix>/{lib,include}, then this module may fail to locate the headers and libraries as
# needed. In this case, the following cached variables can be editted to point to the
# correct locations.
#
# PAPI_INCLUDE_DIR    The path to the PAPI include directory: cached
#
# PAPI_LIBRARY        The path to the PAPI library: cached
#
# You should not need to set these in the vast majority of cases
#

# ----------------------------------------------------------------------------------------#

include(TimemoryFindUtilities)

# ----------------------------------------------------------------------------------------#

timemory_find_root_path(
    PAPI_ROOT
    include/papi.h
    HINTS
    ENV
    PAPI_ROOT
    ENV
    PAPI_ROOT_DIR
    ENV
    CRAY_PAPI_PREFIX
    ENV
    PAPI_ROOT_DIR
    ENV
    LIBRARY_PATH
    ENV
    LD_LIBRARY_PATH
    ENV
    DYLD_LIBRARY_PATH
    CACHE)

# ----------------------------------------------------------------------------------------#

if(PAPI_ROOT)
    set(PAPI_FIND_ARGS NO_CMAKE_ENVIRONMENT_PATH)
    # PAPI_ROOT_DIR is not required but this variable is passed as first arg to
    # find_package_handle_standard_args to display the root directory
    set(PAPI_FIND_VARS PAPI_ROOT_DIR)
endif()

# ----------------------------------------------------------------------------------------#

find_path(
    PAPI_ROOT_DIR
    NAMES include/papi.h
    HINTS ${PAPI_ROOT}
    PATHS ${PAPI_ROOT}
    DOC "PAPI root installation directory" ${PAPI_FIND_ARGS})

# ----------------------------------------------------------------------------------------#

find_path(
    PAPI_INCLUDE_DIR
    NAMES papi.h
    HINTS ${PAPI_ROOT}
    PATHS ${PAPI_ROOT}
    PATH_SUFFIXES include
    DOC "Path to the PAPI headers" ${PAPI_FIND_ARGS})

# ----------------------------------------------------------------------------------------#

find_library(
    PAPI_LIBRARY
    NAMES papi
    HINTS ${PAPI_ROOT}
    PATHS ${PAPI_ROOT}
    PATH_SUFFIXES lib lib64
    DOC "Path to the PAPI library" ${PAPI_FIND_ARGS})

# ----------------------------------------------------------------------------------------#

find_library(
    PAPI_pfm_LIBRARY
    NAMES pfm libpfm.so libpfm.so.4
    HINTS ${PAPI_ROOT}
    PATHS ${PAPI_ROOT}
    PATH_SUFFIXES lib lib64
    DOC "Path to the PAPI pfm library" ${PAPI_FIND_ARGS})

# ----------------------------------------------------------------------------------------#

timemory_find_static_library(
    PAPI_STATIC_LIBRARY
    NAMES papi
    HINTS ${PAPI_ROOT}
    PATHS ${PAPI_ROOT}
    PATH_SUFFIXES lib lib64
    DOC "Path to the PAPI static library" ${PAPI_FIND_ARGS})

#
# ----------------------------------------------------------------------------------------#

timemory_find_static_library(
    PAPI_pfm_STATIC_LIBRARY
    NAMES pfm libpfm.a libpfm
    HINTS ${PAPI_ROOT}
    PATHS ${PAPI_ROOT}
    PATH_SUFFIXES lib lib64
    DOC "Path to the PAPI pfm static library" ${PAPI_FIND_ARGS})

# ----------------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set PAPI_FOUND to TRUE if all listed
# variables are TRUE
find_package_handle_standard_args(PAPI DEFAULT_MSG ${PAPI_FIND_VARS} PAPI_INCLUDE_DIR
                                  PAPI_LIBRARY)

# ----------------------------------------------------------------------------------------#

if(PAPI_FOUND)
    if(NOT TARGET PAPI::papi-shared AND PAPI_LIBRARY)
        add_library(PAPI::papi-shared INTERFACE IMPORTED)
        target_link_libraries(PAPI::papi-shared INTERFACE ${PAPI_LIBRARY})
        target_include_directories(PAPI::papi-shared INTERFACE ${PAPI_INCLUDE_DIR})
        if(PAPI_pfm_LIBRARY)
            target_link_libraries(PAPI::papi-shared INTERFACE ${PAPI_pfm_LIBRARY})
        endif()
    endif()

    if(NOT TARGET PAPI::papi-static AND PAPI_STATIC_LIBRARY)
        add_library(PAPI::papi-static INTERFACE IMPORTED)
        target_link_libraries(PAPI::papi-static INTERFACE ${PAPI_STATIC_LIBRARY})
        target_include_directories(PAPI::papi-static INTERFACE ${PAPI_INCLUDE_DIR})
        if(PAPI_STATIC_LIBRARY)
            target_link_libraries(PAPI::papi-static INTERFACE ${PAPI_STATIC_LIBRARY})
        endif()
        if(PAPI_pfm_STATIC_LIBRARY)
            target_link_libraries(PAPI::papi-static INTERFACE ${PAPI_pfm_STATIC_LIBRARY})
        endif()
    endif()

    get_filename_component(PAPI_INCLUDE_DIRS ${PAPI_INCLUDE_DIR} REALPATH)
    get_filename_component(PAPI_LIBRARIES ${PAPI_LIBRARY} REALPATH)

    if(PAPI_STATIC_LIBRARY)
        list(APPEND PAPI_STATIC_LIBRARIES ${PAPI_STATIC_LIBRARY})
    endif()

    if(PAPI_pfm_LIBRARY)
        list(APPEND PAPI_LIBRARIES ${PAPI_pfm_LIBRARY})
    endif()

    if(PAPI_pfm_STATIC_LIBRARY)
        list(APPEND PAPI_STATIC_LIBRARIES ${PAPI_pfm_STATIC_LIBRARY})
    endif()
endif()

mark_as_advanced(PAPI_INCLUDE_DIR PAPI_LIBRARY)
