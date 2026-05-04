# ------------------------------------------------------------------------------#
#
# Finds headers and libraries for NCCL library
#
# ------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)

# ------------------------------------------------------------------------------#

find_path(
    NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATH_SUFFIXES include
    HINTS ${NCCL_ROOT_DIR}
    PATHS ${NCCL_ROOT_DIR})

# ------------------------------------------------------------------------------#

find_library(
    NCCL_LIBRARY
    NAMES nccl
    PATH_SUFFIXES lib lib64 lib 64 lib/64
    HINTS ${NCCL_ROOT_DIR}
    PATHS ${NCCL_ROOT_DIR})

# ------------------------------------------------------------------------------#

if(NCCL_INCLUDE_DIR)
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
endif()

# ------------------------------------------------------------------------------#

if(NCCL_LIBRARY)
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
endif()

# ------------------------------------------------------------------------------#

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
find_package_handle_standard_args(NCCL REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY)

# ------------------------------------------------------------------------------#
