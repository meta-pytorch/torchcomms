# ------------------------------------------------------------------------------#
#
# Finds headers and libraries for AllineaMAP instrumentation
#
# ------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)

set(AllineaMAP_HINTS "$ENV{ALLINEA_TOOLS_DIR}/$ENV{ALLINEA_TOOLS_VERSION}")

# ------------------------------------------------------------------------------#

find_path(
    AllineaMAP_INCLUDE_DIR
    NAMES mapsampler_api.h
    PATH_SUFFIXES map/wrapper
    HINTS ${AllineaMAP_ROOT_DIR} ${AllineaMAP_HINTS}
    PATHS ${AllineaMAP_ROOT_DIR} ${AllineaMAP_HINTS})

# ------------------------------------------------------------------------------#

find_library(
    AllineaMAP_LIBRARY
    NAMES map-sampler
    PATH_SUFFIXES lib lib64 lib 64 lib/64
    HINTS ${AllineaMAP_ROOT_DIR} ${AllineaMAP_HINTS}
    PATHS ${AllineaMAP_ROOT_DIR} ${AllineaMAP_HINTS})

# ------------------------------------------------------------------------------#

if(AllineaMAP_INCLUDE_DIR)
    set(AllineaMAP_INCLUDE_DIRS ${AllineaMAP_INCLUDE_DIR})
endif()

# ------------------------------------------------------------------------------#

if(AllineaMAP_LIBRARY)
    set(AllineaMAP_LIBRARIES ${AllineaMAP_LIBRARY})
endif()

# ------------------------------------------------------------------------------#

mark_as_advanced(AllineaMAP_INCLUDE_DIR AllineaMAP_LIBRARY)
find_package_handle_standard_args(AllineaMAP REQUIRED_VARS AllineaMAP_INCLUDE_DIR
                                                           AllineaMAP_LIBRARY)

# ------------------------------------------------------------------------------#
