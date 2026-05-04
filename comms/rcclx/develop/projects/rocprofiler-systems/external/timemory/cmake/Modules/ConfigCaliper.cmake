# Try to find the libraries and headers for Caliper optional dependencies Usage of this
# module is as follows
#
# include(ConfigCaliper)
#

set(CALIPER_OPTION_PREFIX
    ON
    CACHE INTERNAL "Prefix caliper options with CALIPER_")

# caliper uses PAPI high-level API so do not enable by default
set(CALIPER_WITH_PAPI
    OFF
    CACHE BOOL "Enable PAPI in Caliper")
set(CALIPER_WITH_MPI
    OFF
    CACHE BOOL "Enable MPI in Caliper")
set(CALIPER_WITH_CUPTI
    OFF
    CACHE BOOL "Enable CUPTI in Caliper")

# always sync with timemory settings
set(CALIPER_INSTALL_CONFIG
    ${TIMEMORY_INSTALL_CONFIG}
    CACHE BOOL "Install cmake and pkg-config files" FORCE)
set(CALIPER_INSTALL_HEADERS
    ${TIMEMORY_INSTALL_HEADERS}
    CACHE BOOL "Install caliper headers" FORCE)

find_path(
    LIBUNWIND_INCLUDE_DIR
    NAMES unwind.h libunwind.h
    PATH_SUFFIXES include)

find_library(
    LIBUNWIND_LIBRARY
    NAMES unwind
    PATH_SUFFIXES lib lib64)

find_library(
    LIBUNWIND_STATIC_LIBRARY
    NAMES libunwind.a
    PATH_SUFFIXES lib lib64)

if(LIBUNWIND_INCLUDE_DIR AND LIBUNWIND_LIBRARY)
    set(CALIPER_WITH_CALLPATH
        ON
        CACHE BOOL "Enable libunwind in Caliper")
else()
    set(CALIPER_WITH_CALLPATH
        OFF
        CACHE BOOL "Enable libunwind in Caliper")
endif()

if(APPLE)
    set(CALIPER_WITH_GOTCHA
        OFF
        CACHE BOOL "Enable gotcha in Caliper")
endif()
