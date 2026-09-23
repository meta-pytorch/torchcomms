# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying file
# Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

# ----------------------------------------------------------------------------------------#

find_path(
    BFD_INCLUDE_DIR
    NAMES bfd.h
    PATH_SUFFIXES include)

mark_as_advanced(BFD_INCLUDE_DIR)

# ----------------------------------------------------------------------------------------#

find_library(
    BFD_LIBRARY
    NAMES bfd
    PATH_SUFFIXES lib lib64)

if(BFD_LIBRARY)
    get_filename_component(BFD_LIBRARY_DIR "${BFD_LIBRARY}" PATH CACHE)
endif()

mark_as_advanced(BFD_LIBRARY)

# ----------------------------------------------------------------------------------------#

find_package_handle_standard_args(BFD DEFAULT_MSG BFD_INCLUDE_DIR BFD_LIBRARY)

# ------------------------------------------------------------------------------#

if(BFD_FOUND)
    add_library(BFD::BFD INTERFACE IMPORTED)
    set(BFD_INCLUDE_DIRS ${BFD_INCLUDE_DIR})
    set(BFD_LIBRARIES ${BFD_LIBRARY})
    set(BFD_LIBRARY_DIRS ${BFD_LIBRARY_DIR})

    target_include_directories(BFD::BFD INTERFACE ${BFD_INCLUDE_DIRS})
    target_link_libraries(BFD::BFD INTERFACE ${BFD_LIBRARIES})
endif()

# ------------------------------------------------------------------------------#
