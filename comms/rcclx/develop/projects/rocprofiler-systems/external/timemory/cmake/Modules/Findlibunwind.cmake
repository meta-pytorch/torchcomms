#
# * Find libunwind
#
# libunwind_PREFIX      - Set to the libunwind installation directory
#
# libunwind_INCLUDE_DIR - Path to libunwind.h libunwind_LIBRARIES   - List of libraries
# for using libunwind libunwind_FOUND       - True if libunwind was found

include(TimemoryFindUtilities)

timemory_find_root_path(
    libunwind_ROOT
    include/libunwind.h
    HINTS
    libunwind_ROOT
    libunwind_ROOT_DIR
    ENV
    libunwind_ROOT
    ENV
    libunwind_ROOT_DIR
    CACHE)

mark_as_advanced(libunwind_ROOT)

find_path(
    libunwind_ROOT_DIR
    NAMES include/libunwind.h
    HINTS ${libunwind_ROOT}
    PATHS ${libunwind_ROOT}
    DOC "libunwind root installation directory")

mark_as_advanced(libunwind_ROOT_DIR)

if(libunwind_ROOT_DIR)
    set(libunwind_FIND_VARS libunwind_ROOT_DIR)
elseif(libunwind_ROOT)
    set(libunwind_FIND_VARS libunwind_ROOT)
endif()

find_path(
    libunwind_INCLUDE_DIR
    NAMES libunwind.h
    HINTS ${libunwind_ROOT}
    PATHS ${libunwind_ROOT}
    PATH_SUFFIXES include
    DOC "Path to the libunwind headers")

mark_as_advanced(libunwind_INCLUDE_DIR)

if(NOT APPLE)
    find_library(
        libunwind_LIBRARY
        NAMES unwind libunwind
        HINTS ${libunwind_ROOT}
        PATHS ${libunwind_ROOT}
        PATH_SUFFIXES lib lib64
        DOC "Path to the libunwind library")

    mark_as_advanced(libunwind_LIBRARY)

    timemory_find_static_library(
        libunwind_STATIC_LIBRARY
        NAMES unwind libunwind
        HINTS ${libunwind_ROOT}
        PATHS ${libunwind_ROOT}
        PATH_SUFFIXES lib lib64
        DOC "Path to the libunwind static library")

    mark_as_advanced(libunwind_STATIC_LIBRARY)
else()
    set(libunwind_LIBRARY
        unwind
        CACHE STRING "libunwind library")

    mark_as_advanced(libunwind_LIBRARY)
endif()

foreach(_COMPONENT ${libunwind_FIND_COMPONENTS})
    find_library(
        libunwind_${_COMPONENT}_LIBRARY
        NAMES unwind-_${_COMPONENT} libunwind-_${_COMPONENT}
        HINTS ${libunwind_ROOT}
        PATHS ${libunwind_ROOT}
        PATH_SUFFIXES lib lib64
        DOC "Path to the libunwind ${_COMPONENT} library")

    mark_as_advanced(libunwind_${_COMPONENT}_LIBRARY)

    if(libunwind_INCLUDE_DIR
       AND libunwind_${_COMPONENT}_LIBRARY
       AND NOT TARGET libunwind::libunwind-${_COMPONENT})
        add_library(libunwind::libunwind-${_COMPONENT} INTERFACE IMPORTED)
        target_link_libraries(libunwind::libunwind-${_COMPONENT}
                              INTERFACE ${libunwind_${_COMPONENT}_LIBRARY})
        target_include_directories(libunwind::libunwind-shared
                                   INTERFACE ${libunwind_INCLUDE_DIR})
    endif()

    if(libunwind_${_COMPONENT}_LIBRARY)
        set(libunwind_${_COMPONENT}_FOUND ON)
    else()
        set(libunwind_${_COMPONENT}_FOUND OFF)
    endif()
endforeach()

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set libunwind_FOUND to TRUE if all listed
# variables are TRUE
find_package_handle_standard_args(
    libunwind
    FOUND_VAR libunwind_FOUND
    REQUIRED_VARS ${libunwind_FIND_VARS} libunwind_INCLUDE_DIR libunwind_LIBRARY
    HANDLE_COMPONENTS)

if(libunwind_FOUND)
    if(NOT TARGET libunwind::libunwind-shared AND libunwind_LIBRARY)
        add_library(libunwind::libunwind-shared INTERFACE IMPORTED)
        target_link_libraries(libunwind::libunwind-shared INTERFACE ${libunwind_LIBRARY})
        target_include_directories(libunwind::libunwind-shared
                                   INTERFACE ${libunwind_INCLUDE_DIR})
    endif()

    if(NOT TARGET libunwind::libunwind-static AND libunwind_STATIC_LIBRARY)
        add_library(libunwind::libunwind-static INTERFACE IMPORTED)
        target_link_libraries(libunwind::libunwind-static
                              INTERFACE ${libunwind_STATIC_LIBRARY})
        target_include_directories(libunwind::libunwind-static
                                   INTERFACE ${libunwind_INCLUDE_DIR})
    endif()

    get_filename_component(libunwind_INCLUDE_DIRS ${libunwind_INCLUDE_DIR} REALPATH)
    if(NOT APPLE)
        get_filename_component(libunwind_LIBRARIES ${libunwind_LIBRARY} REALPATH)
    endif()

    if(NOT TARGET libunwind::libunwind)
        add_library(libunwind::libunwind INTERFACE IMPORTED)
        if(TARGET libunwind::libunwind-shared)
            target_link_libraries(libunwind::libunwind
                                  INTERFACE libunwind::libunwind-shared)
        elseif(TARGET libunwind::libunwind-static)
            target_link_libraries(libunwind::libunwind
                                  INTERFACE libunwind::libunwind-static)
        endif()
    endif()
endif()

mark_as_advanced(libunwind_LIBRARIES libunwind_INCLUDE_DIRS)
