# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

# libtorchcomms links glog::glog. Extensions consume glog::glog_headers and
# inherit its header definitions without adding another library dependency.
function(torchcomms_add_glog_headers)
    get_target_property(_glog_inc glog::glog INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(_glog_defs glog::glog INTERFACE_COMPILE_DEFINITIONS)
    add_library(glog::glog_headers INTERFACE IMPORTED GLOBAL)
    set_target_properties(glog::glog_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_glog_inc}"
    )
    if(_glog_defs AND NOT _glog_defs MATCHES "-NOTFOUND$")
        set_target_properties(glog::glog_headers PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "${_glog_defs}"
        )
    endif()
endfunction()

function(torchcomms_add_glog_direct_sources is_static)
    add_library(glog::glog_direct_sources INTERFACE IMPORTED GLOBAL)
    if(is_static)
        set(_glog_direct_sources_link glog::glog_headers)
    else()
        set(_glog_direct_sources_link glog::glog)
    endif()
    set_target_properties(glog::glog_direct_sources PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_glog_direct_sources_link}"
    )
endfunction()

# Wheels do not bundle glog, so prefer the conda PIC archive when available.
if(EXISTS "${CONDA_INCLUDE}/glog/logging.h" AND
   EXISTS "${CONDA_LIB}/libglog.a")
    set(glog_FOUND FALSE)
else()
    find_package(glog 0.4.0 QUIET CONFIG NO_CMAKE_PACKAGE_REGISTRY
        HINTS "${CONDA_LIB}/cmake/glog")
endif()
if(glog_FOUND)
    message(STATUS "Found system glog: ${glog_VERSION}")
    if(glog_VERSION VERSION_GREATER_EQUAL "0.6.0")
        set_property(TARGET glog::glog APPEND PROPERTY
            INTERFACE_COMPILE_DEFINITIONS
            TORCHCOMMS_GLOG_HAS_PUBLIC_INIT_CHECK)
    endif()
    torchcomms_add_glog_headers()
    get_target_property(_glog_type glog::glog TYPE)
    if(_glog_type STREQUAL "STATIC_LIBRARY")
        torchcomms_add_glog_direct_sources(TRUE)
    else()
        torchcomms_add_glog_direct_sources(FALSE)
    endif()
elseif(EXISTS "${CONDA_INCLUDE}/glog/logging.h")
    # Prefer static, fall back to shared, fall back to -lglog.
    if(EXISTS "${CONDA_LIB}/libglog.a")
        set(_GLOG_LIB "${CONDA_LIB}/libglog.a")
        set(_GLOG_IS_STATIC TRUE)
    elseif(EXISTS "${CONDA_LIB}/libglog.so")
        set(_GLOG_LIB "${CONDA_LIB}/libglog.so")
        set(_GLOG_IS_STATIC FALSE)
    else()
        set(_GLOG_LIB "glog")
        set(_GLOG_IS_STATIC FALSE)
    endif()
    add_library(glog::glog INTERFACE IMPORTED GLOBAL)
    set_target_properties(glog::glog PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CONDA_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "${_GLOG_LIB}"
    )
    add_library(glog::glog_headers INTERFACE IMPORTED GLOBAL)
    set_target_properties(glog::glog_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CONDA_INCLUDE}"
    )
    set(_glog_defs)
    if(EXISTS "${CONDA_INCLUDE}/glog/export.h")
        list(APPEND _glog_defs GLOG_USE_GLOG_EXPORT)
    endif()
    file(STRINGS "${CONDA_INCLUDE}/glog/logging.h" _glog_public_init
        REGEX "bool[ \t]+IsGoogleLoggingInitialized\\(\\)")
    if(_glog_public_init)
        list(APPEND _glog_defs TORCHCOMMS_GLOG_HAS_PUBLIC_INIT_CHECK)
    endif()
    if(_glog_defs)
        set_target_properties(glog::glog_headers PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "${_glog_defs}")
        set_target_properties(glog::glog PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "${_glog_defs}")
    endif()
    torchcomms_add_glog_direct_sources(${_GLOG_IS_STATIC})
    message(STATUS "Using glog: ${_GLOG_LIB}")
else()
    message(STATUS "System glog not found, fetching v0.4.0 via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        glog
        GIT_REPOSITORY https://github.com/google/glog.git
        GIT_TAG v0.4.0
    )
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
    set(_save_archive_dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    set(_save_lib_dir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    FetchContent_Populate(glog)
    add_subdirectory(${glog_SOURCE_DIR} ${glog_BINARY_DIR} EXCLUDE_FROM_ALL)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_save_archive_dir})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${_save_lib_dir})
    torchcomms_add_glog_headers()
    get_target_property(_glog_type glog::glog TYPE)
    if(_glog_type STREQUAL "STATIC_LIBRARY")
        torchcomms_add_glog_direct_sources(TRUE)
    else()
        torchcomms_add_glog_direct_sources(FALSE)
    endif()
endif()
