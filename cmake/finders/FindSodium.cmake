# Copyright (c) Meta Platforms, Inc. and affiliates.
# Stub Find module: short-circuits when sodium is provided via FetchContent/ExternalProject/pkg-config.

if(TARGET PkgConfig::sodium OR TARGET sodium)
    set(sodium_FOUND TRUE)
    set(Sodium_FOUND TRUE)
    if(TARGET PkgConfig::sodium)
        set(sodium_LIBRARY_RELEASE PkgConfig::sodium)
        get_target_property(sodium_INCLUDE_DIR PkgConfig::sodium INTERFACE_INCLUDE_DIRECTORIES)
    elseif(TARGET sodium)
        get_target_property(sodium_LIBRARY_RELEASE sodium IMPORTED_LOCATION)
        get_target_property(sodium_INCLUDE_DIR sodium INTERFACE_INCLUDE_DIRECTORIES)
    endif()
    set(sodium_LIBRARY_DEBUG "${sodium_LIBRARY_RELEASE}")
    if(NOT TARGET sodium::sodium)
        if(TARGET PkgConfig::sodium)
            add_library(sodium::sodium ALIAS PkgConfig::sodium)
        elseif(TARGET sodium)
            # Can't alias IMPORTED, just reference directly
        endif()
    endif()
    return()
endif()

# Fallback: system search
find_path(sodium_INCLUDE_DIR sodium.h)
find_library(sodium_LIBRARY_RELEASE sodium)
set(sodium_LIBRARY_DEBUG "${sodium_LIBRARY_RELEASE}")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sodium DEFAULT_MSG
    sodium_LIBRARY_RELEASE sodium_INCLUDE_DIR)
