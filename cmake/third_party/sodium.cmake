# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# libsodium uses automake, not CMake. Prefer finding a prebuilt version;
# fall back to ExternalProject_Add if needed.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(sodium REQUIRED IMPORTED_TARGET libsodium)
elseif(CONDA_PREFIX)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(sodium QUIET IMPORTED_TARGET libsodium)
    endif()
endif()

if(NOT TARGET PkgConfig::sodium AND NOT TARGET sodium)
    # Try system libsodium before building from source.
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(sodium QUIET IMPORTED_TARGET libsodium)
    endif()
endif()

if(NOT TARGET PkgConfig::sodium AND NOT TARGET sodium)
    include(ExternalProject)
    set(_sodium_install_dir "${CMAKE_BINARY_DIR}/third_party/sodium-install")
    ExternalProject_Add(sodium_external
        GIT_REPOSITORY https://github.com/jedisct1/libsodium.git
        GIT_TAG ${TORCHCOMMS_SODIUM_VERSION}
        GIT_SHALLOW TRUE
        CONFIGURE_COMMAND ./configure
            --prefix=${_sodium_install_dir}
            --disable-pie
        BUILD_COMMAND make -j $ENV{CMAKE_BUILD_PARALLEL_LEVEL}
        INSTALL_COMMAND make install
        BUILD_IN_SOURCE TRUE
        BUILD_BYPRODUCTS "${_sodium_install_dir}/lib/libsodium.a"
    )

    # Create the include dir now so CMake's INTERFACE_INCLUDE_DIRECTORIES
    # validation passes.  ExternalProject populates it at build time.
    file(MAKE_DIRECTORY "${_sodium_install_dir}/include")

    add_library(sodium STATIC IMPORTED)
    set_target_properties(sodium PROPERTIES
        IMPORTED_LOCATION ${_sodium_install_dir}/lib/libsodium.a
        INTERFACE_INCLUDE_DIRECTORIES ${_sodium_install_dir}/include
    )
    add_dependencies(sodium sodium_external)

    # Pre-set variables for folly's FindLibsodium and fizz's FindSodium.
    # The actual library is built by ExternalProject at build time, but
    # cmake needs the paths at configure time for find_library caching.
    set(LIBSODIUM_FOUND TRUE CACHE BOOL "" FORCE)
    set(LIBSODIUM_INCLUDE_DIRS "${_sodium_install_dir}/include" CACHE PATH "" FORCE)
    set(LIBSODIUM_LIBRARIES sodium CACHE STRING "" FORCE)
    # fizz's FindSodium.cmake uses these variable names:
    set(sodium_INCLUDE_DIR "${_sodium_install_dir}/include" CACHE PATH "" FORCE)
    set(sodium_LIBRARY_RELEASE "${_sodium_install_dir}/lib/libsodium.a" CACHE FILEPATH "" FORCE)
    set(sodium_LIBRARY_DEBUG "${_sodium_install_dir}/lib/libsodium.a" CACHE FILEPATH "" FORCE)
endif()
