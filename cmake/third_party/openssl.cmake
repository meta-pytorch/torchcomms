# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# OpenSSL doesn't have native CMake build support, so we prefer
# find_package in all cases (system, conda, or prebuilt).
# ExternalProject is only used as a last resort.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(OpenSSL REQUIRED)
elseif(CONDA_PREFIX)
    set(OPENSSL_ROOT_DIR "${CONDA_PREFIX}" CACHE PATH "" FORCE)
    find_package(OpenSSL QUIET)
endif()

if(NOT TARGET OpenSSL::SSL)
    # Try system OpenSSL before resorting to building from source.
    find_package(OpenSSL QUIET)
endif()

if(NOT TARGET OpenSSL::SSL)
    include(ExternalProject)
    set(_openssl_install_dir "${CMAKE_BINARY_DIR}/third_party/openssl-install")
    ExternalProject_Add(openssl_external
        GIT_REPOSITORY https://github.com/openssl/openssl.git
        GIT_TAG ${TORCHCOMMS_OPENSSL_VERSION}
        GIT_SHALLOW TRUE
        CONFIGURE_COMMAND ./config no-shared
            --prefix=${_openssl_install_dir}
            --openssldir=${_openssl_install_dir}
            --libdir=lib
        BUILD_COMMAND make -j $ENV{CMAKE_BUILD_PARALLEL_LEVEL}
        INSTALL_COMMAND make install
        BUILD_IN_SOURCE TRUE
        BUILD_BYPRODUCTS
            "${_openssl_install_dir}/lib/libssl.a"
            "${_openssl_install_dir}/lib/libcrypto.a"
    )

    add_library(OpenSSL::SSL STATIC IMPORTED)
    set_target_properties(OpenSSL::SSL PROPERTIES
        IMPORTED_LOCATION ${_openssl_install_dir}/lib/libssl.a
        INTERFACE_INCLUDE_DIRECTORIES ${_openssl_install_dir}/include
    )
    add_dependencies(OpenSSL::SSL openssl_external)

    add_library(OpenSSL::Crypto STATIC IMPORTED)
    set_target_properties(OpenSSL::Crypto PROPERTIES
        IMPORTED_LOCATION ${_openssl_install_dir}/lib/libcrypto.a
        INTERFACE_INCLUDE_DIRECTORIES ${_openssl_install_dir}/include
    )
    add_dependencies(OpenSSL::Crypto openssl_external)

    set(OPENSSL_ROOT_DIR ${_openssl_install_dir} CACHE PATH "" FORCE)
    set(OPENSSL_FOUND TRUE CACHE BOOL "" FORCE)
    set(OPENSSL_INCLUDE_DIR "${_openssl_install_dir}/include" CACHE PATH "" FORCE)
    set(OPENSSL_LIBRARIES "OpenSSL::SSL;OpenSSL::Crypto" CACHE STRING "" FORCE)
endif()
