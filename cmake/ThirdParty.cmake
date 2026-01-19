# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Third-party dependency management for torchcomms/NCCLX build.
# This module fetches and builds all third-party dependencies needed by NCCLX.

include(ExternalProject)
include(FetchContent)

set(THIRD_PARTY_TAG "v2025.12.15.00" CACHE STRING "FB OSS library version tag")
set(THIRD_PARTY_BUILD_DIR "${CMAKE_BINARY_DIR}/third-party" CACHE PATH "Third-party build directory")

# Determine install prefix - use CONDA_PREFIX if available, else build directory
if(DEFINED ENV{CONDA_PREFIX})
    set(THIRD_PARTY_INSTALL_PREFIX "$ENV{CONDA_PREFIX}" CACHE PATH "Third-party install prefix")
else()
    set(THIRD_PARTY_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/conda" CACHE PATH "Third-party install prefix")
endif()

set(THIRD_PARTY_LIB_SUFFIX "lib" CACHE STRING "Library directory suffix")
set(THIRD_PARTY_LIB_DIR "${THIRD_PARTY_INSTALL_PREFIX}/${THIRD_PARTY_LIB_SUFFIX}")
set(THIRD_PARTY_INCLUDE_DIR "${THIRD_PARTY_INSTALL_PREFIX}/include")

# Common CMake args for ExternalProject builds
set(THIRD_PARTY_CMAKE_ARGS
    -DCMAKE_PREFIX_PATH=${THIRD_PARTY_INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PREFIX}
    -DCMAKE_MODULE_PATH=${THIRD_PARTY_INSTALL_PREFIX}
    -DCMAKE_INSTALL_DIR=${THIRD_PARTY_INSTALL_PREFIX}
    -DBIN_INSTALL_DIR=${THIRD_PARTY_INSTALL_PREFIX}/bin
    -DLIB_INSTALL_DIR=${THIRD_PARTY_LIB_DIR}
    -DINCLUDE_INSTALL_DIR=${THIRD_PARTY_INCLUDE_DIR}
    -DCMAKE_INSTALL_INCLUDEDIR=${THIRD_PARTY_INCLUDE_DIR}
    -DCMAKE_INSTALL_LIBDIR=${THIRD_PARTY_LIB_DIR}
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_STANDARD=20
    -DCMAKE_POLICY_VERSION_MINIMUM=3.22
    -DCMAKE_BUILD_TYPE=Release
)

# Skip building third-party if USE_SYSTEM_LIBS is set
if(USE_SYSTEM_LIBS)
    message(STATUS "USE_SYSTEM_LIBS is set, skipping third-party builds")
    add_custom_target(third_party_deps)
    return()
endif()

# Skip if NCCL_BUILD_SKIP_DEPS is set
if(DEFINED ENV{NCCL_BUILD_SKIP_DEPS})
    message(STATUS "NCCL_BUILD_SKIP_DEPS is set, skipping third-party builds")
    add_custom_target(third_party_deps)
    return()
endif()

message(STATUS "Building third-party dependencies to ${THIRD_PARTY_INSTALL_PREFIX}")

# ============================================================================
# fmt library
# ============================================================================
ExternalProject_Add(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/fmt
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
        -DFMT_INSTALL=ON
        -DFMT_TEST=OFF
        -DFMT_DOC=OFF
    CMAKE_GENERATOR Ninja
)

# Build shared fmt as well
ExternalProject_Add(fmt_shared
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/fmt_shared
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
        -DFMT_INSTALL=ON
        -DFMT_TEST=OFF
        -DFMT_DOC=OFF
        -DBUILD_SHARED_LIBS=ON
    CMAKE_GENERATOR Ninja
    DEPENDS fmt
)

# ============================================================================
# zlib library
# ============================================================================
ExternalProject_Add(zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.2.13
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/zlib
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
        -DZLIB_BUILD_TESTING=OFF
    CMAKE_GENERATOR Ninja
)

# ============================================================================
# boost library (uses b2 build system)
# ============================================================================
ExternalProject_Add(boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG boost-1.82.0
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/boost
    CONFIGURE_COMMAND ./bootstrap.sh
        --prefix=${THIRD_PARTY_INSTALL_PREFIX}
        --libdir=${THIRD_PARTY_LIB_DIR}
        --without-libraries=python
    BUILD_COMMAND ./b2 -q cxxflags=-fPIC cflags=-fPIC install
    BUILD_IN_SOURCE TRUE
    INSTALL_COMMAND ""
)

# ============================================================================
# openssl library (uses autoconf)
# ============================================================================
ExternalProject_Add(openssl
    GIT_REPOSITORY https://github.com/openssl/openssl.git
    GIT_TAG openssl-3.5.1
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/openssl
    CONFIGURE_COMMAND ./config no-shared
        --prefix=${THIRD_PARTY_INSTALL_PREFIX}
        --openssldir=${THIRD_PARTY_INSTALL_PREFIX}
        --libdir=lib
    BUILD_COMMAND ${CMAKE_COMMAND} -E env make -j
    INSTALL_COMMAND ${CMAKE_COMMAND} -E env make install
    BUILD_IN_SOURCE TRUE
)

# ============================================================================
# xxHash library
# ============================================================================
ExternalProject_Add(xxhash
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG v0.8.0
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/xxhash
    SOURCE_SUBDIR cmake_unofficial
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
    CMAKE_GENERATOR Ninja
)

# ============================================================================
# gflags library (static)
# ============================================================================
ExternalProject_Add(gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG v2.2.2
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/gflags
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
    CMAKE_GENERATOR Ninja
)

# gflags shared (needed by thrift generator)
ExternalProject_Add(gflags_shared
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG v2.2.2
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/gflags_shared
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
        -DBUILD_SHARED_LIBS=ON
    CMAKE_GENERATOR Ninja
    DEPENDS gflags
)

# ============================================================================
# glog library (static)
# ============================================================================
ExternalProject_Add(glog
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG v0.4.0
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/glog
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
    CMAKE_GENERATOR Ninja
    DEPENDS gflags
)

# glog shared (needed by thrift generator)
ExternalProject_Add(glog_shared
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG v0.4.0
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/glog_shared
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
        -DBUILD_SHARED_LIBS=ON
    CMAKE_GENERATOR Ninja
    DEPENDS glog gflags_shared
)

# ============================================================================
# zstd library
# ============================================================================
ExternalProject_Add(zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG v1.5.6
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/zstd
    SOURCE_SUBDIR build/cmake
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
    CMAKE_GENERATOR Ninja
)

# ============================================================================
# libsodium library (uses autoconf)
# ============================================================================
ExternalProject_Add(sodium
    GIT_REPOSITORY https://github.com/jedisct1/libsodium.git
    GIT_TAG 1.0.20-RELEASE
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/sodium
    CONFIGURE_COMMAND ./configure --prefix=${THIRD_PARTY_INSTALL_PREFIX} --disable-pie
    BUILD_COMMAND ${CMAKE_COMMAND} -E env LDFLAGS=-Wl,--allow-shlib-undefined make -j
    INSTALL_COMMAND make install
    BUILD_IN_SOURCE TRUE
)

# ============================================================================
# fast_float library
# ============================================================================
ExternalProject_Add(fast_float
    GIT_REPOSITORY https://github.com/fastfloat/fast_float.git
    GIT_TAG v8.0.2
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/fast_float
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
        -DFASTFLOAT_INSTALL=ON
    CMAKE_GENERATOR Ninja
)

# ============================================================================
# libevent library
# ============================================================================
ExternalProject_Add(libevent
    GIT_REPOSITORY https://github.com/libevent/libevent.git
    GIT_TAG release-2.1.12-stable
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/libevent
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
    CMAKE_GENERATOR Ninja
    DEPENDS openssl
)

# ============================================================================
# double-conversion library
# ============================================================================
ExternalProject_Add(double-conversion
    GIT_REPOSITORY https://github.com/google/double-conversion.git
    GIT_TAG v3.3.1
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/double-conversion
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
    CMAKE_GENERATOR Ninja
)

# ============================================================================
# folly library
# ============================================================================
ExternalProject_Add(folly
    GIT_REPOSITORY https://github.com/facebook/folly.git
    GIT_TAG ${THIRD_PARTY_TAG}
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/folly
    SOURCE_SUBDIR folly
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
        -DUSE_STATIC_DEPS_ON_UNIX=ON
        -DOPENSSL_USE_STATIC_LIBS=ON
    CMAKE_GENERATOR Ninja
    DEPENDS fmt zlib boost openssl xxhash gflags glog zstd sodium fast_float libevent double-conversion
)

# ============================================================================
# fizz library
# ============================================================================
ExternalProject_Add(fizz
    GIT_REPOSITORY https://github.com/facebookincubator/fizz.git
    GIT_TAG ${THIRD_PARTY_TAG}
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/fizz
    SOURCE_SUBDIR fizz
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
        -DBUILD_TESTS=OFF
        -DBUILD_EXAMPLES=OFF
    CMAKE_GENERATOR Ninja
    DEPENDS folly
)

# ============================================================================
# mvfst (quic) library
# ============================================================================
ExternalProject_Add(quic
    GIT_REPOSITORY https://github.com/facebook/mvfst.git
    GIT_TAG ${THIRD_PARTY_TAG}
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/quic
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
    CMAKE_GENERATOR Ninja
    DEPENDS fizz
)

# ============================================================================
# wangle library
# ============================================================================
ExternalProject_Add(wangle
    GIT_REPOSITORY https://github.com/facebook/wangle.git
    GIT_TAG ${THIRD_PARTY_TAG}
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/wangle
    SOURCE_SUBDIR wangle
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
        -DBUILD_TESTS=OFF
    CMAKE_GENERATOR Ninja
    DEPENDS quic
)

# ============================================================================
# fbthrift library
# ============================================================================
ExternalProject_Add(thrift
    GIT_REPOSITORY https://github.com/facebook/fbthrift.git
    GIT_TAG ${THIRD_PARTY_TAG}
    GIT_SHALLOW TRUE
    PREFIX ${THIRD_PARTY_BUILD_DIR}/thrift
    CMAKE_ARGS ${THIRD_PARTY_CMAKE_ARGS}
    CMAKE_GENERATOR Ninja
    DEPENDS wangle glog_shared gflags_shared
)

# Create a target that depends on all third-party libraries
add_custom_target(third_party_deps
    DEPENDS thrift
)
