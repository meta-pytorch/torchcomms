#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -x

function do_cmake_build() {
  local source_dir="$1"
  local extra_flags="$2"
  local ninja_bin
  ninja_bin="$(which ninja)"
  cmake -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$ninja_bin" \
    -DCMAKE_C_COMPILER="${CC:-$(which gcc)}" \
    -DCMAKE_CXX_COMPILER="${CXX:-$(which g++)}" \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX="$CMAKE_PREFIX_PATH" \
    -DCMAKE_MODULE_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_INSTALL_DIR="$CMAKE_PREFIX_PATH" \
    -DBIN_INSTALL_DIR="$CMAKE_PREFIX_PATH/bin" \
    -DLIB_INSTALL_DIR="$CMAKE_PREFIX_PATH/$LIB_SUFFIX" \
    -DINCLUDE_INSTALL_DIR="$CMAKE_PREFIX_PATH/include" \
    -DCMAKE_INSTALL_INCLUDEDIR="$CMAKE_PREFIX_PATH/include" \
    -DCMAKE_INSTALL_LIBDIR="$CMAKE_PREFIX_PATH/$LIB_SUFFIX" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.22 \
    -DCMAKE_CXX_FLAGS="-DFMT_HEADER_ONLY=1" \
    $extra_flags \
    -S "${source_dir}"
  ninja
  ninja install
}

function clean_third_party {
  local library_name="$1"
  if [ "$CLEAN_THIRD_PARTY" == 1 ]; then
    rm -rf "${CONDA_PREFIX}"/include/"${library_name}"*/
    rm -rf "${CONDA_PREFIX}"/include/"${library_name}"*.h
  fi
}

function build_fb_oss_library() {
  local repo_url="$1"
  local repo_tag="$2"
  local library_name="$3"
  local extra_flags="$4"

  clean_third_party "$library_name"

  if [ ! -e "$library_name" ]; then
    git clone --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  local source_dir="../${library_name}/${library_name}"
  if [ -f ${library_name}/CMakeLists.txt ]; then
    source_dir="../${library_name}"
  fi
  if [ -f ${library_name}/build/cmake/CMakeLists.txt ]; then
    source_dir="../${library_name}/build/cmake"
  fi
  if [ -f ${library_name}/cmake_unofficial/CMakeLists.txt ]; then
    source_dir="../${library_name}/cmake_unofficial"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  rm -rf build-output
  mkdir -p build-output
  pushd build-output
  do_cmake_build "$source_dir" "$extra_flags"
  popd
}

function build_automake_library() {
  local repo_url="$1"
  local repo_tag="$2"
  local library_name="$3"
  local extra_flags="$4"

  clean_third_party "$library_name"

  if [ ! -e "$library_name" ]; then
    git clone --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  pushd "$library_name"
  ./configure --prefix="$CMAKE_PREFIX_PATH" --disable-pie

  make -j$(nproc)
  make install
  popd
}

function build_boost() {
  local repo_url="https://github.com/boostorg/boost.git"
  local repo_tag="boost-1.82.0"
  local library_name="boost"
  local extra_flags=""

  # clean up existing boost
  clean_third_party "$library_name"

  if [ ! -e "$library_name" ]; then
    git clone -j 10 --recurse-submodules --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  export LDFLAGS="-Wl,--allow-shlib-undefined"
  pushd "$library_name"
  ./bootstrap.sh --prefix="$CMAKE_PREFIX_PATH" --libdir="$CMAKE_PREFIX_PATH/$LIB_SUFFIX" --without-libraries=python --without-icu
  ./b2 -q cxxflags=-fPIC cflags=-fPIC pch=off install
  popd
}

function build_openssl() {
  local repo_url="https://github.com/openssl/openssl.git"
  local repo_tag="openssl-3.5.1"
  local library_name="openssl"
  local extra_flags=""

  # clean up existing boost
  clean_third_party "$library_name"

  if [ ! -e "$library_name" ]; then
    git clone -j 10 --recurse-submodules --depth 1 -b "$repo_tag" "$repo_url" "$library_name"
  fi

  pushd "$library_name"
  ./config no-shared --prefix="$CMAKE_PREFIX_PATH" --openssldir="$CMAKE_PREFIX_PATH" --libdir=lib

  make -j$(nproc)
  make install
  popd
}

function build_third_party {
  # build third-party libraries
  if [ "$CLEAN_THIRD_PARTY" == 1 ]; then
    rm -f "${CONDA_PREFIX}"/*.cmake 2>/dev/null || true
  fi
  local third_party_tag="v2026.01.19.00"
  local folly_tag="v2026.02.23.00"

  mkdir -p /tmp/third-party
  pushd /tmp/third-party
  if [[ -z "${USE_SYSTEM_LIBS}" ]]; then
    build_fb_oss_library "https://github.com/fmtlib/fmt.git" "11.2.0" fmt "-DFMT_INSTALL=ON -DFMT_TEST=OFF -DFMT_DOC=OFF"
    build_fb_oss_library "https://github.com/fmtlib/fmt.git" "11.2.0" fmt "-DFMT_INSTALL=ON -DFMT_TEST=OFF -DFMT_DOC=OFF -DBUILD_SHARED_LIBS=ON"
    build_fb_oss_library "https://github.com/madler/zlib.git" "v1.2.13" zlib "-DZLIB_BUILD_TESTING=OFF"
    build_boost
    build_openssl
    build_fb_oss_library "https://github.com/Cyan4973/xxHash.git" "v0.8.0" xxhash
    # we need both static and dynamic gflags since thrift generator can't
    # statically link against glog.
    build_fb_oss_library "https://github.com/gflags/gflags.git" "v2.2.2" gflags
    build_fb_oss_library "https://github.com/gflags/gflags.git" "v2.2.2" gflags "-DBUILD_SHARED_LIBS=ON"
    # we need both static and dynamic glog since thrift generator can't
    # statically link against glog.
    build_fb_oss_library "https://github.com/google/glog.git" "v0.4.0" glog
    build_fb_oss_library "https://github.com/google/glog.git" "v0.4.0" glog "-DBUILD_SHARED_LIBS=ON"
    build_fb_oss_library "https://github.com/facebook/zstd.git" "v1.5.6" zstd
    build_automake_library "https://github.com/jedisct1/libsodium.git" "1.0.20-RELEASE" sodium
    build_fb_oss_library "https://github.com/fastfloat/fast_float.git" "v8.0.2" fast_float "-DFASTFLOAT_INSTALL=ON"
    # Build libevent as both static and shared (thrift needs .a, others may need .so)
    build_fb_oss_library "https://github.com/libevent/libevent.git" "release-2.1.12-stable" event "-DEVENT__LIBRARY_TYPE=BOTH"
    build_fb_oss_library "https://github.com/google/double-conversion.git" "v3.3.1" double-conversion
    # Build folly with SSE4.2 to match RCCL's F14 intrinsics mode, and define SO_INCOMING_NAPI_ID
    CXXFLAGS_SAVED="${CXXFLAGS:-}"
    export CXXFLAGS="${CXXFLAGS_SAVED} -DSO_INCOMING_NAPI_ID=56 -msse4.2"
    build_fb_oss_library "https://github.com/facebook/folly.git" "$folly_tag" folly "-DUSE_STATIC_DEPS_ON_UNIX=ON -DOPENSSL_USE_STATIC_LIBS=ON -DLIBEVENT_INCLUDE_DIR=${CONDA_PREFIX}/include -DLIBEVENT_LIB=${CONDA_PREFIX}/lib/libevent.a"
    export CXXFLAGS="${CXXFLAGS_SAVED}"
  else
    if [[ -z "${NCCL_SKIP_CONDA_INSTALL}" ]]; then
      DEPS=(
        cmake
        ninja
        jemalloc
        gtest
        boost
        double-conversion
        libevent
        conda-forge::libsodium
        libunwind
        snappy
        conda-forge::fast_float
        libdwarf-dev
        gflags
        glog==0.4.0
        xxhash
        zstd
        conda-forge::zlib
        conda-forge::libopenssl-static
        conda-forge::folly
        fmt
        pyyaml
      )
      conda install "${DEPS[@]}" --yes
    fi
  fi

  # TODO: migrate out all dependencies for feedstock
  if [[ -z "${NCCL_FEEDSTOCK_BUILD}" ]]; then
    build_fb_oss_library "https://github.com/facebookincubator/fizz.git" "$third_party_tag" fizz "-DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF"
    # Clone mvfst and disable the xsk subdirectory (requires linux/if_xdp.h not available on all systems)
    if [ ! -e "quic" ]; then
      git clone --depth 1 -b "$third_party_tag" "https://github.com/facebook/mvfst" quic
    fi
    sed -i 's|^add_subdirectory(xsk)|# add_subdirectory(xsk) # disabled: requires linux/if_xdp.h|' quic/quic/CMakeLists.txt
    build_fb_oss_library "https://github.com/facebook/mvfst" "$third_party_tag" quic
    build_fb_oss_library "https://github.com/facebook/wangle.git" "$third_party_tag" wangle "-DBUILD_TESTS=OFF"
  fi
  build_fb_oss_library "https://github.com/facebook/fbthrift.git" "$third_party_tag" thrift

  popd
}

function build_comms_tracing_service {
  local include_prefix="comms/analyzer/if"
  local base_dir="${PWD}"
  local build_dir=/tmp/build/comms_tracing_service

  rm -rf "$build_dir"
  mkdir -p "$build_dir"
  pushd "$build_dir"
  # set up the directory structure
  mkdir -p "$include_prefix"
  cp -r "${base_dir}/${include_prefix}"/* "$include_prefix"
  mv "$include_prefix"/CMakeLists.txt .

  # set up the build config
  cp -r /tmp/third-party/thrift/build .

  # build the thrift service library
  cd build
  do_cmake_build ..

  popd
}

if [ -z "$DEV_SIGNATURE" ]; then
    is_git=$(git rev-parse --is-inside-work-tree)
    if [ $is_git ]; then
        DEV_SIGNATURE="git-"$(git rev-parse --short HEAD)
    else
        echo "Cannot detect source repository hash. Skip"
        DEV_SIGNATURE=""
    fi
fi

set -e

export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export LIB_PREFIX="lib64"

BUILDDIR=${BUILDDIR:="${PWD}/build/rcclx"}
AMDGPU_TARGETS=${AMDGPU_TARGETS:="gfx942,gfx950"}

# Add b200 support if CUDA 12.8+ is available
CLEAN_BUILD=${CLEAN_BUILD:=0}
LIB_SUFFIX=${LIB_SUFFIX:-lib}
CONDA_INCLUDE_DIR="${CONDA_PREFIX}/include"
CONDA_LIB_DIR="${CONDA_PREFIX}/lib"
NCCL_HOME=${NCCL_HOME:="${PWD}/comms/rcclx/develop"}
BASE_DIR=${BASE_DIR:="${PWD}"}
THIRD_PARTY_LDFLAGS=""

if [[ -z "${NCCL_BUILD_SKIP_DEPS}" ]]; then
  echo "Building dependencies"
  build_third_party
  build_comms_tracing_service
fi

# set up the third-party ldflags
export PKG_CONFIG_PATH="${CONDA_LIB_DIR}"/pkgconfig
THRIFT_SERVICE_LDFLAGS=(
  "-l:libcomms_tracing_service.a"
  "-Wl,--start-group"
  "-l:libasync.a"
  "-l:libconcurrency.a"
  "-l:libthrift-core.a"
  "-l:libthriftanyrep.a"
  "-l:libthriftcpp2.a"
  "-l:libthriftmetadata.a"
  "-l:libthriftprotocol.a"
  "-l:libthrifttype.a"
  "-l:libthrifttyperep.a"
  "-l:librpcmetadata.a"
  "-l:libruntime.a"
  "-l:libserverdbginfo.a"
  "-l:libtransport.a"
  "-l:libcommon.a"
  "-Wl,--end-group"
  "-l:libwangle.a"
  "-l:libfizz.a"
  "-l:libxxhash.a"
)
THIRD_PARTY_LDFLAGS+="${THRIFT_SERVICE_LDFLAGS[*]} "
THIRD_PARTY_LDFLAGS+="$(pkg-config --libs --static libfolly | sed 's/-lfmt\b//g') "
if [[ -z "${USE_SYSTEM_LIBS}" ]]; then
  THIRD_PARTY_LDFLAGS+="-lglog -lgflags -l:libboost_context.a -l:libssl.a -l:libcrypto.a"
else
  THIRD_PARTY_LDFLAGS+="-lglog -lgflags -lboost_context -lssl -lcrypto"
fi

echo "$THIRD_PARTY_LDFLAGS"

export BUILDDIR
export NCCL_HOME
export BASE_DIR
export CONDA_INCLUDE_DIR
export CONDA_LIB_DIR
export THIRD_PARTY_LDFLAGS

# Generate nccl_cvars files (these are no longer checked into the repo)
# The files are generated by extractcvars.py which reads nccl_cvars.yaml and nccl_cvars.cc.in
echo "Generating nccl_cvars files..."
CVARS_DIR="$BASE_DIR/comms/utils/cvars"

# Validate that the required source files exist
if [ ! -f "$CVARS_DIR/extractcvars.py" ]; then
  echo "ERROR: extractcvars.py not found at $CVARS_DIR/extractcvars.py"
  exit 1
fi
if [ ! -f "$CVARS_DIR/nccl_cvars.yaml" ]; then
  echo "ERROR: nccl_cvars.yaml not found at $CVARS_DIR/nccl_cvars.yaml"
  exit 1
fi
if [ ! -f "$CVARS_DIR/nccl_cvars.cc.in" ]; then
  echo "ERROR: nccl_cvars.cc.in not found at $CVARS_DIR/nccl_cvars.cc.in"
  exit 1
fi

# Install pyyaml if not already installed (required by extractcvars.py)
if [[ -z "${NCCL_SKIP_CONDA_INSTALL}" ]]; then
  conda install pyyaml --yes
fi

# Run the extractcvars.py script directly to generate the files
export NCCL_CVARS_OUTPUT_DIR="$CVARS_DIR"
python3 "$CVARS_DIR/extractcvars.py"

# Verify the files were generated
if [ ! -f "$CVARS_DIR/nccl_cvars.h" ] || [ ! -f "$CVARS_DIR/nccl_cvars.cc" ]; then
  echo "ERROR: Failed to generate nccl_cvars files"
  exit 1
fi
echo "Successfully generated nccl_cvars files in $CVARS_DIR"

if [ "$CLEAN_BUILD" == 1 ]; then
    rm -rf "$BUILDDIR"
fi

# hipify-perl (ROCm 7.0) doesn't map cudaEventWait/Record flags, so they pass
# through unchanged into hipified source. Define them directly.
export CXXFLAGS="-DSO_INCOMING_NAPI_ID=56 -DcudaEventWaitDefault=0x00 -DcudaEventWaitExternal=0x01 -DcudaEventRecordDefault=0x00 -DcudaEventRecordExternal=0x01"

# Create linker version script to hide gflags/glog symbols from librccl.so.
# These get pulled in via static libs (libfolly.a, libthriftcpp2.a) but conflict
# at runtime with libgflags.so/libglog.so loaded by PyTorch.
cat > /tmp/rccl_hide_gflags.lds << 'LDSEOF'
{
  global: *;
  local:
    *gflags*;
    *google*CommandLine*;
    *google*FlagRegisterer*;
    *google*GetAllFlags*;
    *FLAGS_flagfile*;
    *fLS*FLAGS_*;
};
LDSEOF
export LDFLAGS="${LDFLAGS:-} -Wl,--version-script=/tmp/rccl_hide_gflags.lds"

mkdir -p "$BUILDDIR"
pushd "${NCCL_HOME}"

function build_rccl {
  ./install.sh \
    --prefix "$BUILDDIR" \
    --amdgpu_targets "$AMDGPU_TARGETS" \
    --disable-colltrace \
    --disable-msccl-kernel
}

build_rccl

popd
