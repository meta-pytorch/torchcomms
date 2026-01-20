#!/bin/bash
set -x

# Parse command line arguments
AMDGPU_TARGETS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --amdgpu_targets)
      AMDGPU_TARGETS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --amdgpu_targets <targets>"
      exit 1
      ;;
  esac
done

# Set default value for amdgpu_targets if not provided
if [ -z "$AMDGPU_TARGETS" ]; then
  AMDGPU_TARGETS="gfx942;gfx950"
  echo "Using default amdgpu_targets: $AMDGPU_TARGETS"
fi

function do_cmake_build() {
  local source_dir="$1"
  local extra_flags="$2"
  cmake -G Ninja \
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
    "$extra_flags" \
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
  if [ -f "${library_name}/CMakeLists.txt" ]; then
    source_dir="../${library_name}"
  fi
  if [ -f "${library_name}/build/cmake/CMakeLists.txt" ]; then
    source_dir="../${library_name}/build/cmake"
  fi
  if [ -f "${library_name}/cmake_unofficial/CMakeLists.txt" ]; then
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

  make
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
  ./bootstrap.sh --prefix="$CMAKE_PREFIX_PATH" --libdir="$CMAKE_PREFIX_PATH/$LIB_SUFFIX" --without-libraries=python
  ./b2 -q cxxflags=-fPIC cflags=-fPIC install
  popd
}

function build_third_party {
  # build third-party libraries
  if [ "$CLEAN_THIRD_PARTY" == 1 ]; then
    rm -f "${CONDA_PREFIX}"/*.cmake 2>/dev/null || true
  fi
  local third_party_tag="v2025.12.15.00"

  mkdir -p /tmp/third-party
  pushd /tmp/third-party
  # TODO: Move other dependencies into system libs
  build_fb_oss_library "https://github.com/fmtlib/fmt.git" "11.2.0" fmt "-DFMT_INSTALL=ON -DFMT_TEST=OFF -DFMT_DOC=OFF"
  build_fb_oss_library "https://github.com/fmtlib/fmt.git" "11.2.0" fmt "-DFMT_INSTALL=ON -DFMT_TEST=OFF -DFMT_DOC=OFF -DBUILD_SHARED_LIBS=ON"
  if [[ -z "${USE_SYSTEM_LIBS}" ]]; then
    build_fb_oss_library "https://github.com/madler/zlib.git" "v1.2.13" zlib "-DZLIB_BUILD_TESTING=OFF"
    build_boost
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
    build_fb_oss_library "https://github.com/libevent/libevent.git" "release-2.1.12-stable" event
    build_fb_oss_library "https://github.com/google/double-conversion.git" "v3.3.1" double-conversion
    # Enable SSE4.2 for F14 CRC intrinsics to avoid link errors with F14LinkCheck
    build_fb_oss_library "https://github.com/facebook/folly.git" "$third_party_tag" folly "-DUSE_STATIC_DEPS_ON_UNIX=ON -DCMAKE_CXX_FLAGS=-msse4.2"
  else
    DEPS=(
      boost
      double-conversion
      libevent
      conda-forge::libsodium
      libunwind
      snappy
      conda-forge::fast_float
      libdwarf-dev
      gflags
      xxhash
      zstd
      conda-forge::zlib
      fmt
      glog==0.4.0
    )
    conda install "${DEPS[@]}" --yes
    # Enable SSE4.2 for F14 CRC intrinsics to avoid link errors with F14LinkCheck
    build_fb_oss_library "https://github.com/facebook/folly.git" "$third_party_tag" folly "-DCMAKE_CXX_FLAGS=-msse4.2"
  fi
  build_fb_oss_library "https://github.com/facebookincubator/fizz.git" "$third_party_tag" fizz "-DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF"
  build_fb_oss_library "https://github.com/facebook/mvfst" "$third_party_tag" quic
  build_fb_oss_library "https://github.com/facebook/wangle.git" "$third_party_tag" wangle "-DBUILD_TESTS=OFF"
  build_fb_oss_library "https://github.com/facebook/fbthrift.git" "$third_party_tag" thrift
  popd
}

if [ -z "$DEV_SIGNATURE" ]; then
    is_git=$(git rev-parse --is-inside-work-tree)
    if [ "$is_git" ]; then
        DEV_SIGNATURE="git-"$(git rev-parse --short HEAD)
    else
        echo "Cannot detect source repository hash. Skip"
        DEV_SIGNATURE=""
    fi
fi

set -e

export CMAKE_PREFIX_PATH="$CONDA_PREFIX"

BUILDDIR=${BUILDDIR:="${PWD}/build"}
CLEAN_BUILD=${CLEAN_BUILD:=0}
LIB_SUFFIX=${LIB_SUFFIX:-lib}
NCCL_HOME=${NCCL_HOME:="${PWD}/comms/rcclx/develop"}
BASE_DIR=${BASE_DIR:="${PWD}"}

if [[ -z "${NCCL_BUILD_SKIP_DEPS}" ]]; then
  echo "Building dependencies"
  if [[ -z "${NCCL_SKIP_CONDA_INSTALL}" ]]; then
    DEPS=(
      cmake=3.26.4
      ninja
      jemalloc
      gtest
    )
    conda install "${DEPS[@]}" --yes
  fi
  build_third_party
fi

if [ "$CLEAN_BUILD" == 1 ]; then
    rm -rf "$BUILDDIR"
fi

mkdir -p "$BUILDDIR"
pushd "${NCCL_HOME}"

./install.sh \
    --prefix build \
    --amdgpu_targets "$AMDGPU_TARGETS" \
    --disable-colltrace \
    --disable-msccl-kernel \
    -j 16

popd
