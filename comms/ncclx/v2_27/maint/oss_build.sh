#!/bin/bash
set -x
# Note: this script must be run from a conda environment
# in order to have the proper build tools and dependencies.

function usage {
  echo "You must run this script from inside a conda environment."
  echo "See: https://www.internalfb.com/intern/wiki/NCCLX/NCCLX_Developers_Runbook/#building-a-new-conda-pac"
  exit 1
}

function do_ninja() {
  if [ -n "$NINJA_JOBS" ]; then
    ninja -j"$NINJA_JOBS" "$@"
  else
    ninja "$@"
  fi
}

function do_cmake_build() {
  local source_dir="$1"
  cmake -G Ninja \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX="$CMAKE_PREFIX_PATH" \
    -DCMAKE_MODULE_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_INSTALL_DIR="$CMAKE_PREFIX_PATH" \
    -DBIN_INSTALL_DIR="$CMAKE_PREFIX_PATH/bin" \
    -DINCLUDE_INSTALL_DIR="$CMAKE_PREFIX_PATH/include" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=20 \
    "${source_dir}"
  do_ninja
  do_ninja install
}

function build_fb_oss_library() {
  local base_staging_dir="$1"
  local repo_dir="$2"
  local library_name="$3"

  # Make the open source directory structure:
  # library_staging/
  #    library/
  #     ... library contents
  #     ... public contents
  #     build/
  #
  local staging_dir="${base_staging_dir}/${library_name}_staging"
  mkdir -p "$staging_dir"
  pushd "$staging_dir"

  cp -r "${repo_dir}/${library_name}" "${library_name}"
  mkdir -p build
  cp -r "${FBCODE_DIR}/opensource/fbcode_builder" build/fbcode_builder

  if [ -d "${library_name}/public_root" ]; then
    cp -r ${library_name}/public_root/* .
  fi
  if [ -d "${library_name}/public_tld" ]; then
    cp -r ${library_name}/public_tld/* .
  fi

  local source_dir="../${library_name}"
  if [ -f CMakeLists.txt ]; then
    source_dir=".."
  fi
  echo $source_dir
  export LDFLAGS="-Wl,--allow-shlib-undefined"
  mkdir -p build-output
  cd build-output
  do_cmake_build "$source_dir"

  popd
}


function build_folly {
  # TODO: use a conda build recipe instead.
  # We can create a different conda environment, perform the build in there
  # and generate the statically-linked dependencies, then move the final
  # libnccl.so into the production conda environment.

  # dependencies needed by folly
  local folly_conda_deps=(
    boost
    double-conversion
    fmt==9.1.0
    jemalloc
    libevent
    libsodium
    liburing
    libunwind
    snappy
    conda-forge::fast_float
  )
  if [[ -z "${LIBDWARF_INSTALLED}" ]]; then
    folly_conda_deps+=( libdwarf-dev )
  fi

  if [ -z "$SKIP_CONDA_INSTALL" ]; then
    conda install -p "$CONDA_DIR" "${folly_conda_deps[@]}" --yes
  fi

  mkdir -p "$FOLLY_STAGING_DIR"/folly
  pushd "$FOLLY_STAGING_DIR/folly"

  export LD_LIBRARY_PATH="$CMAKE_PREFIX_PATH/lib"
  build_fb_oss_library $PWD $XPLAT_DIR folly
  popd
}

function build_thrift {
  # builds thrift and installs it into the cmake prefix path
  local thrift_conda_deps=(
    gflags
    gtest
    xxhash
    zstd
  )
  if [ -z "$SKIP_CONDA_INSTALL" ]; then
    conda install -p "$CONDA_DIR" "${thrift_conda_deps[@]}" --yes
  fi

  mkdir -p $THRIFT_STAGING_DIR
  pushd $THRIFT_STAGING_DIR

  export LD_LIBRARY_PATH="$CMAKE_PREFIX_PATH/lib"

  build_fb_oss_library $PWD $XPLAT_DIR fizz
  build_fb_oss_library $PWD $XPLAT_DIR quic
  build_fb_oss_library $PWD $XPLAT_DIR wangle
  build_fb_oss_library $PWD $XPLAT_DIR thrift
  popd
}

function build_comms_tracing_service {
  local include_prefix="comms/analyzer/if"
  mkdir -p "$COMMS_TRACING_SERVICE_STAGING_DIR"
  pushd "$COMMS_TRACING_SERVICE_STAGING_DIR"

  # set up the directory structure
  mkdir -p "$include_prefix"
  cp -r "$FBCODE_DIR/$include_prefix"/* "$include_prefix"
  mv "$include_prefix"/CMakeLists.txt .

  # set up the cmake config
  cp "$FBCODE_DIR"/opensource/fbcode_builder/CMake/*.cmake "$CMAKE_PREFIX_PATH"

  # build the thrift service library
  export LD_LIBRARY_PATH="$CMAKE_PREFIX_PATH/lib"
  mkdir -p build
  cd build
  do_cmake_build ..

  popd
}

# Allow bypassing conda environment check for Docker builds
if [[ -z "${SKIP_CONDA_ENV_CHECK}" ]]; then
  conda_env="${1:-${CONDA_PREFIX:-}}"
  if [[ -z "$conda_env" || ! -d "$conda_env" || ! -d "$conda_env/conda-meta" ]]; then
    usage
  fi
fi

if [ -z "$DEV_SIGNATURE" ]; then
    # guess default DEV_SIGNATURE
    is_git=$(git rev-parse --is-inside-work-tree)
    is_hg=$(hg id)
    if [ $is_git ]; then
        DEV_SIGNATURE="git-"$(git rev-parse --short HEAD)
    elif [ $is_hg ]; then
        DEV_SIGNATURE="hg-"$(hg id)
    else
        echo "Cannot detect source repository hash. Skip"
        DEV_SIGNATURE=""
    fi
fi

set -e

export CMAKE_PREFIX_PATH="$CONDA_DIR"

BUILDDIR=${BUILDDIR:=/tmp/${USER}/ncclx/2_27/build}
STAGINGDIR=${STAGINGDIR:=$(mktemp -d "/tmp/ncclx.staging_dir.$(date +%s).XXX")}
NVCC_ARCH=${NVCC_ARCH:="a100,h100"}
CUDA_HOME=${CUDA_HOME:="`realpath ../../../../../third-party/tp2/cuda/12.2.2/x86_64`"}
NCCL_FP8=${NCCL_FP8:=1}
NCCL_ENABLE_IN_TRAINER_TUNE=${NCCL_ENABLE_IN_TRAINER_TUNE:=0}
CLEAN_BUILD=${CLEAN_BUILD:=0}
FBCODE_DIR=${FBCODE_DIR:="`realpath ../../../../../fbcode`"}

FBSOURCE_DIR=${FBSOURCE_DIR:="/mnt/fbsource"}
if [[ -n "${NCCL_FBSOURCE_DIR}" ]]; then
  XPLAT_DIR="${NCCL_FBSOURCE_DIR}/xplat"
elif [[ -d "${FBSOURCE_DIR}" ]]; then
  XPLAT_DIR="${FBSOURCE_DIR}/xplat"
else
  echo "Error: Both NCCL_FBSOURCE_DIR and FBSOURCE_DIR are empty."
  exit 1
fi


NCCL_BUILD_SKIP_SANITY_CHECK=${NCCL_BUILD_SKIP_SANITY_CHECK:=0}
NCCL_PYTHON_PACKAGE_SKIP_SANITY_CHECK=${NCCL_PYTHON_PACKAGE_SKIP_SANITY_CHECK:=0}
FOLLY_STAGING_DIR="${STAGINGDIR}/folly"
THRIFT_STAGING_DIR="${STAGINGDIR}/thrift"
COMMS_TRACING_SERVICE_STAGING_DIR="${STAGINGDIR}/comms_tracing_service"
CONDA_INCLUDE_DIR="${CONDA_PREFIX}/include"
CONDA_LIB_DIR="${CONDA_PREFIX}/lib"
CONDA_BIN_DIR="${CONDA_PREFIX}/bin"
NCCL_HOOK_LIBS=${NCCL_HOOK_LIBS:=0}
CUDARTLIB=cudart_static
THIRD_PARTY_LDFLAGS=""

if [[ -z "${NVCC_BUILD_SKIP_FOLLY}" ]]; then
  build_folly
  build_thrift
  build_comms_tracing_service
fi

echo "BUILDDIR=${BUILDDIR}"
echo "NVCC_ARCH=${NVCC_ARCH}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "DEV_SIGNATURE=${DEV_SIGNATURE}"
echo "NCCL_FP8=${NCCL_FP8}"
echo "NCCL_ENABLE_IN_TRAINER_TUNE=${NCCL_ENABLE_IN_TRAINER_TUNE}"

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
  "-lcrypto"
  "-lssl"
  "-lxxhash"
)
THIRD_PARTY_LDFLAGS+="${THRIFT_SERVICE_LDFLAGS[*]} "
THIRD_PARTY_LDFLAGS+="$(pkg-config --libs --static libfolly) -lboost_context -lfmt -lgflags"

if [[ -z "${NVCC_GENCODE-}" ]]; then
    IFS=',' read -ra arch_array <<< "$NVCC_ARCH"
    arch_gencode=""
    for arch in "${arch_array[@]}"
    do
        case "$arch" in
        "p100")
        arch_gencode="$arch_gencode -gencode=arch=compute_60,code=sm_60"
            ;;
        "v100")
        arch_gencode="$arch_gencode -gencode=arch=compute_70,code=sm_70"
            ;;
        "a100")
        arch_gencode="$arch_gencode -gencode=arch=compute_80,code=sm_80"
        ;;
        "h100")
            arch_gencode="$arch_gencode -gencode=arch=compute_90,code=sm_90"
        ;;
        esac
    done
    NVCC_GENCODE=$arch_gencode
fi

echo "NVCC_GENCODE=${NVCC_GENCODE}"

export BUILDDIR=$BUILDDIR

if [ "$CLEAN_BUILD" == 1 ]; then
    rm -rf $BUILDDIR
fi

mkdir -p $BUILDDIR

# Generate nccl_cvars files (these are no longer checked into the repo)
# The files are generated by extractcvars.py which reads nccl_cvars.yaml and nccl_cvars.cc.in
echo "Generating nccl_cvars files..."
CVARS_DIR="$FBCODE_DIR/comms/utils/cvars"

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

# Install ruamel-yaml if not already installed (required by extractcvars.py)
if [ -z "$SKIP_CONDA_INSTALL" ]; then
  conda install -p "$CONDA_DIR" ruamel.yaml --yes
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

# Use nccl relative to fbcode dir (configurable for Docker builds)
export NCCL_HOME=${NCCL_HOME:-$FBCODE_DIR/comms/ncclx/v2_27}
pushd "${NCCL_HOME}"

# Build hook libraries. Used for fault injection.
if [ "$NCCL_HOOK_LIBS" == 1 ]; then
  make -C $FBCODE_DIR/ai_codesign/gen_ai/cpu_trainer/cpu_nccl_emulation/cuda_hook LIBSDIR=$BUILDDIR libcudart-static
  CUDARTLIB=cudarthook
fi

# Note: debug with: make SHELL="/bin/bash -x"
# build libnccl
make -j \
  src.build \
  NVCC_GENCODE="$NVCC_GENCODE" \
  CUDA_HOME="$CUDA_HOME" \
  NCCL_SUFFIX="x" \
  DEV_SIGNATURE="$DEV_SIGNATURE" \
  NCCL_FP8="$NCCL_FP8" \
  BASE_DIR="$FBCODE_DIR" \
  CONDA_INCLUDE_DIR="$CONDA_INCLUDE_DIR" \
  CONDA_LIB_DIR="$CONDA_LIB_DIR" \
  THIRD_PARTY_LDFLAGS="$THIRD_PARTY_LDFLAGS" \
  NCCL_ENABLE_IN_TRAINER_TUNE="$NCCL_ENABLE_IN_TRAINER_TUNE" \
  CUDARTLIB="$CUDARTLIB"

# Expose python API. Currently it depends on ncclx develop branch, so we ignore
# and continue if it fails.

if [ -z "$SKIP_CONDA_INSTALL" ]; then
  conda install -p "$CONDA_DIR" -c conda-forge pybind11==2.13.6 pybind11-global==2.13.6 --yes
  "$CONDA_BIN_DIR"/pip install --no-build-isolation .; pip_exit_code=$?
fi

if [ "$NCCL_PYTHON_PACKAGE_SKIP_SANITY_CHECK" != 1 ]; then
    export LD_LIBRARY_PATH=$BUILDDIR/lib

    if [ $pip_exit_code -eq 0 ]; then
        # some conda env may not have pytest installed
        "$CONDA_BIN_DIR"/pip install pytest
        "$CONDA_BIN_DIR"/pytest ./examples/trainer_context.py
    else
        echo "pip install ncclx_trainer_context failed, skipping tests"
    fi
fi

# sanity check
if [ "$NCCL_BUILD_SKIP_SANITY_CHECK" != 1 ]; then
    pushd examples
    export NCCL_DEBUG=WARN
    export LD_LIBRARY_PATH=$BUILDDIR/lib

    make all \
      NVCC_GENCODE="$NVCC_GENCODE" \
      CUDA_HOME="$CUDA_HOME" \
      DEV_SIGNATURE="$DEV_SIGNATURE" \
      FBCODE_DIR="$FBCODE_DIR" \
      CONDA_INCLUDE_DIR="$CONDA_INCLUDE_DIR" \
      CONDA_LIB_DIR="$CONDA_LIB_DIR" \
      NCCL_ENABLE_IN_TRAINER_TUNE="$NCCL_ENABLE_IN_TRAINER_TUNE"

    set +e

    TIMEOUT=10s
    timeout $TIMEOUT $BUILDDIR/examples/HelloWorld
    if [ "$?" == "124" ]; then
        echo "Program TIMEOUT in ${TIMEOUT}. Terminate."
    fi
    popd
fi

popd # pop NCCL_HOME
