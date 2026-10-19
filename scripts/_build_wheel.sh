#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs within the docker container invoked by docker_build_wheel.sh .
# Run that script instead.

set -ex

dnf config-manager --set-enabled powertools
dnf install -y almalinux-release-devel
dnf install -y ninja-build cmake

# Nuke conda cmake, ninja and libstdc++ we want to install to use system libraries.
rm -f "$CONDA_PREFIX/lib/libstdc"* || true
conda remove -y cmake ninja || true
rm -f "$CONDA_PREFIX/bin/ninja" || true
rm -f "$CONDA_PREFIX/bin/cmake" || true
rm -f "/opt/conda/bin/ninja" || true
rm -f "/opt/conda/bin/cmake" || true

python --version
which python

pip install -r requirements.txt
# pyyaml is a build-time-only dep (used by extractcvars.py codegen, run from
# CMake); it is intentionally not in requirements.txt/install_requires so the
# runtime wheel resolves from the PyTorch index alone.
pip install pyyaml

export NCCL_SKIP_CONDA_INSTALL=1
export CLEAN_BUILD=1
# Match the NCCLX feedstock and TorchComms iter build. NCCLX headers require
# C++20, and fmt needs its NVCC C++20 compatibility patch applied after the
# environment's dependency installation.
export CXXSTD="-std=c++20"
export NCCL_PATCH_FMT_NVCC_CXX20=1
# CUDA device compilation can exhaust memory at full host parallelism. The
# NCCLX build runs during CMake configuration, before the top-level build, so
# cap both sequential phases independently. Four-way parallelism makes the
# AArch64 wheel builds exceed their two-hour timeout.
export CMAKE_BUILD_PARALLEL_LEVEL=8
export NCCL_BUILD_JOBS=8

# sccache: shared compiler cache for the NCCLX device phase, which is ~45 min
# of an ~85 min build (863 nvcc TUs at 8-way). Objects depend on toolkit +
# source only, not on the Python version, so one cache serves all 28 Python
# legs of a CUDA arch.
#
# Enabled only when a backend is configured: SCCACHE_BUCKET for the shared S3
# cache, or SCCACHE_DIR for a local disk cache (the docker_build_wheel.sh
# validation loop). With neither, everything below is skipped and the build is
# byte-identical to an uncached one. TORCHCOMMS_SCCACHE=0 forces it off.
# No failure path here may red a build.
SCCACHE_VERSION="v0.17.0"
# Default S3 backend: the shared PyTorch CI compiler cache, same
# bucket/prefix/region as the CMake builds (PR #4147). Defaulted only under
# GitHub Actions: this script also runs outside CI (the docker_build_wheel.sh
# loop, possibly local runs), where an enabled-but-credentialless sccache
# would build cold while logging a write error per translation unit. An
# explicit SCCACHE_BUCKET in the environment always wins; SCCACHE_DIR selects
# a local disk cache instead.
if [[ -z "${SCCACHE_BUCKET:-}" && -n "${GITHUB_ACTIONS:-}" ]]; then
  SCCACHE_BUCKET="ossci-compiler-cache-circleci-v2"
fi
SCCACHE_BUCKET="${SCCACHE_BUCKET:-}"
# Written only when a cache is actually in use. The post-script does not inherit
# this shell's environment, so this file is how it tells "this build owns an
# sccache server" from "some sccache binary happens to be on PATH".
SCCACHE_MARKER="${GITHUB_WORKSPACE:-$PWD}/.sccache-enabled"
# A reused workspace can carry one over from a run that had a cache when this one
# does not.
rm -f "$SCCACHE_MARKER"

install_sccache() {
  local arch tarball url dir digest rc=0
  arch=$(uname -m)
  case "$arch" in
    x86_64 | aarch64) ;;
    *)
      echo "sccache: no release binary for $arch" >&2
      return 1
      ;;
  esac
  tarball="sccache-${SCCACHE_VERSION}-${arch}-unknown-linux-musl.tar.gz"
  url="https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${tarball}"
  dir=$(mktemp -d)
  # Cleaned up explicitly rather than with `trap ... RETURN`: traps are
  # shell-global, so a RETURN trap set here would survive this function and fire
  # again on later returns with $dir already out of scope.
  {
    curl -fsSL "$url" -o "$dir/$tarball" &&
      # Integrity, not authenticity: the digest comes from the same origin as
      # the binary, so this catches a truncated or half-served download only.
      # Every published asset has a .sha256 sibling, so a missing one means the
      # release is not what this script expects — refuse the binary rather than
      # guess.
      curl -fsSL "${url}.sha256" -o "$dir/digest" &&
      digest=$(awk '{print $1}' "$dir/digest") &&
      echo "${digest}  $dir/$tarball" | sha256sum -c - &&
      tar -xzf "$dir/$tarball" -C "$dir" &&
      install -m 0755 \
        "$dir/sccache-${SCCACHE_VERSION}-${arch}-unknown-linux-musl/sccache" \
        /usr/local/bin/sccache
  } || rc=1
  rm -rf "$dir"
  return "$rc"
}

setup_sccache() {
  command -v sccache > /dev/null || install_sccache || return 1

  # One server for the whole build, so post-script stats are cumulative.
  export SCCACHE_IDLE_TIMEOUT=0
  # The error log is only written when a log level is configured, and it is the
  # only diagnostic the post-script has on the failure paths.
  export SCCACHE_LOG="${SCCACHE_LOG:-warn}"
  export SCCACHE_ERROR_LOG="${GITHUB_WORKSPACE:-$PWD}/sccache-error.log"
  # Device TUs carry several -gencode pairs; without this they are treated as
  # non-cacheable.
  export SCCACHE_CACHE_MULTIARCH=1
  # sccache normalizes paths against a single directory, spelled SCCACHE_BASEDIR;
  # the plural is ccache's and would be silently ignored. The conda prefix is the
  # root that moves between runs — it embeds the CI run id
  # (conda_environment_<runid>) and shows up in -I paths, which would make every
  # cross-run lookup miss — so that is the one worth normalizing away.
  export SCCACHE_BASEDIR="${CONDA_PREFIX:-${GITHUB_WORKSPACE:-$PWD}}"
  if [[ -n "$SCCACHE_BUCKET" ]]; then
    export SCCACHE_BUCKET
    export SCCACHE_REGION="${SCCACHE_REGION:-us-east-1}"
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-torchcomms}"
    export AWS_REGION="${AWS_REGION:-us-east-1}"
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
    # Without credentials every compile logs a write error. A PR from a fork
    # gets no OIDC token and the runner credentials may not reach the
    # container, so read the cache anonymously in that case rather than
    # failing — matching the CMake builds (PR #4147) and test-infra's
    # linux_job_v3.yml for the same bucket.
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
      echo "sccache: no AWS credentials -- read-only (anonymous) cache" >&2
      export SCCACHE_S3_NO_CREDENTIALS=true
    fi
  fi

  # Prove the backend answers before wiring it into any compiler: a server that
  # cannot reach its bucket must cost a cold build, not the build.
  #
  # A server left behind by an earlier leg on a reused runner never exits
  # (SCCACHE_IDLE_TIMEOUT=0) and makes --start-server report failure, which is
  # not a reason to drop to an uncached build — probe for a live server before
  # giving up. Such a server keeps the configuration it was started with, so the
  # exports above then describe this shell's compilers only, not the cache it
  # talks to; the stats printed here are what says which one is in play.
  if ! sccache --start-server; then
    sccache --show-stats > /dev/null 2>&1 || return 1
    echo "sccache: reusing the server already running on this host" >&2
  fi
  # --start-server and --show-stats never touch S3, so neither proves the
  # bucket is reachable: a wrong bucket or missing credentials would surface
  # only as write errors in the post-script stats — fail-open, but silently
  # cold. Compile one TU through the server instead; that exercises
  # credentials, bucket policy and region end to end (same pattern as PR
  # #4147). A failed probe costs a cold build, never the build.
  local probe="${TMPDIR:-/tmp}/sccache_probe.c"
  echo 'int main(void){return 0;}' > "${probe}" || return 1
  if ! sccache cc -c "${probe}" -o "${probe}.o" >/dev/null 2>&1; then
    echo "sccache: probe compile failed -- building uncached (see ${SCCACHE_ERROR_LOG})" >&2
    sccache --stop-server >/dev/null 2>&1 || true
    rm -f "${probe}" "${probe}.o"
    return 1
  fi
  rm -f "${probe}" "${probe}.o"
  # Zero the counters after the probe so the post-script stats measure the
  # build, not the probe (and not a previous leg's adopted server).
  sccache --zero-stats >/dev/null 2>&1 || true
  sccache --show-stats
}

# Kill switch: TORCHCOMMS_SCCACHE=0 forces an uncached build without a code
# change, so a misbehaving cache rolls back from the workflow dispatch inputs.
# Same convention as the CMake builds.
if [[ "${TORCHCOMMS_SCCACHE:-1}" != "1" ]]; then
  echo "sccache: disabled by TORCHCOMMS_SCCACHE=${TORCHCOMMS_SCCACHE} -- building uncached" >&2
elif [[ -n "$SCCACHE_BUCKET" || -n "${SCCACHE_DIR:-}" ]]; then
  if setup_sccache; then
    export USE_SCCACHE=1
    : > "$SCCACHE_MARKER" || true
    # NCCLX's make takes NVCC from the environment
    # (`NVCC ?= $(CUDA_HOME)/bin/nvcc` in makefiles/common.mk).
    export NVCC="sccache ${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
    # The host compiler is deliberately left unwrapped: NVCUFLAGS passes
    # `-ccbin $(CXX)`, which takes one binary, so CXX="sccache g++" would break
    # every nvcc invocation. The device phase is nvcc-bound regardless.
  else
    echo "sccache setup failed, building uncached" >&2
  fi
fi

python setup.py bdist_wheel
