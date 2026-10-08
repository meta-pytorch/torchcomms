#!/bin/bash
# Set up sccache against the shared PyTorch CI compiler cache.
#
# Sourced (not executed) from setup_env.sh so the exports reach the build.
#
# Scope: the CMake/Ninja builds only -- the third-party libraries built by
# build_ncclx.sh and the C++ test build. CMake picks up the *_COMPILER_LAUNCHER
# environment variables on its own, so no build files change.
#
# Deliberately NOT done here:
#   * PATH compiler stubs (as pytorch/executorch use). NCCLX's common.mk sets
#     `NVCUFLAGS := -ccbin $(CXX)`, and $(CXX) resolves through PATH, so a stub
#     would hand nvcc an sccache wrapper as its host compiler. Stubs would also
#     cover boost/openssl/libsodium, which are autotools/b2 and are missed here.
#   * nvcc caching. common.mk has `NVCC ?= $(CUDA_HOME)/bin/nvcc`, so an
#     exported NVCC would override it, but multi-arch -gencode is the least
#     reliable sccache case. Measure the hit rate below before adding it.
#
# Opt out with TORCHCOMMS_SCCACHE=0.

SCCACHE_INSTALL_DIR="${SCCACHE_INSTALL_DIR:-/opt/cache/bin}"
SCCACHE_RELEASE="${SCCACHE_RELEASE:-v0.13.0}"

sccache_disable() {
  echo "sccache: disabled ($1) -- building uncached"
  unset SCCACHE_BUCKET SCCACHE_S3_KEY_PREFIX SCCACHE_REGION
  unset CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER
}

setup_sccache() {
  if [ "${TORCHCOMMS_SCCACHE:-1}" != "1" ]; then
    sccache_disable "TORCHCOMMS_SCCACHE=0"
    return 0
  fi

  local arch
  case "$(uname -m)" in
    x86_64)  arch=x86_64 ;;
    aarch64) arch=aarch64 ;;
    *)       sccache_disable "unsupported arch $(uname -m)"; return 0 ;;
  esac

  if ! command -v sccache >/dev/null 2>&1; then
    local tarball="sccache-${SCCACHE_RELEASE}-${arch}-unknown-linux-musl"
    if ! mkdir -p "${SCCACHE_INSTALL_DIR}" 2>/dev/null; then
      sccache_disable "cannot create ${SCCACHE_INSTALL_DIR}"
      return 0
    fi
    if ! curl -fsSL --retry 3 \
        "https://github.com/mozilla/sccache/releases/download/${SCCACHE_RELEASE}/${tarball}.tar.gz" \
        | tar xz -C "${SCCACHE_INSTALL_DIR}" --strip-components=1 "${tarball}/sccache"; then
      sccache_disable "download failed"
      return 0
    fi
    chmod a+x "${SCCACHE_INSTALL_DIR}/sccache"
    export PATH="${SCCACHE_INSTALL_DIR}:${PATH}"
  fi

  export SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2
  export SCCACHE_S3_KEY_PREFIX=torchcomms
  export SCCACHE_REGION=us-east-1
  export AWS_REGION="${AWS_REGION:-us-east-1}"
  export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
  # The build outlives sccache's default idle timeout; a restart mid-build
  # loses the in-memory stats we report at the end.
  export SCCACHE_IDLE_TIMEOUT=0
  export SCCACHE_ERROR_LOG=/tmp/sccache_error.log
  export RUST_LOG=sccache::server=error

  # A PR from a fork gets no OIDC token and the runner credentials may not
  # reach the container, so writes would fail on every compile. Read the cache
  # anonymously in that case rather than failing, matching what
  # test-infra's linux_job_v3.yml does for the same bucket.
  if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "sccache: no AWS credentials -- read-only (anonymous) cache"
    export SCCACHE_S3_NO_CREDENTIALS=true
  fi

  sccache --stop-server >/dev/null 2>&1 || true
  rm -f "${SCCACHE_ERROR_LOG}" || true
  if ! sccache --start-server >/dev/null 2>&1; then
    sccache_disable "server failed to start"
    return 0
  fi

  # Prove the bucket is actually reachable before pointing the build at it.
  local probe="${TMPDIR:-/tmp}/sccache_probe.c"
  echo 'int main(void){return 0;}' > "${probe}"
  if ! sccache cc -c "${probe}" -o "${probe}.o" >/dev/null 2>&1; then
    echo "sccache: probe compile failed; see ${SCCACHE_ERROR_LOG}"
    sccache --stop-server >/dev/null 2>&1 || true
    sccache_disable "probe failed"
    rm -f "${probe}" "${probe}.o"
    return 0
  fi
  rm -f "${probe}" "${probe}.o"

  # CMake reads these itself, so build_ncclx.sh and the test build are covered
  # without touching any CMakeLists.
  export CMAKE_C_COMPILER_LAUNCHER=sccache
  export CMAKE_CXX_COMPILER_LAUNCHER=sccache

  sccache --zero-stats >/dev/null 2>&1 || true
  echo "sccache: enabled -- s3://${SCCACHE_BUCKET}, prefix ${SCCACHE_S3_KEY_PREFIX}"
}

setup_sccache
