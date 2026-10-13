#!/usr/bin/env bash
# Builds and installs the commsutils libraries -- libcommsutils.so.1
# (comms/utils) and libobservatory.so.1 (comms/observatory) -- from
# comms/utils/Makefile into $PREFIX/lib. This is the one place that derives the
# library's third-party link line; the conda feedstock, the wheel build and
# build_ncclx.sh's dependency step all run it.
#
# Required env:
#   PREFIX        - install prefix; the libraries land in $PREFIX/lib
# Optional env:
#   BASE_DIR      - directory containing comms/ (default: $PWD)
#   CONDA_PREFIX  - where fmt/folly/abseil/glog/gflags live (default: $PREFIX)
#   USE_SYSTEM_LIBS - set when the third-party libraries are the environment's
#                   shared ones (conda); unset when they were built from source
#                   as static archives (build_ncclx.sh's OSS path)
#   BUILDDIR      - object/library output tree (default: $PWD/build/commsutils)
#   CUDA_HOME (/usr/local/cuda), NVCC_ARCH (a100,h100; b200 appended on
#   CUDA >= 12.8) or NVCC_GENCODE, CUDARTLIB (cudart_static), CXXSTD, CXXFLAGS,
#   PYTHON (python3; needs pyyaml for cvars/extractcvars.py)
set -euo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=nvcc_gencode.sh
. "${SELF_DIR}/nvcc_gencode.sh"

: "${PREFIX:?PREFIX must name the install prefix}"
BASE_DIR="${BASE_DIR:-$PWD}"
CONDA_PREFIX="${CONDA_PREFIX:-$PREFIX}"
BUILDDIR="${BUILDDIR:-$PWD/build/commsutils}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDARTLIB="${CUDARTLIB:-cudart_static}"
CONDA_INCLUDE_DIR="${CONDA_PREFIX}/include"
CONDA_LIB_DIR="${CONDA_PREFIX}/lib"

if [ ! -f "${BASE_DIR}/comms/utils/Makefile" ]; then
  echo "BASE_DIR does not contain comms/utils/Makefile: ${BASE_DIR}" >&2
  exit 2
fi

if [[ -z "${NVCC_GENCODE-}" ]]; then
  NVCC_ARCH="${NVCC_ARCH:-a100,h100}"
  NVCC_ARCH=$(nvcc_arch_with_b200 "${NVCC_ARCH}" "${CUDA_HOME}")
  NVCC_GENCODE=$(nvcc_gencode_from_arch "${NVCC_ARCH}")
fi

# The same third-party set libnccl links, except that gflags and glog are
# static archives hidden inside libcommsutils (--exclude-libs). Both register
# global state at load: libnccl and libcommsutils each carry their own static
# folly, and two folly copies defining the same flags against one shared
# libgflags abort at load ("linked both statically and dynamically"). Every
# environment this runs in provides libgflags.a/libglog.a beside the shared
# ones: conda's folly package installs them, and build_ncclx.sh's from-source
# path builds them.
export PKG_CONFIG_PATH="${CONDA_LIB_DIR}/pkgconfig"
THIRD_PARTY_LDFLAGS="$(pkg-config --libs --static libfolly) "
THIRD_PARTY_LDFLAGS+="$(pkg-config --libs --static absl_log absl_check) "
# liburing: folly's cmake config declares this dependency but pkg-config omits
# it. Add -luring if the system has liburing (folly uses io_uring for async I/O).
if pkg-config --exists liburing 2>/dev/null; then
  THIRD_PARTY_LDFLAGS+="-luring "
fi
# libfmt: comms/utils compiles with FMT_HEADER_ONLY=1, but the prebuilt folly
# static libs were built without it and reference non-inline fmt symbols; the
# conda libfolly.pc does not list fmt as a dep, so name it explicitly.
if [[ -z "${USE_SYSTEM_LIBS:-}" ]]; then
  THIRD_PARTY_LDFLAGS+="-l:libboost_context.a -l:libssl.a -l:libcrypto.a -l:libfmt.a "
else
  THIRD_PARTY_LDFLAGS+="-lboost_context -lssl -lcrypto -lfmt "
fi
THIRD_PARTY_LDFLAGS="$(echo "${THIRD_PARTY_LDFLAGS}" | sed -E 's/(^| )-l(glog|gflags)\b/\1/g') -l:libglog.a -l:libgflags.a"
echo "THIRD_PARTY_LDFLAGS=${THIRD_PARTY_LDFLAGS}"

make -C "${BASE_DIR}/comms/utils" -j"$(nproc)" install \
  BASE_DIR="${BASE_DIR}" \
  BUILDDIR="${BUILDDIR}" \
  PREFIX="${PREFIX}" \
  CUDA_HOME="${CUDA_HOME}" \
  NVCC_GENCODE="${NVCC_GENCODE}" \
  CUDARTLIB="${CUDARTLIB}" \
  CONDA_INCLUDE_DIR="${CONDA_INCLUDE_DIR}" \
  CONDA_LIB_DIR="${CONDA_LIB_DIR}" \
  THIRD_PARTY_LDFLAGS="${THIRD_PARTY_LDFLAGS}" \
  ${CXXSTD:+CXXSTD="${CXXSTD}"} \
  ${PYTHON:+PYTHON="${PYTHON}"}

# The static contract above, checked on the artifact rather than trusted. The
# shared dependencies resolve from the build env's lib dir here; whoever
# packages the library (conda: same prefix; wheel: bundled beside it) gives it
# its runtime search path.
for lib in libcommsutils.so.1 libobservatory.so.1; do
  path="${PREFIX}/lib/${lib}"
  [ -f "${path}" ] || { echo "ERROR: ${path} was not installed" >&2; exit 1; }
  if LD_LIBRARY_PATH="${CONDA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ldd "${path}" | grep -q 'not found'; then
    echo "ERROR: ${path} has unresolved dependencies:" >&2
    LD_LIBRARY_PATH="${CONDA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ldd "${path}" | grep 'not found' >&2
    exit 1
  fi
done
if readelf -d "${PREFIX}/lib/libcommsutils.so.1" | grep -qE 'NEEDED.*lib(glog|gflags)\.so'; then
  echo "ERROR: libcommsutils.so.1 links shared glog/gflags; they must be static inside it" >&2
  exit 1
fi
# Nothing from the third-party namespaces may be exported, or a consumer with
# its own copy binds to it (comms/utils/libcommsutils.map). The two folly
# specializations for CommLogData are comms/utils' own and stay exported.
THIRD_PARTY_EXPORT_RE='^_Z([GT][VRISTC])?Z{0,2}NK?(5folly|3fmt|6spdlog|6google|4absl|5boost|3fLB|3fLS|3fLI|3fLU|3fLD|4fL64)|^_ZT[hvc][^_]*_N(5folly|3fmt|6spdlog|6google|4absl|5boost)'
for lib in libcommsutils.so.1 libobservatory.so.1; do
  leaked=$(nm -D --defined-only "${PREFIX}/lib/${lib}" | awk '{print $3}' \
    | grep -E "${THIRD_PARTY_EXPORT_RE}" \
    | grep -vE '^_ZN5folly1[68]Dynamic(Converter|Constructor)I11CommLogData' || true)
  if [ -n "${leaked}" ]; then
    echo "ERROR: ${lib} exports third-party symbols (the export map must hide them):" >&2
    echo "${leaked}" | head -20 >&2
    exit 1
  fi
done
# The C++ runtime's exception entry points must come from libstdc++, not from
# a folly exception-tracer copy linked in as a local interposer (its RTLD_NEXT
# lookup fails under Python, where libstdc++ is already loaded, and the first
# throw in the library crashes the process). nm reads .symtab, so this runs
# before any stripping.
for lib in libcommsutils.so.1 libobservatory.so.1; do
  interposed=$(nm "${PREFIX}/lib/${lib}" 2>/dev/null | awk '$2 ~ /^[TtWw]$/ && $3 ~ /^__cxa_(throw|rethrow|begin_catch|end_catch)$/ {print $3}')
  if [ -n "${interposed}" ]; then
    echo "ERROR: ${lib} defines ${interposed//$'\n'/ }; link libstdc++ ahead of the folly archive" >&2
    exit 1
  fi
done
echo "commsutils installed into ${PREFIX}/lib"
