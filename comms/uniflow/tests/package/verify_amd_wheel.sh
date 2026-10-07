#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rocm_path="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}"
hipcc="${rocm_path}/bin/hipcc"
if [[ ! -x "${hipcc}" ]]; then
  hipcc="$(command -v hipcc || true)"
  if [[ -n "${hipcc}" ]]; then
    rocm_path="$(cd -- "$(dirname -- "${hipcc}")/.." && pwd)"
  fi
fi
if [[ -z "${hipcc}" ]]; then
  echo "hipcc was not found; set ROCM_PATH to a ROCm SDK" >&2
  exit 1
fi

cmake_prefix_path="${rocm_path}"
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  cmake_prefix_path="${rocm_path}:${CMAKE_PREFIX_PATH}"
fi

CMAKE_PREFIX_PATH="${cmake_prefix_path}" \
  CXX="${hipcc}" \
  CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_PREFIX_PATH=${rocm_path} -DCMAKE_CXX_COMPILER=${hipcc}" \
  UNIFLOW_PACKAGE_DEPENDENCY_PREFIX_PATH="${rocm_path}" \
  UNIFLOW_PACKAGE_GPU_PLATFORM=HIP \
  exec "${script_dir}/verify_wheel.sh"
