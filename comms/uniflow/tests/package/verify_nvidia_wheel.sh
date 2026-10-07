#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIFLOW_PACKAGE_GPU_PLATFORM=CUDA exec "${script_dir}/verify_wheel.sh"
