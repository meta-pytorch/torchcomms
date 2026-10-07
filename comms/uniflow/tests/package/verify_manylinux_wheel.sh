#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIFLOW_EXPECT_MANYLINUX=manylinux_2_28 \
  UNIFLOW_PACKAGE_AUDITWHEEL=ON \
  exec "${script_dir}/verify_wheel.sh"
