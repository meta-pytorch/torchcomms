#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -x

# Set default ROCM_HOME if not already set
if [ -z "$ROCM_HOME" ]; then
  export ROCM_HOME="/opt/rocm"
  echo "ROCM_HOME not set, using default: $ROCM_HOME"
else
  echo "Using ROCM_HOME: $ROCM_HOME"
fi

DEFAULT_BRANCH="rocm-7.0.0"
FORCE_BUILD=false
CUSTOM_BRANCH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --custom-branch)
      CUSTOM_BRANCH="$2"
      FORCE_BUILD=true
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--custom-branch <branch_name>]"
      echo ""
      echo "Options:"
      echo "  --custom-branch <branch_name>  The branch of oss rccl that you want to install. This will ensure the system library is not used."
      exit 1
      ;;
  esac
done

BRANCH_TO_USE="${CUSTOM_BRANCH:-$DEFAULT_BRANCH}"

function build_rccl_oss_library() {
  local repo_url="$1"
  local repo_tag="$2"
  local library_name="$3"

  rm -rf "$library_name"
  git clone -b "$repo_tag" "$repo_url" "$library_name"

  rm -rf build-output
  mkdir -p build-output
  cd "$library_name" || exit 1

  # We want to disable mscclpp as it is not needed for torchcomms_rccl
  # rocm-7.0.0 branch has an option to disable mscclpp (--disable-mscclpp)
  # develop branch does not have this option, since mscclpp is disabled by default
  if ./install.sh --help 2>&1 | grep -q "\-\-disable-mscclpp"; then
    echo "Using --disable-mscclpp option (found in install.sh)"
    ./install.sh --disable-mscclpp --fast --disable-msccl-kernel
  else
    echo "Using --fast --disable-msccl-kernel options (--disable-mscclpp not found in install.sh)"
    ./install.sh --install --prefix /lib64/rccl  --fast --disable-msccl-kernel
  fi

  cd .. || exit 1
}

# Use system library if available
if [ "$FORCE_BUILD" = true ] || [ ! -f "$ROCM_HOME/lib/librccl.so" ]; then
  if [ "$FORCE_BUILD" = true ]; then
    echo "Force build enabled, building OSS library with branch: $BRANCH_TO_USE"
  else
    echo "librccl.so not found in $ROCM_HOME/lib, building OSS library with branch: $BRANCH_TO_USE"
  fi
  build_rccl_oss_library "https://github.com/ROCm/rccl.git" "$BRANCH_TO_USE" "rccl"
  export RCCL_HOME=/lib64/rccl/lib
else
  echo "librccl.so found in $ROCM_HOME/lib, skipping OSS library build (use --custom-branch to override)"
  export RCCL_HOME=$ROCM_HOME/lib
fi

popd || exit 1
# TODO build torchcomms_rccl
