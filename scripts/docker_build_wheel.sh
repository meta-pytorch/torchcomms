#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script is used to build manylinux releases of torchcomms. The output from
# this script should be able to be used on most modern OSes as long as the
# Python version and CUDA version match.

set -ex

docker stop torchcomms || true
docker rm torchcomms || true

# SCCACHE_DIR turns on the local-disk compiler cache inside the container. The
# host directory gets its own mount at a fixed container path, so the cache
# survives the `docker rm` above wherever the caller put it (that is what makes a
# second run a warm-cache measurement); forwarding the host path alone would
# leave a cache outside `.` living and dying inside the container.
docker_args=()
if [ -n "${SCCACHE_DIR:-}" ]; then
    mkdir -p "$SCCACHE_DIR"
    docker_args+=(-v "$SCCACHE_DIR:/sccache" -e "SCCACHE_DIR=/sccache")
fi
# Forward an explicitly configured S3 backend so the shared cache can be
# exercised locally. Unset by default, so local runs stay uncached; without
# this, an exported SCCACHE_BUCKET would silently do nothing in the container.
if [ -n "${SCCACHE_BUCKET:-}" ]; then
    docker_args+=(-e "SCCACHE_BUCKET=$SCCACHE_BUCKET")
fi
if [ -n "${SCCACHE_REGION:-}" ]; then
    docker_args+=(-e "SCCACHE_REGION=$SCCACHE_REGION")
fi
if [ -n "${SCCACHE_S3_KEY_PREFIX:-}" ]; then
    docker_args+=(-e "SCCACHE_S3_KEY_PREFIX=$SCCACHE_S3_KEY_PREFIX")
fi
# Forward the kill switch so a local run can be forced uncached the same way.
if [ -n "${TORCHCOMMS_SCCACHE:-}" ]; then
    docker_args+=(-e "TORCHCOMMS_SCCACHE=$TORCHCOMMS_SCCACHE")
fi

docker run --name torchcomms \
    --net=host \
    "${docker_args[@]}" \
    -i \
    -t \
    -v ".:/torchcomms" \
    pytorch/manylinux2_28-builder:cuda13.2-main \
    bash /torchcomms/scripts/_emulate_build_wheel.sh
