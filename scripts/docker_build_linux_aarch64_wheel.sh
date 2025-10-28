#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script is used to build manylinux releases of torchcomms. The output from
# this script should be able to be used on most modern OSes as long as the
# Python version and CUDA version match.

set -ex

docker stop torchcomms || true
docker rm torchcomms || true

docker run --name torchcomms \
    --arch=aarch64 \
    --net=host \
    -i \
    -t \
    -v ".:/torchcomms" \
    pytorch/manylinuxaarch64-builder:cuda12.9 \
    bash /torchcomms/scripts/_emulate_build_wheel.sh
