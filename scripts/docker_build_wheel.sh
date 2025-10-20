#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script is used to build manylinux releases of torchcomms. The output from
# this script should be able to be used on most modern OSes as long as the
# Python version and CUDA version match.

set -ex

docker run --name torchcomms \
    --net=host \
    -i \
    -t \
    -v ".:/torchcomms" \
    pytorch/manylinux2_28-builder:cuda12.8-main \
    bash /torchcomms/scripts/_emulate_build_wheel.sh
