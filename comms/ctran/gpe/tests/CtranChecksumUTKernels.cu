// Copyright (c) Meta Platforms, Inc. and affiliates.

// FIXME [REBASE]: update the path once moved to fbcode/comms
#include "comms/ctran/gpe/tests/CtranChecksumUTKernels.h"

__global__ void DummyChecksumKernel(
    const uint8_t* __restrict__ in,
    const uint32_t size,
    uint32_t* __restrict__ out) {
  *out = size;
}
