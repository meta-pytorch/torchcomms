// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/Checksum.cuh"

#define DECL_CHECKSUM_KERN(thread)                 \
  template __global__ void checksumKernel<thread>( \
      const uint8_t* __restrict__ in,              \
      const uint32_t size,                         \
      uint32_t* __restrict__ out);

DECL_CHECKSUM_KERN(1024);
DECL_CHECKSUM_KERN(512);
