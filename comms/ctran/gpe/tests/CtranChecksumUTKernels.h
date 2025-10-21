// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>

__global__ void DummyChecksumKernel(
    const uint8_t* __restrict__ in,
    const uint32_t size,
    uint32_t* __restrict__ out);
