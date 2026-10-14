/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hipipc_probe.h"

namespace {
constexpr int kProbeThreads = 64;
typedef unsigned int v4u __attribute__((ext_vector_type(4)));

__global__ void hipipcRelayProbeKernel(
    const unsigned char* const* addresses,
    int count,
    unsigned long long* out) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) {
    return;
  }
  // Match the relay's peer loads without retaining nonce canaries in cache.
  const v4u value =
      __builtin_nontemporal_load(reinterpret_cast<const v4u*>(addresses[i]));
  out[2 * i] = (static_cast<unsigned long long>(value[1]) << 32) | value[0];
  out[2 * i + 1] = (static_cast<unsigned long long>(value[3]) << 32) | value[2];
}
} // namespace

hipError_t hipipcRelayProbeRead(
    const unsigned char* const* addresses,
    int count,
    unsigned long long* out,
    hipStream_t stream) {
  if (addresses == nullptr || out == nullptr || count <= 0) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(
      hipipcRelayProbeKernel,
      dim3((count + kProbeThreads - 1) / kProbeThreads),
      dim3(kProbeThreads),
      0,
      stream,
      addresses,
      count,
      out);
  return hipPeekAtLastError();
}
