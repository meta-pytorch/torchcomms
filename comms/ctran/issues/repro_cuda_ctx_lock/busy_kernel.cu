// Copyright (c) Meta Platforms, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0

__global__ void busyKernel(long long ticks) {
  long long s = clock64();
  while (clock64() - s < ticks) {
  }
}
