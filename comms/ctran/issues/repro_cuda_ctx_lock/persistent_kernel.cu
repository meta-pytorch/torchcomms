// Copyright (c) Meta Platforms, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0

__global__ void persistentKernel(volatile int* shutdown) {
  while (*shutdown == 0) {
  }
}
