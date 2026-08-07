// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

namespace meta::comms::stagedrdma_test {

// Enqueues (non-blocking) a clock64-spin kernel sized to fill every SM on the
// current device, keeping GPU compute ~100% busy for ~spinTicks SM-clock cycles
// per launch. Used by the staged-RDMA microbenchmark to model a compute-bound
// trainer that saturates the SMs (as opposed to allocator-lock churn).
void launchBusyKernel(cudaStream_t stream, long long spinTicks);

} // namespace meta::comms::stagedrdma_test
