// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/device/cuda/AtomicAddKernel.h"

namespace torch::comms {

// CPU stand-in for the real atomic-add CUDA kernel so DeviceCounter (used by
// McclGraphEventTracker for its replay counter) works in host-only unit tests.
// link_whole in BUCK makes this strong symbol override the weak symbol from the
// real kernel (atomic-add-kernel) transitively linked via torchcomms-mccl-cpp.
cudaError_t
launchAtomicAdd(cudaStream_t /*stream*/, uint64_t* d_counter, uint64_t amount) {
  *d_counter += amount;
  return cudaSuccess;
}

} // namespace torch::comms
