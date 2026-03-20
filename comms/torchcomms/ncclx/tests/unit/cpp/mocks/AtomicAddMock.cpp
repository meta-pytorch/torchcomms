// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/torchcomms/device/cuda/AtomicAddKernel.h"

namespace torch::comms {

cudaError_t
launchAtomicAdd(cudaStream_t /*stream*/, uint64_t* d_counter, uint64_t amount) {
  *d_counter += amount;
  return cudaSuccess;
}

} // namespace torch::comms
