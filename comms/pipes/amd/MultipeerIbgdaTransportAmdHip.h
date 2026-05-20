// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "PipesGdaShared.h"
#include "verbs/VerbsDev.h"

namespace pipes_gda {

struct P2pIbgdaTransportBuildParamsAmd {
  pipes_gda_gpu_dev_verbs_qp* gpuQp{nullptr};
  NetworkLKey sinkLkey{};
  void* sinkBufPtr{nullptr};
};

// Build array of P2pIbgdaTransportDevice on GPU memory.
// Returns opaque pointer (cast to P2pIbgdaTransportDevice* in device code).
void* buildDeviceTransportsOnGpuAmd(
    const P2pIbgdaTransportBuildParamsAmd* params,
    int numPeers);

// Free GPU transport memory
void freeDeviceTransportsOnGpuAmd(void* ptr);

// Get sizeof(P2pIbgdaTransportDevice)
std::size_t getP2pIbgdaTransportDeviceSizeAmd();

} // namespace pipes_gda
