#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <hip/hip_runtime.h>

#include "MultipeerIbgdaTransportAmdHip.h"
#include "P2pIbgdaTransportDeviceAmd.h"

namespace pipes_gda {

void* buildDeviceTransportsOnGpuAmd(
    const P2pIbgdaTransportBuildParamsAmd* params,
    int numPeers) {
  std::size_t elemSize = sizeof(P2pIbgdaTransportDevice);
  void* gpuPtr = nullptr;

  hipError_t err = hipMalloc(&gpuPtr, numPeers * elemSize);
  if (err != hipSuccess)
    return nullptr;

  for (int i = 0; i < numPeers; i++) {
    P2pIbgdaTransportDevice hostTransport(
        params[i].gpuQp, nullptr, params[i].sinkLkey, params[i].sinkBufPtr);
    err = hipMemcpy(
        reinterpret_cast<char*>(gpuPtr) + i * elemSize,
        &hostTransport,
        elemSize,
        hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      hipFree(gpuPtr);
      return nullptr;
    }
  }

  return gpuPtr;
}

void freeDeviceTransportsOnGpuAmd(void* ptr) {
  if (ptr)
    hipFree(ptr);
}

std::size_t getP2pIbgdaTransportDeviceSizeAmd() {
  return sizeof(P2pIbgdaTransportDevice);
}

} // namespace pipes_gda
#endif
