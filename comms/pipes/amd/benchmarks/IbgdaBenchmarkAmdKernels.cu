#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD port of comms/pipes/benchmarks/IbgdaBenchmark.cu
// Same kernel logic but uses pipes_gda::P2pIbgdaTransportDevice and HIP.

#include <hip/hip_runtime.h>

#include "IbgdaBenchmarkAmdKernels.h"
#include "P2pIbgdaTransportDeviceAmd.h"

namespace pipes_gda::benchmark {

// =============================================================================
// GPU Kernels
// =============================================================================

__global__ void ibgdaPutSignalWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work = transport->put_signal(
        localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId, 1);
    transport->wait_local(work);
  }
}

__global__ void ibgdaPutWaitLocalBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    for (int i = 0; i < 10; i++) {
      auto work = transport->put(localBuf, remoteBuf, nbytes);
      transport->wait_local(work);
    }

    unsigned long long startCycle = wall_clock64();
    for (int i = 0; i < numIters; i++) {
      auto work = transport->put(localBuf, remoteBuf, nbytes);
      transport->wait_local(work);
    }
    unsigned long long endCycle = wall_clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaPutSignalWaitLocalBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    for (int i = 0; i < 10; i++) {
      auto work = transport->put_signal(
          localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId, 1);
      transport->wait_local(work);
    }

    unsigned long long startCycle = wall_clock64();
    for (int i = 0; i < numIters; i++) {
      auto work = transport->put_signal(
          localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId, 1);
      transport->wait_local(work);
    }
    unsigned long long endCycle = wall_clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaSignalOnlyBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    for (int i = 0; i < 10; i++) {
      auto work = transport->signal_remote(remoteSignalBuf, signalId, 1);
      transport->wait_local(work);
    }

    unsigned long long startCycle = wall_clock64();
    for (int i = 0; i < numIters; i++) {
      auto work = transport->signal_remote(remoteSignalBuf, signalId, 1);
      transport->wait_local(work);
    }
    unsigned long long endCycle = wall_clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaPutCqPollWaitBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    for (int i = 0; i < 10; i++) {
      auto work = transport->put(localBuf, remoteBuf, nbytes);
      transport->wait_local(work);
    }

    unsigned long long startCycle = wall_clock64();
    for (int i = 0; i < numIters; i++) {
      auto work = transport->put(localBuf, remoteBuf, nbytes);
      transport->wait_local(work);
    }
    unsigned long long endCycle = wall_clock64();
    *totalCycles = endCycle - startCycle;
  }
}

// =============================================================================
// Launch Wrappers
// =============================================================================

// AMD wavefront size = 64
constexpr int kWavefrontSize = 64;

void launchIbgdaPutSignalSingle(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    hipStream_t stream) {
  ibgdaPutSignalWaitLocalKernel<<<1, kWavefrontSize, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, remoteSignalBuf, signalId);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + hipGetErrorString(err));
}

void launchIbgdaPutWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    hipStream_t stream) {
  ibgdaPutWaitLocalBatchKernel<<<1, kWavefrontSize, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, numIters, totalCycles);
}

void launchIbgdaPutSignalWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    hipStream_t stream) {
  ibgdaPutSignalWaitLocalBatchKernel<<<1, kWavefrontSize, 0, stream>>>(
      transport,
      localBuf,
      remoteBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      numIters,
      totalCycles);
}

void launchIbgdaSignalOnlyBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    hipStream_t stream) {
  ibgdaSignalOnlyBatchKernel<<<1, kWavefrontSize, 0, stream>>>(
      transport, remoteSignalBuf, signalId, numIters, totalCycles);
}

void launchIbgdaPutCqPollWaitBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    hipStream_t stream) {
  ibgdaPutCqPollWaitBatchKernel<<<1, kWavefrontSize, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, numIters, totalCycles);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + hipGetErrorString(err));
}

} // namespace pipes_gda::benchmark
#endif
