// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/P2pIbgdaTransportDistributedTest.cuh"

#include <cuda_runtime.h>

namespace comms::pipes::tests {

// =============================================================================
// CUDA Check Macro for test code
// =============================================================================

#define PIPES_CUDA_CHECK_KERNEL(EXPR) \
  do {                                \
    const cudaError_t err = EXPR;     \
    if (err != cudaSuccess) {         \
      return;                         \
    }                                 \
  } while (0)

// =============================================================================
// Device Transport Allocation Helpers
// =============================================================================

P2pIbgdaTransportDevice* allocateDeviceTransport(
    doca_gpu_dev_verbs_qp* qp,
    const IbgdaLocalBuffer& localSignalBuf,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int numSignals) {
  P2pIbgdaTransportDevice* d_transport = nullptr;
  cudaError_t err = cudaMalloc(&d_transport, sizeof(P2pIbgdaTransportDevice));
  if (err != cudaSuccess) {
    return nullptr;
  }

  // Create host-side transport and copy to device
  P2pIbgdaTransportDevice hostTransport(
      qp, localSignalBuf, remoteSignalBuf, numSignals);
  err = cudaMemcpy(
      d_transport,
      &hostTransport,
      sizeof(P2pIbgdaTransportDevice),
      cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_transport);
    return nullptr;
  }

  return d_transport;
}

void freeDeviceTransport(P2pIbgdaTransportDevice* d_transport) {
  if (d_transport) {
    cudaFree(d_transport);
  }
}

// =============================================================================
// CUDA Kernel Implementations
// =============================================================================

__global__ void putSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  // Single-threaded kernel
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    IbgdaWork work = transport->put_signal(
        localDataBuf, remoteDataBuf, nbytes, signalId, signalVal);
    transport->wait_local(work);
  }
}

__global__ void putSignalNonAdaptiveKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  // Single-threaded kernel
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    IbgdaWork work = transport->put_signal_non_adaptive(
        localDataBuf, remoteDataBuf, nbytes, signalId, signalVal);
    transport->wait_local(work);
  }
}

__global__ void waitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t expectedSignal,
    bool* success) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    transport->wait_signal(signalId, IbgdaCmpOp::GE, expectedSignal);
    *success = true;
  }
}

__global__ void signalOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t signalVal) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    IbgdaWork work = transport->signal(signalId, signalVal);
    transport->wait_local(work);
  }
}

__global__ void verifyDataKernel(
    void* data,
    std::size_t nbytes,
    uint8_t expectedPattern,
    bool* success) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint8_t* bytes = static_cast<uint8_t*>(data);
    *success = true;
    for (std::size_t i = 0; i < nbytes; i++) {
      if (bytes[i] != expectedPattern) {
        *success = false;
        return;
      }
    }
  }
}

__global__ void
fillDataKernel(void* data, std::size_t nbytes, uint8_t pattern) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint8_t* bytes = static_cast<uint8_t*>(data);
    for (std::size_t i = 0; i < nbytes; i++) {
      bytes[i] = pattern;
    }
  }
}

// =============================================================================
// Kernel Wrapper Implementations
// =============================================================================

void runPutSignalKernel(
    P2pIbgdaTransportDevice* d_transport,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  putSignalKernel<<<1, 1>>>(
      d_transport, localDataBuf, remoteDataBuf, nbytes, signalId, signalVal);
  PIPES_CUDA_CHECK_KERNEL(cudaDeviceSynchronize());
}

void runPutSignalNonAdaptiveKernel(
    P2pIbgdaTransportDevice* d_transport,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  putSignalNonAdaptiveKernel<<<1, 1>>>(
      d_transport, localDataBuf, remoteDataBuf, nbytes, signalId, signalVal);
  PIPES_CUDA_CHECK_KERNEL(cudaDeviceSynchronize());
}

void runWaitSignalKernel(
    P2pIbgdaTransportDevice* d_transport,
    int signalId,
    uint64_t expectedSignal,
    bool* d_success) {
  waitSignalKernel<<<1, 1>>>(d_transport, signalId, expectedSignal, d_success);
  PIPES_CUDA_CHECK_KERNEL(cudaDeviceSynchronize());
}

void runSignalOnlyKernel(
    P2pIbgdaTransportDevice* d_transport,
    int signalId,
    uint64_t signalVal) {
  signalOnlyKernel<<<1, 1>>>(d_transport, signalId, signalVal);
  PIPES_CUDA_CHECK_KERNEL(cudaDeviceSynchronize());
}

void runVerifyDataKernel(
    void* d_data,
    std::size_t nbytes,
    uint8_t expectedPattern,
    bool* d_success) {
  verifyDataKernel<<<1, 1>>>(d_data, nbytes, expectedPattern, d_success);
  PIPES_CUDA_CHECK_KERNEL(cudaDeviceSynchronize());
}

void runFillDataKernel(void* d_data, std::size_t nbytes, uint8_t pattern) {
  fillDataKernel<<<1, 1>>>(d_data, nbytes, pattern);
  PIPES_CUDA_CHECK_KERNEL(cudaDeviceSynchronize());
}

} // namespace comms::pipes::tests
