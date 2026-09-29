// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/llx/tests/LLIbgdaSendRecv.cuh"

#include <stdexcept>
#include <string>

#include "comms/common/fault_tolerance/TestAbort.h"
#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims::test {

using comms::fault_tolerance::testing::testAbortDevice;

namespace {

void throwOnLaunchError(const char* what) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string(what) + " launch failed: " + cudaGetErrorString(err));
  }
}

__global__ void
fillPatternKernel(uint8_t* buffer, std::size_t nbytes, uint8_t baseValue) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = idx; i < nbytes; i += stride) {
    buffer[i] = static_cast<uint8_t>(baseValue + (i % 256));
  }
}

__global__ void verifyPatternKernel(
    const uint8_t* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int* errorCount) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = idx; i < nbytes; i += stride) {
    const uint8_t expected = static_cast<uint8_t>(baseValue + (i % 256));
    if (buffer[i] != expected) {
      atomicAdd(errorCount, 1);
    }
  }
}

// One block group drives send/recv over the transport's channel state via the
// protocol-generic detail::send/recv<Transport, Memcpy, Proto>. Proto = Simple
// or LL selects the wire format. (The P2pIbgdaTransportDevice class methods are
// hard-wired to Simple, so LL is exercised through detail:: directly.)
template <typename Proto>
__global__ void sendRecvKernel(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    int activeBlocks,
    std::size_t maxSignalBytes,
    bool send,
    AbortDevice abortDevice) {
  (void)activeBlocks; // master's detail::send/recv has no active_blocks param
  auto group = make_block_group();
  abortDevice.start();
  if (send) {
    detail::send<P2pIbgdaTransportDevice, Memcpy, Proto>(
        *transport, group, buffer, nbytes, maxSignalBytes, abortDevice);
  } else {
    detail::recv<P2pIbgdaTransportDevice, Memcpy, Proto>(
        *transport, group, buffer, nbytes, maxSignalBytes, abortDevice);
  }
}

} // namespace

void fillPattern(
    void* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int numBlocks,
    int blockSize) {
  fillPatternKernel<<<numBlocks, blockSize>>>(
      static_cast<uint8_t*>(buffer), nbytes, baseValue);
  throwOnLaunchError("fillPattern");
}

void verifyPattern(
    const void* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int* errorCount,
    int numBlocks,
    int blockSize) {
  verifyPatternKernel<<<numBlocks, blockSize>>>(
      static_cast<const uint8_t*>(buffer), nbytes, baseValue, errorCount);
  throwOnLaunchError("verifyPattern");
}

void launchLLSendRecv(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    int activeBlocks,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize) {
  sendRecvKernel<protocol::LL><<<numBlocks, blockSize>>>(
      transport,
      buffer,
      nbytes,
      activeBlocks,
      maxSignalBytes,
      send,
      testAbortDevice());
  throwOnLaunchError("sendRecvKernel<protocol::LL>");
}

void launchSimpleSendRecv(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    int activeBlocks,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize) {
  sendRecvKernel<protocol::Simple><<<numBlocks, blockSize>>>(
      transport,
      buffer,
      nbytes,
      activeBlocks,
      maxSignalBytes,
      send,
      testAbortDevice());
  throwOnLaunchError("sendRecvKernel<protocol::Simple>");
}

} // namespace comms::prims::test
