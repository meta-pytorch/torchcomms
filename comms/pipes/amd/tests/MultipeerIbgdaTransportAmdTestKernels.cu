#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD/HIP port of comms/pipes/tests/MultipeerIbgdaTransportTest.cu
// Kernel implementations for the multipeer IBGDA integration tests.

#include "MultipeerIbgdaTransportAmdTestKernels.h"
#include "P2pIbgdaTransportDeviceAmd.h"
#include "PipesGdaShared.h"

#include <hip/hip_runtime.h>

namespace pipes_gda::tests {

// =============================================================================
// Data verification kernels
// =============================================================================

__global__ void
fillBufferWithPatternKernel(uint8_t* buf, std::size_t nbytes, uint8_t seed) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = gridDim.x * blockDim.x;
  for (std::size_t i = idx; i < nbytes; i += stride) {
    buf[i] = static_cast<uint8_t>((i % 251) ^ seed);
  }
}

__global__ void verifyBufferPatternKernel(
    const uint8_t* buf,
    std::size_t nbytes,
    uint8_t seed,
    bool* success) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = gridDim.x * blockDim.x;
  for (std::size_t i = idx; i < nbytes; i += stride) {
    uint8_t expected = static_cast<uint8_t>((i % 251) ^ seed);
    if (buf[i] != expected) {
      *success = false;
      return;
    }
  }
}

void fillBufferWithPattern(void* d_buf, std::size_t nbytes, uint8_t seed) {
  uint32_t threads = 256;
  uint32_t blocks = (nbytes + threads - 1) / threads;
  if (blocks > 1024)
    blocks = 1024;
  fillBufferWithPatternKernel<<<blocks, threads>>>(
      static_cast<uint8_t*>(d_buf), nbytes, seed);
  (void)hipDeviceSynchronize();
}

void verifyBufferPattern(
    const void* d_buf,
    std::size_t nbytes,
    uint8_t seed,
    bool* d_success) {
  bool initVal = true;
  (void)hipMemcpy(d_success, &initVal, sizeof(bool), hipMemcpyHostToDevice);
  uint32_t threads = 256;
  uint32_t blocks = (nbytes + threads - 1) / threads;
  if (blocks > 1024)
    blocks = 1024;
  verifyBufferPatternKernel<<<blocks, threads>>>(
      static_cast<const uint8_t*>(d_buf), nbytes, seed, d_success);
  (void)hipDeviceSynchronize();
}

// =============================================================================
// Sender-side kernels
// =============================================================================

__global__ void putAndSignalKernel(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto& transport = *transportPtr;
  IbgdaWork work = transport.put_signal(
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal);
  transport.wait_local(work);
}

void runTestPutAndSignal(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  putAndSignalKernel<<<1, 1>>>(
      transportPtr,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal, );
  (void)hipDeviceSynchronize();
}

// ---------------------------------------------------------------------------

__global__ void putAndSignalGroupKernel(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  pipes_gda::ThreadGroup group = pipes_gda::make_wavefront_group();
  auto& transport = *transportPtr;

  IbgdaWork work = transport.put_signal_group_local(
      group,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal);

  if (group.is_leader()) {
    transport.wait_local(work);
  }
  group.sync();
}

void runTestPutAndSignalGroup(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  putAndSignalGroupKernel<<<1, pipes_gda::kWavefrontSize>>>(
      transportPtr,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal, );
  (void)hipDeviceSynchronize();
}

// ---------------------------------------------------------------------------

__global__ void putAndSignalGroupMultiWarpKernel(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  pipes_gda::ThreadGroup group = pipes_gda::make_wavefront_group();
  auto& transport = *transportPtr;

  IbgdaWork work = transport.put_signal_group_global(
      group,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal);

  if (group.is_leader()) {
    transport.wait_local(work);
  }
  group.sync();
}

void runTestPutAndSignalGroupMultiWarp(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  // 4 wavefronts = 256 threads, using wavefront groups for global partitioning
  putAndSignalGroupMultiWarpKernel<<<1, 4 * pipes_gda::kWavefrontSize>>>(
      transportPtr,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal, );
  (void)hipDeviceSynchronize();
}

// ---------------------------------------------------------------------------

__global__ void putAndSignalGroupBlockKernel(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  pipes_gda::ThreadGroup group = pipes_gda::make_block_group();
  auto& transport = *transportPtr;

  IbgdaWork work = transport.put_signal_group_global(
      group,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal);

  if (group.is_leader()) {
    transport.wait_local(work);
  }
  group.sync();
}

void runTestPutAndSignalGroupBlock(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  // 2 blocks × 256 threads each, block-scope groups
  putAndSignalGroupBlockKernel<<<2, 256>>>(
      transportPtr,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal, );
  (void)hipDeviceSynchronize();
}

// ---------------------------------------------------------------------------

__global__ void multiplePutAndSignalKernel(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalValPerIter,
    uint32_t numIters) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto& transport = *transportPtr;

  for (uint32_t i = 0; i < numIters; ++i) {
    IbgdaWork work = transport.put_signal(
        localDataBuf,
        remoteDataBuf,
        nbytes,
        remoteSignalBuf,
        signalId,
        signalValPerIter);
    transport.wait_local(work);
  }
}

void runTestMultiplePutAndSignal(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalValPerIter,
    uint32_t numIters) {
  multiplePutAndSignalKernel<<<1, 1>>>(
      transportPtr,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalValPerIter,
      numIters);
  (void)hipDeviceSynchronize();
}

// ---------------------------------------------------------------------------

__global__ void signalOnlyKernel(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto& transport = *transportPtr;
  IbgdaWork work =
      transport.signal_remote(remoteSignalBuf, signalId, signalVal);
  transport.wait_local(work);
}

void runTestSignalOnly(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal) {
  signalOnlyKernel<<<1, 1>>>(
      transportPtr, remoteSignalBuf, signalId, signalVal);
  (void)hipDeviceSynchronize();
}

// ---------------------------------------------------------------------------

__global__ void putOnlyKernel(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto& transport = *transportPtr;
  IbgdaWork work = transport.put(localDataBuf, remoteDataBuf, nbytes);
  transport.wait_local(work);
}

void runTestPutOnly(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes) {
  putOnlyKernel<<<1, 1>>>(transportPtr, localDataBuf, remoteDataBuf, nbytes);
  (void)hipDeviceSynchronize();
}

// ---------------------------------------------------------------------------

__global__ void resetSignalKernel(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto& transport = *transportPtr;
  transport.reset_signal(remoteSignalBuf, signalId);
}

void runTestResetSignal(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId) {
  resetSignalKernel<<<1, 1>>>(transportPtr, remoteSignalBuf, signalId);
  (void)hipDeviceSynchronize();
}

// =============================================================================
// Receiver-side kernels
// =============================================================================

__global__ void waitSignalKernel(
    IbgdaLocalBuffer localSignalBuf,
    int signalId,
    uint64_t expectedVal,
    bool* success) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  volatile uint64_t* sig =
      static_cast<volatile uint64_t*>(localSignalBuf.ptr) + signalId;
  while (*sig < expectedVal) {
    // spin
  }
  __threadfence_system();
  *success = true;
}

void runTestWaitSignal(
    IbgdaLocalBuffer localSignalBuf,
    int signalId,
    uint64_t expectedVal,
    bool* d_success) {
  bool initVal = false;
  (void)hipMemcpy(d_success, &initVal, sizeof(bool), hipMemcpyHostToDevice);
  waitSignalKernel<<<1, 1>>>(localSignalBuf, signalId, expectedVal, d_success);
  (void)hipDeviceSynchronize();
}

// ---------------------------------------------------------------------------

__global__ void waitCounterKernel(
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    uint64_t expectedVal,
    bool* success) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  volatile uint64_t* counter =
      static_cast<volatile uint64_t*>(localCounterBuf.ptr) + counterId;
  while (*counter < expectedVal) {
    // spin
  }
  __threadfence_system();
  *success = true;
}

void runTestWaitCounter(
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    uint64_t expectedVal,
    bool* d_success) {
  bool initVal = false;
  (void)hipMemcpy(d_success, &initVal, sizeof(bool), hipMemcpyHostToDevice);
  waitCounterKernel<<<1, 1>>>(
      localCounterBuf, counterId, expectedVal, d_success);
  (void)hipDeviceSynchronize();
}

// =============================================================================
// Compound kernels
// =============================================================================

__global__ void waitReadyThenPutAndSignalKernel(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaLocalBuffer localSignalBuf,
    int readySignalId,
    uint64_t readyExpected,
    IbgdaRemoteBuffer remoteSignalBuf,
    int dataSignalId,
    uint64_t dataSignalVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto& transport = *transportPtr;

  // Wait for ready signal from remote
  transport.wait_signal(localSignalBuf, readySignalId, readyExpected);

  // Send data + signal
  IbgdaWork work = transport.put_signal(
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      dataSignalId,
      dataSignalVal);
  transport.wait_local(work);
}

void runTestWaitReadyThenPutAndSignal(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaLocalBuffer localSignalBuf,
    int readySignalId,
    uint64_t readyExpected,
    IbgdaRemoteBuffer remoteSignalBuf,
    int dataSignalId,
    uint64_t dataSignalVal) {
  waitReadyThenPutAndSignalKernel<<<1, 1>>>(
      transportPtr,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      localSignalBuf,
      readySignalId,
      readyExpected,
      remoteSignalBuf,
      dataSignalId,
      dataSignalVal, );
  (void)hipDeviceSynchronize();
}

// ---------------------------------------------------------------------------

__global__ void bidirectionalPutAndWaitKernel(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int sendSignalId,
    uint64_t sendSignalVal,
    IbgdaLocalBuffer localSignalBuf,
    int recvSignalId,
    uint64_t recvExpected,
    bool* success) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto& transport = *transportPtr;

  // Send data + signal to remote
  IbgdaWork work = transport.put_signal(
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      sendSignalId,
      sendSignalVal);
  transport.wait_local(work);

  // Wait for incoming signal from remote
  transport.wait_signal(localSignalBuf, recvSignalId, recvExpected);

  *success = true;
}

void runTestBidirectionalPutAndWait(
    P2pIbgdaTransportDevice* transportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int sendSignalId,
    uint64_t sendSignalVal,
    IbgdaLocalBuffer localSignalBuf,
    int recvSignalId,
    uint64_t recvExpected,
    bool* d_success) {
  bool initVal = false;
  (void)hipMemcpy(d_success, &initVal, sizeof(bool), hipMemcpyHostToDevice);
  bidirectionalPutAndWaitKernel<<<1, 1>>>(
      transportPtr,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      sendSignalId,
      sendSignalVal,
      localSignalBuf,
      recvSignalId,
      recvExpected,
      d_success);
  (void)hipDeviceSynchronize();
}

// ---------------------------------------------------------------------------

__global__ void putSignalCounterKernel(
    P2pIbgdaTransportDevice* transportPtr,
    P2pIbgdaTransportDevice* companionTransportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    uint64_t counterVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto& transport = *transportPtr;
  transport.put_signal_counter_remote(
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal,
      localCounterBuf,
      counterId,
      counterVal);
}

void runTestPutSignalCounter(
    P2pIbgdaTransportDevice* transportPtr,
    P2pIbgdaTransportDevice* companionTransportPtr,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    uint64_t counterVal) {
  putSignalCounterKernel<<<1, 1>>>(
      transportPtr,
      companionTransportPtr,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      remoteSignalBuf,
      signalId,
      signalVal,
      localCounterBuf,
      counterId,
      counterVal);
  (void)hipDeviceSynchronize();
}

} // namespace pipes_gda::tests
#endif
