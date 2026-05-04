#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD/HIP port of comms/pipes/tests/P2pIbgdaTransportDeviceTest.cu
// Same kernel logic, adapted for pipes_gda::P2pIbgdaTransportDevice (AMD).

#include "P2pIbgdaTransportDeviceAmd.h"
#include "P2pIbgdaTransportDeviceAmdTestKernels.h"
#include "PipesGdaShared.h"

#include <hip/hip_runtime.h>

namespace pipes_gda::tests {

// =============================================================================
// Device-side test kernels
// =============================================================================

__global__ void testP2pTransportConstruction(bool* success) {
  P2pIbgdaTransportDevice transport(nullptr);

  *success = true;

  if (transport.getQp() != nullptr) {
    *success = false;
  }
}

__global__ void testP2pTransportDefaultConstruction(bool* success) {
  P2pIbgdaTransportDevice transport;

  *success = true;

  if (transport.getQp() != nullptr) {
    *success = false;
  }
}

__global__ void testP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    int numSignals,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr);

  *success = true;

  for (int i = 0; i < numSignals; ++i) {
    uint64_t expected = static_cast<uint64_t>(i + 1) * 100;
    uint64_t actual = transport.read_signal(localBuf, i);
    if (actual != expected) {
      *success = false;
    }
  }
}

__global__ void testIbgdaWork(bool* success) {
  *success = true;

  IbgdaWork defaultWork;
  if (defaultWork.value != 0) {
    *success = false;
  }

  uint64_t testTicket = 12345;
  IbgdaWork workWithValue(testTicket);
  if (workWithValue.value != testTicket) {
    *success = false;
  }

  IbgdaWork copiedWork = workWithValue;
  if (copiedWork.value != testTicket) {
    *success = false;
  }
}

// =============================================================================
// wait_signal test kernels (GE-only comparison)
// =============================================================================

__global__ void testWaitSignalGE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr);
  transport.wait_signal(localBuf, 0, targetValue);
  *success = true;
}

__global__ void testWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    int numSignals,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr);

  *success = true;

  for (int i = 0; i < numSignals; ++i) {
    uint64_t expectedValue = static_cast<uint64_t>(i + 1) * 100;
    transport.wait_signal(localBuf, i, expectedValue);

    uint64_t readValue = transport.read_signal(localBuf, i);
    if (readValue != expectedValue) {
      *success = false;
    }
  }
}

// =============================================================================
// wait_signal timeout test kernels
// =============================================================================

__global__ void testWaitSignalTimeout(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    Timeout timeout) {
  timeout.start();
  P2pIbgdaTransportDevice transport(nullptr);
  transport.wait_signal(localBuf, 0, 999, timeout);
}

__global__ void testWaitSignalNoTimeout(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    Timeout timeout,
    bool* success) {
  timeout.start();
  P2pIbgdaTransportDevice transport(nullptr);
  transport.wait_signal(localBuf, 0, 42, timeout);
  *success = true;
}

// =============================================================================
// Wrapper functions (called from TestMain)
// =============================================================================

void runTestP2pTransportConstruction(bool* d_success) {
  testP2pTransportConstruction<<<1, 1>>>(d_success);
}

void runTestP2pTransportDefaultConstruction(bool* d_success) {
  testP2pTransportDefaultConstruction<<<1, 1>>>(d_success);
}

void runTestP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    int numSignals,
    bool* d_success) {
  testP2pTransportReadSignal<<<1, 1>>>(
      d_signalBuf, localBuf, numSignals, d_success);
}

void runTestIbgdaWork(bool* d_success) {
  testIbgdaWork<<<1, 1>>>(d_success);
}

void runTestWaitSignalGE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalGE<<<1, 1>>>(d_signalBuf, localBuf, targetValue, d_success);
}

void runTestWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    int numSignals,
    bool* d_success) {
  testWaitSignalMultipleSlots<<<1, 1>>>(
      d_signalBuf, localBuf, numSignals, d_success);
}

void runTestWaitSignalTimeout(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    int device,
    uint32_t timeout_ms) {
  (void)hipSetDevice(device);
  Timeout timeout =
      pipes_gda::make_timeout_us(static_cast<uint64_t>(timeout_ms) * 1000);
  testWaitSignalTimeout<<<1, 1>>>(d_signalBuf, localBuf, timeout);
  (void)hipDeviceSynchronize();
}

void runTestWaitSignalNoTimeout(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    int device,
    uint32_t timeout_ms,
    bool* d_success) {
  (void)hipSetDevice(device);
  Timeout timeout =
      pipes_gda::make_timeout_us(static_cast<uint64_t>(timeout_ms) * 1000);
  testWaitSignalNoTimeout<<<1, 1>>>(d_signalBuf, localBuf, timeout, d_success);
}

// =============================================================================
// put() latency benchmark kernel (mock QP, no real NIC)
// =============================================================================

static constexpr uint16_t kMockSqSize = 4096;

struct MockQpDevice {
  pipes_gda_gpu_dev_verbs_qp qp;
  pipes_gda_gpu_dev_verbs_wqe wqeBuf[kMockSqSize];
  __be32 dbrec[2];
  uint64_t dbReg;
  uint64_t signalBuf;
};

__device__ void initMockQp(MockQpDevice* mock) {
  memset(mock, 0, sizeof(MockQpDevice));

  mock->qp.sq_wqe_daddr = reinterpret_cast<uint8_t*>(mock->wqeBuf);
  mock->qp.sq_wqe_num = kMockSqSize;
  mock->qp.sq_wqe_mask = kMockSqSize - 1;
  mock->qp.sq_num = 1;
  mock->qp.sq_num_shift8 = 1 << 8;
  mock->qp.sq_dbrec = mock->dbrec;
  mock->qp.sq_db = &mock->dbReg;
  mock->qp.sq_rsvd_index = 0;
  mock->qp.sq_ready_index = 0;
}

__global__ void testPutLatency(
    MockQpDevice* mock,
    std::size_t nbytes,
    uint32_t numWarmup,
    uint32_t numIters,
    uint64_t* latencies) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  initMockQp(mock);

  IbgdaLocalBuffer localDataBuf(
      mock->wqeBuf, NetworkLKeys{NetworkLKey(0x3333)});
  IbgdaRemoteBuffer remoteDataBuf(
      mock->wqeBuf, NetworkRKeys{NetworkRKey(0x4444)});

  P2pIbgdaTransportDevice transport(&mock->qp);

  uint32_t totalIters = numWarmup + numIters;
  for (uint32_t iter = 0; iter < totalIters; iter++) {
    mock->qp.sq_rsvd_index = 0;
    mock->qp.sq_ready_index = 0;

    uint64_t t0 = wall_clock64();
    transport.put(localDataBuf, remoteDataBuf, nbytes);
    uint64_t t1 = wall_clock64();

    if (iter >= numWarmup) {
      latencies[iter - numWarmup] = t1 - t0;
    }
  }
}

void runTestPutLatency(
    std::size_t nbytes,
    uint32_t numWarmup,
    uint32_t numIters,
    uint64_t* d_latencies) {
  MockQpDevice* d_mock = nullptr;
  (void)hipMalloc(&d_mock, sizeof(MockQpDevice));

  testPutLatency<<<1, 1>>>(d_mock, nbytes, numWarmup, numIters, d_latencies);
  (void)hipDeviceSynchronize();

  (void)hipFree(d_mock);
}

// =============================================================================
// put() latency benchmark kernel (REAL QP + NIC)
// =============================================================================

__global__ void testPutLatencyReal(
    pipes_gda_gpu_dev_verbs_qp* qp,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    uint32_t numWarmup,
    uint32_t numIters,
    uint64_t* latencies) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  P2pIbgdaTransportDevice transport(qp);

  uint32_t totalIters = numWarmup + numIters;
  for (uint32_t iter = 0; iter < totalIters; iter++) {
    uint64_t t0 = wall_clock64();
    IbgdaWork work = transport.put(localDataBuf, remoteDataBuf, nbytes);
    transport.wait_local(work);
    uint64_t t1 = wall_clock64();

    if (iter >= numWarmup) {
      latencies[iter - numWarmup] = t1 - t0;
    }
  }
}

void runTestPutLatencyReal(
    pipes_gda_gpu_dev_verbs_qp* qp,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    uint32_t numWarmup,
    uint32_t numIters,
    uint64_t* d_latencies) {
  testPutLatencyReal<<<1, 1>>>(
      qp,
      localDataBuf,
      remoteDataBuf,
      nbytes,
      numWarmup,
      numIters,
      d_latencies);
  (void)hipDeviceSynchronize();
}

} // namespace pipes_gda::tests

// =============================================================================
// ThreadGroup / group-level API test kernels
// =============================================================================

namespace pipes_gda::tests {

__global__ void testPutGroupPartitioning(bool* success) {
  pipes_gda::ThreadGroup group = pipes_gda::make_wavefront_group();

  *success = true;

  const std::size_t totalBytes = 1024;
  char dummyLocal[8];
  char dummyRemote[8];
  pipes_gda::IbgdaLocalBuffer localBuf(
      dummyLocal, pipes_gda::NetworkLKeys{pipes_gda::NetworkLKey(0x1111)});
  pipes_gda::IbgdaRemoteBuffer remoteBuf(
      dummyRemote, pipes_gda::NetworkRKeys{pipes_gda::NetworkRKey(0x2222)});

  std::size_t chunkSize = totalBytes / group.group_size;
  std::size_t expectedOffset = group.thread_id_in_group * chunkSize;
  std::size_t expectedBytes = (group.thread_id_in_group == group.group_size - 1)
      ? (totalBytes - expectedOffset)
      : chunkSize;
  (void)expectedBytes;

  pipes_gda::IbgdaLocalBuffer laneBuf = localBuf.subBuffer(expectedOffset);
  pipes_gda::IbgdaRemoteBuffer laneRemoteBuf =
      remoteBuf.subBuffer(expectedOffset);

  char* expectedLocalPtr = static_cast<char*>(localBuf.ptr) + expectedOffset;
  char* expectedRemotePtr = static_cast<char*>(remoteBuf.ptr) + expectedOffset;

  if (laneBuf.ptr != expectedLocalPtr) {
    *success = false;
  }
  if (laneRemoteBuf.ptr != expectedRemotePtr) {
    *success = false;
  }
  if (laneBuf.lkey_per_device[0] != localBuf.lkey_per_device[0]) {
    *success = false;
  }
  if (laneRemoteBuf.rkey_per_device[0] != remoteBuf.rkey_per_device[0]) {
    *success = false;
  }
}

void runTestPutGroupPartitioning(bool* d_success) {
  testPutGroupPartitioning<<<1, pipes_gda::kWavefrontSize>>>(d_success);
}

__global__ void testPutSignalGroupBroadcast(bool* success) {
  pipes_gda::ThreadGroup group = pipes_gda::make_wavefront_group();

  *success = true;

  const uint64_t leaderTicket = 0xDEADBEEF42ULL;
  uint64_t ticket = 0;

  if (group.is_leader()) {
    ticket = leaderTicket;
  }

  ticket = group.broadcast<uint64_t>(ticket);

  if (ticket != leaderTicket) {
    *success = false;
  }
}

void runTestPutSignalGroupBroadcast(bool* d_success) {
  testPutSignalGroupBroadcast<<<1, pipes_gda::kWavefrontSize>>>(d_success);
}

__global__ void testBroadcast64Block(bool* success) {
  pipes_gda::ThreadGroup group = pipes_gda::make_block_group();

  const uint64_t leaderVal = 0xCAFEBABE12345678ULL;
  uint64_t val = 0;

  if (group.is_leader()) {
    val = leaderVal;
  }

  val = group.broadcast<uint64_t>(val);

  if (val != leaderVal) {
    atomicExch(reinterpret_cast<int*>(success), 0);
  }
}

void runTestBroadcast64Block(bool* d_success) {
  bool initVal = true;
  (void)hipMemcpy(d_success, &initVal, sizeof(bool), hipMemcpyHostToDevice);
  testBroadcast64Block<<<4, 256>>>(d_success);
}

// ---------------------------------------------------------------------------
// Broadcast64Multiwarp: verify multiwarp-level broadcast (4 wavefronts)
// AMD equivalent of NVIDIA's Broadcast64Multiwarp (4 warps = 128 threads).
// On AMD CDNA, kMultiwarpSize = 4 × 64 = 256 threads.
// ---------------------------------------------------------------------------
__global__ void testBroadcast64Multiwarp(bool* success) {
  pipes_gda::ThreadGroup group = pipes_gda::make_multiwarp_group();

  // Each multiwarp leader produces a unique value based on group_id
  uint64_t val = 0;
  if (group.is_leader()) {
    val = 0xAAAABBBB00000000ULL + group.group_id;
  }

  val = group.broadcast<uint64_t>(val);

  // All threads in the multiwarp should see their leader's value
  uint64_t expected = 0xAAAABBBB00000000ULL + group.group_id;
  if (val != expected) {
    atomicExch(reinterpret_cast<int*>(success), 0);
  }
}

void runTestBroadcast64Multiwarp(bool* d_success) {
  bool initVal = true;
  (void)hipMemcpy(d_success, &initVal, sizeof(bool), hipMemcpyHostToDevice);
  // 2 blocks × 512 threads = 4 multiwarps (each 256 threads)
  testBroadcast64Multiwarp<<<2, 512>>>(d_success);
}

__global__ void testBroadcast64DoubleSafety(bool* success) {
  pipes_gda::ThreadGroup group = pipes_gda::make_block_group();

  const uint64_t firstVal = 111ULL;
  const uint64_t secondVal = 222ULL;

  uint64_t v1 = 0, v2 = 0;
  if (group.is_leader()) {
    v1 = firstVal;
  }
  v1 = group.broadcast<uint64_t>(v1);

  if (group.is_leader()) {
    v2 = secondVal;
  }
  v2 = group.broadcast<uint64_t>(v2);

  if (v1 != firstVal || v2 != secondVal) {
    atomicExch(reinterpret_cast<int*>(success), 0);
  }
}

void runTestBroadcast64DoubleSafety(bool* d_success) {
  bool initVal = true;
  (void)hipMemcpy(d_success, &initVal, sizeof(bool), hipMemcpyHostToDevice);
  testBroadcast64DoubleSafety<<<4, 256>>>(d_success);
}

__global__ void testPutGroupPartitioningBlock(bool* success) {
  pipes_gda::ThreadGroup group = pipes_gda::make_block_group();

  const std::size_t totalBytes = 4096;
  char dummyLocal[8];
  char dummyRemote[8];
  pipes_gda::IbgdaLocalBuffer localBuf(
      dummyLocal, pipes_gda::NetworkLKeys{pipes_gda::NetworkLKey(0x1111)});
  pipes_gda::IbgdaRemoteBuffer remoteBuf(
      dummyRemote, pipes_gda::NetworkRKeys{pipes_gda::NetworkRKey(0x2222)});

  std::size_t chunkSize = totalBytes / group.group_size;
  std::size_t expectedOffset = group.thread_id_in_group * chunkSize;

  pipes_gda::IbgdaLocalBuffer laneBuf = localBuf.subBuffer(expectedOffset);

  char* expectedPtr = static_cast<char*>(localBuf.ptr) + expectedOffset;
  if (laneBuf.ptr != expectedPtr) {
    atomicExch(reinterpret_cast<int*>(success), 0);
  }
  if (laneBuf.lkey_per_device[0] != localBuf.lkey_per_device[0]) {
    atomicExch(reinterpret_cast<int*>(success), 0);
  }
}

void runTestPutGroupPartitioningBlock(bool* d_success) {
  bool initVal = true;
  (void)hipMemcpy(d_success, &initVal, sizeof(bool), hipMemcpyHostToDevice);
  testPutGroupPartitioningBlock<<<4, 256>>>(d_success);
}

} // namespace pipes_gda::tests
#endif
