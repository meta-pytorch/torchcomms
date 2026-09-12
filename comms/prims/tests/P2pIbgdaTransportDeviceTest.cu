// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// CudaHipCompat must come before Checks.h so the `cuda*` -> `hip*`
// macro renames apply on AMD builds (Checks.h uses `cudaError_t` /
// `cudaSuccess` / `cudaGetErrorString` / `cudaGetLastError` directly).
#include "comms/prims/transport/amd/HipHostCompat.h"

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/tests/P2pIbgdaTransportDeviceTest.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

#include <chrono>
#include <thread>

namespace comms::prims::tests {

// =============================================================================
// Device-side test kernels
// =============================================================================

__global__ void testP2pTransportConstruction(bool* success) {
  // Create transport on device with empty NIC span
  P2pIbgdaTransportDevice transport(DeviceSpan<NicDeviceIbgdaResources>{});

  // If we get here, construction succeeded
  *success = true;
}

__global__ void testP2pTransportDefaultConstruction(bool* success) {
  // Default construction should initialize all members
  P2pIbgdaTransportDevice transport;

  // If we get here, default construction succeeded
  *success = true;
}

__global__ void testP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    int numSignals,
    bool* success) {
  // Construct transport with ownedLocalSignalBuf pointing to d_signalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      numSignals);

  *success = true;

  // Test read_signal for each slot via slot-index API
  for (int i = 0; i < numSignals; ++i) {
    uint64_t expected = static_cast<uint64_t>(i + 1) * 100;
    uint64_t actual = transport.read_signal(i);
    if (actual != expected) {
      *success = false;
    }
  }
}

// =============================================================================
// wait_signal test kernels
// =============================================================================

__global__ void
testWaitSignalGE(uint64_t* d_signalBuf, uint64_t targetValue, bool* success) {
  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  // Signal buffer is pre-set to a value >= targetValue by host
  // wait_signal should return immediately (slot 0)
  transport.wait_signal(0, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    int numSignals,
    bool* success) {
  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      numSignals);

  *success = true;

  // Signal buffer is pre-set: slot[i] = (i + 1) * 100
  // Test wait_signal on each slot with matching GE condition
  for (int i = 0; i < numSignals; ++i) {
    uint64_t expectedValue = static_cast<uint64_t>(i + 1) * 100;
    transport.wait_signal(i, expectedValue);

    // Verify read_signal returns the same value
    uint64_t readValue = transport.read_signal(i);
    if (readValue != expectedValue) {
      *success = false;
    }
  }
}

__global__ void testWaitSignalWithDisabledAbort(
    uint64_t* d_signalBuf,
    bool* success) {
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  comms::fault_tolerance::AbortDevice abort;
  transport.wait_signal(0, 0, abort);
  *success = true;
}

__global__ void testWaitSignalUntilAbort(
    uint64_t* d_signalBuf,
    comms::fault_tolerance::AbortDevice abort,
    bool* success,
    uint32_t* enteredWait) {
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  // Arm the deadline the way a production kernel does. Without this the
  // handle observes explicit aborts only, and a communicator abortDevice never
  // reaches the wait.
  abort.start();
  // Published after the handle is armed, so a host that sees it knows every
  // precondition of the wait is already in place.
  if (enteredWait != nullptr) {
    __threadfence_system();
    *static_cast<volatile uint32_t*>(enteredWait) = 1U;
  }
  transport.wait_signal(0, 1, abort);
  // Reaching this line is the liveness guarantee: the wait reports no status,
  // so one that failed to terminate hangs the kernel instead. The signal slot
  // is what distinguishes the two ways out -- it is still short of the expected
  // value here, so the abort released the wait rather than the signal landing.
  *success = *static_cast<volatile uint64_t*>(localSigBuf.ptr) < 1;
}

// =============================================================================
// Wrapper functions to launch the kernels (called from .cc test file)
// =============================================================================

void runTestP2pTransportConstruction(bool* d_success) {
  testP2pTransportConstruction<<<1, 1>>>(d_success);
}

void runTestP2pTransportDefaultConstruction(bool* d_success) {
  testP2pTransportDefaultConstruction<<<1, 1>>>(d_success);
}

void runTestP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    int numSignals,
    bool* d_success) {
  testP2pTransportReadSignal<<<1, 1>>>(d_signalBuf, numSignals, d_success);
}

void runTestWaitSignalGE(
    uint64_t* d_signalBuf,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalGE<<<1, 1>>>(d_signalBuf, targetValue, d_success);
}

void runTestWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    int numSignals,
    bool* d_success) {
  testWaitSignalMultipleSlots<<<1, 1>>>(d_signalBuf, numSignals, d_success);
}

void runTestWaitSignalWithDisabledAbort(
    uint64_t* d_signalBuf,
    bool* d_success) {
  testWaitSignalWithDisabledAbort<<<1, 1>>>(d_signalBuf, d_success);
}

void runTestWaitSignalUntilAbort(
    uint64_t* d_signalBuf,
    comms::fault_tolerance::AbortDevice abort,
    bool* d_success,
    uint32_t* d_enteredWait) {
  testWaitSignalUntilAbort<<<1, 1>>>(
      d_signalBuf, abort, d_success, d_enteredWait);
}

// =============================================================================
// Group-level API test kernels
// =============================================================================

__global__ void testPutCooperativePartitioning(bool* success) {
  *success = true;

  auto group = comms::prims::make_warp_group();
  if (group.group_size != comms::prims::kWarpSize) {
    *success = false;
    return;
  }

  constexpr std::size_t kTotalBytes = 1024; // 1KB
  constexpr std::size_t kChunkSize = kTotalBytes / comms::prims::kWarpSize;

  std::size_t expectedOffset = group.thread_id_in_group * kChunkSize;
  std::size_t expectedChunk = kChunkSize;

  char baseData[8];
  void* basePtr = baseData;

  comms::prims::IbgdaLocalBuffer baseBuf(
      basePtr, comms::prims::NetworkLKeys{comms::prims::NetworkLKey(0x1111)});
  comms::prims::IbgdaLocalBuffer laneBuf = baseBuf.subBuffer(expectedOffset);

  auto* expectedPtr = static_cast<char*>(basePtr) + expectedOffset;
  if (laneBuf.ptr != expectedPtr) {
    *success = false;
  }

  if (laneBuf.lkey_per_device[0] != baseBuf.lkey_per_device[0]) {
    *success = false;
  }

  if (expectedChunk != kChunkSize) {
    *success = false;
  }
}

__global__ void testPutSignalGroupBroadcast(bool* success) {
  *success = true;

  auto group = comms::prims::make_warp_group();
  if (group.group_size != comms::prims::kWarpSize) {
    *success = false;
    return;
  }

  uint64_t signalTicket = 0;
  if (group.is_leader()) {
    signalTicket = 0xCAFEBABE12345678ULL;
  }

  signalTicket = group.broadcast<uint64_t>(signalTicket);

  if (signalTicket != 0xCAFEBABE12345678ULL) {
    *success = false;
  }
}

// =============================================================================
// Group-level test wrapper functions
// =============================================================================

void runTestPutCooperativePartitioning(bool* d_success) {
  testPutCooperativePartitioning<<<1, comms::prims::kWarpSize>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void runTestPutSignalGroupBroadcast(bool* d_success) {
  testPutSignalGroupBroadcast<<<1, comms::prims::kWarpSize>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// broadcast test kernels for BLOCK and MULTIWARP scopes
// =============================================================================

__global__ void testBroadcast64Block(bool* success) {
  auto group = comms::prims::make_block_group();

  uint64_t val = 0;
  if (group.is_leader()) {
    val = 0xDEADBEEF42424242ULL;
  }

  val = group.broadcast<uint64_t>(val);

  if (val != 0xDEADBEEF42424242ULL) {
    *success = false;
  }
}

__global__ void testBroadcast64Multiwarp(bool* success) {
  auto group = comms::prims::make_multiwarp_group();

  uint64_t val = 0;
  if (group.is_leader()) {
    val = 0xAAAABBBB00000000ULL + group.group_id;
  }

  val = group.broadcast<uint64_t>(val);

  uint64_t expected = 0xAAAABBBB00000000ULL + group.group_id;
  if (val != expected) {
    *success = false;
  }
}

__global__ void testBroadcast64DoubleSafety(bool* success) {
  auto group = comms::prims::make_block_group();

  uint64_t val1 = 0;
  if (group.is_leader()) {
    val1 = 0x1111111111111111ULL;
  }
  val1 = group.broadcast<uint64_t>(val1);

  if (val1 != 0x1111111111111111ULL) {
    *success = false;
  }

  uint64_t val2 = 0;
  if (group.is_leader()) {
    val2 = 0x2222222222222222ULL;
  }
  val2 = group.broadcast<uint64_t>(val2);

  if (val2 != 0x2222222222222222ULL) {
    *success = false;
  }
}

__global__ void testPutCooperativePartitioningBlock(bool* success) {
  auto group = comms::prims::make_block_group();

  constexpr std::size_t kTotalBytes = 4096; // 4KB
  std::size_t chunkSize = kTotalBytes / group.group_size;
  std::size_t expectedOffset = group.thread_id_in_group * chunkSize;

  char baseData[8];
  void* basePtr = baseData;

  comms::prims::IbgdaLocalBuffer baseBuf(
      basePtr, comms::prims::NetworkLKeys{comms::prims::NetworkLKey(0x1111)});
  comms::prims::IbgdaLocalBuffer laneBuf = baseBuf.subBuffer(expectedOffset);

  auto* expectedPtr = static_cast<char*>(basePtr) + expectedOffset;
  if (laneBuf.ptr != expectedPtr) {
    *success = false;
  }

  if (laneBuf.lkey_per_device[0] != baseBuf.lkey_per_device[0]) {
    *success = false;
  }
}

// =============================================================================
// broadcast / block-scope test wrapper functions
// =============================================================================

void runTestBroadcast64Block(bool* d_success) {
  testBroadcast64Block<<<4, 256>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void runTestBroadcast64Multiwarp(bool* d_success) {
  testBroadcast64Multiwarp<<<2, 512>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void runTestBroadcast64DoubleSafety(bool* d_success) {
  testBroadcast64DoubleSafety<<<4, 256>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void runTestPutCooperativePartitioningBlock(bool* d_success) {
  testPutCooperativePartitioningBlock<<<4, 256>>>(d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// trace_ibgda_event test kernel
// =============================================================================

__global__ void testTraceIbgdaEvent(PipesTraceHandle trace) {
#if PIPES_IS_DEVICE_COMPILE
  trace_ibgda_event(
      trace,
      /*self_rank=*/7,
      PipesTraceEventType::kIbSendBegin,
      /*step=*/0x12345678,
      /*group_id=*/0x4321);
#endif
}

void runTestTraceIbgdaEvent(PipesTraceHandle trace) {
  testTraceIbgdaEvent<<<1, 1>>>(trace);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// wait_signal abortDevice test kernels
// =============================================================================

__global__ void testWaitSignalTimeout(
    uint64_t* d_signalBuf,
    AbortDevice abortDevice) {
  // Start the abortDevice timer
  abortDevice.start();

  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  // Signal buffer is pre-set to 0 by host.
  // Waiting for >= 999 will never succeed, so abortDevice should fire.
  transport.wait_signal(0, 999, abortDevice);
}

__global__ void testWaitSignalNoTimeout(
    uint64_t* d_signalBuf,
    AbortDevice abortDevice,
    bool* success) {
  // Start the abortDevice timer
  abortDevice.start();

  // Construct transport with ownedLocalSignalBuf
  IbgdaLocalBuffer localSigBuf(d_signalBuf, NetworkLKeys{});
  P2pIbgdaTransportDevice transport(
      DeviceSpan<NicDeviceIbgdaResources>{},
      IbgdaRemoteBuffer{},
      localSigBuf,
      IbgdaLocalBuffer{},
      1);

  // Signal buffer is pre-set to 42 by host.
  // Waiting for >= 42 will succeed immediately, no abort handle.
  transport.wait_signal(0, 42, abortDevice);

  *success = true;
}

// =============================================================================
// wait_signal abortDevice test wrapper functions
// =============================================================================

cudaError_t runTestWaitSignalTimeout(
    uint64_t* d_signalBuf,
    int device,
    uint32_t timeout_ms) {
  auto status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }
  comms::fault_tolerance::Abort abort{
      /*enabled=*/true, comms::fault_tolerance::AbortBehavior::TRAP};
  abort.setDefaultTimeout(std::chrono::milliseconds{timeout_ms});
  AbortDevice abortDevice = abort.getDeviceHandle();

  // Intentionally unchecked - we expect the kernel to trap
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  testWaitSignalTimeout<<<1, 1>>>(d_signalBuf, abortDevice);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  return cudaDeviceSynchronize();
}

void runTestWaitSignalNoTimeout(
    uint64_t* d_signalBuf,
    int /*device*/,
    uint32_t /*timeout_ms*/,
    bool* d_success) {
  AbortDevice abortDevice;

  testWaitSignalNoTimeout<<<1, 1>>>(d_signalBuf, abortDevice, d_success);
  PIPES_KERNEL_LAUNCH_CHECK();
}

#ifndef __HIP_PLATFORM_AMD__
__global__ void testCollapsedCqPoll(
    doca_gpu_dev_verbs_cq* cq,
    uint64_t ticket,
    bool blocking,
    bool abortAware,
    bool collapsedCq,
    bool gpuSharing,
    comms::fault_tolerance::AbortDevice abort,
    CollapsedCqPollResult* result) {
  if (blocking) {
    result->status = prims_ibgda_wait_collapsed_cq<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>(cq, ticket);
    result->aborted = 0;
  } else if (abortAware) {
    const auto pollResult = gpuSharing
        ? detail::pollIbgdaSqOnce<
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
              DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>(
              cq, ticket, collapsedCq, abort)
        : detail::pollIbgdaSqOnce<
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
              DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>(
              cq, ticket, collapsedCq, abort);
    result->status = pollResult.status;
    result->aborted = pollResult.aborted ? 1U : 0U;
  } else {
    result->status = prims_ibgda_poll_collapsed_cq_once<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>(cq, ticket);
    result->aborted = 0;
  }
  result->finalConsumerIndex = cq->cqe_ci;
}

namespace {

cudaError_t runTestIbgdaSqPoll(
    const CollapsedCqPollCase& testCase,
    bool abortAware,
    bool collapsedCq,
    bool gpuSharing,
    comms::fault_tolerance::AbortDevice abort,
    CollapsedCqPollResult* result) {
  doca_gpunetio_ib_mlx5_cqe64 hostCqe{};
  hostCqe.wqe_counter = static_cast<__be16>(
      (testCase.wqeCounter >> 8) | (testCase.wqeCounter << 8));
  hostCqe.op_own = static_cast<uint8_t>(
      testCase.opcode << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT);
  if (!collapsedCq) {
    hostCqe.op_own |= DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK;
  }

  doca_gpunetio_ib_mlx5_cqe64* deviceCqe = nullptr;
  doca_gpu_dev_verbs_cq* deviceCq = nullptr;
  CollapsedCqPollResult* deviceResult = nullptr;
  cudaError_t status = cudaMalloc(&deviceCqe, sizeof(hostCqe));
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaMalloc(&deviceCq, sizeof(doca_gpu_dev_verbs_cq));
  if (status != cudaSuccess) {
    cudaFree(deviceCqe);
    return status;
  }
  status = cudaMalloc(&deviceResult, sizeof(CollapsedCqPollResult));
  if (status != cudaSuccess) {
    cudaFree(deviceCq);
    cudaFree(deviceCqe);
    return status;
  }

  doca_gpu_dev_verbs_cq hostCq{};
  hostCq.cqe_daddr = reinterpret_cast<uint8_t*>(deviceCqe);
  hostCq.cqe_num = testCase.cqeCount;
  hostCq.cqe_ci = testCase.initialConsumerIndex;
  status =
      cudaMemcpy(deviceCqe, &hostCqe, sizeof(hostCqe), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    status =
        cudaMemcpy(deviceCq, &hostCq, sizeof(hostCq), cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    testCollapsedCqPoll<<<1, 1>>>(
        deviceCq,
        testCase.ticket,
        testCase.blocking,
        abortAware,
        collapsedCq,
        gpuSharing,
        abort,
        deviceResult);
    status = cudaDeviceSynchronize();
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(
        result,
        deviceResult,
        sizeof(CollapsedCqPollResult),
        cudaMemcpyDeviceToHost);
  }

  const cudaError_t resultFreeStatus = cudaFree(deviceResult);
  const cudaError_t cqFreeStatus = cudaFree(deviceCq);
  const cudaError_t cqeFreeStatus = cudaFree(deviceCqe);
  if (status != cudaSuccess) {
    return status;
  }
  if (resultFreeStatus != cudaSuccess) {
    return resultFreeStatus;
  }
  if (cqFreeStatus != cudaSuccess) {
    return cqFreeStatus;
  }
  return cqeFreeStatus;
}

} // namespace

cudaError_t runTestCollapsedCqPoll(
    const CollapsedCqPollCase& testCase,
    CollapsedCqPollResult* result) {
  return runTestIbgdaSqPoll(
      testCase,
      /*abortAware=*/false,
      /*collapsedCq=*/true,
      /*gpuSharing=*/true,
      comms::fault_tolerance::AbortDevice{},
      result);
}

cudaError_t runTestIbgdaSqPollWithAbort(
    const CollapsedCqPollCase& testCase,
    bool collapsedCq,
    bool gpuSharing,
    comms::fault_tolerance::AbortDevice abort,
    CollapsedCqPollResult* result) {
  return runTestIbgdaSqPoll(
      testCase,
      /*abortAware=*/true,
      collapsedCq,
      gpuSharing,
      abort,
      result);
}

namespace {

constexpr uint8_t kDataOnlyWqeCanary = 0xa5;

struct DataOnlySqAbortState {
  doca_gpu_dev_verbs_qp qp{};
  doca_gpu_dev_verbs_qp* qps[kIbDirections]{};
  doca_gpu_dev_verbs_qp* companionQps[kIbDirections]{};
  IbLocalChannel channel{};
  doca_gpunetio_ib_mlx5_cqe64 cqe{};
  doca_gpu_dev_verbs_wqe wqe{};
  __be32 doorbellRecord{};
  uint64_t doorbell{};
  uint8_t localData{};
  uint8_t remoteData{};
  DataOnlySqAbortResult result{};
};

__global__ void testDataOnlySqReservationAbort(
    DataOnlySqAbortState* state,
    uint32_t* enteredReservation,
    comms::fault_tolerance::AbortDevice abort) {
  if (threadIdx.x == 0) {
    auto* wqeBytes = reinterpret_cast<uint8_t*>(&state->wqe);
    for (std::size_t i = 0; i < sizeof(state->wqe); ++i) {
      wqeBytes[i] = kDataOnlyWqeCanary;
    }

    state->cqe.op_own = static_cast<uint8_t>(
        (15U << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) |
        DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK);
    state->qp.sq_rsvd_index = 1;
    state->qp.sq_ready_index = 0;
    state->qp.sq_wqe_pi = 0;
    state->qp.sq_wqe_num = 1;
    state->qp.sq_wqe_mask = 0;
    state->qp.sq_wqe_daddr = reinterpret_cast<uint8_t*>(&state->wqe);
    state->qp.sq_dbrec = &state->doorbellRecord;
    state->qp.sq_db = &state->doorbell;
    state->qp.cq_sq.cqe_daddr = reinterpret_cast<uint8_t*>(&state->cqe);
    state->qp.cq_sq.cqe_num = 1;
    state->qp.cq_sq.cqe_mask = 0;
    state->qp.cq_sq.cqe_ci = 0;
    for (int i = 0; i < kIbDirections; ++i) {
      state->qps[i] = &state->qp;
      state->companionQps[i] = &state->qp;
    }
  }
  __syncthreads();

  if (threadIdx.x < comms::prims::kWarpSize) {
    auto group = comms::prims::make_warp_group();
    NicDeviceIbgdaResources nic{
        .qps = DeviceSpan<doca_gpu_dev_verbs_qp*>(state->qps, kIbDirections),
        .companion_qps = DeviceSpan<doca_gpu_dev_verbs_qp*>(
            state->companionQps, kIbDirections),
    };
    P2pIbgdaTransportDevice transport(
        DeviceSpan<NicDeviceIbgdaResources>(&nic, 1),
        IbgdaRemoteBuffer{},
        IbgdaLocalBuffer{},
        IbgdaLocalBuffer{},
        /*numSignalSlots=*/0,
        /*numCounterSlots=*/0,
        /*maxChannels=*/1,
        /*qpsPerConnection=*/1,
        /*qpDirectionCount=*/kIbDirections,
        DeviceSpan<IbLocalChannel>(&state->channel, 1),
        IbChannelLayout{},
        /*collapsedCq=*/false);
    const IbgdaLocalBuffer localBuf(
        &state->localData, NetworkLKeys{NetworkLKey{0x1111}});
    const IbgdaRemoteBuffer remoteBuf(
        &state->remoteData, NetworkRKeys{NetworkRKey{0x2222}});

    if (group.is_leader()) {
      state->result.prePutAbortClear = abort.isAborted() ? 0U : 1U;
    }
    group.sync();
    const auto completion = transport.put(
        group,
        localBuf,
        remoteBuf,
        /*nbytes=*/1,
        IbgdaRemoteBuffer{},
        /*signalVal=*/1,
        IbgdaLocalBuffer{},
        /*counterVal=*/1,
        /*signalPerLane=*/false,
        abort);

    if (group.is_leader()) {
      state->result.posted = completion.posted ? 1U : 0U;
      state->result.reservedIndex = state->qp.sq_rsvd_index;
      state->result.readyIndex = state->qp.sq_ready_index;
      state->result.producerIndex = state->qp.sq_wqe_pi;
      state->result.doorbellRecord = state->doorbellRecord;
      state->result.doorbell = state->doorbell;
      state->result.pendingFlushLanesMask =
          state->channel.sendQp.pendingFlushLanesMask;
      state->result.wqeUnchanged = 1U;
      const auto* wqeBytes = reinterpret_cast<const uint8_t*>(&state->wqe);
      for (std::size_t i = 0; i < sizeof(state->wqe); ++i) {
        if (wqeBytes[i] != kDataOnlyWqeCanary) {
          state->result.wqeUnchanged = 0U;
          break;
        }
      }
    }
    return;
  }

  if (threadIdx.x == comms::prims::kWarpSize) {
    while (atomicAdd(
               reinterpret_cast<unsigned long long*>(&state->qp.sq_rsvd_index),
               0ULL) < 2ULL) {
      if (abort.checkExpired()) {
        return;
      }
    }
    __threadfence_system();
    *static_cast<volatile uint32_t*>(enteredReservation) = 1U;
  }
}

} // namespace

cudaError_t runTestDataOnlySqReservationAbort(
    comms::fault_tolerance::Abort& abort,
    DataOnlySqAbortResult* result) {
  DataOnlySqAbortState* state = nullptr;
  uint32_t* hostEnteredReservation = nullptr;
  uint32_t* deviceEnteredReservation = nullptr;

  cudaError_t status = cudaMalloc(&state, sizeof(*state));
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaMemset(state, 0, sizeof(*state));
  if (status != cudaSuccess) {
    cudaFree(state);
    return status;
  }
  status = cudaHostAlloc(
      reinterpret_cast<void**>(&hostEnteredReservation),
      sizeof(*hostEnteredReservation),
      cudaHostAllocMapped);
  if (status != cudaSuccess) {
    cudaFree(state);
    return status;
  }
  *hostEnteredReservation = 0;
  status = cudaHostGetDevicePointer(
      reinterpret_cast<void**>(&deviceEnteredReservation),
      hostEnteredReservation,
      0);
  if (status == cudaSuccess) {
    testDataOnlySqReservationAbort<<<1, 2 * comms::prims::kWarpSize>>>(
        state, deviceEnteredReservation, abort.getDeviceHandle());
    status = cudaGetLastError();
  }

  bool reservationObserved = false;
  if (status == cudaSuccess) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
      if (__atomic_load_n(hostEnteredReservation, __ATOMIC_ACQUIRE) != 0U) {
        reservationObserved = true;
        break;
      }
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  abort.setAbort();
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(
        result, &state->result, sizeof(*result), cudaMemcpyDeviceToHost);
  }
  if (status == cudaSuccess) {
    result->reservationObserved = reservationObserved ? 1U : 0U;
  }

  const cudaError_t flagFreeStatus = cudaFreeHost(hostEnteredReservation);
  const cudaError_t stateFreeStatus = cudaFree(state);
  if (status != cudaSuccess) {
    return status;
  }
  if (flagFreeStatus != cudaSuccess) {
    return flagFreeStatus;
  }
  return stateFreeStatus;
}
#endif

} // namespace comms::prims::tests
