// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/tests/P2pIbTransportDeviceAbortTest.cuh"

#include <cstddef>
#include <type_traits>

#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/common/fault_tolerance/AbortMacros.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/transport/P2pIbTransportDevice.cuh"
#include "comms/prims/transport/P2pIbTransportProgressImpl.cuh"
#include "comms/prims/transport/ibrc/P2pIbrcTransportDevice.cuh"

namespace comms::prims::test {

namespace {

struct PrepareSendSlotProbeChannel {
  IbSendCompletionSlot sendCompletionSlots[1];
};

class PrepareSendSlotProbeTransport {
 public:
  __device__ explicit PrepareSendSlotProbeTransport(
      PrepareSendSlotProbeChannel* slot,
      PrepareSendSlotAbortObservation* observation)
      : slot_(slot), observation_(observation) {}

  template <typename P>
  __device__ PrepareSendSlotProbeChannel& local_channel_slot(
      uint32_t /*channelId*/) {
    return *slot_;
  }

  __device__ uint32_t send_completion_lane_count() const {
    return 1;
  }

  __device__ void wait_local_completion(
      uint32_t /*channelId*/,
      const IbLocalCompletionTicket& /*ticket*/,
      const comms::fault_tolerance::AbortDevice& abort) {
    observation_->waitReason = static_cast<uint32_t>(abort.reason());
  }

  __device__ bool is_local_completion_ready(
      uint32_t /*channelId*/,
      const IbLocalCompletionTicket& /*ticket*/) = delete;

  __device__ bool is_local_completion_ready(
      uint32_t /*channelId*/,
      const IbLocalCompletionTicket& /*ticket*/,
      const comms::fault_tolerance::AbortDevice& abort) {
    observation_->confirmationReason = static_cast<uint32_t>(abort.reason());
    return false;
  }

  __device__ const PrepareSendSlotProbeChannel& slot() const {
    return *slot_;
  }

 private:
  PrepareSendSlotProbeChannel* slot_{nullptr};
  PrepareSendSlotAbortObservation* observation_{nullptr};
};

__global__ void prepareSendSlotAbortForwardingKernel(
    PrepareSendSlotAbortObservation* observation,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ PrepareSendSlotProbeChannel slot;
  if (group.is_leader()) {
    auto& completion = slot.sendCompletionSlots[0];
    completion.generation = 0;
    completion.laneMask = 1;
    completion.values[0] = 1;
  }
  group.sync();
  PrepareSendSlotProbeTransport transport(&slot, observation);

  const bool slotUnretired = detail::prepare_send_slot<protocol::Simple>(
      transport,
      group,
      /*slotId=*/0,
      /*generation=*/1,
      abort);
  if (group.is_leader()) {
    const auto& completion = transport.slot().sendCompletionSlots[0];
    observation->slotUnretired = static_cast<uint32_t>(slotUnretired);
    observation->remainingLaneMask = completion.laneMask;
    observation->generation = completion.generation;
  }
}

// Queue and channel state backing the transport under test.
//
// Must be one block-wide object: `wait_signal` is a group primitive whose
// leader reads this state on behalf of every thread. Per-thread locals would
// give each thread its own copy, so the leader would be polling storage no
// other thread can see.
struct IbrcScratch {
  IbrcCmdQueueDevice queues[2];
  IbLocalChannel channels[1];
};

template <typename Scratch>
__device__ void zeroScratch(ThreadGroup& group, Scratch& scratch) {
  auto* raw = reinterpret_cast<char*>(&scratch);
  for (std::size_t i = group.thread_id_in_group; i < sizeof(Scratch);
       i += group.group_size) {
    raw[i] = 0;
  }
  group.sync();
}

struct VariableWaitScratch {
  IbChannelLayout layout;
  IbLocalChannel channel;
  IbSendCompletionSlot sendCompletions[1];
  alignas(512) char staging[1024];
  alignas(8) char signals[4 * kSendRecvSignalSlotStride];
  alignas(8) char counters[2 * kSendRecvSignalSlotStride];
};

__device__ void initializeVariableWaitScratch(
    ThreadGroup& group,
    VariableWaitScratch& scratch) {
  zeroScratch(group, scratch);
  if (group.is_leader()) {
    scratch.layout.sendStagingBuf =
        IbgdaLocalBuffer{scratch.staging, NetworkLKeys{}};
    scratch.layout.recvStagingBuf =
        IbgdaRemoteBuffer{scratch.staging, NetworkRKeys{}};
    scratch.layout.sendStagingPtr = scratch.staging;
    scratch.layout.recvStagingPtr = scratch.staging;
    scratch.layout.localSignalBuf =
        IbgdaLocalBuffer{scratch.signals, NetworkLKeys{}};
    scratch.layout.remoteSignalBuf =
        IbgdaRemoteBuffer{scratch.signals, NetworkRKeys{}};
    scratch.layout.localCounterBuf =
        IbgdaLocalBuffer{scratch.counters, NetworkLKeys{}};
    scratch.layout.localCounterCompletionBuf =
        IbgdaLocalBuffer{scratch.counters, NetworkLKeys{}};
    scratch.layout.maxChannels = kNumProtoSlots;
    scratch.layout.numChannels = 1;
    scratch.layout.numLanes = 1;
    scratch.layout.pipelineDepth = 1;
    scratch.layout.perChannelSize = 512;
    scratch.layout.perChannelBufferSize = 512;
    scratch.channel = makeIbLocalChannel(
        scratch.layout, /*channelId=*/0, scratch.sendCompletions);
  }
  group.sync();
}

class VariableWaitProbeTransport {
 public:
  __device__ VariableWaitProbeTransport(
      VariableWaitScratch* scratch,
      VariableWaitAbortObservation* observation)
      : scratch_(scratch), observation_(observation) {}

  __device__ const IbChannelLayout& channel_layout() const {
    return scratch_->layout;
  }

  __device__ IbLocalChannel& local_channel(uint32_t /*channelId*/) {
    return scratch_->channel;
  }

  template <typename P>
  __device__ IbChannelProtoSlot& local_channel_slot(uint32_t /*channelId*/) {
    return scratch_->channel.protos[P::kProtoSlot];
  }

  template <typename P>
  __device__ IbChannelProtoSlot& local_channel_slot(ThreadGroup& group) {
    return local_channel_slot<P>(group.group_id);
  }

  __device__ uint32_t send_completion_lane_count() const {
    return 1;
  }

  __device__ void wait_local_completion(
      uint32_t /*channelId*/,
      const IbLocalCompletionTicket& /*ticket*/,
      const comms::fault_tolerance::AbortDevice& /*abort*/) {}

  __device__ bool is_local_completion_ready(
      uint32_t /*channelId*/,
      const IbLocalCompletionTicket& /*ticket*/,
      const comms::fault_tolerance::AbortDevice& /*abort*/) {
    return true;
  }

  __device__ void wait_signal(
      ThreadGroup& group,
      const IbgdaLocalBuffer& /*signalBuf*/,
      uint64_t /*signalVal*/,
      const comms::fault_tolerance::AbortDevice& abort) {
    if (group.is_leader()) {
      abort.setAbort();
      // This is far above the mapped-state polling interval but still turns a
      // broken probe into an observation failure instead of a wedged test.
      constexpr uint64_t kObservationBoundCycles = 10'000'000'000ULL;
      const uint64_t start = clock64();
      ++observation_->waitCallCount;
      bool observedAbort = false;
      while (clock64() - start < kObservationBoundCycles) {
        if (abort.checkExpired()) {
          observedAbort = true;
          break;
        }
      }
      if (observedAbort) {
        ++observation_->waitObservedAbortCount;
      } else {
        ++observation_->waitBoundExpiredCount;
      }
    }
    group.sync();
  }

  __device__ IbLocalCompletionTicket
  put(ThreadGroup& group,
      const IbgdaLocalBuffer& /*localBuf*/,
      const IbgdaRemoteBuffer& /*remoteBuf*/,
      std::size_t /*nbytes*/,
      const IbgdaRemoteBuffer& /*signalBuf*/,
      uint64_t /*signalVal*/,
      const IbgdaLocalBuffer& /*counterBuf*/,
      uint64_t /*counterVal*/,
      bool /*signalPerLane*/,
      const comms::fault_tolerance::AbortDevice& /*abort*/) {
    if (group.is_leader()) {
      ++observation_->putCount;
    }
    return IbLocalCompletionTicket{
        .completionId = 0, .posted = true, .value = 1};
  }

  template <bool HasSignal>
  __device__ IbLocalCompletionTicket put_staged(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& signalBuf,
      uint64_t signalVal,
      const comms::fault_tolerance::AbortDevice& abort) {
    return put(
        group,
        localBuf,
        remoteBuf,
        nbytes,
        HasSignal ? signalBuf : IbgdaRemoteBuffer{},
        HasSignal ? signalVal : 0,
        /*counterBuf=*/{},
        /*counterVal=*/0,
        /*signalPerLane=*/true,
        abort);
  }

  __device__ bool try_signal(
      ThreadGroup& group,
      const IbgdaRemoteBuffer& /*signalBuf*/,
      uint64_t /*signalVal*/,
      IbDirection /*direction*/,
      const comms::fault_tolerance::AbortDevice& /*abort*/) {
    if (group.is_leader()) {
      ++observation_->signalCount;
    }
    return group.broadcast<uint32_t>(1U) != 0U;
  }

 private:
  VariableWaitScratch* scratch_{nullptr};
  VariableWaitAbortObservation* observation_{nullptr};
};

struct VariableWaitProbeCopyOp {
  static constexpr bool kVariableSize = true;

  __device__ static std::size_t max_safe_chunk_size_for_slot(
      std::size_t slotBytes) {
    return slotBytes;
  }

  __device__ static std::size_t worst_case_chunk_stride(
      std::size_t chunkBytes) {
    return chunkBytes;
  }

  __device__ static std::size_t send(
      char* /*dst*/,
      const char* /*src*/,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*dataOff*/,
      VariableWaitAbortObservation* observation) {
    if (group.is_leader()) {
      ++observation->sendCopyCount;
    }
    return nbytes;
  }

  __device__ static void recv(
      char* /*dst*/,
      const char* /*src*/,
      std::size_t /*nbytes*/,
      ThreadGroup& group,
      std::size_t /*dataOff*/,
      VariableWaitAbortObservation* observation) {
    if (group.is_leader()) {
      ++observation->recvCopyCount;
    }
  }
};

class ProgressPostRefusalProbeTransport {
 public:
  __device__ ProgressPostRefusalProbeTransport(
      VariableWaitScratch* scratch,
      ProgressPostRefusalObservation* observation,
      bool refuseDataPost,
      bool refuseCreditPost)
      : scratch_(scratch),
        observation_(observation),
        refuseDataPost_(refuseDataPost),
        refuseCreditPost_(refuseCreditPost) {}

  __device__ const IbChannelLayout& channel_layout() const {
    return scratch_->layout;
  }

  __device__ IbLocalChannel& local_channel(uint32_t /*channelId*/) {
    return scratch_->channel;
  }

  template <typename P>
  __device__ IbChannelProtoSlot& local_channel_slot(uint32_t /*channelId*/) {
    return scratch_->channel.protos[P::kProtoSlot];
  }

  template <typename P>
  __device__ IbChannelProtoSlot& local_channel_slot(ThreadGroup& group) {
    return local_channel_slot<P>(group.group_id);
  }

  __device__ uint32_t send_completion_lane_count() const {
    return 1;
  }

  __device__ void wait_local_completion(
      uint32_t /*channelId*/,
      const IbLocalCompletionTicket& /*ticket*/,
      const comms::fault_tolerance::AbortDevice& /*abort*/) {}

  __device__ bool is_local_completion_ready(
      uint32_t /*channelId*/,
      const IbLocalCompletionTicket& /*ticket*/,
      const comms::fault_tolerance::AbortDevice& /*abort*/) {
    return true;
  }

  __device__ uint64_t read_signal(const IbgdaLocalBuffer& /*signalBuf*/) {
    return UINT64_MAX;
  }

  __device__ IbLocalCompletionTicket
  put(ThreadGroup& group,
      const IbgdaLocalBuffer& /*localBuf*/,
      const IbgdaRemoteBuffer& /*remoteBuf*/,
      std::size_t /*nbytes*/,
      const IbgdaRemoteBuffer& /*signalBuf*/,
      uint64_t /*signalVal*/,
      const IbgdaLocalBuffer& /*counterBuf*/,
      uint64_t /*counterVal*/,
      bool /*signalPerLane*/) {
    if (group.is_leader()) {
      ++observation_->legacyPutCount;
    }
    return IbLocalCompletionTicket{
        .completionId = 0, .posted = true, .value = 9};
  }

  template <bool HasSignal>
  __device__ IbLocalCompletionTicket put_staged(
      ThreadGroup& group,
      const IbgdaLocalBuffer& /*localBuf*/,
      const IbgdaRemoteBuffer& /*remoteBuf*/,
      std::size_t /*nbytes*/,
      const IbgdaRemoteBuffer& /*signalBuf*/,
      uint64_t /*signalVal*/,
      const comms::fault_tolerance::AbortDevice& abort) {
    if (group.is_leader()) {
      ++observation_->abortAwarePutCount;
      if (refuseDataPost_) {
        abort.setAbort();
      }
    }
    const uint32_t posted =
        group.broadcast<uint32_t>(refuseDataPost_ ? 0U : 1U);
    return IbLocalCompletionTicket{
        .completionId = 0, .posted = posted != 0U, .value = 9};
  }

  __device__ void signal(
      ThreadGroup& group,
      const IbgdaRemoteBuffer& /*signalBuf*/,
      uint64_t /*signalVal*/,
      IbDirection /*direction*/) {
    if (group.is_leader()) {
      ++observation_->legacySignalCount;
    }
    group.sync();
  }

  __device__ bool try_signal(
      ThreadGroup& group,
      const IbgdaRemoteBuffer& /*signalBuf*/,
      uint64_t /*signalVal*/,
      IbDirection /*direction*/,
      const comms::fault_tolerance::AbortDevice& abort) {
    if (group.is_leader()) {
      ++observation_->abortAwareSignalCount;
      if (refuseCreditPost_) {
        abort.setAbort();
      }
    }
    return group.broadcast<uint32_t>(refuseCreditPost_ ? 0U : 1U) != 0U;
  }

 private:
  VariableWaitScratch* scratch_{nullptr};
  ProgressPostRefusalObservation* observation_{nullptr};
  bool refuseDataPost_{false};
  bool refuseCreditPost_{false};
};

template <typename Transport>
__device__ bool releaseProgressChunk(
    Transport& transport,
    ThreadGroup& group,
    const comms::fault_tolerance::AbortDevice& abort,
    const detail::RecvChunkAcquisition& view) {
  using Result =
      decltype(detail::progress_recv_release_once<Transport, protocol::Simple>(
          transport, group, abort, view));
  if constexpr (std::is_same_v<Result, bool>) {
    return detail::progress_recv_release_once<Transport, protocol::Simple>(
        transport, group, abort, view);
  } else {
    detail::progress_recv_release_once<Transport, protocol::Simple>(
        transport, group, abort, view);
    return true;
  }
}

struct LlForwardProbeScratch {
  VariableWaitScratch recv;
  VariableWaitScratch fwd;
};

__device__ __align__(512) char gLlForwardRecvStaging[1024];
__device__ __align__(512) char gLlForwardSendStaging[1024];

class LlForwardProbeTransport {
 public:
  __device__ LlForwardProbeTransport(
      VariableWaitScratch* scratch,
      PrepareSendSlotAbortObservation* observation,
      bool refuseCompletion)
      : scratch_(scratch),
        observation_(observation),
        refuseCompletion_(refuseCompletion) {}

  __device__ const IbChannelLayout& channel_layout() const {
    return scratch_->layout;
  }

  __device__ IbLocalChannel& local_channel(uint32_t /*channelId*/) {
    return scratch_->channel;
  }

  template <typename P>
  __device__ IbChannelProtoSlot& local_channel_slot(uint32_t /*channelId*/) {
    return scratch_->channel.protos[P::kProtoSlot];
  }

  template <typename P>
  __device__ IbChannelProtoSlot& local_channel_slot(ThreadGroup& group) {
    return local_channel_slot<P>(group.group_id);
  }

  __device__ uint32_t send_completion_lane_count() const {
    return 1;
  }

  __device__ void wait_local_completion(
      uint32_t /*channelId*/,
      const IbLocalCompletionTicket& /*ticket*/,
      const comms::fault_tolerance::AbortDevice& abort) {
    if (refuseCompletion_) {
      abort.setAbort();
      observation_->waitReason = static_cast<uint32_t>(abort.reason());
    }
  }

  __device__ bool is_local_completion_ready(
      uint32_t /*channelId*/,
      const IbLocalCompletionTicket& /*ticket*/,
      const comms::fault_tolerance::AbortDevice& abort) {
    observation_->confirmationReason = static_cast<uint32_t>(abort.reason());
    return !refuseCompletion_;
  }

  __device__ void wait_signal(
      ThreadGroup& group,
      const IbgdaLocalBuffer& /*signalBuf*/,
      uint64_t /*signalVal*/,
      const comms::fault_tolerance::AbortDevice& /*abort*/) {
    group.sync();
  }

  __device__ bool try_signal(
      ThreadGroup& group,
      const IbgdaRemoteBuffer& /*signalBuf*/,
      uint64_t /*signalVal*/,
      IbDirection /*direction*/,
      const comms::fault_tolerance::AbortDevice& /*abort*/) {
    if (group.is_leader()) {
      ++observation_->predecessorCreditCount;
    }
    return group.broadcast<uint32_t>(1U) != 0U;
  }

  template <bool HasSignal>
  __device__ IbLocalCompletionTicket put_staged(
      ThreadGroup& group,
      const IbgdaLocalBuffer& /*localBuf*/,
      const IbgdaRemoteBuffer& /*remoteBuf*/,
      std::size_t /*nbytes*/,
      const IbgdaRemoteBuffer& /*signalBuf*/,
      uint64_t /*signalVal*/,
      const comms::fault_tolerance::AbortDevice& /*abort*/) {
    static_assert(!HasSignal, "LL forwarding carries no DATA_READY signal");
    if (group.is_leader()) {
      ++observation_->successorPutCount;
    }
    return IbLocalCompletionTicket{
        .completionId = 0, .posted = true, .value = kPostedCompletionValue};
  }

  static constexpr uint64_t kPostedCompletionValue = 9;

 private:
  VariableWaitScratch* scratch_{nullptr};
  PrepareSendSlotAbortObservation* observation_{nullptr};
  bool refuseCompletion_{false};
};

struct LlForwardProbeCopyOp {
  template <typename P>
  __device__ static void forwardLL(
      ThreadGroup& group,
      char* /*dst*/,
      char* /*fwdStaging*/,
      const char* /*recvStaging*/,
      std::size_t /*nbytes*/,
      std::size_t /*dataOff*/,
      typename P::FlagType /*recvFlagVal*/,
      typename P::FlagType /*fwdFlagVal*/,
      PrepareSendSlotAbortObservation* observation = nullptr) {
    if (group.is_leader() && observation != nullptr) {
      ++observation->llForwardCount;
    }
  }
};

__global__ void llForwardPreparationRetirementRefusalKernel(
    PrepareSendSlotAbortObservation* observation,
    comms::fault_tolerance::AbortDevice abort) {
  using P = LlxPacket<4, 4>;
  auto group = make_block_group();
  __shared__ LlForwardProbeScratch scratch;
  initializeVariableWaitScratch(group, scratch.recv);
  initializeVariableWaitScratch(group, scratch.fwd);

  constexpr uint64_t kUnretiredGeneration = 1;
  if (group.is_leader()) {
    scratch.recv.layout.sendStagingBuf =
        IbgdaLocalBuffer{gLlForwardRecvStaging, NetworkLKeys{}};
    scratch.recv.layout.recvStagingBuf =
        IbgdaRemoteBuffer{gLlForwardRecvStaging, NetworkRKeys{}};
    scratch.recv.layout.sendStagingPtr = gLlForwardRecvStaging;
    scratch.recv.layout.recvStagingPtr = gLlForwardRecvStaging;
    scratch.fwd.layout.sendStagingBuf =
        IbgdaLocalBuffer{gLlForwardSendStaging, NetworkLKeys{}};
    scratch.fwd.layout.recvStagingBuf =
        IbgdaRemoteBuffer{gLlForwardSendStaging, NetworkRKeys{}};
    scratch.fwd.layout.sendStagingPtr = gLlForwardSendStaging;
    scratch.fwd.layout.recvStagingPtr = gLlForwardSendStaging;
    auto& completion = scratch.fwd.sendCompletions[0];
    completion.generation = kUnretiredGeneration;
    completion.laneMask = 1;
    completion.values[0] = 1;
    char* recvStaging = gLlForwardRecvStaging +
        static_cast<std::size_t>(protocol::LL::kProtoSlot) *
            detail::pipeline_window(scratch.recv.layout);
    for (std::size_t packet = 0; packet < 2; ++packet) {
      LLImpl<P>::store_flag(
          recvStaging + packet * static_cast<std::size_t>(P::kPacketBytes),
          static_cast<typename P::FlagType>(1));
    }
  }
  group.sync();

  if (group.is_leader()) {
    (void)abort.checkExpired();
  }
  group.sync();

  LlForwardProbeTransport recvTransport(
      &scratch.recv, observation, /*refuseCompletion=*/false);
  LlForwardProbeTransport fwdTransport(
      &scratch.fwd, observation, /*refuseCompletion=*/true);
  detail::forward<LlForwardProbeCopyOp, LlForwardProbeTransport, protocol::LL>(
      recvTransport,
      group,
      /*dst=*/nullptr,
      fwdTransport,
      /*nbytes=*/8,
      /*max_signal_bytes=*/0,
      abort,
      observation);

  if (group.is_leader()) {
    const auto& completion = scratch.fwd.sendCompletions[0];
    observation->slotUnretired =
        completion.generation == kUnretiredGeneration &&
            completion.laneMask == 1
        ? 1U
        : 0U;
    observation->completionRecordCount =
        completion.values[0] == LlForwardProbeTransport::kPostedCompletionValue
        ? 1U
        : 0U;
    observation->remainingLaneMask = completion.laneMask;
    observation->generation = completion.generation;
  }
}

__global__ void llForwardPreparationDataReadyAbortKernel(
    PrepareSendSlotAbortObservation* observation,
    comms::fault_tolerance::AbortDevice abort) {
  using P = LlxPacket<4, 4>;
  auto group = make_block_group();
  __shared__ LlForwardProbeScratch scratch;
  initializeVariableWaitScratch(group, scratch.recv);
  initializeVariableWaitScratch(group, scratch.fwd);

  if (group.is_leader()) {
    scratch.recv.layout.sendStagingBuf =
        IbgdaLocalBuffer{gLlForwardRecvStaging, NetworkLKeys{}};
    scratch.recv.layout.recvStagingBuf =
        IbgdaRemoteBuffer{gLlForwardRecvStaging, NetworkRKeys{}};
    scratch.recv.layout.sendStagingPtr = gLlForwardRecvStaging;
    scratch.recv.layout.recvStagingPtr = gLlForwardRecvStaging;
    scratch.fwd.layout.sendStagingBuf =
        IbgdaLocalBuffer{gLlForwardSendStaging, NetworkLKeys{}};
    scratch.fwd.layout.recvStagingBuf =
        IbgdaRemoteBuffer{gLlForwardSendStaging, NetworkRKeys{}};
    scratch.fwd.layout.sendStagingPtr = gLlForwardSendStaging;
    scratch.fwd.layout.recvStagingPtr = gLlForwardSendStaging;
    char* recvStaging = gLlForwardRecvStaging +
        static_cast<std::size_t>(protocol::LL::kProtoSlot) *
            detail::pipeline_window(scratch.recv.layout);
    for (std::size_t packet = 0; packet < 2; ++packet) {
      LLImpl<P>::store_flag(
          recvStaging + packet * static_cast<std::size_t>(P::kPacketBytes),
          static_cast<typename P::FlagType>(0));
    }
  }
  group.sync();

  LlForwardProbeTransport recvTransport(
      &scratch.recv, observation, /*refuseCompletion=*/false);
  LlForwardProbeTransport fwdTransport(
      &scratch.fwd, observation, /*refuseCompletion=*/false);
  detail::forward<LlForwardProbeCopyOp, LlForwardProbeTransport, protocol::LL>(
      recvTransport,
      group,
      /*dst=*/nullptr,
      fwdTransport,
      /*nbytes=*/8,
      /*max_signal_bytes=*/0,
      abort,
      observation);

  if (group.is_leader()) {
    observation->recvLaneCursor = scratch.recv.channel.recvDataReadyLaneCursor;
  }
}

__device__ P2pIbrcTransportDevice makeLocalIbrcTransport(
    IbrcScratch& scratch,
    comms::fault_tolerance::AbortDevice abort = {}) {
  return P2pIbrcTransportDevice(
      DeviceSpan<IbrcCmdQueueDevice>{scratch.queues, 2},
      /*nics=*/1,
      /*maxChannels=*/1,
      /*qpsPerConnection=*/1,
      DeviceSpan<IbLocalChannel>{scratch.channels, 1},
      /*ownedRemoteSignalBuf=*/{},
      /*ownedLocalSignalBuf=*/{},
      /*ownedCounterDeviceBuf=*/{},
      /*ownedCounterHostBuf=*/{},
      /*numSignalSlots=*/0,
      /*numCounterSlots=*/0,
      /*channelLayout=*/IbChannelLayout{.numChannels = 1},
      abort);
}

constexpr uint32_t kIbrcTestQueueDepth = 4;

// Ring storage for the queue-full test. Global rather than shared because the
// transport reaches pi/ci with system-scope atomics, which are meaningful only
// on memory the host could also map; a __shared__ ring would make the test
// depend on undefined behavior rather than on the code under test.
__device__ IbrcDesc gTestDescs[kIbrcTestQueueDepth];
__device__ uint64_t gTestPi;
__device__ uint64_t gTestCi;

// Which of the two production call paths reaches the same IBRC wait.
enum class IbEntryPoint { Wrapper, Ibrc };

template <typename Transport>
__device__ bool releaseMemberProgressChunk(
    Transport& transport,
    ThreadGroup& group,
    const comms::fault_tolerance::AbortDevice& abort,
    const detail::RecvChunkAcquisition& view) {
  using Result =
      decltype(transport.progress_recv_release_once(group, abort, view));
  if constexpr (std::is_same_v<Result, bool>) {
    return transport.progress_recv_release_once(group, abort, view);
  } else {
    transport.progress_recv_release_once(group, abort, view);
    return true;
  }
}

template <IbEntryPoint kEntry>
__global__ void waitSignalKernel(
    uint64_t* signal,
    bool* waitResult,
    uint64_t expected,
    comms::fault_tolerance::AbortDevice abort,
    uint32_t* enteredWait) {
  auto group = make_block_group();
  __shared__ IbrcScratch scratch;
  zeroScratch(group, scratch);

  P2pIbrcTransportDevice ibrc = makeLocalIbrcTransport(scratch);
  IbgdaLocalBuffer localSignal{signal, NetworkLKeys{}};

  abort.start();
  // Published last, after the handle is armed, so a host that sees it knows
  // every precondition of the wait is already in place.
  if (group.is_leader() && enteredWait != nullptr) {
    __threadfence_system();
    *static_cast<volatile uint32_t*>(enteredWait) = 1U;
  }
  group.sync();

  if constexpr (kEntry == IbEntryPoint::Wrapper) {
    P2pIbTransportDevice transport(&ibrc);
    transport.wait_signal(group, localSignal, expected, abort);
  } else {
    ibrc.wait_signal(group, localSignal, expected, abort);
  }
  // Reaching this line at all is the liveness guarantee under test: the wait
  // reports no status, so a wait that failed to terminate hangs the kernel
  // rather than returning something for the host to inspect.
  //
  // Whether the condition actually held on exit is read back from the signal
  // itself. That is what separates "the signal arrived" from "the abort
  // released us", without the wait having to hand back a status.
  if (group.is_leader()) {
    const uint64_t current = *static_cast<volatile uint64_t*>(signal);
    *waitResult = current >= expected;
  }
}

// Body of the queue-full producer, shared by the two kernels below. The
// transport is built with `abort` exactly as handed in -- unstarted -- because
// that is what a real IBRC transport holds: the communicator flag, never a
// deadline.
__device__ void runPutUntilQueueFull(
    ThreadGroup& group,
    IbrcScratch& scratch,
    uint64_t* dataBuf,
    uint32_t* postedOut,
    uint32_t attempts,
    comms::fault_tolerance::AbortDevice abort) {
  if (group.is_leader()) {
    gTestPi = 0;
    gTestCi = 0;
    for (uint32_t i = 0; i < kIbrcTestQueueDepth; ++i) {
      gTestDescs[i].ready_seq = kIbrcInvalidReadySeq;
    }
    // Lane 0 of channel 0 is where a single-NIC, single-QP put lands.
    scratch.queues[0].descs = gTestDescs;
    scratch.queues[0].pi = &gTestPi;
    scratch.queues[0].ci = &gTestCi;
    scratch.queues[0].status = nullptr;
    scratch.queues[0].depth = kIbrcTestQueueDepth;
    scratch.queues[0].mask = kIbrcTestQueueDepth - 1;
  }
  group.sync();

  P2pIbrcTransportDevice ibrc = makeLocalIbrcTransport(scratch, abort);
  IbgdaLocalBuffer localBuf{dataBuf, NetworkLKeys{/*n=*/1}};
  IbgdaRemoteBuffer remoteBuf{dataBuf, NetworkRKeys{/*n=*/1}};

  // Deliberately no start() here. The local copy is not the transport's, so
  // arming it would only look like a deadline while changing nothing -- which
  // is what made the earlier version of these tests pass on the explicit host
  // abort alone.
  uint32_t posted = 0;
  for (uint32_t i = 0; i < attempts; ++i) {
    // A skipped put returns an unposted ticket; a successful enqueue marks it
    // posted before the descriptor can be tracked as an outstanding send.
    const IbLocalCompletionTicket ticket = ibrc.put(
        group,
        localBuf,
        remoteBuf,
        sizeof(uint64_t),
        /*signalBuf=*/IbgdaRemoteBuffer{},
        /*signalVal=*/0);
    if (ticket.posted) {
      ++posted;
    }
  }
  if (group.is_leader()) {
    *postedOut = posted;
  }
}

__global__ void putUntilQueueFullKernel(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    uint32_t attempts,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ IbrcScratch scratch;
  zeroScratch(group, scratch);
  runPutUntilQueueFull(group, scratch, dataBuf, postedOut, attempts, abort);
}

__global__ void wrapperTrySignalKernel(
    uint64_t* signal,
    uint32_t* postedCount,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ IbrcScratch scratch;
  zeroScratch(group, scratch);

  if (group.is_leader()) {
    gTestPi = 0;
    gTestCi = 0;
    for (uint32_t i = 0; i < kIbrcTestQueueDepth; ++i) {
      gTestDescs[i].ready_seq = kIbrcInvalidReadySeq;
    }
    // Recv-direction control traffic uses queue 1 in this one-channel fixture.
    scratch.queues[1].descs = gTestDescs;
    scratch.queues[1].pi = &gTestPi;
    scratch.queues[1].ci = &gTestCi;
    scratch.queues[1].status = nullptr;
    scratch.queues[1].depth = kIbrcTestQueueDepth;
    scratch.queues[1].mask = kIbrcTestQueueDepth - 1;
  }
  group.sync();

  P2pIbrcTransportDevice ibrc = makeLocalIbrcTransport(scratch, abort);
  P2pIbTransportDevice transport(&ibrc);
  IbgdaRemoteBuffer remoteSignal{signal, NetworkRKeys{/*n=*/1}};
  const bool result = transport.try_signal(
      group,
      remoteSignal,
      /*signalVal=*/1,
      IbDirection::Recv,
      abort);
  if (result) {
    atomicAdd(postedCount, 1U);
  }
}

template <IbEntryPoint kEntry>
__global__ void recvReleaseKernel(
    uint64_t* data,
    bool* releaseResult,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ IbrcScratch scratch;
  zeroScratch(group, scratch);
  P2pIbrcTransportDevice ibrc = makeLocalIbrcTransport(scratch, abort);
  const detail::RecvChunkAcquisition view{
      .staging = reinterpret_cast<const char*>(data),
      .validBytes = sizeof(*data),
      .dataOff = 0,
      .protocolBytes = sizeof(*data),
  };
  bool result = false;
  if constexpr (kEntry == IbEntryPoint::Wrapper) {
    P2pIbTransportDevice transport(&ibrc);
    result = releaseMemberProgressChunk(transport, group, abort, view);
  } else {
    result = releaseMemberProgressChunk(ibrc, group, abort, view);
  }
  if (group.is_leader()) {
    *releaseResult = result;
  }
}

__global__ void variableSendWaitAbortKernel(
    VariableWaitAbortObservation* observation,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ VariableWaitScratch scratch;
  initializeVariableWaitScratch(group, scratch);

  VariableWaitProbeTransport transport(&scratch, observation);
  detail::send_impl<
      VariableWaitProbeTransport,
      VariableWaitProbeCopyOp,
      void,
      protocol::Simple>(
      transport,
      group,
      nullptr,
      scratch.staging,
      /*nbytes=*/1024,
      /*max_signal_bytes=*/512,
      abort,
      nullptr,
      observation);
}

__global__ void variableRecvWaitAbortKernel(
    VariableWaitAbortObservation* observation,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ VariableWaitScratch scratch;
  initializeVariableWaitScratch(group, scratch);

  VariableWaitProbeTransport transport(&scratch, observation);
  detail::recv_impl<
      VariableWaitProbeTransport,
      VariableWaitProbeCopyOp,
      void,
      protocol::Simple>(
      transport,
      group,
      nullptr,
      scratch.staging,
      /*nbytes=*/512,
      /*max_signal_bytes=*/512,
      abort,
      nullptr,
      observation);
}

__global__ void progressSendPostRefusalKernel(
    ProgressPostRefusalObservation* observation,
    comms::fault_tolerance::AbortDevice abort,
    bool refuse) {
  auto group = make_block_group();
  __shared__ VariableWaitScratch scratch;
  initializeVariableWaitScratch(group, scratch);
  ProgressPostRefusalProbeTransport transport(
      &scratch,
      observation,
      /*refuseDataPost=*/refuse,
      /*refuseCreditPost=*/false);

  detail::
      init_send_progress<ProgressPostRefusalProbeTransport, protocol::Simple>(
          transport,
          group,
          scratch.staging,
          /*nbytes=*/64,
          /*max_signal_bytes=*/64);
  const auto status = detail::progress_send_once<
      ProgressPostRefusalProbeTransport,
      Memcpy,
      protocol::Simple>(transport, group, abort);
  if (group.is_leader()) {
    const auto& slot = scratch.channel.protos[protocol::Simple::kProtoSlot];
    observation->status = static_cast<uint32_t>(status);
    observation->finalStage =
        static_cast<uint32_t>(slot.sendProgress.activeStage);
    observation->completionRecordCount =
        slot.sendCompletionSlots[0].laneMask != 0 ? 1U : 0U;
    observation->completed = 1U;
  }
}

__global__ void registeredProgressSendPostRefusalKernel(
    ProgressPostRefusalObservation* observation,
    comms::fault_tolerance::AbortDevice abort,
    bool refuse) {
  auto group = make_block_group();
  __shared__ VariableWaitScratch scratch;
  initializeVariableWaitScratch(group, scratch);
  ProgressPostRefusalProbeTransport transport(
      &scratch,
      observation,
      /*refuseDataPost=*/refuse,
      /*refuseCreditPost=*/false);

  detail::init_registered_send_progress(
      transport,
      group,
      IbgdaLocalBuffer{scratch.staging, NetworkLKeys{}},
      /*nbytes=*/64,
      /*max_signal_bytes=*/64);
  const auto status =
      detail::progress_registered_send_once(transport, group, abort);
  if (group.is_leader()) {
    const auto& slot = scratch.channel.protos[protocol::Simple::kProtoSlot];
    observation->status = static_cast<uint32_t>(status);
    observation->finalStage =
        static_cast<uint32_t>(slot.sendProgress.activeStage);
    observation->completionRecordCount =
        slot.sendCompletionSlots[0].laneMask != 0 ? 1U : 0U;
    observation->completed = 1U;
  }
}

__global__ void progressRecvCreditRefusalKernel(
    ProgressPostRefusalObservation* observation,
    comms::fault_tolerance::AbortDevice abort,
    bool refuse) {
  auto group = make_block_group();
  __shared__ VariableWaitScratch scratch;
  initializeVariableWaitScratch(group, scratch);
  ProgressPostRefusalProbeTransport transport(
      &scratch,
      observation,
      /*refuseDataPost=*/false,
      /*refuseCreditPost=*/refuse);

  detail::
      init_recv_progress<ProgressPostRefusalProbeTransport, protocol::Simple>(
          transport,
          group,
          scratch.staging,
          /*nbytes=*/64,
          /*max_signal_bytes=*/64);
  const auto status = detail::progress_recv_once<
      ProgressPostRefusalProbeTransport,
      Memcpy,
      protocol::Simple>(transport, group, abort);
  if (group.is_leader()) {
    const auto& slot = scratch.channel.protos[protocol::Simple::kProtoSlot];
    observation->status = static_cast<uint32_t>(status);
    observation->finalStage =
        static_cast<uint32_t>(slot.recvProgress.activeStage);
    observation->completed = 1U;
  }
}

__global__ void progressRecvReleaseSequenceKernel(
    ProgressPostRefusalObservation* observation,
    comms::fault_tolerance::AbortDevice abort,
    bool refuse) {
  auto group = make_block_group();
  __shared__ VariableWaitScratch scratch;
  initializeVariableWaitScratch(group, scratch);
  ProgressPostRefusalProbeTransport transport(
      &scratch,
      observation,
      /*refuseDataPost=*/false,
      /*refuseCreditPost=*/refuse);
  if (group.is_leader()) {
    scratch.channel.protos[protocol::Simple::kProtoSlot]
        .recvProgress.activeStage =
        detail::IbSendRecvProgressStage::WaitDataReady;
  }
  group.sync();
  const detail::RecvChunkAcquisition view{
      .staging = scratch.staging,
      .validBytes = 64,
      .dataOff = 0,
      .protocolBytes = 64,
  };
  const bool firstPosted = releaseProgressChunk(transport, group, abort, view);
  if (firstPosted) {
    (void)releaseProgressChunk(transport, group, abort, view);
  }
  if (group.is_leader()) {
    const auto& slot =
        scratch.channel.protos[protocol::Simple::kProtoSlot].recvProgress;
    observation->status = firstPosted ? 1U : 0U;
    observation->finalStage = static_cast<uint32_t>(slot.activeStage);
    observation->completed = 1U;
  }
}

// The division of labour the FT contract actually specifies, in one kernel.
//
// Block 0 is an IBRC producer parked in `reserve()` on a full ring, holding a
// flag-only handle. Block 1 is the collective: it owns the deadline, arms the
// only started handle in the launch, and waits on a signal that never arrives.
//
// A block parked in a proxy-facing wait cannot latch its own timeout -- that is
// the property under test. Block 1's deadline expires, latches TIMED_OUT into
// the shared state, and block 0 leaves on the flag. Nothing calls setAbort().
//
// The two blocks race only in the benign direction: block 0 fills a 4-entry
// ring in microseconds against block 1's millisecond deadline, and if that
// order were ever inverted the test fails loudly on the posted count rather
// than hanging.
//
// Block 0 is the producer because the transport derives its channel id from
// the group id, and this fixture is wired for a single channel.
//
// The collective block spins on the signal directly rather than through a
// transport: all it has to model is "a wait in this kernel owns the armed
// handle", and a bare loop does that without needing a second channel's worth
// of queue geometry.
__global__ void queueFullReleasedByCollectiveDeadlineKernel(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    uint32_t attempts,
    uint64_t* signal,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ IbrcScratch scratch;
  zeroScratch(group, scratch);

  if (blockIdx.x == 0) {
    runPutUntilQueueFull(group, scratch, dataBuf, postedOut, attempts, abort);
    return;
  }

  abort.start();
  if (group.is_leader()) {
    const auto* observed = static_cast<volatile uint64_t*>(signal);
    while (*observed < 1U) {
      FT_ABORT_BREAK(abort, "test collective wait on a signal that never sets");
    }
  }
  group.sync();
}

// Ben's case: a kernel that *ends* in flush() rather than parking in reserve().
//
// The proxy never advances `ci`, so the drain cannot complete. Before the
// watchdog was made unconditional, FT-on removed the only bound here -- the
// legacy cycle deadline was gated on `!abort.isEnabled()` and the caller's
// deadline was dropped on the IBRC branch of P2pIbTransportDevice::flush --
// leaving an explicit host abort as the sole exit. Nothing calls setAbort().
__global__ void flushNeverDrainsKernel(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    comms::fault_tolerance::AbortDevice abort) {
  auto group = make_block_group();
  __shared__ IbrcScratch scratch;
  zeroScratch(group, scratch);

  // One successful put, so the drain has something to wait on.
  runPutUntilQueueFull(
      group, scratch, dataBuf, postedOut, /*attempts=*/1, abort);

  P2pIbrcTransportDevice ibrc = makeLocalIbrcTransport(scratch, abort);
  // `ci` is never advanced by anyone, so this can only end on the fixed proxy
  // watchdog -- flush takes no deadline, by design.
  ibrc.flush(group, IbDirection::Send);
}

} // namespace

uint32_t ibrcTestQueueDepth() {
  return kIbrcTestQueueDepth;
}

uint32_t progressSendRecvDoneStatus() {
  return static_cast<uint32_t>(IbgdaSendRecvProgressStatus::Done);
}

uint32_t progressSendRecvAbortedStatus() {
  return static_cast<uint32_t>(IbgdaSendRecvProgressStatus::Aborted);
}

uint32_t registeredSendAbortedStatus() {
  return static_cast<uint32_t>(IbgdaRegisteredSendProgressStatus::Aborted);
}

uint32_t progressDoneStage() {
  return static_cast<uint32_t>(detail::IbSendRecvProgressStage::Done);
}

void launchPrepareSendSlotAbortForwarding(
    PrepareSendSlotAbortObservation* observation,
    comms::fault_tolerance::AbortDevice abort) {
  prepareSendSlotAbortForwardingKernel<<<1, kTestBlockSize>>>(
      observation, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchLlForwardPreparationRetirementRefusal(
    PrepareSendSlotAbortObservation* observation,
    comms::fault_tolerance::AbortDevice abort) {
  llForwardPreparationRetirementRefusalKernel<<<1, 32>>>(observation, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchLlForwardPreparationDataReadyAbort(
    PrepareSendSlotAbortObservation* observation,
    comms::fault_tolerance::AbortDevice abort) {
  llForwardPreparationDataReadyAbortKernel<<<1, 32>>>(observation, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcPutUntilQueueFull(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    uint32_t attempts,
    comms::fault_tolerance::AbortDevice abort) {
  putUntilQueueFullKernel<<<1, kTestBlockSize>>>(
      dataBuf, postedOut, attempts, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcFlushNeverDrains(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    comms::fault_tolerance::AbortDevice abort) {
  flushNeverDrainsKernel<<<1, kTestBlockSize>>>(dataBuf, postedOut, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcQueueFullReleasedByCollectiveDeadline(
    uint64_t* dataBuf,
    uint32_t* postedOut,
    uint32_t attempts,
    uint64_t* signal,
    comms::fault_tolerance::AbortDevice abort) {
  queueFullReleasedByCollectiveDeadlineKernel<<<2, kTestBlockSize>>>(
      dataBuf, postedOut, attempts, signal, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbWrapperWaitSignal(
    uint64_t* signal,
    bool* waitResult,
    uint64_t expected,
    comms::fault_tolerance::AbortDevice abort,
    uint32_t* enteredWait) {
  waitSignalKernel<IbEntryPoint::Wrapper>
      <<<1, kTestBlockSize>>>(signal, waitResult, expected, abort, enteredWait);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcWaitSignal(
    uint64_t* signal,
    bool* waitResult,
    uint64_t expected,
    comms::fault_tolerance::AbortDevice abort,
    uint32_t* enteredWait) {
  waitSignalKernel<IbEntryPoint::Ibrc>
      <<<1, kTestBlockSize>>>(signal, waitResult, expected, abort, enteredWait);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbWrapperTrySignal(
    uint64_t* signal,
    uint32_t* postedCount,
    comms::fault_tolerance::AbortDevice abort) {
  wrapperTrySignalKernel<<<1, kTestBlockSize>>>(signal, postedCount, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbWrapperRecvRelease(
    uint64_t* data,
    bool* releaseResult,
    comms::fault_tolerance::AbortDevice abort) {
  recvReleaseKernel<IbEntryPoint::Wrapper>
      <<<1, kTestBlockSize>>>(data, releaseResult, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchIbrcRecvRelease(
    uint64_t* data,
    bool* releaseResult,
    comms::fault_tolerance::AbortDevice abort) {
  recvReleaseKernel<IbEntryPoint::Ibrc>
      <<<1, kTestBlockSize>>>(data, releaseResult, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchVariableSendWaitAbort(
    VariableWaitAbortObservation* observation,
    comms::fault_tolerance::AbortDevice abort) {
  variableSendWaitAbortKernel<<<1, kTestBlockSize>>>(observation, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchVariableRecvWaitAbort(
    VariableWaitAbortObservation* observation,
    comms::fault_tolerance::AbortDevice abort) {
  variableRecvWaitAbortKernel<<<1, kTestBlockSize>>>(observation, abort);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchProgressSendPostRefusal(
    ProgressPostRefusalObservation* observation,
    comms::fault_tolerance::AbortDevice abort,
    bool refuse) {
  progressSendPostRefusalKernel<<<1, kTestBlockSize>>>(
      observation, abort, refuse);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchRegisteredProgressSendPostRefusal(
    ProgressPostRefusalObservation* observation,
    comms::fault_tolerance::AbortDevice abort,
    bool refuse) {
  registeredProgressSendPostRefusalKernel<<<1, kTestBlockSize>>>(
      observation, abort, refuse);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchProgressRecvCreditRefusal(
    ProgressPostRefusalObservation* observation,
    comms::fault_tolerance::AbortDevice abort,
    bool refuse) {
  progressRecvCreditRefusalKernel<<<1, kTestBlockSize>>>(
      observation, abort, refuse);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchProgressRecvReleaseSequence(
    ProgressPostRefusalObservation* observation,
    comms::fault_tolerance::AbortDevice abort,
    bool refuse) {
  progressRecvReleaseSequenceKernel<<<1, kTestBlockSize>>>(
      observation, abort, refuse);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
