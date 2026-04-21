// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// AMD GPU (HIP/ROCm) P2pIbgdaTransportDevice
// =============================================================================
//
// AMD/HIP port of comms::pipes::P2pIbgdaTransportDevice.
//
// Provides the same device-side per-peer RDMA transport API as the NVIDIA
// version, but uses AMD GCN/CDNA intrinsics and direct WQE construction
// instead of DOCA GPUNetIO high-level APIs.
//
// TEMPLATE ARCHITECTURE:
// P2pIbgdaTransportDeviceImpl<NicBackend> is parameterized on a NIC backend
// that provides all NIC-specific WQE format, doorbell, and CQ polling logic.
// See nic/Mlx5NicBackend.h, nic/BnxtNicBackend.h, nic/IonicNicBackend.h.
//
// nic/NicSelector.h provides the compile-time type alias:
//   using P2pIbgdaTransportDevice =
//   P2pIbgdaTransportDeviceImpl<ActiveNicBackend>;
//
// IMPLEMENTATION STYLE:
// Methods use pipes_gda_gpu_dev_verbs_* helper functions from verbs/VerbsOps.h
// (equivalent to DOCA GPUNetIO doca_gpu_dev_verbs_* APIs) so that the code
// reads similarly to comms::pipes::P2pIbgdaTransportDevice.
//
// The public API is identical to comms::pipes::P2pIbgdaTransportDevice.
// =============================================================================

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "PipesGdaShared.h" // @manual
#include "nic/NicSelector.h" // @manual
#include "verbs/VerbsDev.h" // @manual
#include "verbs/VerbsOps.h" // @manual

namespace pipes_gda {

// Default timeout for internal synchronous waits (e.g., reset_signal).
// 10 billion cycles ≈ 5-7 seconds on typical GPU clocks (~1.5-1.8 GHz).
inline constexpr uint64_t kDefaultDeviceTimeoutCycles = 10'000'000'000ULL;

// =============================================================================
// IbgdaWork - Operation handle for tracking RDMA completion
// =============================================================================

struct IbgdaWork {
  uint64_t value{0};

  IbgdaWork() = default;

  __device__ explicit IbgdaWork(uint64_t ticket) : value(ticket) {}
};

// =============================================================================
// P2pIbgdaTransportDeviceImpl - Device-side per-peer RDMA transport handle
// =============================================================================

template <typename NicBackend>
class P2pIbgdaTransportDeviceImpl {
 public:
  P2pIbgdaTransportDeviceImpl() = default;

  __host__ __device__ P2pIbgdaTransportDeviceImpl(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_qp* companionQp = nullptr,
      NetworkLKey sinkLkey = NetworkLKey{},
      void* sinkBufPtr = nullptr)
      : qp_(qp),
        companionQp_(companionQp),
        sinkLkey_(sinkLkey),
        sinkBufPtr_(sinkBufPtr) {}

  // ===========================================================================
  // put - RDMA Write without signal (non-blocking)
  // ===========================================================================
  __device__ IbgdaWork
  put(const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    uint64_t ticket;

    pipes_gda_gpu_dev_verbs_addr localAddr = {
        .addr = reinterpret_cast<uint64_t>(localBuf.ptr),
        .key = localBuf.lkey.value};
    pipes_gda_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(remoteBuf.ptr),
        .key = remoteBuf.rkey.value};

    pipes_gda_gpu_dev_verbs_put(
        nic_, qp_, remoteAddr, localAddr, nbytes, &ticket);

    return IbgdaWork(ticket);
  }

  // ===========================================================================
  // put_group_local - Group-collaborative RDMA Write (group-local data)
  // ===========================================================================
  __device__ IbgdaWork put_group_local(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    std::size_t chunkSize = nbytes / group.group_size;
    std::size_t offset = group.thread_id_in_group * chunkSize;
    std::size_t laneBytes = (group.thread_id_in_group == group.group_size - 1)
        ? (nbytes - offset)
        : chunkSize;

    IbgdaLocalBuffer laneBuf = localBuf.subBuffer(offset);
    IbgdaRemoteBuffer laneRemoteBuf = remoteBuf.subBuffer(offset);

    if (group.group_size == 1) {
      return put(laneBuf, laneRemoteBuf, laneBytes);
    }
    return put_group_impl(group, laneBuf, laneRemoteBuf, laneBytes);
  }

  // ===========================================================================
  // put_group_global - Group-collaborative RDMA Write (global data)
  // ===========================================================================
  __device__ IbgdaWork put_group_global(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes) {
    std::size_t chunkPerGroup = nbytes / group.total_groups;
    std::size_t groupOffset = group.group_id * chunkPerGroup;
    std::size_t groupBytes = (group.group_id == group.total_groups - 1)
        ? (nbytes - groupOffset)
        : chunkPerGroup;

    IbgdaLocalBuffer groupLocalBuf = localBuf.subBuffer(groupOffset);
    IbgdaRemoteBuffer groupRemoteBuf = remoteBuf.subBuffer(groupOffset);

    return put_group_local(group, groupLocalBuf, groupRemoteBuf, groupBytes);
  }

  // ===========================================================================
  // Compound Put + Signal APIs (caller-provided signal buffers)
  // ===========================================================================

  __device__ IbgdaWork put_signal(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId,
      uint64_t signalVal) {
    put(localBuf, remoteBuf, nbytes);
    return signal_remote_with_fence(remoteSignalBuf, signalId, signalVal);
  }

  __device__ IbgdaWork put_signal_group_local(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId,
      uint64_t signalVal) {
    std::size_t chunkSize = nbytes / group.group_size;
    std::size_t offset = group.thread_id_in_group * chunkSize;
    std::size_t laneBytes = (group.thread_id_in_group == group.group_size - 1)
        ? (nbytes - offset)
        : chunkSize;

    IbgdaLocalBuffer laneBuf = localBuf.subBuffer(offset);
    IbgdaRemoteBuffer laneRemoteBuf = remoteBuf.subBuffer(offset);

    if (group.group_size == 1) {
      put(laneBuf, laneRemoteBuf, laneBytes);
      return signal_remote_with_fence(remoteSignalBuf, signalId, signalVal);
    }

    put_group_impl(group, laneBuf, laneRemoteBuf, laneBytes);

    uint64_t signalTicket = 0;
    if (group.is_leader()) {
      IbgdaWork signalWork =
          signal_remote_with_fence(remoteSignalBuf, signalId, signalVal);
      signalTicket = signalWork.value;
    }
    signalTicket = group.broadcast<uint64_t>(signalTicket);
    return IbgdaWork(signalTicket);
  }

  __device__ IbgdaWork put_signal_group_global(
      ThreadGroup& group,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId,
      uint64_t signalVal) {
    std::size_t chunkPerGroup = nbytes / group.total_groups;
    std::size_t groupOffset = group.group_id * chunkPerGroup;
    std::size_t groupBytes = (group.group_id == group.total_groups - 1)
        ? (nbytes - groupOffset)
        : chunkPerGroup;

    IbgdaLocalBuffer groupLocalBuf = localBuf.subBuffer(groupOffset);
    IbgdaRemoteBuffer groupRemoteBuf = remoteBuf.subBuffer(groupOffset);

    return put_signal_group_local(
        group,
        groupLocalBuf,
        groupRemoteBuf,
        groupBytes,
        remoteSignalBuf,
        signalId,
        signalVal);
  }

  // ===========================================================================
  // Local Signal Operations (caller-provided local signal buffer)
  // ===========================================================================

  __device__ void wait_signal(
      const IbgdaLocalBuffer& localSignalBuf,
      int signalId,
      uint64_t expected,
      const Timeout& timeout = Timeout()) {
    volatile uint64_t* sig =
        static_cast<volatile uint64_t*>(localSignalBuf.ptr) + signalId;
    while (*sig < expected) {
      TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
          timeout,
          "wait_signal(GE): signalId=%d, expected>=%llu, current=%llu",
          signalId,
          static_cast<unsigned long long>(expected),
          static_cast<unsigned long long>(*sig));
    }
    __threadfence_system();
  }

  __device__ uint64_t
  read_signal(const IbgdaLocalBuffer& localSignalBuf, int signalId) const {
    volatile uint64_t* sig =
        static_cast<volatile uint64_t*>(localSignalBuf.ptr) + signalId;
    return *sig;
  }

  __device__ void reset_signal(
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId) {
    fence();

    uint64_t ticket;
    pipes_gda_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(remoteSignalBuf.ptr) + signalId),
        .key = remoteSignalBuf.rkey.value};

    pipes_gda_gpu_dev_verbs_p<uint64_t>(
        nic_, qp_, remoteAddr, static_cast<uint64_t>(0), &ticket);

    Timeout timeout(kDefaultDeviceTimeoutCycles);
    timeout.start();
    wait_local(IbgdaWork(ticket), timeout);
  }

  // ===========================================================================
  // Remote Signal / Counter Operations (for window-owned buffers)
  // ===========================================================================

  __device__ IbgdaWork signal_remote(
      const IbgdaRemoteBuffer& remoteBuf,
      int signalId,
      uint64_t value) {
    uint64_t ticket;

    pipes_gda_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(remoteBuf.ptr) + signalId),
        .key = remoteBuf.rkey.value};
    pipes_gda_gpu_dev_verbs_addr sinkAddr = {
        .addr = reinterpret_cast<uint64_t>(sinkBufPtr_),
        .key = sinkLkey_.value};

    pipes_gda_gpu_dev_verbs_signal(
        nic_, qp_, remoteAddr, sinkAddr, value, &ticket);

    return IbgdaWork(ticket);
  }

  __device__ IbgdaWork signal_remote_with_fence(
      const IbgdaRemoteBuffer& remoteBuf,
      int signalId,
      uint64_t value) {
    pipes_gda_gpu_dev_verbs_addr remoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(remoteBuf.ptr) + signalId),
        .key = remoteBuf.rkey.value};
    pipes_gda_gpu_dev_verbs_addr sinkAddr = {
        .addr = reinterpret_cast<uint64_t>(sinkBufPtr_),
        .key = sinkLkey_.value};

    uint64_t wqeIdx = pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic_, qp_, 1);
    auto* wqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic_, qp_, wqeIdx);

    pipes_gda_gpu_dev_verbs_wqe_prepare_atomic(
        nic_,
        qp_,
        wqe,
        wqeIdx,
        static_cast<uint8_t>(
            PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE |
            PIPES_GDA_IB_MLX5_WQE_CTRL_FENCE),
        remoteAddr.addr,
        remoteAddr.key,
        sinkAddr.addr,
        sinkAddr.key,
        sizeof(uint64_t),
        value,
        0);

    pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic_, qp_, wqeIdx, wqeIdx);
    pipes_gda_gpu_dev_verbs_submit(nic_, qp_, wqeIdx + 1);

    return IbgdaWork(wqeIdx);
  }

  __device__ void put_signal_counter_remote(
      const IbgdaLocalBuffer& localDataBuf,
      const IbgdaRemoteBuffer& remoteDataBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId,
      uint64_t signalVal,
      const IbgdaLocalBuffer& localCounterBuf,
      int counterId,
      uint64_t counterVal) {
    pipes_gda_gpu_dev_verbs_addr laddr = {
        .addr = reinterpret_cast<uint64_t>(localDataBuf.ptr),
        .key = localDataBuf.lkey.value};
    pipes_gda_gpu_dev_verbs_addr raddr = {
        .addr = reinterpret_cast<uint64_t>(remoteDataBuf.ptr),
        .key = remoteDataBuf.rkey.value};

    pipes_gda_gpu_dev_verbs_addr sigRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(remoteSignalBuf.ptr) + signalId),
        .key = remoteSignalBuf.rkey.value};
    pipes_gda_gpu_dev_verbs_addr sigSinkAddr = {
        .addr = 0, .key = sinkLkey_.value};

    pipes_gda_gpu_dev_verbs_addr counterRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(localCounterBuf.ptr) + counterId),
        .key = localCounterBuf.lkey.value};
    pipes_gda_gpu_dev_verbs_addr counterSinkAddr = {
        .addr = 0, .key = sinkLkey_.value};

    pipes_gda_gpu_dev_verbs_put_signal_counter(
        nic_,
        qp_,
        raddr,
        laddr,
        nbytes,
        sigRemoteAddr,
        sigSinkAddr,
        signalVal,
        companionQp_,
        counterRemoteAddr,
        counterSinkAddr,
        counterVal);
  }

  __device__ void signal_counter_remote(
      const IbgdaRemoteBuffer& remoteSignalBuf,
      int signalId,
      uint64_t signalVal,
      const IbgdaLocalBuffer& localCounterBuf,
      int counterId,
      uint64_t counterVal) {
    pipes_gda_gpu_dev_verbs_addr sigRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(remoteSignalBuf.ptr) + signalId),
        .key = remoteSignalBuf.rkey.value};
    pipes_gda_gpu_dev_verbs_addr sigSinkAddr = {
        .addr = 0, .key = sinkLkey_.value};

    pipes_gda_gpu_dev_verbs_addr counterRemoteAddr = {
        .addr = reinterpret_cast<uint64_t>(
            static_cast<uint64_t*>(localCounterBuf.ptr) + counterId),
        .key = localCounterBuf.lkey.value};
    pipes_gda_gpu_dev_verbs_addr counterSinkAddr = {
        .addr = 0, .key = sinkLkey_.value};

    pipes_gda_gpu_dev_verbs_signal_counter(
        nic_,
        qp_,
        sigRemoteAddr,
        sigSinkAddr,
        signalVal,
        companionQp_,
        counterRemoteAddr,
        counterSinkAddr,
        counterVal);
  }

  // ===========================================================================
  // wait_local - Wait for local completion of an RDMA operation
  // ===========================================================================
  __device__ void wait_local(
      const IbgdaWork& work,
      Timeout timeout = Timeout()) {
    if (!timeout.isEnabled()) {
      pipes_gda_gpu_dev_verbs_wait(nic_, qp_, work.value);
    } else {
      int status;
      do {
        status = pipes_gda_gpu_dev_verbs_poll_one_cq_at(
            nic_, pipes_gda_gpu_dev_verbs_qp_get_cq_sq(qp_), work.value);
        if (status == EBUSY) {
          TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
              timeout,
              "P2pIbgdaTransportDevice::wait_local timed out "
              "(ticket=%llu)",
              static_cast<unsigned long long>(work.value));
        }
      } while (status == EBUSY);
    }
  }

  // ===========================================================================
  // fence - Wait for all pending RDMA operations to complete at the NIC
  // ===========================================================================
  __device__ void fence() {
    pipes_gda_fence(nic_, qp_);
  }

  __host__ __device__ pipes_gda_gpu_dev_verbs_qp* getQp() const {
    return qp_;
  }

 private:
  // ===========================================================================
  // put_group_impl - Group-collaborative RDMA Write (private helper)
  // ===========================================================================
  __device__ IbgdaWork put_group_impl(
      ThreadGroup& group,
      const IbgdaLocalBuffer& laneBuf,
      const IbgdaRemoteBuffer& laneRemoteBuf,
      std::size_t laneBytes) {
    uint64_t baseWqeIdx = 0;
    if (group.is_leader()) {
      baseWqeIdx =
          pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic_, qp_, group.group_size);
    }

    baseWqeIdx = group.broadcast<uint64_t>(baseWqeIdx);

    uint64_t wqeIdx = baseWqeIdx + group.thread_id_in_group;
    auto* wqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic_, qp_, wqeIdx);

    pipes_gda_gpu_dev_verbs_wqe_prepare_write(
        nic_,
        qp_,
        wqe,
        wqeIdx,
        PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        reinterpret_cast<uint64_t>(laneRemoteBuf.ptr),
        laneRemoteBuf.rkey.value,
        reinterpret_cast<uint64_t>(laneBuf.ptr),
        laneBuf.lkey.value,
        static_cast<uint32_t>(laneBytes));

    group.sync();

    if (group.is_leader()) {
      pipes_gda_gpu_dev_verbs_mark_wqes_ready(
          nic_, qp_, baseWqeIdx, baseWqeIdx + group.group_size - 1);
      pipes_gda_gpu_dev_verbs_submit(nic_, qp_, baseWqeIdx + group.group_size);
    }

    group.sync();

    return IbgdaWork(wqeIdx);
  }

  NicBackend nic_;
  pipes_gda_gpu_dev_verbs_qp* qp_{nullptr};
  pipes_gda_gpu_dev_verbs_qp* companionQp_{nullptr};
  NetworkLKey sinkLkey_{};
  void* sinkBufPtr_{nullptr};
};

// =============================================================================
// Convenience type alias: P2pIbgdaTransportDevice
// =============================================================================

using P2pIbgdaTransportDevice = P2pIbgdaTransportDeviceImpl<ActiveNicBackend>;

} // namespace pipes_gda
