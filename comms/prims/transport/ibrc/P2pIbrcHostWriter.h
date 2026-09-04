// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <fmt/core.h>

#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibrc/IbrcTypes.h"

namespace comms::prims {

/**
 * Host-side writer into a single IBRC command-queue ring.
 *
 * The IBRC command queue is host-mapped memory: the same ring is addressable
 * from GPU device code (via IbrcCmdQueueDevice) and from the CPU (via the host
 * aliases descsHost/piHost/ciHost). Device code posts RDMA work by writing
 * IbrcDesc entries and publishing them; the CPU proxy thread drains the ring
 * (acquire on pi/ready_seq), posts the verbs, and advances ci — and it is
 * agnostic to whether the GPU or the host produced a descriptor.
 *
 * This class lets a CPU thread produce descriptors using the exact same
 * reserve -> fill -> publish protocol as P2pIbrcTransportDevice, so a
 * host-driven collective can post RDMA put/signal with no kernel launched.
 *
 * One instance wraps ONE (qpSlot, nic) ring. IMPORTANT: do not concurrently
 * drive the same ring from device code and this host writer — both atomically
 * increment `pi`, so interleaving is memory-safe but the backpressure/ordering
 * model assumes a single logical producer per ring. Dedicate specific
 * (qpSlot, nic) lanes to the host writer, or quiesce the GPU.
 *
 * Completion:
 *  - counter ("my put completed"): host-mapped u64 the proxy fetch-adds after
 *    the CQE; poll it directly with poll_counter().
 *  - signal ("remote's data arrived"): an RDMA fetch-add the remote peer posts
 *    into a local inbox. Where that inbox lives is the caller's choice and it
 *    is a correctness decision, not a preference:
 *      * host-pinned (allocateIbCounterBuffer, registered through its device
 *        alias) when the host is the only consumer -- poll_counter() reads it
 *        directly;
 *      * device memory when a CUDA stream must also gate on arrival, since
 *        cuStreamWaitValue64 only reads GPU-visible memory. That placement is
 *        also the stronger one: signal and payload then share a PCIe
 *        completer, so observing the signal implies the data is visible in
 *        HBM. A host-pinned signal in that case would need a receiver-side
 *        flush before consuming the payload, and IBRC has no such primitive.
 *    A device-resident signal is not host-readable; bridge it to a host flag
 *    (e.g. a Copy-Engine write) if the host must also observe it.
 */
class P2pIbrcHostWriter {
 public:
  /**
   * Non-owning view of one host-mapped IBRC ring.
   * @param descsHost host alias of the descriptor array
   * @param piHost    host alias of the producer index
   * @param ciHost    host alias of the consumer index (advanced by the proxy)
   * @param statusHost host alias of the NIC status/error block (may be null)
   * @param depth     ring depth (power of two)
   * @param nicId     NIC index this ring targets (selects lkey/rkey per device)
   */
  P2pIbrcHostWriter(
      IbrcDesc* descsHost,
      uint64_t* piHost,
      uint64_t* ciHost,
      IbrcNicStatus* statusHost,
      uint32_t depth,
      uint32_t nicId)
      : descs_(descsHost),
        pi_(piHost),
        ci_(ciHost),
        status_(statusHost),
        depth_(depth),
        mask_(depth - 1),
        nic_(nicId) {
    if (descsHost == nullptr || piHost == nullptr || ciHost == nullptr) {
      throw std::runtime_error("P2pIbrcHostWriter: null command queue pointer");
    }
    // A non-power-of-two depth would desynchronize `seq & mask_` from the
    // `seq - ci < depth_` backpressure check, silently aliasing live slots.
    if (depth == 0 || (depth & (depth - 1)) != 0) {
      throw std::runtime_error(
          fmt::format(
              "P2pIbrcHostWriter: ring depth must be a power of two, got {}",
              depth));
    }
  }

  ~P2pIbrcHostWriter() = default;
  P2pIbrcHostWriter(const P2pIbrcHostWriter&) = delete;
  P2pIbrcHostWriter& operator=(const P2pIbrcHostWriter&) = delete;
  P2pIbrcHostWriter(P2pIbrcHostWriter&& other) noexcept {
    *this = std::move(other);
  }
  P2pIbrcHostWriter& operator=(P2pIbrcHostWriter&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    descs_ = std::exchange(other.descs_, nullptr);
    pi_ = std::exchange(other.pi_, nullptr);
    ci_ = std::exchange(other.ci_, nullptr);
    status_ = std::exchange(other.status_, nullptr);
    depth_ = std::exchange(other.depth_, 0);
    mask_ = std::exchange(other.mask_, 0);
    nic_ = std::exchange(other.nic_, 0);
    waitTimeout_ = other.waitTimeout_;
    aborted_ = std::move(other.aborted_);
    return *this;
  }

  /**
   * RDMA-write localBuf -> remoteBuf (nbytes), with an optional remote signal
   * (fetch-add) and an optional local completion counter.
   * @return the reserved sequence number for this descriptor.
   */
  uint64_t put(
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      std::size_t nbytes,
      const IbgdaRemoteBuffer* signalBuf = nullptr,
      uint64_t signalVal = 1,
      const IbgdaLocalBuffer* counterBuf = nullptr,
      uint64_t counterVal = 1) {
    const bool hasData = nbytes > 0;
    if (signalBuf != nullptr && signalBuf->ptr == nullptr) {
      throw std::runtime_error("P2pIbrcHostWriter::put: null signal buffer");
    }
    if (counterBuf != nullptr && counterBuf->ptr == nullptr) {
      throw std::runtime_error("P2pIbrcHostWriter::put: null counter buffer");
    }
    const bool hasSignal = signalBuf != nullptr;
    const bool hasCounter = counterBuf != nullptr;
    if (!hasData && !hasSignal) {
      throw std::runtime_error(
          "P2pIbrcHostWriter::put: empty data buffer without signal");
    }

    IbrcDesc desc{};
    desc.op = static_cast<uint16_t>(hasData ? IbrcOp::PUT : IbrcOp::SIGNAL);

    if (hasData) {
      if (localBuf.ptr == nullptr || remoteBuf.ptr == nullptr) {
        throw std::runtime_error("P2pIbrcHostWriter::put: null data buffer");
      }
      desc.local_addr = reinterpret_cast<uint64_t>(localBuf.ptr);
      desc.remote_addr = reinterpret_cast<uint64_t>(remoteBuf.ptr);
      desc.bytes = nbytes;
      check_key_index(
          "local data buffer",
          "lkey_per_device",
          localBuf.lkey_per_device.size);
      check_key_index(
          "remote data buffer",
          "rkey_per_device",
          remoteBuf.rkey_per_device.size);
      desc.lkey_device_order = localBuf.lkey_per_device[nic_].value;
      desc.rkey_device_order = remoteBuf.rkey_per_device[nic_].value;
    }

    if (hasSignal) {
      desc.signal_addr = reinterpret_cast<uint64_t>(signalBuf->ptr);
      desc.signal_value = signalVal;
      check_key_index(
          "signal buffer", "rkey_per_device", signalBuf->rkey_per_device.size);
      desc.signal_rkey_device_order = signalBuf->rkey_per_device[nic_].value;
      desc.flags |= IBRC_HAS_SIGNAL | IBRC_SIGNAL_ADD;
    }

    if (hasCounter) {
      desc.counter_addr = reinterpret_cast<uint64_t>(counterBuf->ptr);
      desc.counter_value = counterVal;
      desc.flags |= IBRC_HAS_COUNTER;
    }

    return enqueue(desc);
  }

  /** Post a standalone SIGNAL (RDMA fetch-add into the remote signal slot). */
  uint64_t signal(const IbgdaRemoteBuffer& signalBuf, uint64_t signalVal = 1) {
    if (signalBuf.ptr == nullptr) {
      throw std::runtime_error("P2pIbrcHostWriter::signal: null signal buffer");
    }
    IbrcDesc desc{};
    desc.op = static_cast<uint16_t>(IbrcOp::SIGNAL);
    desc.signal_addr = reinterpret_cast<uint64_t>(signalBuf.ptr);
    desc.signal_value = signalVal;
    check_key_index(
        "signal buffer", "rkey_per_device", signalBuf.rkey_per_device.size);
    desc.signal_rkey_device_order = signalBuf.rkey_per_device[nic_].value;
    desc.flags = IBRC_HAS_SIGNAL | IBRC_SIGNAL_ADD;
    return enqueue(desc);
  }

  /** Deadline every wait uses unless set_wait_policy() overrides it. */
  static constexpr std::chrono::milliseconds kDefaultWaitTimeout{
      std::chrono::minutes(10)};

  /**
   * Bound every wait this writer performs (poll_counter, fence, and the
   * enqueue backpressure loop).
   *
   * All of them spin on the calling thread, which is the collective's own
   * thread, so a dead peer or a wedged NIC would otherwise hang it with no way
   * out. Callers set their operation timeout here, plus an optional abort
   * predicate so fault tolerance can break a wait before the deadline.
   *
   * @param timeout deadline for each individual wait; zero or negative waits
   *                forever (only appropriate for tests)
   * @param aborted polled while spinning; the wait throws once it returns true
   */
  void set_wait_policy(
      std::chrono::milliseconds timeout,
      std::function<bool()> aborted = nullptr) {
    waitTimeout_ = timeout;
    aborted_ = std::move(aborted);
  }

  /**
   * Spin on a host-mapped completion counter until it reaches `expected`.
   * The proxy bumps this counter (release) after polling the CQE for the
   * corresponding PUT, so an acquire load here observes that the put completed.
   *
   * @return the observed counter value
   */
  uint64_t poll_counter(const uint64_t* counterHost, uint64_t expected) const {
    if (counterHost == nullptr) {
      throw std::runtime_error("P2pIbrcHostWriter::poll_counter: null counter");
    }
    uint64_t v = 0;
    spin_until(
        [&] {
          v = __atomic_load_n(counterHost, __ATOMIC_ACQUIRE);
          return v >= expected;
        },
        "counter to reach",
        expected);
    return v;
  }

  /** Block until the proxy has drained everything posted so far (ci == pi). */
  void fence() const {
    check_ring();
    const uint64_t target = __atomic_load_n(pi_, __ATOMIC_ACQUIRE);
    spin_until(
        [&] { return __atomic_load_n(ci_, __ATOMIC_ACQUIRE) >= target; },
        "queue to drain to",
        target);
  }

  uint32_t nic() const {
    return nic_;
  }

 private:
  static constexpr uint64_t kNoWaitTarget =
      std::numeric_limits<uint64_t>::max();

  void check_ring() const {
    if (descs_ == nullptr || pi_ == nullptr || ci_ == nullptr) {
      throw std::runtime_error(
          "P2pIbrcHostWriter: invalid or moved-from writer");
    }
  }

  void check_key_index(
      const char* bufferName,
      const char* keyName,
      int keyCount) const {
    if (keyCount <= 0 || nic_ >= static_cast<uint32_t>(keyCount)) {
      throw std::runtime_error(
          fmt::format(
              "P2pIbrcHostWriter: {} has {} {} entries, missing NIC {}",
              bufferName,
              keyCount,
              keyName,
              nic_));
    }
  }

  /**
   * Wait for a free slot, reserve it, copy the descriptor body, then publish
   * ready_seq with release ordering — the same reserve -> fill -> publish
   * protocol as P2pIbrcTransportDevice::reserve()/enqueue().
   *
   * Backpressure runs *before* the fetch-add, unlike the device path. The
   * fetch-add cannot be rolled back, so waiting after it would let a throw
   * (timeout, abort, NIC error) leave a reserved seq that is never published,
   * and the in-order proxy would wait on that hole forever. Publishing a
   * filler descriptor instead is not an option: the slot we failed to acquire
   * still holds a live, undrained descriptor. Checking first is sound under
   * the single-logical-producer-per-ring model this class documents — nobody
   * else advances pi, and ci only moves forward, so space observed here is
   * still free at reserve time.
   */
  uint64_t enqueue(IbrcDesc& desc) {
    check_ring();
    check_status();
    desc.ready_seq = kIbrcInvalidReadySeq;

    spin_until(
        [&] {
          return __atomic_load_n(pi_, __ATOMIC_RELAXED) -
              __atomic_load_n(ci_, __ATOMIC_ACQUIRE) <
              depth_;
        },
        "a free ring slot",
        kNoWaitTarget);

    // reserve: relaxed fetch-add on the producer index (matches device).
    const uint64_t seq = __atomic_fetch_add(pi_, 1, __ATOMIC_RELAXED);

    IbrcDesc& slot = descs_[seq & mask_];
    // Fill body (with ready_seq still invalid) ...
    slot = desc;
    // ... then publish. The release store orders the body writes above before
    // the proxy's acquire load of ready_seq observes `seq`.
    __atomic_store_n(&slot.ready_seq, seq, __ATOMIC_RELEASE);
    return seq;
  }

  /**
   * Spin until `ready()` holds, re-checking the NIC status, the abort
   * predicate and the deadline as it goes. `what`/`target` describe the awaited
   * condition; they are only formatted on the throwing paths, since fence() and
   * enqueue() run per step and must not allocate when the wait is satisfied
   * immediately.
   */
  template <typename Ready>
  void spin_until(Ready&& ready, const char* what, uint64_t target) const {
    // Reading the clock and calling the abort predicate on every spin would
    // dominate the wait, so only do it once per kSpinsPerCheck iterations.
    constexpr uint64_t kSpinsPerCheck = 1024;
    const bool bounded = waitTimeout_ > std::chrono::milliseconds::zero();
    // Armed on the first failed ready() rather than up front, so the fast path
    // -- fence() per ring step, already satisfied -- never reads the clock.
    std::chrono::steady_clock::time_point deadline{};
    bool armed = false;

    for (uint64_t spins = 0; !ready(); ++spins) {
      if (spins % kSpinsPerCheck != 0) {
        continue;
      }
      check_status();
      if (aborted_ && aborted_()) {
        throw std::runtime_error(
            fmt::format(
                "P2pIbrcHostWriter: aborted while waiting for {}",
                wait_description(what, target)));
      }
      std::this_thread::yield();
      if (!bounded) {
        continue;
      }
      const auto now = std::chrono::steady_clock::now();
      if (!armed) {
        deadline = now + waitTimeout_;
        armed = true;
      } else if (now >= deadline) {
        throw std::runtime_error(
            fmt::format(
                "P2pIbrcHostWriter: timed out after {}ms waiting for {}",
                waitTimeout_.count(),
                wait_description(what, target)));
      }
    }
  }

  static std::string wait_description(const char* what, uint64_t target) {
    if (target == kNoWaitTarget) {
      return what;
    }
    return fmt::format("{} {}", what, target);
  }

  void check_status() const {
    if (status_ != nullptr &&
        __atomic_load_n(&status_->error, __ATOMIC_ACQUIRE) != 0) {
      const uint32_t errorCode =
          __atomic_load_n(&status_->error_code, __ATOMIC_RELAXED);
      throw std::runtime_error(
          fmt::format("P2pIbrcHostWriter: IBRC NIC error, code={}", errorCode));
    }
  }

  IbrcDesc* descs_{nullptr};
  uint64_t* pi_{nullptr};
  uint64_t* ci_{nullptr};
  IbrcNicStatus* status_{nullptr};
  uint32_t depth_{0};
  uint32_t mask_{0};
  uint32_t nic_{0};
  std::chrono::milliseconds waitTimeout_{kDefaultWaitTimeout};
  std::function<bool()> aborted_{nullptr};
};

} // namespace comms::prims
