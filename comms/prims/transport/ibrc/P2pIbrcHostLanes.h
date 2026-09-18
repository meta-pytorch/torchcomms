// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibrc/P2pIbrcHostWriter.h"

namespace comms::prims {

/**
 * Host-side fan-out of one transfer across several IBRC rings.
 *
 * A single P2pIbrcHostWriter drives ONE (qpSlot, nic) ring, which caps a
 * transfer at one NIC and one QP. On a GB200 rail that measured 44 GB/s -- 88%
 * of a single 400G NDR port -- while the transport had two ports open and idle.
 * Splitting the same transfer across four rings (2 NICs x 2 QPs) reached
 * 93 GB/s, 2.1x, with no transport change: the rings already existed.
 *
 * Command queues are laid out [qpSlot * numNics + nic] with nic varying
 * fastest, so consecutive lane indices alternate NICs before reusing one.
 * That is the same interleave the device path applies via nic_for_queue(); a
 * sequential walk would instead fill one NIC's QPs before touching the second
 * and leave half the fabric idle for small lane counts.
 *
 *
 * A put and its fetch-add signal ride the same QP, so RC ordering makes a
 * bumped counter proof that that lane's bytes landed. Across lanes there
 * is no such guarantee: a signal on lane A says nothing about lane B's
 * bytes, since they are different QPs and (usually) different NICs, i.e.
 * different PCIe requesters at the receiver. Therefore each lane MUST signal
 * its own counter and the receiver MUST wait on every lane. A single shared
 * counter is NOT sufficient.
 *
 *
 * Both endpoints must agree on the lane count, since the receiver waits on
 * exactly the lanes the sender posts and a mismatch hangs rather than fails.
 */
class P2pIbrcHostLanes {
 public:
  /** Maximum number of command-queue lanes in one host-striped transfer. */
  static constexpr int kMaxLanes = 128;

  /*
   * QPs to drive per NIC by default. One QP tops out at ~88% of a 400G NDR
   * port and a second reaches ~93%; a third and fourth measured no better, so
   * this stops at two rather than following CTRAN's eight, which is sized for
   * a different traffic pattern.
   */
  static constexpr int kDefaultQpsPerNic = 2;

  /** Per-lane byte range within a transfer. */
  struct LaneRange {
    std::size_t offset{0};
    std::size_t bytes{0};
  };

  /**
   * Below this per-lane size a transfer is latency-bound: one lane
   * handles it and striping only adds cross-lane straggler cost. Mirrors the
   * equivalent gate on CTRAN's interleave path (shouldInterleaveQp /
   * minWqeSize).
   */
  static constexpr std::size_t kDefaultMinLaneBytes = 256 * 1024;

  /** Lane offsets are kept 128B-aligned so no two lanes share a cache
   * line. */
  static constexpr std::size_t kLaneAlignment = 128;

  /*
   * `owner` is the transport's claim on this peer's rings, released when this
   * object dies. Held opaquely and never dereferenced -- the transport tracks
   * it through a weak_ptr, so nothing here has to reach back into it. Default
   * null for tests that build lanes over their own rings.
   */
  explicit P2pIbrcHostLanes(
      std::vector<P2pIbrcHostWriter> writers,
      std::shared_ptr<void> owner = nullptr)
      : writers_(std::move(writers)), owner_(std::move(owner)) {
    if (writers_.empty()) {
      throw std::runtime_error("P2pIbrcHostLanes: needs at least one lane");
    }
  }

  int lanes() const {
    return static_cast<int>(writers_.size());
  }

  /** NIC behind lane `l`; distinct across the first numNics lanes. */
  uint32_t nic(int lane) const {
    return writers_.at(lane).nic();
  }

  P2pIbrcHostWriter& writer(int lane) {
    return writers_.at(lane);
  }

  /** Apply one per-wait bound and abort predicate to every lane. */
  void set_wait_policy(
      std::chrono::milliseconds timeout,
      const std::function<bool()>& aborted) {
    for (auto& w : writers_) {
      w.set_wait_policy(timeout, aborted);
    }
  }

  /**
   * Per-wait bound only, leaving each lane's abort predicate alone.
   *
   * Deliberately separate from set_wait_policy(), mirroring the writer. The
   * transport arms every lane with the communicator's abort when it builds
   * them, so a defaulted `aborted` on the two-argument form would silently
   * disarm all of them and leave each wait spinning out its full deadline on
   * an aborted job instead of unwinding.
   */
  void set_timeout(std::chrono::milliseconds timeout) {
    for (auto& w : writers_) {
      w.set_timeout(timeout);
    }
  }

  /**
   * One operation budget spanning every lane, ending when the returned guards
   * die. A striped transfer waits per lane -- a poll_counter() each, then a
   * fence each -- and the per-wait bound restarts for all of them, so a caller
   * forwarding one operation timeout needs every lane sharing one deadline
   * rather than holding its own copy.
   *
   * The deadline is absolute because the lanes must agree on it; deriving each
   * lane's from its own start would hand the last lane the same budget the
   * first already spent.
   */
  [[nodiscard]] std::vector<P2pIbrcHostWriter::OpBudget> begin_op_until(
      std::chrono::steady_clock::time_point deadline) {
    std::vector<P2pIbrcHostWriter::OpBudget> budgets;
    budgets.reserve(writers_.size());
    for (auto& w : writers_) {
      budgets.push_back(w.begin_op_until(deadline));
    }
    return budgets;
  }

  /**
   * How `nbytes` divides over lanes. Every lane but the last carries the same
   * aligned size and the remainder rides the last lane, rather than being
   * spread, so the mapping stays trivially reproducible on both sides.
   *
   * A transfer too small to give every lane `minLaneBytes` uses FEWER
   * lanes rather than falling back to one. Collapsing to a single lane
   * would throw away real bandwidth in the middle of the range
   */
  static std::vector<LaneRange> split(
      std::size_t nbytes,
      int lanes,
      std::size_t minLaneBytes = kDefaultMinLaneBytes) {
    if (lanes <= 1) {
      return {LaneRange{0, nbytes}};
    }
    const int affordable =
        minLaneBytes > 0 ? static_cast<int>(nbytes / minLaneBytes) : lanes;
    const int used = std::max(1, std::min(lanes, affordable));
    if (used <= 1) {
      return {LaneRange{0, nbytes}};
    }
    const std::size_t per = (nbytes / static_cast<std::size_t>(used)) &
        ~(kLaneAlignment - std::size_t{1});
    if (per == 0) {
      return {LaneRange{0, nbytes}};
    }
    std::vector<LaneRange> ranges;
    ranges.reserve(static_cast<std::size_t>(used));
    for (int l = 0; l < used; ++l) {
      const std::size_t offset = per * static_cast<std::size_t>(l);
      const std::size_t bytes = (l == used - 1) ? nbytes - offset : per;
      ranges.push_back(LaneRange{offset, bytes});
    }
    return ranges;
  }

  /**
   * The lane division for one transfer: which lane carries which bytes, and
   * how far apart the per-lane signal slots sit.
   *
   * Both endpoints must arrive at the same one. The receiver waits on exactly
   * the lanes the sender posted, so a mismatch hangs rather than fails -- and
   * a count each side works out for itself, from arguments it passes
   * separately to the post and the wait, is a mismatch waiting to happen.
   * Carrying it in one value is what stops the two from drifting.
   */
  class LaneLayout {
   public:
    LaneLayout(
        std::size_t nbytes,
        int lanes,
        std::size_t signalStride,
        std::size_t minLaneBytes = kDefaultMinLaneBytes)
        : ranges_(split(nbytes, lanes, minLaneBytes)), stride_(signalStride) {
      /*
       * split() yields at least one range on every path, including a zero-byte
       * transfer. Enforced rather than assumed, because a layout covering no
       * lanes makes poll_all() return without waiting on anything and the
       * caller reads a buffer nobody wrote.
       */
      if (ranges_.empty()) {
        throw std::runtime_error(
            "P2pIbrcHostLanes: lane layout covers no lanes");
      }
      /*
       * A zero stride aliases every lane onto one counter, so the first
       * arrival satisfies every wait while the rest are still in flight and
       * the receiver reads a half-written buffer with nothing reporting an
       * error. Checked here rather than per lane because put_striped() cannot
       * recall the lanes it has already put on the wire.
       */
      checkSignalStride(signalStride, "LaneLayout");
    }

    int used() const {
      return static_cast<int>(ranges_.size());
    }

    const std::vector<LaneRange>& ranges() const {
      return ranges_;
    }

    std::size_t signalStride() const {
      return stride_;
    }

   private:
    std::vector<LaneRange> ranges_;
    std::size_t stride_;
  };

  /**
   * The layout for a transfer of `nbytes` over THIS object's lanes. The peer
   * builds its own from the lane count the sender used, which is why
   * LaneLayout stays constructible directly.
   */
  [[nodiscard]] LaneLayout layout_for(
      std::size_t nbytes,
      std::size_t signalStride,
      std::size_t minLaneBytes = kDefaultMinLaneBytes) const {
    return LaneLayout(nbytes, lanes(), signalStride, minLaneBytes);
  }

  /**
   * RDMA-write localBuf -> remoteBuf over the lanes `layout` names, each lane
   * fetch-adding `signalVal` into ITS OWN slot of the remote signal array.
   *
   * The layout is the single source of the lane count, so the wait side cannot
   * end up on a different one.
   */
  void put_striped(
      const LaneLayout& layout,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      const IbgdaRemoteBuffer& remoteSignalBase,
      uint64_t signalVal = 1) {
    checkLayout(layout, "put_striped");
    const auto& ranges = layout.ranges();
    for (std::size_t l = 0; l < ranges.size(); ++l) {
      const IbgdaRemoteBuffer sig =
          remoteSignalBase.subBuffer(l * layout.signalStride());
      writers_[l].put(
          localBuf.subBuffer(ranges[l].offset),
          remoteBuf.subBuffer(ranges[l].offset),
          ranges[l].bytes,
          &sig,
          signalVal);
    }
  }

  /**
   * Spin until every lane `layout` names has reached `expected`.
   *
   * The lane count comes from the layout the post used, so a stale or
   * mismatched count cannot be supplied here.
   */
  void poll_all(
      const LaneLayout& layout,
      const uint64_t* localSignalBase,
      uint64_t expected) {
    checkLayout(layout, "poll_all");
    const auto* base = reinterpret_cast<const char*>(localSignalBase);
    for (int l = 0; l < layout.used(); ++l) {
      writers_.at(l).poll_counter(
          reinterpret_cast<const uint64_t*>(
              base + static_cast<std::size_t>(l) * layout.signalStride()),
          expected);
    }
  }

  /** Block until every lane's proxy has drained what this rank posted. */
  void fence_all() {
    for (auto& w : writers_) {
      w.fence();
    }
  }

 private:
  /*
   * A stride of zero puts every lane's signal slot at the same address, so the
   * first arrival satisfies the wait for all of them while the rest are still
   * in flight and the receiver reads a half-written buffer with nothing
   * reporting an error. A misaligned stride is caught by the individual
   * writer, but only when that lane's turn comes round -- by which point the
   * earlier lanes are already on the wire and cannot be recalled. Both are
   * rejected here instead, before any lane is touched.
   */
  /*
   * A layout built against a different (usually stale) lane count would index
   * past writers_. It cannot be caught at construction because LaneLayout has
   * to stay constructible from the SENDER's lane count on the receiving side.
   */
  void checkLayout(const LaneLayout& layout, const char* what) const {
    if (layout.used() > lanes()) {
      throw std::runtime_error(
          fmt::format(
              "P2pIbrcHostLanes::{}: layout spans {} lanes, this peer has {}",
              what,
              layout.used(),
              lanes()));
    }
  }

  static void checkSignalStride(std::size_t signalStride, const char* what) {
    if (signalStride < sizeof(uint64_t) ||
        signalStride % alignof(uint64_t) != 0) {
      throw std::runtime_error(
          fmt::format(
              "P2pIbrcHostLanes::{}: signal stride {} must be at least {} "
              "bytes and {}-byte aligned",
              what,
              signalStride,
              sizeof(uint64_t),
              alignof(uint64_t)));
    }
  }

  std::vector<P2pIbrcHostWriter> writers_;
  std::shared_ptr<void> owner_;
};

} // namespace comms::prims
