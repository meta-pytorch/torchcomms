// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "comms/uniflow/Result.h"

namespace uniflow {

class CudaApi;
class TcpPinnedSlabPool;

/// A borrow of one pinned staging slab. Move-only, and returned to the pool
/// when it is destroyed, so a slab is held for exactly as long as some object
/// owns the lease. Outbound frames retain it through socket send; inbound
/// frames retain it through the destination copy that still reads from it.
class TcpPinnedSlab {
 public:
  TcpPinnedSlab() = default;
  TcpPinnedSlab(
      std::shared_ptr<TcpPinnedSlabPool> pool,
      size_t index,
      uint8_t* data,
      size_t capacity);

  ~TcpPinnedSlab();

  TcpPinnedSlab(TcpPinnedSlab&& other) noexcept;
  TcpPinnedSlab& operator=(TcpPinnedSlab&& other) noexcept;

  TcpPinnedSlab(const TcpPinnedSlab&) = delete;
  TcpPinnedSlab& operator=(const TcpPinnedSlab&) = delete;

  explicit operator bool() const {
    return pool_ != nullptr;
  }
  uint8_t* data() const {
    return data_;
  }
  size_t capacity() const {
    return capacity_;
  }

  /// Returns the slab to the pool early. A no-op on an empty lease.
  void reset();

 private:
  std::shared_ptr<TcpPinnedSlabPool> pool_;
  size_t index_{0};
  uint8_t* data_{nullptr};
  size_t capacity_{0};
};

/// A fixed set of equally-sized pinned host slabs, used to stage payloads
/// between device memory and the socket.
///
/// Pinned rather than pageable because a device-to-host `cudaMemcpyAsync` into
/// pageable memory is documented to complete synchronously: the thread that
/// issues it blocks for the copy. On the get() responder that thread is the
/// reader, and blocking it is what the staging queue exists to avoid.
///
/// `reservedForReader` slabs are withheld from the bulk `acquire()` path so a
/// saturated put() cannot leave the responder with nothing to stage into.
/// "Reserved" bounds put(), not the reader: when puts are idle the reader may
/// use every slab, which keeps a multi-chunk get overlapping copy and transmit
/// instead of serialising on one slab.
class TcpPinnedSlabPool
    : public std::enable_shared_from_this<TcpPinnedSlabPool> {
 public:
  /// Allocates `slabCount * slabSize` bytes of pinned host memory in one
  /// region. Fails rather than throws, so a caller can fall back to failing one
  /// transfer instead of the whole transport.
  static Result<std::shared_ptr<TcpPinnedSlabPool>> create(
      std::shared_ptr<CudaApi> cudaApi,
      size_t slabSize,
      size_t slabCount,
      size_t reservedForReader);

  ~TcpPinnedSlabPool();

  TcpPinnedSlabPool(const TcpPinnedSlabPool&) = delete;
  TcpPinnedSlabPool& operator=(const TcpPinnedSlabPool&) = delete;
  TcpPinnedSlabPool(TcpPinnedSlabPool&&) = delete;
  TcpPinnedSlabPool& operator=(TcpPinnedSlabPool&&) = delete;

  size_t slabSize() const {
    return slabSize_;
  }
  size_t slabCount() const {
    return slabCount_;
  }

  /// Never blocks. `allowReserved` is for the reader thread, which must not
  /// wait; put() passes false and so sees the pool as exhausted while the
  /// reserved slabs are all that is left. An empty lease means "none
  /// available", which is not an error.
  TcpPinnedSlab tryAcquire(bool allowReserved);

  /// Blocks until `count` slabs can be handed out together, drawing only on the
  /// unreserved slabs. All-or-nothing: a caller that took what it could and
  /// waited for the rest could deadlock against another doing the same, so a
  /// waiter here holds nothing.
  ///
  /// The waiter-holds-nothing rule is a contract on CALLERS, not something this
  /// class can enforce, and it is load-bearing: it is the whole reason the bulk
  /// acquire is deadlock-free for any number of them. TcpTransport::put() has
  /// to hold a launched wave across its acquire to overlap staging with
  /// transmission, which would break it, so put() bounds its own concurrency
  /// with a permit derived from this pool's geometry -- see
  /// kMaxConcurrentPutStaging. A future caller that holds slabs across
  /// acquire() owes the same bound.
  ///
  /// Separately, a maximum-size acquire needs the READER to be idle, not just
  /// the other bulk callers. `reservedForReader` only withholds slabs from this
  /// path; it does not stop the reader taking unreserved ones through
  /// tryAcquire(allowReserved=true). So `count == slabCount -
  /// reservedForReader` can only be satisfied while the reader holds none,
  /// which is a property of live traffic rather than of this pool's geometry --
  /// and unlike the rule above, it is not something a caller can discharge by
  /// bounding itself.
  ///
  /// Fails if `count` exceeds the unreserved capacity (it could never be
  /// satisfied), if the pool has been closed, or if `timeout` elapses first.
  ///
  /// The deadline is what keeps an exhausted pool from turning a caller into a
  /// permanent hang. This is the only blocking entry point on the pool, put()
  /// calls it on the application's own thread, and shutdown() never joins that
  /// thread -- so before the deadline existed, `closed_` was the sole escape
  /// and close() runs only from shutdown(). Concurrent puts reach that state
  /// without anything going wrong on the wire: the pool is sized for one put()
  /// in flight, so several of them each hold a wave and none can release
  /// without returning from the acquire it is parked in. A timeout converts
  /// that from a wedged application thread into a failed transfer the caller
  /// can see.
  ///
  /// The default is deliberately far above any legitimate wait. One wave is at
  /// most `kMaxPutWaveChunks * kMaxChunkSize`, which drains in milliseconds on
  /// a healthy link, so a wait measured in tens of seconds already means the
  /// sender has stopped making progress rather than fallen behind. It happens
  /// to equal the DEFAULT connected-socket read timeout, and for the same
  /// reason -- both answer "the peer has stopped" -- but they are two
  /// independent numbers, not one: TcpSocketConfig::connTimeout is
  /// configurable, so a caller that changes it makes them diverge. Do not read
  /// this as a derived value.
  ///
  /// It is a defaulted parameter, but no production path passes anything else
  /// -- launchPutWave() takes the default and nothing threads a value in from a
  /// transport config. Treat 30s as fixed in deployment; the parameter exists
  /// for tests, which do pass their own.
  static constexpr std::chrono::seconds kDefaultAcquireTimeout{30};
  Result<std::vector<TcpPinnedSlab>> acquire(
      size_t count,
      std::chrono::milliseconds timeout = kDefaultAcquireTimeout);

  /// Wakes every waiter and refuses further acquisition. Outstanding leases
  /// stay valid; this only stops new ones, so a shutdown does not pull memory
  /// out from under a copy that is still running.
  void close();

 private:
  friend class TcpPinnedSlab;

  TcpPinnedSlabPool(
      std::shared_ptr<CudaApi> cudaApi,
      void* base,
      size_t slabSize,
      size_t slabCount,
      size_t reservedForReader);

  void release(size_t index);

  std::shared_ptr<CudaApi> cudaApi_;
  void* base_{nullptr};
  size_t slabSize_{0};
  size_t slabCount_{0};
  size_t reservedForReader_{0};

  std::mutex mu_;
  // Notified on every release, not once per waiter: waiters want different
  // counts, so the thread a notify_one picked may be unable to proceed while
  // another could.
  std::condition_variable freed_;
  std::vector<size_t> free_;
  bool closed_{false};
};

} // namespace uniflow
