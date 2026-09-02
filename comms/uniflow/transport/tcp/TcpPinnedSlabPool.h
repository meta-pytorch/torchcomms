// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

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
/// owns the lease -- which on the outbound path means until the frame built in
/// it has been handed to the socket and the queue entry is gone.
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
  /// Fails if `count` exceeds the unreserved capacity (it could never be
  /// satisfied) or if the pool has been closed.
  Result<std::vector<TcpPinnedSlab>> acquire(size_t count);

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
