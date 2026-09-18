// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpPinnedSlabPool.h"

#include <limits>
#include <string>
#include <utility>

#include "comms/uniflow/drivers/cuda/CudaApi.h"

namespace uniflow {

// ---------------------------------------------------------------------------
// TcpPinnedSlab
// ---------------------------------------------------------------------------

TcpPinnedSlab::TcpPinnedSlab(
    std::shared_ptr<TcpPinnedSlabPool> pool,
    size_t index,
    uint8_t* data,
    size_t capacity)
    : pool_(std::move(pool)), index_(index), data_(data), capacity_(capacity) {}

TcpPinnedSlab::~TcpPinnedSlab() {
  reset();
}

TcpPinnedSlab::TcpPinnedSlab(TcpPinnedSlab&& other) noexcept
    : pool_(std::move(other.pool_)),
      index_(other.index_),
      data_(other.data_),
      capacity_(other.capacity_) {
  other.data_ = nullptr;
  other.capacity_ = 0;
}

TcpPinnedSlab& TcpPinnedSlab::operator=(TcpPinnedSlab&& other) noexcept {
  if (this != &other) {
    reset();
    pool_ = std::move(other.pool_);
    index_ = other.index_;
    data_ = other.data_;
    capacity_ = other.capacity_;
    other.data_ = nullptr;
    other.capacity_ = 0;
  }
  return *this;
}

void TcpPinnedSlab::reset() {
  if (pool_ != nullptr) {
    // Moved out first: release() can wake a waiter that immediately reacquires
    // this slab, and the pool must not be kept alive by this lease afterwards.
    auto pool = std::move(pool_);
    data_ = nullptr;
    capacity_ = 0;
    pool->release(index_);
  }
}

// ---------------------------------------------------------------------------
// TcpPinnedSlabPool
// ---------------------------------------------------------------------------

Result<std::shared_ptr<TcpPinnedSlabPool>> TcpPinnedSlabPool::create(
    std::shared_ptr<CudaApi> cudaApi,
    size_t slabSize,
    size_t slabCount,
    size_t reservedForReader) {
  if (cudaApi == nullptr) {
    return Err(
        ErrCode::InvalidArgument, "TcpPinnedSlabPool: no CUDA API available");
  }
  if (slabSize == 0 || slabCount == 0) {
    return Err(
        ErrCode::InvalidArgument,
        "TcpPinnedSlabPool: slabSize and slabCount must both be non-zero");
  }
  if (reservedForReader >= slabCount) {
    return Err(
        ErrCode::InvalidArgument,
        "TcpPinnedSlabPool: reservedForReader must leave at least one slab");
  }
  // create() is the geometry's validation boundary, and the product below is
  // the one input to it that can wrap. A wrapped product allocates a short
  // region and the pool then hands out slabs pointing past its end -- and the
  // pointer-bounds assertions in SlabsAreDistinctWindowsOntoTheRegion would
  // still pass for the slabs that happen to land inside it. Unreachable while
  // every caller passes compile-time constants; checked here because the
  // geometry is heading toward caller-configurable.
  if (slabCount > std::numeric_limits<size_t>::max() / slabSize) {
    return Err(
        ErrCode::InvalidArgument,
        "TcpPinnedSlabPool: slabCount " + std::to_string(slabCount) +
            " times slabSize " + std::to_string(slabSize) + " overflows");
  }
  // cudaHostAllocPortable so the region is pinned with respect to every device
  // context, not just whichever was current here: one transport stages for
  // whatever devices its peer's segments live on.
  auto base = cudaApi->hostAlloc(slabCount * slabSize, cudaHostAllocPortable);
  if (!base) {
    return std::move(base).error();
  }
  return std::shared_ptr<TcpPinnedSlabPool>(new TcpPinnedSlabPool(
      std::move(cudaApi),
      base.value(),
      slabSize,
      slabCount,
      reservedForReader));
}

TcpPinnedSlabPool::TcpPinnedSlabPool(
    std::shared_ptr<CudaApi> cudaApi,
    void* base,
    size_t slabSize,
    size_t slabCount,
    size_t reservedForReader)
    : cudaApi_(std::move(cudaApi)),
      base_(base),
      slabSize_(slabSize),
      slabCount_(slabCount),
      reservedForReader_(reservedForReader) {
  free_.reserve(slabCount_);
  for (size_t i = 0; i < slabCount_; ++i) {
    free_.push_back(i);
  }
}

TcpPinnedSlabPool::~TcpPinnedSlabPool() {
  if (base_ != nullptr) {
    (void)cudaApi_->hostFree(base_);
  }
}

TcpPinnedSlab TcpPinnedSlabPool::tryAcquire(bool allowReserved) {
  size_t index = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_ || free_.empty()) {
      return TcpPinnedSlab{};
    }
    if (!allowReserved && free_.size() <= reservedForReader_) {
      return TcpPinnedSlab{};
    }
    index = free_.back();
    free_.pop_back();
  }
  return TcpPinnedSlab{
      shared_from_this(),
      index,
      static_cast<uint8_t*>(base_) + index * slabSize_,
      slabSize_};
}

Result<std::vector<TcpPinnedSlab>> TcpPinnedSlabPool::acquire(
    size_t count,
    std::chrono::milliseconds timeout) {
  if (count == 0) {
    return std::vector<TcpPinnedSlab>{};
  }
  if (count > slabCount_ - reservedForReader_) {
    return Err(
        ErrCode::InvalidArgument,
        "TcpPinnedSlabPool: asked for " + std::to_string(count) +
            " slabs, only " + std::to_string(slabCount_ - reservedForReader_) +
            " are available to bulk callers");
  }
  std::vector<size_t> indices;
  {
    std::unique_lock<std::mutex> lk(mu_);
    // wait_for returns the predicate's value, so `satisfied` is false only if
    // the deadline expired with the pool still short. closed_ is checked first
    // because a shutdown is the more specific answer: close() makes the
    // predicate true, so a pool closed while waiting reports closure rather
    // than a coincidental expiry.
    const bool satisfied = freed_.wait_for(lk, timeout, [this, count]() {
      return closed_ || free_.size() >= count + reservedForReader_;
    });
    if (closed_) {
      return Err(ErrCode::NotConnected, "TcpPinnedSlabPool: pool is closed");
    }
    if (!satisfied) {
      // Says what was wanted, what was there, and what it means, because the
      // whole point of the deadline is that this is diagnosable where a hang
      // was not. A caller seeing this has either lost its peer or is running
      // more concurrent transfers than the pool was sized for.
      // NOT the benign, retryable Timeout that readerLoop and the benchmark
      // rendezvous continue on: that one means "an idle socket had nothing
      // yet", whereas this means "the pool stayed exhausted for the whole
      // deadline", which a retry would only repeat. ResourceExhausted is
      // already taken here for a request that could NEVER be satisfied (count
      // above unreserved capacity); collapsing the two would lose that
      // distinction. A caller must not apply the idle-socket retry convention
      // to this code.
      return Err(
          ErrCode::Timeout,
          "TcpPinnedSlabPool: waited " + std::to_string(timeout.count()) +
              "ms for " + std::to_string(count) + " slabs, only " +
              std::to_string(
                  free_.size() > reservedForReader_
                      ? free_.size() - reservedForReader_
                      : 0) +
              " of " + std::to_string(slabCount_ - reservedForReader_) +
              " unreserved slabs are free; the peer has stopped draining or "
              "more transfers are in flight than the pool is sized for");
    }
    indices.assign(free_.end() - static_cast<ptrdiff_t>(count), free_.end());
    free_.resize(free_.size() - count);
  }
  auto self = shared_from_this();
  std::vector<TcpPinnedSlab> slabs;
  slabs.reserve(count);
  for (auto index : indices) {
    slabs.emplace_back(
        self,
        index,
        static_cast<uint8_t*>(base_) + index * slabSize_,
        slabSize_);
  }
  return slabs;
}

void TcpPinnedSlabPool::close() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
  }
  freed_.notify_all();
}

void TcpPinnedSlabPool::release(size_t index) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    free_.push_back(index);
  }
  freed_.notify_all();
}

} // namespace uniflow
