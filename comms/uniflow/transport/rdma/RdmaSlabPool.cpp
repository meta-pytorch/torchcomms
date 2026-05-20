// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>

#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/rdma/RdmaResources.h"

namespace uniflow {

// --- PinnedBuffer: RAII host-pinned memory with device pointer ---

class PinnedBuffer {
 public:
  PinnedBuffer(size_t size, std::shared_ptr<CudaApi> cudaApi)
      : cudaApi_(std::move(cudaApi)) {
    auto result =
        cudaApi_->hostAlloc(size, cudaHostAllocMapped | cudaHostAllocPortable);
    if (!result) {
      throw std::runtime_error(
          "PinnedBuffer: cudaHostAlloc failed: " + result.error().toString());
    }
    hostPtr_ = result.value();

    auto devResult = cudaApi_->hostGetDevicePointer(hostPtr_);
    if (!devResult) {
      cudaApi_->hostFree(hostPtr_);
      throw std::runtime_error(
          "PinnedBuffer: cudaHostGetDevicePointer failed: " +
          devResult.error().toString());
    }
    devicePtr_ = reinterpret_cast<uintptr_t>(devResult.value());
  }

  ~PinnedBuffer() {
    if (hostPtr_) {
      cudaApi_->hostFree(hostPtr_);
    }
  }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& other) noexcept
      : cudaApi_(std::move(other.cudaApi_)),
        hostPtr_(other.hostPtr_),
        devicePtr_(other.devicePtr_) {
    other.hostPtr_ = nullptr;
    other.devicePtr_ = 0;
  }
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
      if (hostPtr_) {
        cudaApi_->hostFree(hostPtr_);
      }
      cudaApi_ = std::move(other.cudaApi_);
      hostPtr_ = other.hostPtr_;
      devicePtr_ = other.devicePtr_;
      other.hostPtr_ = nullptr;
      other.devicePtr_ = 0;
    }
    return *this;
  }

  void* hostPtr() const {
    return hostPtr_;
  }
  uintptr_t devicePtr() const {
    return devicePtr_;
  }

 private:
  std::shared_ptr<CudaApi> cudaApi_;
  void* hostPtr_{nullptr};
  uintptr_t devicePtr_{0};
};

// --- MrSet: RAII per-NIC MR registration ---

class MrSet {
 public:
  MrSet(
      void* buffer,
      size_t size,
      std::shared_ptr<IbvApi> ibvApi,
      std::span<NicResources> nics)
      : ibvApi_(std::move(ibvApi)) {
    constexpr int kAccess = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ;
    mrs_.reserve(nics.size());
    for (const auto& nic : nics) {
      auto result = ibvApi_->regMr(nic.pd, buffer, size, kAccess);
      if (!result) {
        for (auto* mr : mrs_) {
          ibvApi_->deregMr(mr);
        }
        throw std::runtime_error(
            "MrSet: ibv_reg_mr failed: " + result.error().toString());
      }
      mrs_.push_back(result.value());
    }
  }

  ~MrSet() {
    for (auto* mr : mrs_) {
      ibvApi_->deregMr(mr);
    }
  }

  MrSet(const MrSet&) = delete;
  MrSet& operator=(const MrSet&) = delete;

  uint32_t lkey(size_t nicIdx) const {
    return mrs_[nicIdx]->lkey;
  }
  uint32_t rkey(size_t nicIdx) const {
    return mrs_[nicIdx]->rkey;
  }

 private:
  std::shared_ptr<IbvApi> ibvApi_;
  std::vector<ibv_mr*> mrs_;
};

// --- SlabAllocator: abstract allocation strategy ---

class SlabAllocator {
 public:
  virtual ~SlabAllocator() = default;
  virtual Result<uint16_t> acquire() = 0;
  virtual Result<std::vector<uint16_t>> acquire(uint32_t count) = 0;
  virtual void release(uint16_t slabIdx) = 0;
};

// --- AtomicBitmapAllocator: lock-free for ≤128 slabs ---

class AtomicBitmapAllocator : public SlabAllocator {
 public:
  explicit AtomicBitmapAllocator(size_t slabNum) {
    assert(slabNum <= 128);
    if (slabNum <= 64) {
      bitmapLow_.store(
          slabNum == 64 ? 0ULL : ~((1ULL << slabNum) - 1),
          std::memory_order_relaxed);
      bitmapHigh_.store(~0ULL, std::memory_order_relaxed);
    } else {
      uint64_t highSlabs = slabNum - 64;
      bitmapHigh_.store(
          highSlabs == 64 ? 0ULL : ~((1ULL << highSlabs) - 1),
          std::memory_order_relaxed);
    }
  }

  Result<uint16_t> acquire() override {
    if (auto idx = tryAcquireFrom(bitmapLow_)) {
      return idx.value();
    }
    if (auto idx = tryAcquireFrom(bitmapHigh_)) {
      return static_cast<uint16_t>(idx.value() + 64);
    }
    return Err(ErrCode::ResourceExhausted, "RdmaSlabPool: no free slabs");
  }

  Result<std::vector<uint16_t>> acquire(uint32_t count) override {
    std::vector<uint16_t> acquired;
    acquired.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      auto r = acquire();
      if (!r) {
        for (auto idx : acquired) {
          release(idx);
        }
        return Err(
            ErrCode::ResourceExhausted, "RdmaSlabPool: insufficient slabs");
      }
      acquired.push_back(r.value());
    }
    return acquired;
  }

  void release(uint16_t slabIdx) override {
    if (slabIdx < 64) {
      bitmapLow_.fetch_and(~(1ULL << slabIdx), std::memory_order_release);
    } else {
      bitmapHigh_.fetch_and(
          ~(1ULL << (slabIdx - 64)), std::memory_order_release);
    }
  }

 private:
  static std::optional<uint16_t> tryAcquireFrom(std::atomic<uint64_t>& bitmap) {
    uint64_t current;
    uint64_t next;
    uint16_t bitIndex;
    do {
      current = bitmap.load(std::memory_order_relaxed);
      uint64_t available = ~current;
      if (available == 0) {
        return std::nullopt;
      }
      bitIndex = __builtin_ctzll(available);
      next = current | (1ULL << bitIndex);
    } while (!bitmap.compare_exchange_weak(
        current, next, std::memory_order_acquire, std::memory_order_relaxed));
    return bitIndex;
  }

  std::atomic<uint64_t> bitmapLow_{};
  std::atomic<uint64_t> bitmapHigh_{};
};

// --- MutexSlabAllocator: mutex-guarded for >128 slabs ---

class MutexSlabAllocator : public SlabAllocator {
 public:
  explicit MutexSlabAllocator(size_t slabNum) {
    freeList_.reserve(slabNum);
    for (uint16_t i = 0; i < slabNum; ++i) {
      freeList_.push_back(i);
    }
  }

  Result<uint16_t> acquire() override {
    std::lock_guard<std::mutex> lock(mu_);
    if (freeList_.empty()) {
      return Err(ErrCode::ResourceExhausted, "RdmaSlabPool: no free slabs");
    }
    uint16_t idx = freeList_.back();
    freeList_.pop_back();
    return idx;
  }

  Result<std::vector<uint16_t>> acquire(uint32_t count) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (freeList_.size() < count) {
      return Err(
          ErrCode::ResourceExhausted,
          "RdmaSlabPool: need " + std::to_string(count) + " slabs, have " +
              std::to_string(freeList_.size()));
    }
    std::vector<uint16_t> acquired(freeList_.end() - count, freeList_.end());
    freeList_.resize(freeList_.size() - count);
    return acquired;
  }

  void release(uint16_t slabIdx) override {
    std::lock_guard<std::mutex> lock(mu_);
    freeList_.push_back(slabIdx);
  }

 private:
  std::mutex mu_;
  std::vector<uint16_t> freeList_;
};

// --- RdmaSlab ---

RdmaSlab::RdmaSlab(std::shared_ptr<RdmaSlabPool> pool, uint16_t index)
    : pool_(std::move(pool)), index_(index) {}

RdmaSlab::~RdmaSlab() {
  if (pool_) {
    pool_->release(index_);
  }
}

RdmaSlab::RdmaSlab(RdmaSlab&& other) noexcept
    : pool_(std::move(other.pool_)), index_(other.index_) {}

RdmaSlab& RdmaSlab::operator=(RdmaSlab&& other) noexcept {
  if (this != &other) {
    if (pool_) {
      pool_->release(index_);
    }
    pool_ = std::move(other.pool_);
    index_ = other.index_;
  }
  return *this;
}

// --- RdmaSlabPool ---

RdmaSlabPool::RdmaSlabPool(
    const RdmaSlabPoolConfig& config,
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<std::vector<NicResources>> nics)
    : config_(config), nics_(std::move(nics)) {
  if (config_.slabSize == 0 || config_.slabNum == 0) {
    throw std::invalid_argument(
        "RdmaSlabPool: slabSize must be > 0 and slabNum must be > 0");
  }

  const size_t bufferSize = config_.slabNum * config_.slabSize;
  const size_t stateSize = config_.slabNum * sizeof(uint64_t);
  const size_t totalSize = bufferSize + stateSize;

  buffer_ = std::make_unique<PinnedBuffer>(totalSize, cudaApi);
  std::memset(static_cast<char*>(buffer_->hostPtr()), 0, totalSize);

  mrSet_ = std::make_unique<MrSet>(
      buffer_->hostPtr(), totalSize, std::move(ibvApi), std::span(*nics_));

  if (config_.slabNum <= 128) {
    allocator_ = std::make_unique<AtomicBitmapAllocator>(config_.slabNum);
  } else {
    allocator_ = std::make_unique<MutexSlabAllocator>(config_.slabNum);
  }

  UNIFLOW_LOG_INFO(
      "RdmaSlabPool: {} slabs x {} bytes = {} bytes",
      config_.slabNum,
      config_.slabSize,
      totalSize);
}

RdmaSlabPool::~RdmaSlabPool() = default;

Result<RdmaSlab> RdmaSlabPool::acquire() {
  auto result = allocator_->acquire();
  if (!result) {
    return Err(result.error().code(), result.error().toString());
  }
  return RdmaSlab(shared_from_this(), result.value());
}

Result<std::vector<RdmaSlab>> RdmaSlabPool::acquire(uint32_t count) {
  auto result = allocator_->acquire(count);
  if (!result) {
    return Err(result.error().code(), result.error().toString());
  }
  auto self = shared_from_this();
  std::vector<RdmaSlab> slabs;
  slabs.reserve(count);
  for (auto idx : result.value()) {
    slabs.emplace_back(RdmaSlab(self, idx));
  }
  return slabs;
}

void RdmaSlabPool::release(uint16_t slabIdx) {
  allocator_->release(slabIdx);
}

void* RdmaSlabPool::slabPtr(uint16_t idx) const {
  return static_cast<char*>(buffer_->hostPtr()) +
      static_cast<size_t>(idx) * config_.slabSize;
}

uint64_t RdmaSlabPool::slabAddr(uint16_t idx) const {
  return reinterpret_cast<uint64_t>(slabPtr(idx));
}

uint32_t RdmaSlabPool::slabLkey(size_t nicIdx) const {
  return mrSet_->lkey(nicIdx);
}

uint32_t RdmaSlabPool::slabRkey(size_t nicIdx) const {
  return mrSet_->rkey(nicIdx);
}

void* RdmaSlabPool::statePtr(uint16_t idx) const {
  return static_cast<char*>(buffer_->hostPtr()) +
      config_.slabNum * config_.slabSize + idx * sizeof(uint64_t);
}

uint64_t RdmaSlabPool::stateAddr(uint16_t idx) const {
  return reinterpret_cast<uint64_t>(statePtr(idx));
}

uint64_t RdmaSlabPool::stateDeviceAddr(uint16_t idx) const {
  return buffer_->devicePtr() + config_.slabNum * config_.slabSize +
      idx * sizeof(uint64_t);
}

} // namespace uniflow
