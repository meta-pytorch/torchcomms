// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"

namespace uniflow {

struct NicResources;
class SlabAllocator;
class PinnedBuffer;
class MrSet;

struct RdmaSlabPoolConfig {
  size_t slabNum{128};
  size_t slabSize{512 * 1024}; // 512KB
};

class RdmaSlabPool;

/// RAII wrapper for an acquired slab. Auto-releases back to pool on
/// destruction. Movable but not copyable. Call release() for explicit early
/// release without waiting for destruction.
class RdmaSlab {
 public:
  ~RdmaSlab();
  RdmaSlab(RdmaSlab&& other) noexcept;
  RdmaSlab& operator=(RdmaSlab&& other) noexcept;
  RdmaSlab(const RdmaSlab&) = delete;
  RdmaSlab& operator=(const RdmaSlab&) = delete;

  uint16_t index() const {
    return index_;
  }

 private:
  friend class RdmaSlabPool;
  RdmaSlab(std::shared_ptr<RdmaSlabPool> pool, uint16_t index);
  std::shared_ptr<RdmaSlabPool> pool_;
  uint16_t index_;
};

/// Factory-level shared pool of host-pinned staging buffers for copy-based
/// send/recv. Composed from three RAII subsystems:
///   - PinnedBuffer: host-pinned memory allocation + CUDA device pointer
///   - MrSet: per-NIC ibv_mr registration
///   - SlabAllocator: index allocation (atomic bitmap ≤128 slabs, mutex
///     free-list otherwise)
///
/// acquire() returns RdmaSlab RAII handles that auto-release on destruction.
class RdmaSlabPool : public std::enable_shared_from_this<RdmaSlabPool> {
 public:
  RdmaSlabPool(
      const RdmaSlabPoolConfig& config,
      std::shared_ptr<CudaApi> cudaApi,
      std::shared_ptr<IbvApi> ibvApi,
      std::shared_ptr<std::vector<NicResources>> nics);
  ~RdmaSlabPool();

  RdmaSlabPool(const RdmaSlabPool&) = delete;
  RdmaSlabPool& operator=(const RdmaSlabPool&) = delete;
  RdmaSlabPool(RdmaSlabPool&&) = delete;
  RdmaSlabPool& operator=(RdmaSlabPool&&) = delete;

  Result<RdmaSlab> acquire();
  Result<std::vector<RdmaSlab>> acquire(uint32_t count);

  void* slabPtr(uint16_t idx) const;
  uint64_t slabAddr(uint16_t idx) const;
  uint32_t slabLkey(size_t nicIdx) const;
  uint32_t slabRkey(size_t nicIdx) const;

  void* statePtr(uint16_t idx) const;
  uint64_t stateAddr(uint16_t idx) const;
  uint64_t stateDeviceAddr(uint16_t idx) const;

  size_t slabSize() const {
    return config_.slabSize;
  }
  size_t numSlabs() const {
    return config_.slabNum;
  }

 private:
  friend class RdmaSlab;
  void release(uint16_t slabIdx);

  RdmaSlabPoolConfig config_;
  std::shared_ptr<std::vector<NicResources>> nics_;
  std::unique_ptr<PinnedBuffer> buffer_;
  std::unique_ptr<MrSet> mrSet_;
  std::unique_ptr<SlabAllocator> allocator_;
};

} // namespace uniflow
