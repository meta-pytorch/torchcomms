// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace meta::comms::memtrace {

enum class GpuMemoryComponent : uint8_t {
  kMccl,
  kCommon,
  kNvl,
  kIb,
  kIbgda,
  kUnclassified,
  kCount,
};

enum class GpuMemoryResourceType : uint8_t {
  kMcclAllGatherOverlapCounters,

  kCommonTransportDispatchTable,

  kNvlP2pDataStaging,
  kNvlP2pSignal,
  kNvlP2pChannelState,
  kNvlP2pChannelProgress,
  kNvlP2pBarrier,
  kNvlP2pLl,
  kNvlP2pLl128,
  kNvlP2pTransportTable,
  kNvlMultimemCombinedBacking,
  kNvlMultimemPeerSignalPtrTable,

  kIbEagerSendBuffer,
  kIbEagerRecvBuffer,
  kIbEagerControlBuffer,
  kIbSendRecvPeerBulk,
  kIbSlotSignalInbox,
  kIbSlotCounter,
  kIbSlotDiscardSignal,

  kIbgdaAtomicReturnSink,
  kIbgdaPeerTransportArray,

  kUnclassified,

  kCount,
};

inline constexpr std::size_t kGpuMemoryComponentCount =
    static_cast<std::size_t>(GpuMemoryComponent::kCount);
inline constexpr std::size_t kGpuMemoryResourceTypeCount =
    static_cast<std::size_t>(GpuMemoryResourceType::kCount);

std::string_view gpuMemoryComponentName(GpuMemoryComponent component) noexcept;
std::string_view gpuMemoryResourceTypeName(
    GpuMemoryResourceType resource) noexcept;
GpuMemoryComponent gpuMemoryResourceTypeComponent(
    GpuMemoryResourceType resource) noexcept;

struct GpuMemoryContext {
  uint64_t commHash{0};
  int rank{-1};
  int device{-1};

  bool operator==(const GpuMemoryContext&) const = default;
};

struct GpuMemoryUsage {
  uint64_t currentLogicalBytes{0};
  uint64_t currentBytes{0};
  uint64_t peakBytes{0};
  uint64_t totalAllocatedBytes{0};
  uint64_t totalFreedBytes{0};

  bool operator==(const GpuMemoryUsage&) const = default;
};

struct GpuMemorySnapshot {
  GpuMemoryContext context;
  std::array<GpuMemoryUsage, kGpuMemoryComponentCount> components{};
  std::array<GpuMemoryUsage, kGpuMemoryResourceTypeCount> resources{};
  // Exact aggregate across common, NVL, IB, and IBGDA.
  GpuMemoryUsage prims{};
  // Aggregate across unclassified, direct MCCL-owned, and Prims GPU memory.
  GpuMemoryUsage total{};
  uint64_t activeAllocations{0};
  uint64_t duplicateAllocations{0};

  bool operator==(const GpuMemorySnapshot&) const = default;
};

enum class GpuMemoryBackingKind : uint8_t {
  kRuntimeAllocation,
  kVirtualMemoryHandle,
  kCount,
};

std::string formatGpuMemorySnapshot(
    std::string_view stage,
    const GpuMemorySnapshot& snapshot);

class GpuMemoryTracker : public std::enable_shared_from_this<GpuMemoryTracker> {
 public:
  explicit GpuMemoryTracker(GpuMemoryContext context);
  ~GpuMemoryTracker();

  GpuMemoryTracker(const GpuMemoryTracker&) = delete;
  GpuMemoryTracker& operator=(const GpuMemoryTracker&) = delete;
  GpuMemoryTracker(GpuMemoryTracker&&) = delete;
  GpuMemoryTracker& operator=(GpuMemoryTracker&&) = delete;

  void updateContext(GpuMemoryContext context);
  GpuMemorySnapshot snapshot() const;

 private:
  friend void recordGpuMemoryAllocation(
      GpuMemoryResourceType resource,
      uintptr_t backingId,
      uint64_t logicalBytes,
      uint64_t accountedBytes,
      GpuMemoryBackingKind backingKind) noexcept;
  friend void recordGpuMemoryFree(
      uintptr_t backingId,
      GpuMemoryBackingKind backingKind) noexcept;
  friend void finishGpuMemoryFree(
      uintptr_t backingId,
      GpuMemoryBackingKind backingKind,
      uint64_t generation,
      bool succeeded) noexcept;

  struct Impl;

  bool recordAllocation(
      GpuMemoryResourceType resource,
      uint64_t logicalBytes,
      uint64_t accountedBytes) noexcept;
  void recordDuplicateAllocation() noexcept;
  void recordFree(
      GpuMemoryResourceType resource,
      uint64_t logicalBytes,
      uint64_t accountedBytes) noexcept;

  std::unique_ptr<Impl> impl_;
};

class ScopedGpuMemoryContext {
 public:
  // The tracker is non-owning to keep the enabled collective fast path free
  // of shared_ptr reference-count traffic. It must be shared-owned, outlive
  // this scope, and the scope must be destroyed on its creating thread.
  explicit ScopedGpuMemoryContext(GpuMemoryTracker* gpuMemoryTracker) noexcept;
  ~ScopedGpuMemoryContext() noexcept;

  ScopedGpuMemoryContext(const ScopedGpuMemoryContext&) = delete;
  ScopedGpuMemoryContext& operator=(const ScopedGpuMemoryContext&) = delete;
  ScopedGpuMemoryContext(ScopedGpuMemoryContext&&) = delete;
  ScopedGpuMemoryContext& operator=(ScopedGpuMemoryContext&&) = delete;

 private:
  GpuMemoryTracker* previousGpuMemoryTracker_;
  bool restorePrevious_{false};
};

// `backingId` must identify the owning local allocation, not an imported or
// peer mapping alias.
// Allocation attribution uses the innermost ScopedGpuMemoryContext on the
// calling thread. Free attribution is recovered from the allocation registry
// and is independent of the calling thread's current scope.
void recordGpuMemoryAllocation(
    GpuMemoryResourceType resource,
    uintptr_t backingId,
    uint64_t logicalBytes,
    uint64_t accountedBytes,
    GpuMemoryBackingKind backingKind =
        GpuMemoryBackingKind::kRuntimeAllocation) noexcept;
void recordGpuMemoryFree(
    uintptr_t backingId,
    GpuMemoryBackingKind backingKind =
        GpuMemoryBackingKind::kRuntimeAllocation) noexcept;

// The mcclCuda* wrappers bracket the physical free with these calls so a
// concurrently reused pointer or handle cannot consume the old allocation's
// registry entry. A zero generation means the backing was not tracked.
uint64_t beginGpuMemoryFree(
    uintptr_t backingId,
    GpuMemoryBackingKind backingKind =
        GpuMemoryBackingKind::kRuntimeAllocation) noexcept;
void finishGpuMemoryFree(
    uintptr_t backingId,
    GpuMemoryBackingKind backingKind,
    uint64_t generation,
    bool succeeded) noexcept;

} // namespace meta::comms::memtrace
