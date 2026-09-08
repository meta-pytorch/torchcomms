// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/memtrace/GpuMemoryTracker.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <optional>
#include <string>
#include <utility>

#include <fmt/format.h>
#include <folly/CppAttributes.h>
#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>

namespace meta::comms::memtrace {
namespace {

constexpr std::size_t componentIndex(GpuMemoryComponent component) {
  return static_cast<std::size_t>(component);
}

constexpr std::size_t resourceIndex(GpuMemoryResourceType resource) {
  return static_cast<std::size_t>(resource);
}

constexpr std::size_t backingKindIndex(GpuMemoryBackingKind backingKind) {
  return static_cast<std::size_t>(backingKind);
}

struct ComponentDescriptor {
  GpuMemoryComponent component;
  std::string_view name;
  bool isPrims;
};

struct ResourceDescriptor {
  GpuMemoryResourceType resource;
  GpuMemoryComponent component;
  std::string_view name;
};

constexpr std::array<ComponentDescriptor, kGpuMemoryComponentCount>
    kComponentDescriptors{{
        {GpuMemoryComponent::kMccl, "mccl", false},
        {GpuMemoryComponent::kCommon, "common", true},
        {GpuMemoryComponent::kNvl, "nvl", true},
        {GpuMemoryComponent::kIb, "ib", true},
        {GpuMemoryComponent::kIbgda, "ibgda", true},
        {GpuMemoryComponent::kUnclassified, "unclassified", false},
    }};

constexpr std::array<ResourceDescriptor, kGpuMemoryResourceTypeCount>
    kResourceDescriptors{{
        {GpuMemoryResourceType::kMcclAllGatherOverlapCounters,
         GpuMemoryComponent::kMccl,
         "mccl.allgather.overlap_counters"},

        {GpuMemoryResourceType::kCommonTransportDispatchTable,
         GpuMemoryComponent::kCommon,
         "common.transport_dispatch_table"},

        {GpuMemoryResourceType::kNvlP2pDataStaging,
         GpuMemoryComponent::kNvl,
         "nvl.p2p.data_staging"},
        {GpuMemoryResourceType::kNvlP2pSignal,
         GpuMemoryComponent::kNvl,
         "nvl.p2p.signal"},
        {GpuMemoryResourceType::kNvlP2pChannelState,
         GpuMemoryComponent::kNvl,
         "nvl.p2p.channel_state"},
        {GpuMemoryResourceType::kNvlP2pChannelProgress,
         GpuMemoryComponent::kNvl,
         "nvl.p2p.channel_progress"},
        {GpuMemoryResourceType::kNvlP2pBarrier,
         GpuMemoryComponent::kNvl,
         "nvl.p2p.barrier"},
        {GpuMemoryResourceType::kNvlP2pLl,
         GpuMemoryComponent::kNvl,
         "nvl.p2p.ll"},
        {GpuMemoryResourceType::kNvlP2pLl128,
         GpuMemoryComponent::kNvl,
         "nvl.p2p.ll128"},
        {GpuMemoryResourceType::kNvlP2pTransportTable,
         GpuMemoryComponent::kNvl,
         "nvl.p2p.transport_table"},
        {GpuMemoryResourceType::kNvlMultimemCombinedBacking,
         GpuMemoryComponent::kNvl,
         "nvl.multimem.combined_backing"},
        {GpuMemoryResourceType::kNvlMultimemPeerSignalPtrTable,
         GpuMemoryComponent::kNvl,
         "nvl.multimem.peer_signal_ptr_table"},

        {GpuMemoryResourceType::kIbEagerSendBuffer,
         GpuMemoryComponent::kIb,
         "ib.eager.send_buffer"},
        {GpuMemoryResourceType::kIbEagerRecvBuffer,
         GpuMemoryComponent::kIb,
         "ib.eager.recv_buffer"},
        {GpuMemoryResourceType::kIbEagerControlBuffer,
         GpuMemoryComponent::kIb,
         "ib.eager.control_buffer"},
        {GpuMemoryResourceType::kIbSendRecvPeerBulk,
         GpuMemoryComponent::kIb,
         "ib.sendrecv.peer_bulk"},
        {GpuMemoryResourceType::kIbSlotSignalInbox,
         GpuMemoryComponent::kIb,
         "ib.slot.signal_inbox"},
        {GpuMemoryResourceType::kIbSlotCounter,
         GpuMemoryComponent::kIb,
         "ib.slot.counter"},
        {GpuMemoryResourceType::kIbSlotDiscardSignal,
         GpuMemoryComponent::kIb,
         "ib.slot.discard_signal"},

        {GpuMemoryResourceType::kIbgdaAtomicReturnSink,
         GpuMemoryComponent::kIbgda,
         "ibgda.atomic_return_sink"},
        {GpuMemoryResourceType::kIbgdaPeerTransportArray,
         GpuMemoryComponent::kIbgda,
         "ibgda.peer_transport_array"},

        {GpuMemoryResourceType::kUnclassified,
         GpuMemoryComponent::kUnclassified,
         "unclassified"},
    }};

constexpr bool componentDescriptorsAreIndexed() {
  for (std::size_t i = 0; i < kComponentDescriptors.size(); ++i) {
    if (componentIndex(kComponentDescriptors[i].component) != i) {
      return false;
    }
  }
  return true;
}

constexpr bool resourceDescriptorsAreIndexed() {
  for (std::size_t i = 0; i < kResourceDescriptors.size(); ++i) {
    if (resourceIndex(kResourceDescriptors[i].resource) != i) {
      return false;
    }
  }
  return true;
}

static_assert(componentDescriptorsAreIndexed());
static_assert(resourceDescriptorsAreIndexed());

struct RegisteredAllocation {
  GpuMemoryTracker* owner;
  std::weak_ptr<GpuMemoryTracker> gpuMemoryTracker;
  GpuMemoryResourceType resource;
  uint64_t logicalBytes;
  uint64_t accountedBytes;
  uint64_t generation;
  uint32_t freeOperationsInProgress{0};
};

struct AllocationRegistry {
  folly::Synchronized<folly::F14FastMap<uintptr_t, RegisteredAllocation>>
      allocations;
  std::atomic<uint64_t> allocationCount{0};
  std::atomic<uint64_t> nextGeneration{1};
};

constexpr std::size_t kGpuMemoryBackingKindCount =
    backingKindIndex(GpuMemoryBackingKind::kCount);
using AllocationRegistries =
    std::array<AllocationRegistry, kGpuMemoryBackingKindCount>;

thread_local GpuMemoryTracker* currentGpuMemoryTracker = nullptr;
// Published registries intentionally live until process exit so GPU-owning
// statics can release safely during late teardown.
constinit std::atomic<AllocationRegistries*> gpuMemoryAllocationRegistries{
    nullptr};

AllocationRegistry* FOLLY_NULLABLE
findAllocationRegistry(GpuMemoryBackingKind backingKind) noexcept {
  const auto index = backingKindIndex(backingKind);
  if (index >= kGpuMemoryBackingKindCount) {
    return nullptr;
  }
  auto* const registries =
      gpuMemoryAllocationRegistries.load(std::memory_order_acquire);
  return registries == nullptr ? nullptr : &(*registries)[index];
}

AllocationRegistry* FOLLY_NULLABLE
getOrCreateAllocationRegistry(GpuMemoryBackingKind backingKind) noexcept {
  const auto index = backingKindIndex(backingKind);
  if (index >= kGpuMemoryBackingKindCount) {
    return nullptr;
  }
  if (auto* const registry = findAllocationRegistry(backingKind)) {
    return registry;
  }

  try {
    auto candidate = std::make_unique<AllocationRegistries>();
    auto* expected = static_cast<AllocationRegistries*>(nullptr);
    if (gpuMemoryAllocationRegistries.compare_exchange_strong(
            expected,
            candidate.get(),
            std::memory_order_release,
            std::memory_order_acquire)) {
      expected = candidate.release();
    }
    return expected == nullptr ? nullptr : &(*expected)[index];
  } catch (...) {
    return findAllocationRegistry(backingKind);
  }
}

bool isValidResource(GpuMemoryResourceType resource) {
  return resourceIndex(resource) < kGpuMemoryResourceTypeCount;
}

bool isValidBackingKind(GpuMemoryBackingKind backingKind) {
  return backingKindIndex(backingKind) < kGpuMemoryBackingKindCount;
}

constexpr bool isPrimsComponent(GpuMemoryComponent component) noexcept {
  const auto index = componentIndex(component);
  return index < kComponentDescriptors.size() &&
      kComponentDescriptors[index].isPrims;
}

void addAllocationUsage(
    GpuMemoryUsage& usage,
    uint64_t logicalBytes,
    uint64_t accountedBytes) {
  usage.currentLogicalBytes += logicalBytes;
  usage.currentBytes += accountedBytes;
  usage.peakBytes = std::max(usage.peakBytes, usage.currentBytes);
  usage.totalAllocatedBytes += accountedBytes;
}

void removeAllocationUsage(
    GpuMemoryUsage& usage,
    uint64_t logicalBytes,
    uint64_t accountedBytes) {
  usage.currentLogicalBytes -= logicalBytes;
  usage.currentBytes -= accountedBytes;
  usage.totalFreedBytes += accountedBytes;
}

double bytesToMiB(uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

} // namespace

struct GpuMemoryTracker::Impl {
  struct State {
    GpuMemoryContext context;
    std::array<GpuMemoryUsage, kGpuMemoryComponentCount> components{};
    std::array<GpuMemoryUsage, kGpuMemoryResourceTypeCount> resources{};
    GpuMemoryUsage prims{};
    GpuMemoryUsage total{};
    uint64_t activeAllocations{0};
    uint64_t duplicateAllocations{0};
  };

  explicit Impl(GpuMemoryContext gpuMemoryContext) {
    state.wlock()->context = gpuMemoryContext;
  }
  folly::Synchronized<State> state;
};

std::string_view gpuMemoryComponentName(GpuMemoryComponent component) noexcept {
  const auto index = componentIndex(component);
  return index < kComponentDescriptors.size()
      ? kComponentDescriptors[index].name
      : "unknown";
}

std::string_view gpuMemoryResourceTypeName(
    GpuMemoryResourceType resource) noexcept {
  const auto index = resourceIndex(resource);
  return index < kResourceDescriptors.size() ? kResourceDescriptors[index].name
                                             : "unknown";
}

GpuMemoryComponent gpuMemoryResourceTypeComponent(
    GpuMemoryResourceType resource) noexcept {
  const auto index = resourceIndex(resource);
  return index < kResourceDescriptors.size()
      ? kResourceDescriptors[index].component
      : GpuMemoryComponent::kCount;
}

std::string formatGpuMemorySnapshot(
    std::string_view stage,
    const GpuMemorySnapshot& snapshot) {
  std::string report = fmt::format(
      "MCCL GPU memory [{}] commHash={:x} rank={} gpu={}\n"
      "  Prims transport-owned GPU memory:\n",
      stage,
      snapshot.context.commHash,
      snapshot.context.rank,
      snapshot.context.device);
  for (const auto& descriptor : kComponentDescriptors) {
    if (!descriptor.isPrims) {
      continue;
    }
    const auto& usage =
        snapshot.components[componentIndex(descriptor.component)];
    report += fmt::format(
        "    {}: {:.2f} MiB ({} bytes)\n",
        descriptor.name,
        bytesToMiB(usage.currentBytes),
        usage.currentBytes);
  }
  const auto& mcclUsage =
      snapshot.components[componentIndex(GpuMemoryComponent::kMccl)];
  const auto& unclassifiedUsage =
      snapshot.components[componentIndex(GpuMemoryComponent::kUnclassified)];
  report += fmt::format(
      "    total: {:.2f} MiB ({} bytes)\n"
      "    peak: {:.2f} MiB ({} bytes)\n"
      "  direct MCCL-owned GPU memory: current={:.2f} MiB ({} bytes) peak={:.2f} MiB ({} bytes)\n"
      "  unclassified GPU memory: current={:.2f} MiB ({} bytes) peak={:.2f} MiB ({} bytes)\n"
      "  total accounted GPU memory: current={:.2f} MiB ({} bytes) peak={:.2f} MiB ({} bytes)\n",
      bytesToMiB(snapshot.prims.currentBytes),
      snapshot.prims.currentBytes,
      bytesToMiB(snapshot.prims.peakBytes),
      snapshot.prims.peakBytes,
      bytesToMiB(mcclUsage.currentBytes),
      mcclUsage.currentBytes,
      bytesToMiB(mcclUsage.peakBytes),
      mcclUsage.peakBytes,
      bytesToMiB(unclassifiedUsage.currentBytes),
      unclassifiedUsage.currentBytes,
      bytesToMiB(unclassifiedUsage.peakBytes),
      unclassifiedUsage.peakBytes,
      bytesToMiB(snapshot.total.currentBytes),
      snapshot.total.currentBytes,
      bytesToMiB(snapshot.total.peakBytes),
      snapshot.total.peakBytes);
  for (size_t i = 0; i < kGpuMemoryResourceTypeCount; ++i) {
    const auto& usage = snapshot.resources[i];
    if (usage.currentBytes == 0 && usage.peakBytes == 0) {
      continue;
    }
    report += fmt::format(
        "  resource {}: current={:.2f} MiB ({} bytes) peak={:.2f} MiB ({} bytes)\n",
        gpuMemoryResourceTypeName(static_cast<GpuMemoryResourceType>(i)),
        bytesToMiB(usage.currentBytes),
        usage.currentBytes,
        bytesToMiB(usage.peakBytes),
        usage.peakBytes);
  }
  if (!report.empty() && report.back() == '\n') {
    report.pop_back();
  }
  return report;
}

GpuMemoryTracker::GpuMemoryTracker(GpuMemoryContext context)
    : impl_{std::make_unique<Impl>(context)} {}

GpuMemoryTracker::~GpuMemoryTracker() {
  for (std::size_t i = 0; i < kGpuMemoryBackingKindCount; ++i) {
    auto* const registry =
        findAllocationRegistry(static_cast<GpuMemoryBackingKind>(i));
    if (registry == nullptr) {
      continue;
    }
    auto allocations = registry->allocations.wlock();
    uint64_t erased = 0;
    for (auto it = allocations->begin(); it != allocations->end();) {
      if (it->second.owner == this) {
        it = allocations->erase(it);
        ++erased;
      } else {
        ++it;
      }
    }
    registry->allocationCount.fetch_sub(erased, std::memory_order_release);
  }
}

ScopedGpuMemoryContext::ScopedGpuMemoryContext(
    GpuMemoryTracker* gpuMemoryTracker) noexcept
    : previousGpuMemoryTracker_{currentGpuMemoryTracker} {
  if (gpuMemoryTracker != nullptr || currentGpuMemoryTracker != nullptr) {
    currentGpuMemoryTracker = gpuMemoryTracker;
    restorePrevious_ = true;
  }
}

ScopedGpuMemoryContext::~ScopedGpuMemoryContext() noexcept {
  if (restorePrevious_) {
    currentGpuMemoryTracker = previousGpuMemoryTracker_;
  }
}

void GpuMemoryTracker::updateContext(GpuMemoryContext context) {
  impl_->state.wlock()->context = context;
}

bool GpuMemoryTracker::recordAllocation(
    GpuMemoryResourceType resource,
    uint64_t logicalBytes,
    uint64_t accountedBytes) noexcept {
  try {
    auto state = impl_->state.wlock();
    const auto component = gpuMemoryResourceTypeComponent(resource);
    auto& resourceUsage = state->resources[resourceIndex(resource)];
    auto& componentUsage = state->components[componentIndex(component)];
    addAllocationUsage(resourceUsage, logicalBytes, accountedBytes);
    addAllocationUsage(componentUsage, logicalBytes, accountedBytes);
    if (isPrimsComponent(component)) {
      addAllocationUsage(state->prims, logicalBytes, accountedBytes);
    }
    addAllocationUsage(state->total, logicalBytes, accountedBytes);
    ++state->activeAllocations;
    return true;
  } catch (...) {
    // Accounting is best-effort and must not fail a communicator allocation.
    return false;
  }
}

void GpuMemoryTracker::recordDuplicateAllocation() noexcept {
  try {
    ++impl_->state.wlock()->duplicateAllocations;
  } catch (...) {
    return;
  }
}

void GpuMemoryTracker::recordFree(
    GpuMemoryResourceType resource,
    uint64_t logicalBytes,
    uint64_t accountedBytes) noexcept {
  try {
    auto state = impl_->state.wlock();
    const auto component = gpuMemoryResourceTypeComponent(resource);
    auto& resourceUsage = state->resources[resourceIndex(resource)];
    auto& componentUsage = state->components[componentIndex(component)];
    removeAllocationUsage(resourceUsage, logicalBytes, accountedBytes);
    removeAllocationUsage(componentUsage, logicalBytes, accountedBytes);
    if (isPrimsComponent(component)) {
      removeAllocationUsage(state->prims, logicalBytes, accountedBytes);
    }
    removeAllocationUsage(state->total, logicalBytes, accountedBytes);
    --state->activeAllocations;
  } catch (...) {
    // Accounting must remain safe during communicator teardown.
    return;
  }
}

GpuMemorySnapshot GpuMemoryTracker::snapshot() const {
  auto state = impl_->state.rlock();
  return GpuMemorySnapshot{
      .context = state->context,
      .components = state->components,
      .resources = state->resources,
      .prims = state->prims,
      .total = state->total,
      .activeAllocations = state->activeAllocations,
      .duplicateAllocations = state->duplicateAllocations,
  };
}

void recordGpuMemoryAllocation(
    GpuMemoryResourceType resource,
    uintptr_t backingId,
    uint64_t logicalBytes,
    uint64_t accountedBytes,
    GpuMemoryBackingKind backingKind) noexcept {
  if (!isValidResource(resource) || !isValidBackingKind(backingKind) ||
      backingId == 0 || accountedBytes == 0 ||
      currentGpuMemoryTracker == nullptr) {
    return;
  }

  const auto gpuMemoryTracker =
      currentGpuMemoryTracker->weak_from_this().lock();
  if (!gpuMemoryTracker) {
    return;
  }
  auto* const registry = getOrCreateAllocationRegistry(backingKind);
  if (registry == nullptr) {
    return;
  }
  std::optional<RegisteredAllocation> replacedAllocation;
  std::shared_ptr<GpuMemoryTracker> replacedTracker;
  bool duplicate = false;
  try {
    {
      auto allocations = registry->allocations.wlock();
      const auto existing = allocations->find(backingId);
      if (existing != allocations->end()) {
        if (existing->second.freeOperationsInProgress != 0) {
          replacedAllocation = existing->second;
          replacedTracker = existing->second.gpuMemoryTracker.lock();
        } else if (!existing->second.gpuMemoryTracker.expired()) {
          duplicate = true;
        }
        if (!duplicate) {
          allocations->erase(existing);
          registry->allocationCount.fetch_sub(1, std::memory_order_relaxed);
          if (replacedTracker) {
            replacedTracker->recordFree(
                replacedAllocation->resource,
                replacedAllocation->logicalBytes,
                replacedAllocation->accountedBytes);
          }
        }
      }

      if (!duplicate) {
        const auto generation =
            registry->nextGeneration.fetch_add(1, std::memory_order_relaxed);
        const auto [it, inserted] = allocations->emplace(
            backingId,
            RegisteredAllocation{
                .owner = gpuMemoryTracker.get(),
                .gpuMemoryTracker = gpuMemoryTracker,
                .resource = resource,
                .logicalBytes = logicalBytes,
                .accountedBytes = accountedBytes,
                .generation = generation,
            });
        if (!inserted) {
          duplicate = true;
        } else if (!gpuMemoryTracker->recordAllocation(
                       resource, logicalBytes, accountedBytes)) {
          allocations->erase(it);
        } else {
          registry->allocationCount.fetch_add(1, std::memory_order_release);
        }
      }
    }
  } catch (...) {
    return;
  }
  if (duplicate) {
    gpuMemoryTracker->recordDuplicateAllocation();
  }
}

void recordGpuMemoryFree(
    uintptr_t backingId,
    GpuMemoryBackingKind backingKind) noexcept {
  const auto generation = beginGpuMemoryFree(backingId, backingKind);
  finishGpuMemoryFree(backingId, backingKind, generation, true);
}

uint64_t beginGpuMemoryFree(
    uintptr_t backingId,
    GpuMemoryBackingKind backingKind) noexcept {
  if (!isValidBackingKind(backingKind) || backingId == 0) {
    return 0;
  }

  auto* const registry = findAllocationRegistry(backingKind);
  if (registry == nullptr ||
      registry->allocationCount.load(std::memory_order_acquire) == 0) {
    return 0;
  }

  try {
    auto allocations = registry->allocations.wlock();
    const auto it = allocations->find(backingId);
    if (it == allocations->end()) {
      return 0;
    }
    ++it->second.freeOperationsInProgress;
    return it->second.generation;
  } catch (...) {
    return 0;
  }
}

void finishGpuMemoryFree(
    uintptr_t backingId,
    GpuMemoryBackingKind backingKind,
    uint64_t generation,
    bool succeeded) noexcept {
  if (!isValidBackingKind(backingKind) || backingId == 0 || generation == 0) {
    return;
  }

  auto* const registry = findAllocationRegistry(backingKind);
  if (registry == nullptr) {
    return;
  }
  std::optional<RegisteredAllocation> allocation;
  std::shared_ptr<GpuMemoryTracker> gpuMemoryTracker;
  try {
    {
      auto allocations = registry->allocations.wlock();
      const auto it = allocations->find(backingId);
      if (it == allocations->end() || it->second.generation != generation) {
        return;
      }
      if (!succeeded) {
        if (it->second.freeOperationsInProgress != 0) {
          --it->second.freeOperationsInProgress;
        }
        return;
      }
      allocation = it->second;
      gpuMemoryTracker = allocation->gpuMemoryTracker.lock();
      allocations->erase(it);
      registry->allocationCount.fetch_sub(1, std::memory_order_release);
      if (gpuMemoryTracker) {
        gpuMemoryTracker->recordFree(
            allocation->resource,
            allocation->logicalBytes,
            allocation->accountedBytes);
      }
    }
  } catch (...) {
    return;
  }
}

} // namespace meta::comms::memtrace
