// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/mccl/TorchCommMCCLCCA.hpp"
#include "comms/torchcomms/utils/Logging.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string_view>

namespace torch::comms {
namespace {

struct AddressRange {
  uintptr_t begin;
  uintptr_t end;

  static std::optional<AddressRange> from(void* addr, size_t len) {
    if (len == 0) {
      return std::nullopt;
    }

    const uintptr_t begin = reinterpret_cast<uintptr_t>(addr);
    if (len > std::numeric_limits<uintptr_t>::max() - begin) {
      return std::nullopt;
    }

    return AddressRange{begin, begin + len};
  }

  bool overlaps(const AddressRange& other) const {
    return begin < other.end && other.begin < end;
  }

  bool contains(const AddressRange& other) const {
    return begin <= other.begin && other.end <= end;
  }

  void* beginPtr() const {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(begin);
  }

  size_t size() const {
    return end - begin;
  }
};

} // namespace

// Global function to be registered as a hook
void mcclCachingAllocatorHookFn(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // Forward to the singleton instance

  // WARNING: This is a temporary solution to avoid the race condition between
  // the destruction of the hook and use of the hook. Tasks: T250902082
  if (McclCachingAllocatorHook::isDestroyed()) {
    return;
  }
  McclCachingAllocatorHook::getInstance().regDeregMem(te);
}

McclCachingAllocatorHookImpl& McclCachingAllocatorHook::getInstance() {
  // Use std::call_once for thread-safe singleton initialization
  // NOLINTNEXTLINE(facebook-hte-std::call_once)
  std::call_once(init_flag_, createInstance);
  return *instance_;
}

void McclCachingAllocatorHook::setDestroyed() {
  destroyed_ = true;
}

bool McclCachingAllocatorHook::isDestroyed() {
  return destroyed_;
}

bool McclCachingAllocatorHook::isEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("TORCHCOMM_MCCL_ENABLE_CCA_HOOK");
    bool e = env != nullptr && std::string_view(env) == "1";
    TC_LOG(INFO) << "TorchCommMCCL: CCA hook " << (e ? "ENABLED" : "DISABLED")
                 << " (TORCHCOMM_MCCL_ENABLE_CCA_HOOK="
                 << (env ? env : "<unset>") << ")";
    return e;
  }();
  return enabled;
}

DefaultMcclCachingAllocatorHookImpl::DefaultMcclCachingAllocatorHookImpl() {
  // Setup memory registration hooks
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  registerMemPreHook();
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
      &mcclCachingAllocatorHookFn);
}

DefaultMcclCachingAllocatorHookImpl::~DefaultMcclCachingAllocatorHookImpl() {
  McclCachingAllocatorHook::setDestroyed();
}

void McclCachingAllocatorHookImpl::registerMemPreHook() {
  // We assume no mem pool and no comm has been created yet, we just loop up the
  // snapshot of the default pool for all devices.
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& segmentInfo : snapshot.segments) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(segmentInfo.address);
    size_t len = segmentInfo.total_size;

    if (registeredMemMap_.contains(addr)) {
      continue;
    }
    registeredMemMap_.emplace(addr, MemInfo{len, segmentInfo.device});
  }
}

void McclCachingAllocatorHookImpl::regDeregMem(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  std::lock_guard<std::mutex> lock(mutex_);
  constexpr int kSegAlloc = static_cast<int>(
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC);
  constexpr int kSegFree = static_cast<int>(
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE);
  constexpr int kSegMap = static_cast<int>(
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_MAP);
  constexpr int kSegUnmap = static_cast<int>(
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_UNMAP);
  const int action = static_cast<int>(te.action_);
  const bool register_mem = action == kSegAlloc || action == kSegMap;
  const bool unregister_mem = action == kSegFree || action == kSegUnmap;

  if (register_mem) {
    // Memory got allocated, register it with MCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;

    if (len == 0 || isRegisteredRangeCovered(addr, len, te.device_)) {
      return;
    }
    queuePendingOperation(
        {PendingOperationType::Register, addr, len, te.device_});
    registeredMemMap_.emplace(addr, MemInfo{len, te.device_});

  } else if (unregister_mem) {
    // Memory got freed, deregister it with MCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;

    if (len == 0) {
      return;
    }
    if (!removeRegisteredRange(addr, len, te.device_)) {
      // Memory was allocated before the hook was installed — skip
      return;
    }
  }
}

void McclCachingAllocatorHookImpl::registerComm(TorchCommMCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (registeredComms_.contains(comm)) {
    return;
  }

  // Register all memory that has already been allocated
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      TorchCommMCCL::global_register_address(
          TorchCommMCCL::AddressWithLen(addr, mem_info.len));
    }
  }

  registeredComms_.insert(comm);
}

void McclCachingAllocatorHookImpl::deregisterComm(TorchCommMCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  drainPendingRegistrationOpsLocked();

  if (!registeredComms_.contains(comm)) {
    TC_LOG(WARNING)
        << "TorchCommMCCL: " << comm->commName_
        << " was called deregister from memory allocator while the comm is not registered."
        << " Please, double check as it could be a bug. However, in dynamic regime (FT enabled) it could happen.";
    return;
  }

  // De-register all memory that has already been allocated
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      TorchCommMCCL::global_deregister_address(
          TorchCommMCCL::AddressWithLen(addr, mem_info.len));
    }
  }

  registeredComms_.erase(comm);
}

void McclCachingAllocatorHookImpl::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  drainPendingRegistrationOpsLocked();

  for (const auto& [addr, mem_info] : registeredMemMap_) {
    TorchCommMCCL::global_deregister_address(
        TorchCommMCCL::AddressWithLen(addr, mem_info.len));
  }
  registeredMemMap_.clear();
  registeredComms_.clear();
}

bool McclCachingAllocatorHookImpl::isCommRegistered(TorchCommMCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  return registeredComms_.contains(comm);
}

void McclCachingAllocatorHookImpl::queuePendingOperation(
    PendingOperation operation) {
  std::lock_guard<std::mutex> lock(pendingMutex_);
  pendingOps_.push_back(operation);
}

bool McclCachingAllocatorHookImpl::removePendingRegistrations(
    void* addr,
    size_t len) {
  auto rangeToRemove = AddressRange::from(addr, len);
  if (!rangeToRemove) {
    return false;
  }

  std::lock_guard<std::mutex> lock(pendingMutex_);
  const auto oldSize = pendingOps_.size();
  pendingOps_.erase(
      std::remove_if(
          pendingOps_.begin(),
          pendingOps_.end(),
          [&rangeToRemove](const PendingOperation& operation) {
            auto pendingRange =
                AddressRange::from(operation.addr, operation.len);
            return operation.type == PendingOperationType::Register &&
                pendingRange && pendingRange->overlaps(*rangeToRemove);
          }),
      pendingOps_.end());
  return pendingOps_.size() != oldSize;
}

bool McclCachingAllocatorHookImpl::isRegisteredRangeCovered(
    void* addr,
    size_t len,
    int32_t device) const {
  auto targetRange = AddressRange::from(addr, len);
  if (!targetRange) {
    return false;
  }

  return std::any_of(
      registeredMemMap_.begin(),
      registeredMemMap_.end(),
      [&targetRange, device](const auto& entry) {
        const auto& [registeredAddr, memInfo] = entry;
        auto registeredRange = AddressRange::from(registeredAddr, memInfo.len);
        return memInfo.device == device && registeredRange &&
            registeredRange->contains(*targetRange);
      });
}

bool McclCachingAllocatorHookImpl::removeRegisteredRange(
    void* addr,
    size_t len,
    int32_t device) {
  auto rangeToRemove = AddressRange::from(addr, len);
  if (!rangeToRemove) {
    return false;
  }

  struct Range {
    void* addr;
    size_t len;
    int32_t device;
  };

  std::vector<Range> rangesToErase;
  std::vector<Range> rangesToRegister;

  for (const auto& [registeredAddr, memInfo] : registeredMemMap_) {
    if (memInfo.device != device) {
      continue;
    }

    auto registeredRange = AddressRange::from(registeredAddr, memInfo.len);
    if (!registeredRange || !registeredRange->overlaps(*rangeToRemove)) {
      continue;
    }

    rangesToErase.push_back({registeredAddr, memInfo.len, memInfo.device});
    if (registeredRange->begin < rangeToRemove->begin) {
      rangesToRegister.push_back({
          registeredRange->beginPtr(),
          rangeToRemove->begin - registeredRange->begin,
          memInfo.device,
      });
    }
    if (rangeToRemove->end < registeredRange->end) {
      const AddressRange rightRange{rangeToRemove->end, registeredRange->end};
      rangesToRegister.push_back({
          rightRange.beginPtr(),
          rightRange.size(),
          memInfo.device,
      });
    }
  }

  removePendingRegistrations(addr, len);

  if (rangesToErase.empty()) {
    return false;
  }

  for (const auto& range : rangesToErase) {
    registeredMemMap_.erase(range.addr);
    queuePendingOperation({
        PendingOperationType::Deregister,
        range.addr,
        range.len,
        range.device,
    });
  }

  for (const auto& range : rangesToRegister) {
    registeredMemMap_.emplace(range.addr, MemInfo{range.len, range.device});
    queuePendingOperation({
        PendingOperationType::Register,
        range.addr,
        range.len,
        range.device,
    });
  }

  return true;
}

void McclCachingAllocatorHookImpl::drainPendingRegistrations() {
  if (!McclCachingAllocatorHook::isEnabled() ||
      McclCachingAllocatorHook::isDestroyed()) {
    return;
  }
  McclCachingAllocatorHook::getInstance().drainPendingRegistrationOps();
}

void McclCachingAllocatorHookImpl::drainPendingRegistrationOps() {
  std::lock_guard<std::mutex> mapLock(mutex_);
  drainPendingRegistrationOpsLocked();
}

void McclCachingAllocatorHookImpl::drainPendingRegistrationOpsLocked() {
  std::vector<PendingOperation> operations;
  {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (pendingOps_.empty()) {
      return;
    }
    operations.swap(pendingOps_);
  }

  for (const auto& operation : operations) {
    if (operation.type == PendingOperationType::Register) {
      auto registeredMemIt = registeredMemMap_.find(operation.addr);
      if (registeredMemIt == registeredMemMap_.end() ||
          registeredMemIt->second.len != operation.len ||
          registeredMemIt->second.device != operation.device) {
        continue;
      }
      TorchCommMCCL::global_register_address(
          TorchCommMCCL::AddressWithLen(operation.addr, operation.len));
    } else {
      TorchCommMCCL::global_deregister_address(
          TorchCommMCCL::AddressWithLen(operation.addr, operation.len));
    }
  }
}

} // namespace torch::comms
