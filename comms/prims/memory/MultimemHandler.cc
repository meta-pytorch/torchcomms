// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/memory/MultimemHandler.h"

#include "comms/prims/bootstrap/TeamStatusAgreement.h"
#include "comms/prims/core/Checks.h"
#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/memory/CuMulticastAllocation.h"
#include "comms/prims/memory/NvlMemExchange.h"

#if !defined(__HIP_PLATFORM_AMD__)
#include "comms/prims/platform/CudaDriverLazy.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <unistd.h>

namespace comms::prims {

namespace {

constexpr int kCachedMultimemSupportDevices = 8;

bool isPowerOfTwo(std::size_t x) {
  return x != 0 && (x & (x - 1)) == 0;
}

enum class MultimemSupportCacheState : uint8_t {
  kUnknown,
  kUnsupported,
  kSupported,
};

const char* handleTypeName(MultimemHandler::HandleType handleType) {
  switch (handleType) {
    case ShareableHandleType::kUnsupported:
      return "unsupported";
    case ShareableHandleType::kFabric:
      return "fabric";
    case ShareableHandleType::kPosixFd:
      return "posix_fd";
  }
  return "unknown";
}

bool multicastDriverApisAvailable() {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  return false;
#else
  return pfn_cuMulticastCreate != nullptr &&
      pfn_cuMulticastAddDevice != nullptr &&
      pfn_cuMulticastBindMem != nullptr && pfn_cuMulticastUnbind != nullptr &&
      pfn_cuMulticastGetGranularity != nullptr;
#endif
}

bool multicastSupported(CUdevice cuDev) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)cuDev;
  return false;
#else
  int multicastSupported = 0;
  auto err = pfn_cuDeviceGetAttribute(
      &multicastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cuDev);
  return err == CUDA_SUCCESS && multicastSupported != 0;
#endif
}

// Builds the local-backing / multicast allocation prop for a single chosen
// shareable handle type, reusing the shared makeVmmAllocationProp() helper.
CUmemAllocationProp makeLocalAllocationProp(
    CUdevice cuDev,
    MultimemHandler::HandleType handleType) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  return makeVmmAllocationProp(
      cuDev, static_cast<unsigned int>(toCudaHandleType(handleType)));
#else
  (void)handleType;
  (void)cuDev;
  return CUmemAllocationProp{};
#endif
}

// Single O(n) pass that validates `nvlRankToCommRank` (non-empty, no negative
// ranks, no duplicates, `commRank` present) and returns this rank's NVL-local
// index. Throws std::runtime_error on any violation.
int32_t computeNvlRankAndValidate(
    int32_t commRank,
    const std::vector<int>& nvlRankToCommRank) {
  if (nvlRankToCommRank.empty()) {
    throw std::runtime_error(
        "MultimemHandler: nvlRankToCommRank must be non-empty");
  }
  std::unordered_set<int> seen;
  seen.reserve(nvlRankToCommRank.size());
  int32_t nvlRank = -1;
  for (std::size_t i = 0; i < nvlRankToCommRank.size(); ++i) {
    const int rank = nvlRankToCommRank[i];
    if (rank < 0) {
      throw std::runtime_error(
          "MultimemHandler: nvlRankToCommRank contains a negative rank");
    }
    if (!seen.insert(rank).second) {
      throw std::runtime_error(
          "MultimemHandler: nvlRankToCommRank contains duplicate ranks");
    }
    if (rank == commRank) {
      nvlRank = static_cast<int32_t>(i);
    }
  }
  if (nvlRank < 0) {
    throw std::runtime_error(
        "MultimemHandler: commRank must appear in nvlRankToCommRank");
  }
  return nvlRank;
}

// Selects the shareable handle type for multicast on `cudaDevice`. Layers the
// multicast-specific requirements (multicast driver APIs + multicast device
// attribute) on top of the shared selectShareableHandleType() probe.
MultimemHandler::HandleType selectMultimemHandleTypeImpl(int cudaDevice) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)cudaDevice;
  return ShareableHandleType::kUnsupported;
#else
  if (cudaDevice < 0) {
    return ShareableHandleType::kUnsupported;
  }
  if (cuda_driver_lazy_init() != 0) {
    return ShareableHandleType::kUnsupported;
  }
  if (!multicastDriverApisAvailable()) {
    return ShareableHandleType::kUnsupported;
  }

  CUdevice cuDev = 0;
  if (pfn_cuDeviceGet(&cuDev, cudaDevice) != CUDA_SUCCESS) {
    return ShareableHandleType::kUnsupported;
  }

  if (!multicastSupported(cuDev)) {
    return ShareableHandleType::kUnsupported;
  }

  return selectShareableHandleType(cudaDevice);
#endif
}

enum class MultimemExchangePhase {
  kNone,
  kInitializeDevice,
  kSetupAgreement,
  kComputeAllocationLayout,
  kCreateMulticastHandle,
  kExportMulticastHandle,
  kHandleAgreement,
  kImportMulticastHandle,
  kAddLocalDeviceToMulticast,
  kJoinAgreement,
  kBindLocalMemoryToMulticast,
  kMapMulticastMemory,
  kBindMapAgreement,
};

const char* phaseName(MultimemExchangePhase phase) {
  switch (phase) {
    case MultimemExchangePhase::kNone:
      return "none";
    case MultimemExchangePhase::kInitializeDevice:
      return "initializeDevice";
    case MultimemExchangePhase::kSetupAgreement:
      return "agreeOnSetup";
    case MultimemExchangePhase::kComputeAllocationLayout:
      return "computeAllocationLayout";
    case MultimemExchangePhase::kCreateMulticastHandle:
      return "createMulticastHandle";
    case MultimemExchangePhase::kExportMulticastHandle:
      return "exportMulticastHandle";
    case MultimemExchangePhase::kHandleAgreement:
      return "exchangeMulticastHandle";
    case MultimemExchangePhase::kImportMulticastHandle:
      return "importMulticastHandle";
    case MultimemExchangePhase::kAddLocalDeviceToMulticast:
      return "addLocalDeviceToMulticast";
    case MultimemExchangePhase::kJoinAgreement:
      return "agreeOnStage:join";
    case MultimemExchangePhase::kBindLocalMemoryToMulticast:
      return "bindLocalMemoryToMulticast";
    case MultimemExchangePhase::kMapMulticastMemory:
      return "mapMulticastMemory";
    case MultimemExchangePhase::kBindMapAgreement:
      return "agreeOnStage:bind-map";
  }
  return "unknown";
}

class MultimemExchangeProgress {
 public:
  template <typename Action>
  void run(MultimemExchangePhase phase, Action&& action) {
    active_ = phase;
    std::forward<Action>(action)();
    completed_ = phase;
  }

  void beginAgreement(MultimemExchangePhase phase) {
    active_ = phase;
  }

  void completeAgreement() {
    completed_ = active_;
  }

  const char* activeName() const {
    return phaseName(active_);
  }

  const char* completedName() const {
    return phaseName(completed_);
  }

 private:
  MultimemExchangePhase active_{MultimemExchangePhase::kNone};
  MultimemExchangePhase completed_{MultimemExchangePhase::kNone};
};

template <typename LocalWork, typename Agreement>
void runAgreedStage(
    MultimemExchangeProgress& progress,
    MultimemExchangePhase agreementPhase,
    LocalWork&& localWork,
    Agreement&& agreement) {
  std::exception_ptr localError;
  try {
    std::forward<LocalWork>(localWork)();
  } catch (...) {
    localError = std::current_exception();
  }

  if (!localError) {
    progress.beginAgreement(agreementPhase);
  }
  std::forward<Agreement>(agreement)(localError);
  progress.completeAgreement();
}

} // namespace

std::shared_ptr<CuMemAllocation> MultimemHandler::requireBacking(
    std::shared_ptr<CuMemAllocation> backing) {
  if (!backing) {
    throw std::runtime_error(
        "MultimemHandler: backing CuMemAllocation must be non-null");
  }
  return backing;
}

MultimemHandler::MultimemHandler(
    std::shared_ptr<CuMemAllocation> backing,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    int32_t commRank,
    std::vector<int> nvlRankToCommRank,
    int cudaDevice,
    MulticastExchangeContract contract)
    : bootstrap_(std::move(bootstrap)),
      commRank_(commRank),
      nvlRank_(computeNvlRankAndValidate(commRank, nvlRankToCommRank)),
      nvlRanks_(static_cast<int32_t>(nvlRankToCommRank.size())),
      nvlRankToCommRank_(std::move(nvlRankToCommRank)),
      cudaDevice_(cudaDevice),
      requestedSize_(requireBacking(backing)->size()),
      contract_(contract),
      backing_(std::move(backing)) {
  if (requestedSize_ == 0) {
    throw std::runtime_error("MultimemHandler: backing size must be non-zero");
  }
  if (!bootstrap_) {
    throw std::runtime_error("MultimemHandler: bootstrap must be non-null");
  }
  handleType_ = selectMultimemHandleTypeImpl(cudaDevice_);
  if (handleType_ == HandleType::kUnsupported) {
    throw std::runtime_error(
        "MultimemHandler: multicast multimem is not supported. Requires CUDA "
        "12.3+, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, and fabric or POSIX "
        "FD allocation handles.");
  }
}

MultimemHandler::~MultimemHandler() {
  cleanup();
}

bool MultimemHandler::isMultimemSupported(int cudaDevice) {
  if (cudaDevice < 0) {
    return false;
  }

  static std::array<
      std::atomic<MultimemSupportCacheState>,
      kCachedMultimemSupportDevices>
      cachedResults{};
  if (cudaDevice >= kCachedMultimemSupportDevices) {
    return selectMultimemHandleTypeImpl(cudaDevice) != HandleType::kUnsupported;
  }

  auto& cachedResult = cachedResults[static_cast<std::size_t>(cudaDevice)];
  const auto cachedState = cachedResult.load(std::memory_order_acquire);
  if (cachedState != MultimemSupportCacheState::kUnknown) {
    return cachedState == MultimemSupportCacheState::kSupported;
  }

  const bool supported =
      selectMultimemHandleTypeImpl(cudaDevice) != HandleType::kUnsupported;
  cachedResult.store(
      supported ? MultimemSupportCacheState::kSupported
                : MultimemSupportCacheState::kUnsupported,
      std::memory_order_release);
  return supported;
}

std::size_t MultimemHandler::backingGranularity(int cudaDevice, int nvlRanks) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)cudaDevice;
  (void)nvlRanks;
  return 0;
#else
  const HandleType handleType = selectMultimemHandleTypeImpl(cudaDevice);
  if (handleType == HandleType::kUnsupported) {
    return 0;
  }

  CUdevice cuDev = 0;
  if (pfn_cuDeviceGet(&cuDev, cudaDevice) != CUDA_SUCCESS) {
    return 0;
  }

  auto localProp = makeLocalAllocationProp(cuDev, handleType);
  std::size_t localGranularity = 0;
  checkCuError(
      pfn_cuMemGetAllocationGranularity(
          &localGranularity, &localProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
      "cuMemGetAllocationGranularity failed");

  CUmulticastObjectProp multicastProp = {};
  multicastProp.numDevices = static_cast<unsigned int>(nvlRanks);
  multicastProp.size = localGranularity;
  multicastProp.handleTypes = toCudaHandleType(handleType);
  multicastProp.flags = 0;

  std::size_t multicastGranularity = 0;
  checkCuError(
      pfn_cuMulticastGetGranularity(
          &multicastGranularity,
          &multicastProp,
          CU_MULTICAST_GRANULARITY_RECOMMENDED),
      "cuMulticastGetGranularity failed");

  if (!isPowerOfTwo(localGranularity)) {
    throw std::runtime_error(
        "MultimemHandler: local allocation granularity is not a power of two");
  }
  if (!isPowerOfTwo(multicastGranularity)) {
    throw std::runtime_error(
        "MultimemHandler: multicast granularity is not a power of two");
  }

  return std::max(localGranularity, multicastGranularity);
#endif
}

void MultimemHandler::exchange() {
  if (exchanged_) {
    return;
  }
  // A previous exchange() attempt threw and cleanup() tore down `backing_`
  // / `overlay_` / `multicastMapping_`. Re-entering the body would
  // dereference the now-null `backing_` in computeAllocationLayout() /
  // bindLocalMemoryToMulticast(). Make this a terminal state: retry
  // requires constructing a fresh MultimemHandler over the same
  // (caller-owned) backing.
  if (failed_) {
    throw std::runtime_error(
        "MultimemHandler::exchange: this handler is in a terminal failed "
        "state after a prior exchange() attempt threw. cleanup() released "
        "the multicast object and the shared backing reference; construct "
        "a new MultimemHandler with the same backing to retry.");
  }

#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  throw std::runtime_error("MultimemHandler requires CUDA 12.3+");
#else
  MultimemExchangeProgress progress;

  const auto runStatusStage = [&](MultimemExchangePhase agreementPhase,
                                  const char* operation,
                                  auto&& localWork) {
    runAgreedStage(
        progress,
        agreementPhase,
        std::forward<decltype(localWork)>(localWork),
        [&](const std::exception_ptr& localError) {
          std::vector<detail::TeamStatus> records(
              static_cast<std::size_t>(nvlRanks_));
          detail::gatherAndAgree(
              nvlRank_,
              nvlRanks_,
              records,
              localError,
              operation,
              [&](void* data, int size) {
                return bootstrap_
                    ->allGatherNvlDomain(
                        data, size, nvlRank_, nvlRanks_, nvlRankToCommRank_)
                    .get();
              });
        });
  };

  try {
    // Multicast setup is host-synchronous locally, but it still has ordering
    // requirements across ranks:
    //
    // 1. initializeDevice() resolves cudaDevice_ to a CUdevice. The caller
    //    must have already made cudaDevice_ current; this handler does not
    //    call cudaSetDevice().
    //
    // 2. computeAllocationLayout() takes the shared backing's size (which the
    //    caller sized to the multicast granularity) and queries the multicast
    //    granularity for the VA mapping.
    //
    // 3. create/export/exchange/import shares only the multicast object
    //    handle. Team rank 0 owns object creation, exports either a fabric
    //    handle or a POSIX FD for it, all ranks exchange metadata, and the
    //    other ranks import rank 0's object before joining the multicast team.
    //
    // 4. addLocalDeviceToMulticast() registers this rank's device with the
    //    shared object. CUDA requires every participating device to be added
    //    before memory can be bound to the multicast object.
    //
    // 5. The join agreement serializes object create/import + addDevice across
    //    ranks before any rank calls cuMulticastBindMem. It also prevents
    //    successful ranks from entering bind after a peer returned an error.
    //
    // 6. bindLocalMemoryToMulticast() attaches this rank's shared backing
    //    allocation at offset 0 in the multicast object. After all ranks bind
    //    the same range, multimem stores to that range can be replicated by
    //    NVSwitch. The caller is responsible for zeroing the backing.
    //
    // 7. mapMulticastMemory() reserves and maps this rank's multicast VA, then
    //    grants device access. Kernels use this VA for multimem instructions.
    //
    // 8. The bind/map agreement is the "ready to use" contract for exchange():
    //    no rank returns and launches a kernel using multimem while another
    //    rank is still binding or mapping the multicast VA or has failed.
    runAgreedStage(
        progress,
        MultimemExchangePhase::kSetupAgreement,
        [&] {
          progress.run(MultimemExchangePhase::kInitializeDevice, [&] {
            cuDev_ = initializeDevice();
            deviceInitialized_ = true;
          });
        },
        [&](const std::exception_ptr& localError) {
          // Every rank independently selects handleType_ in the constructor;
          // verify they all picked the same one before rank 0 creates the
          // multicast object.
          agreeOnSetup(localError);
        });

    AllocationLayout layout{};
    ShareableHandle multicastHandle{};
    runAgreedStage(
        progress,
        MultimemExchangePhase::kHandleAgreement,
        [&] {
          progress.run(MultimemExchangePhase::kComputeAllocationLayout, [&] {
            layout = computeAllocationLayout();
            allocatedSize_ = layout.allocatedSize;
          });

          progress.run(MultimemExchangePhase::kCreateMulticastHandle, [&] {
            createMulticastHandle(layout.allocatedSize);
          });

          if (nvlRank_ == 0) {
            progress.run(MultimemExchangePhase::kExportMulticastHandle, [&] {
              multicastHandle = exportMulticastHandle();
            });
          }
        },
        [&](const std::exception_ptr& localError) {
          multicastHandle =
              exchangeMulticastHandle(multicastHandle, localError);
        });

    // nvlRank_ is the rank inside this NVLink multicast team. Team rank 0
    // already owns the CUDA multicast handle it created; every other team rank
    // imports rank 0's exchanged multicast handle.
    runStatusStage(
        MultimemExchangePhase::kJoinAgreement,
        "MultimemHandler multicast join",
        [&] {
          if (nvlRank_ != 0) {
            progress.run(MultimemExchangePhase::kImportMulticastHandle, [&] {
              importMulticastHandle(multicastHandle);
            });
          }

          // Every participating rank, including rank 0, must add its local
          // device before any rank binds memory to the multicast object.
          progress.run(MultimemExchangePhase::kAddLocalDeviceToMulticast, [&] {
            addLocalDeviceToMulticast(cuDev_);
          });
        });

    if (multicastExportedFd_ >= 0) {
      close(multicastExportedFd_);
      multicastExportedFd_ = -1;
    }

    runStatusStage(
        MultimemExchangePhase::kBindMapAgreement,
        "MultimemHandler multicast bind/map",
        [&] {
          progress.run(MultimemExchangePhase::kBindLocalMemoryToMulticast, [&] {
            bindLocalMemoryToMulticast(layout.allocatedSize);
          });
          progress.run(MultimemExchangePhase::kMapMulticastMemory, [&] {
            mapMulticastMemory(layout);
          });
        });

    exchanged_ = true;
  } catch (const std::exception& ex) {
    const auto state =
        describeState(progress.activeName(), progress.completedName());
    const std::string message =
        std::string("MultimemHandler::exchange failed: ") + ex.what() + "\n" +
        state;
    std::fprintf(stderr, "%s\n", message.c_str());
    failed_ = true;
    cleanup();
    std::throw_with_nested(std::runtime_error(message));
  } catch (...) {
    const auto state =
        describeState(progress.activeName(), progress.completedName());
    const std::string message =
        "MultimemHandler::exchange failed with unknown exception\n" + state;
    std::fprintf(stderr, "%s\n", message.c_str());
    failed_ = true;
    cleanup();
    std::throw_with_nested(std::runtime_error(message));
  }
#endif
}

void* MultimemHandler::getMultimemDeviceMemPtr() const {
  if (!exchanged_ || !multicastMapping_) {
    throw std::runtime_error(
        "MultimemHandler: exchange() must complete before using multimem ptr");
  }
  // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer type
  return reinterpret_cast<void*>(multicastMapping_->devicePtr());
}

std::size_t MultimemHandler::getAllocatedSize() const {
  if (!exchanged_) {
    throw std::runtime_error(
        "MultimemHandler: exchange() must complete before reading allocated size");
  }
  return allocatedSize_;
}

CUdevice MultimemHandler::initializeDevice() const {
  CUdevice cuDev = 0;
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  checkCuError(pfn_cuDeviceGet(&cuDev, cudaDevice_), "cuDeviceGet failed");
#endif
  return cuDev;
}

MultimemHandler::AllocationLayout MultimemHandler::computeAllocationLayout()
    const {
  AllocationLayout layout;

  layout.granularity =
      MultimemHandler::backingGranularity(cudaDevice_, nvlRanks_);
  // backingGranularity() returns 0 on the unsupported / failed-probe path
  // (no CUDA 12.3+, no multicast device attribute, or cuDeviceGet failure).
  // Computing `allocatedSize % 0` below would be UB / SIGFPE; surface a
  // legible error instead so the calling phase ("computeAllocationLayout")
  // gets attributed cleanly.
  if (layout.granularity == 0) {
    throw std::runtime_error(
        "MultimemHandler::computeAllocationLayout: multicast unsupported on "
        "this device / toolkit (backingGranularity returned 0)");
  }

  // The shared backing already exists. Use its size verbatim, but verify it can
  // be bound into the multicast object.
  layout.allocatedSize = backing_->size();
  if (layout.allocatedSize % layout.granularity != 0) {
    throw std::runtime_error(
        fmt::format(
            "shared CuMemAllocation size {} is not multicast-granularity aligned {}",
            layout.allocatedSize,
            layout.granularity));
  }
  return layout;
}

void MultimemHandler::createMulticastHandle(std::size_t allocatedSize) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  if (nvlRank_ != 0) {
    return;
  }

  CUmulticastObjectProp multicastProp = {};
  multicastProp.numDevices = static_cast<unsigned int>(nvlRanks_);
  multicastProp.size = allocatedSize;
  multicastProp.handleTypes = toCudaHandleType(handleType_);
  multicastProp.flags = 0;

  overlay_ = std::make_shared<CuMulticastAllocation>(
      CuMulticastAllocation::create(multicastProp));
#else
  (void)allocatedSize;
#endif
}

ShareableHandle MultimemHandler::exportMulticastHandle() {
  ShareableHandle handle = {};
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  if (nvlRank_ == 0) {
    handle = exportShareableHandle(overlay_->handle(), handleType_);
    // Retain the exported POSIX FD on rank 0 until cleanup so peers can
    // duplicate it during import.
    if (handle.type == HandleType::kPosixFd) {
      multicastExportedFd_ = handle.fd;
    }
  }
#endif
  return handle;
}

ShareableHandle MultimemHandler::exchangeMulticastHandle(
    ShareableHandle handle,
    const std::exception_ptr& localError) {
  struct HandleRecord {
    ShareableHandle handle{};
    detail::TeamStatus status{};
  };
  static_assert(std::is_trivially_copyable_v<HandleRecord>);

  std::vector<HandleRecord> records(static_cast<std::size_t>(nvlRanks_));
  // This trivially copyable wire record is exchanged as raw bytes. Zero its
  // padding before populating the local payload.
  // @lint-ignore CLANGTIDY facebook-hte-BadMemset
  std::memset(records.data(), 0, records.size() * sizeof(HandleRecord));
  records[static_cast<std::size_t>(nvlRank_)].handle = handle;
  detail::gatherAndAgree(
      nvlRank_,
      nvlRanks_,
      records,
      localError,
      "MultimemHandler multicast create/export",
      [&](void* data, int size) {
        return bootstrap_
            ->allGatherNvlDomain(
                data, size, nvlRank_, nvlRanks_, nvlRankToCommRank_)
            .get();
      });
  return records.front().handle;
}

void MultimemHandler::importMulticastHandle(const ShareableHandle& handle) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  overlay_ =
      std::make_shared<CuMulticastAllocation>(CuMulticastAllocation::adopt(
          importShareableHandle(handle), allocatedSize_));
#else
  (void)handle;
#endif
}

void MultimemHandler::addLocalDeviceToMulticast(CUdevice cuDev) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  overlay_->addDevice(cuDev);
#else
  (void)cuDev;
#endif
}

void MultimemHandler::bindLocalMemoryToMulticast(std::size_t allocatedSize) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  // Bind this rank's shared backing allocation at offset 0 in the multicast
  // object. CuMulticastAllocation records the binding so its destructor
  // unbinds.
  overlay_->bindMem(
      backing_->handle(), /*mcOffset=*/0, /*physOffset=*/0, allocatedSize);
#else
  (void)allocatedSize;
#endif
}

void MultimemHandler::mapMulticastMemory(const AllocationLayout& layout) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  // The multicast VA mapping co-owns overlay_ (the multicast object), so the
  // multicast handle stays alive at least as long as this VA mapping.
  // `CuMemMapping::overMulticast` reads the CUdevice from `overlay_` and
  // builds its own access descriptor internally -- no per-phase wiring of
  // device state needed here.
  multicastMapping_ =
      std::make_unique<CuMemMapping>(CuMemMapping::overMulticast(
          overlay_, layout.allocatedSize, layout.granularity));
#endif
}

void MultimemHandler::agreeOnSetup(const std::exception_ptr& localError) {
  struct SetupRecord {
    uint64_t handleType{0};
    uint64_t requestedSize{0};
    MulticastExchangeContract contract;
    detail::TeamStatus status{};
  };
  static_assert(std::is_trivially_copyable_v<SetupRecord>);
  static_assert(std::is_standard_layout_v<SetupRecord>);

  std::vector<SetupRecord> records(static_cast<std::size_t>(nvlRanks_));
  // This trivially copyable wire record is exchanged as raw bytes. Zero its
  // padding before populating the local payload.
  // @lint-ignore CLANGTIDY facebook-hte-BadMemset
  std::memset(records.data(), 0, records.size() * sizeof(SetupRecord));
  auto& local = records[static_cast<std::size_t>(nvlRank_)];
  local.handleType = static_cast<uint64_t>(handleType_);
  local.requestedSize = requestedSize_;
  local.contract = contract_;
  // Use the NVL-domain allGather (not the global one): nvlRank_/nvlRanks_ are
  // the NVLink-team indices, and the bootstrap's plain allGather expects
  // global comm ranks. Mirrors exchangeMulticastHandle's allGatherNvlDomain
  // call. A mismatched scope would either corrupt the buffer (if the global
  // rank count exceeded nvlRanks_) or skip the agreement check entirely (if
  // peers outside the NVL team contributed zeros).
  detail::gatherAndAgree(
      nvlRank_,
      nvlRanks_,
      records,
      localError,
      "MultimemHandler setup",
      [&](void* data, int size) {
        return bootstrap_
            ->allGatherNvlDomain(
                data, size, nvlRank_, nvlRanks_, nvlRankToCommRank_)
            .get();
      });
  for (int32_t r = 0; r < nvlRanks_; ++r) {
    const auto& peer = records[static_cast<std::size_t>(r)];
    if (peer.handleType != local.handleType ||
        peer.requestedSize != local.requestedSize ||
        !(peer.contract == local.contract)) {
      throw std::runtime_error(
          fmt::format(
              "MultimemHandler: ranks disagree on multicast setup "
              "(this rank nvl={} type={} size={} protocol={} version={} "
              "parameters=[{}], peer nvl={} type={} size={} protocol={} "
              "version={} parameters=[{}])",
              nvlRank_,
              handleTypeName(handleType_),
              local.requestedSize,
              local.contract.protocol,
              local.contract.version,
              fmt::join(local.contract.parameters, ", "),
              r,
              handleTypeName(static_cast<HandleType>(peer.handleType)),
              peer.requestedSize,
              peer.contract.protocol,
              peer.contract.version,
              fmt::join(peer.contract.parameters, ", ")));
    }
  }
}

std::string MultimemHandler::describeState(
    const char* failedPhase,
    const char* completedPhase) const {
  // Capture the handle / pointer values as plain integers for logging. The
  // driver typedefs are integer-valued on NVIDIA, but HIPify rewrites some of
  // them to opaque `hip*` POINTER types on the AMD build (and does so
  // inconsistently across the sibling allocation classes), so normalize each
  // value to `unsigned long long` whether it comes back as an integer or a
  // pointer.
  const auto toU64 = [](auto handle) -> unsigned long long {
    if constexpr (std::is_pointer_v<decltype(handle)>) {
      return reinterpret_cast<std::uintptr_t>(handle);
    } else {
      return static_cast<unsigned long long>(handle);
    }
  };
  const unsigned long long localAllocHandle =
      backing_ ? toU64(backing_->handle()) : 0ULL;
  const unsigned long long multicastHandle =
      overlay_ ? toU64(overlay_->handle()) : 0ULL;
  const unsigned long long multicastPtr =
      multicastMapping_ ? toU64(multicastMapping_->devicePtr()) : 0ULL;
  return fmt::format(
      "MultimemHandler state:\n"
      "  progress={{failedPhase={} completedPhase={}}}\n"
      "  team={{commRank={} nvlRank={} nvlRanks={} rankMap=[{}]}}\n"
      "  device={{cudaDevice={} cuDev={}}}\n"
      "  allocation={{requestedSize={} allocatedSize={} handleType={}}}\n"
      "  resources={{multicastExportedFd={} exchanged={} failed={} "
      "deviceInitialized={} backing={} overlay={} multicastMapping={}}}\n"
      "  handles={{multicastPtr=0x{:x} localAllocHandle=0x{:x} "
      "multicastHandle=0x{:x}}}",
      failedPhase != nullptr ? failedPhase : "unknown",
      completedPhase != nullptr ? completedPhase : "unknown",
      commRank_,
      nvlRank_,
      nvlRanks_,
      fmt::join(nvlRankToCommRank_, ","),
      cudaDevice_,
      cuDev_,
      requestedSize_,
      allocatedSize_,
      handleTypeName(handleType_),
      multicastExportedFd_,
      exchanged_,
      failed_,
      deviceInitialized_,
      static_cast<bool>(backing_),
      static_cast<bool>(overlay_),
      static_cast<bool>(multicastMapping_),
      multicastPtr,
      localAllocHandle,
      multicastHandle);
}

void MultimemHandler::cleanup() {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  if (multicastExportedFd_ >= 0) {
    close(multicastExportedFd_);
    multicastExportedFd_ = -1;
  }

  // Tear down the multicast VA mapping before dropping the multicast object
  // (whose destructor unbinds + releases), then drop the shared backing. The
  // VA mapping co-owns overlay_, so even out-of-order resets keep the
  // unmap-before-unbind ordering; reset explicitly for clarity.
  multicastMapping_.reset();
  overlay_.reset();
  backing_.reset();
#endif
}

} // namespace comms::prims
