// Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and
// proprietary.

#include "comms/ctran/utils/CtranMulticast.h"

#include "comms/ctran/utils/Checks.h" // FB_COMMCHECK
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::utils {

#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12010

CtranMulticast::CtranMulticast(int nvlLocalRank, int nLocalRanks, int cudaDev)
    : nvlLocalRank_(nvlLocalRank),
      nLocalRanks_(nLocalRanks),
      cudaDev_(cudaDev) {}

CtranMulticast::~CtranMulticast() {
  // Safe teardown: cuMulticastUnbind can return an RM error if the backing
  // buffer was already freed by the user (see NCCL's "safe to ignore" note);
  // ignore all teardown errors so a partially-created or user-freed overlay
  // never aborts. Order (unmap -> unbind -> release) is a belt; correctness
  // rests on ignoring the unbind error.
  if (mcVA_ != 0) {
    FB_CUCHECKIGNORE(cuMemUnmap(mcVA_, mcSize_));
    FB_CUCHECKIGNORE(cuMemAddressFree(mcVA_, mcSize_));
  }
  for (const auto& b : bound_) {
    FB_CUCHECKIGNORE(cuMulticastUnbind(mcHandle_, cuDev_, b.offset, b.size));
  }
  if (mcHandle_ != 0) {
    FB_CUCHECKIGNORE(cuMemRelease(mcHandle_));
  }
  // Release our own retained segment handles (from import()). Because these are
  // this object's own references, teardown order relative to any other owner of
  // the same backing allocation does not matter.
  for (const auto& seg : segments_) {
    FB_CUCHECKIGNORE(cuMemRelease(seg.handle));
  }
}

bool CtranMulticast::isSupported(int cudaDev) {
  // Memoized: driver PFN presence + the device multicast attribute are
  // invariant for a rank's device, so query once and log the unsupported
  // reason at most once (one device per rank in practice, so a function-local
  // static is effectively once-per-device-per-rank).
  static const bool supported = [cudaDev]() -> bool {
    // Driver-version gate: the multicast entry points are loaded (non-null).
    if (FB_CUPFN(cuMulticastCreate) == nullptr ||
        FB_CUPFN(cuMulticastAddDevice) == nullptr ||
        FB_CUPFN(cuMulticastBindMem) == nullptr ||
        FB_CUPFN(cuMulticastGetGranularity) == nullptr ||
        FB_CUPFN(cuMulticastUnbind) == nullptr) {
      CLOGF(
          WARN,
          "CTRAN-MC: multicast disabled -- driver multicast entry points unavailable (needs a newer CUDA driver)");
      return false;
    }
    // HW gate: the device reports multicast support. No arch conditional.
    CUdevice cuDev = 0;
    FB_CUCHECK_RETURN(cuDeviceGet(&cuDev, cudaDev), false);
    int sup = 0;
    FB_CUCHECK_RETURN(
        cuDeviceGetAttribute(
            &sup, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cuDev),
        false);
    if (sup == 0) {
      CLOGF(
          WARN,
          "CTRAN-MC: multicast disabled -- device {} reports no multicast support",
          cudaDev);
    }
    return sup != 0;
  }();
  return supported;
}

commResult_t
CtranMulticast::granularity(int cudaDev, int nLocalRanks, size_t& gran) {
  (void)cudaDev;
  CUmulticastObjectProp prop = {};
  prop.numDevices = static_cast<unsigned int>(nLocalRanks);
  prop.size = 0;
  prop.handleTypes = 0;
  prop.flags = 0;
  // MINIMUM is the actual bind/map alignment requirement. (RECOMMENDED is a
  // padding hint for allocation sizing -- e.g. ncclMemAlloc under
  // NCCL_MEM_ENABLE_MC_ALIGNMENT -- and is irrelevant here: binding/mapping an
  // already-allocated buffer succeeds as long as its segments meet MINIMUM.)
  FB_CUCHECK(cuMulticastGetGranularity(
      &gran, &prop, CU_MULTICAST_GRANULARITY_MINIMUM));
  return commSuccess;
}

commResult_t CtranMulticast::createRoot(
    size_t mcSize,
    CUmemAllocationHandleType handleType,
    CUmemGenericAllocationHandle& outHandle) {
  CUmulticastObjectProp prop = {};
  prop.numDevices = static_cast<unsigned int>(nLocalRanks_);
  prop.size = mcSize;
  prop.handleTypes = handleType;
  prop.flags = 0;
  FB_CUCHECK(cuMulticastCreate(&mcHandle_, &prop));
  mcSize_ = mcSize;
  outHandle = mcHandle_;
  return commSuccess;
}

void CtranMulticast::adoptImported(CUmemGenericAllocationHandle handle) {
  // Single-use: an instance is either the root (createRoot) or a non-root
  // (adoptImported), each exactly once. Release any handle we already hold
  // before overwriting so a misuse (double-adopt, or adopt after createRoot)
  // cannot silently drop -- and leak -- the previously-held multicast object.
  if (mcHandle_ != 0) {
    FB_CUCHECKIGNORE(cuMemRelease(mcHandle_));
  }
  mcHandle_ = handle;
}

commResult_t CtranMulticast::import(const void* dptr, size_t len) {
  // Single-use: one instance imports exactly one buffer's segments (like
  // createRoot/adoptImported). Re-importing would silently drop the prior
  // import's segments, so reject it as an explicit misuse.
  if (!segments_.empty()) {
    return commInvalidUsage;
  }
  // enumerateCuMemSegments retains one allocation handle per segment; segments_
  // now owns them and the dtor releases each (even on a partial failure here,
  // since any handles retained before the error are left in segments_).
  FB_COMMCHECK(enumerateCuMemSegments(dptr, len, segments_));
  if (segments_.empty()) {
    // Zero-length or non-cuMem-backed buffer: nothing to bind into a multicast
    // object.
    return commInvalidUsage;
  }
  totalSize_ = 0;
  for (const auto& seg : segments_) {
    totalSize_ += seg.size;
  }
  return commSuccess;
}

bool CtranMulticast::segmentsAlignedTo(size_t gran) const {
  if (gran == 0) {
    return false;
  }
  for (const auto& seg : segments_) {
    if ((seg.size % gran) != 0) {
      return false;
    }
  }
  return true;
}

commResult_t CtranMulticast::addDeviceAndBind() {
  FB_CUCHECK(cuDeviceGet(&cuDev_, cudaDev_));
  FB_CUCHECK(cuMulticastAddDevice(mcHandle_, cuDev_));
  size_t offset = 0;
  for (const auto& seg : segments_) {
    FB_CUCHECK(cuMulticastBindMem(
        mcHandle_, offset, seg.handle, /*memOffset=*/0, seg.size, /*flags=*/0));
    bound_.push_back({offset, seg.size});
    offset += seg.size;
  }
  // Non-root ranks learn the object size from their local segments.
  if (mcSize_ == 0) {
    mcSize_ = offset;
  }
  return commSuccess;
}

commResult_t CtranMulticast::mapVA(size_t mcSize, size_t gran) {
  FB_CUCHECK(cuMemAddressReserve(
      &mcVA_, mcSize, /*alignment=*/gran, /*addr=*/0, /*flags=*/0));
  FB_CUCHECK(cuMemMap(mcVA_, mcSize, /*offset=*/0, mcHandle_, /*flags=*/0));
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev_;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  FB_CUCHECK(cuMemSetAccess(mcVA_, mcSize, &accessDesc, /*count=*/1));
  mcSize_ = mcSize;
  return commSuccess;
}

#else // multicast unsupported at compile time (AMD or CUDA < 12.1)

CtranMulticast::CtranMulticast(int nvlLocalRank, int nLocalRanks, int cudaDev)
    : nvlLocalRank_(nvlLocalRank),
      nLocalRanks_(nLocalRanks),
      cudaDev_(cudaDev) {}
CtranMulticast::~CtranMulticast() = default;
bool CtranMulticast::isSupported(int /*cudaDev*/) {
  return false;
}
commResult_t CtranMulticast::granularity(
    int /*cudaDev*/,
    int /*nLocalRanks*/,
    size_t& gran) {
  gran = 0;
  return commInternalError;
}
commResult_t CtranMulticast::createRoot(
    size_t /*mcSize*/,
    CUmemAllocationHandleType /*handleType*/,
    CUmemGenericAllocationHandle& /*outHandle*/) {
  return commInternalError;
}
void CtranMulticast::adoptImported(CUmemGenericAllocationHandle /*handle*/) {}
commResult_t CtranMulticast::import(const void* /*dptr*/, size_t /*len*/) {
  return commInternalError;
}
bool CtranMulticast::segmentsAlignedTo(size_t /*gran*/) const {
  return false;
}
commResult_t CtranMulticast::addDeviceAndBind() {
  return commInternalError;
}
commResult_t CtranMulticast::mapVA(size_t /*mcSize*/, size_t /*gran*/) {
  return commInternalError;
}

#endif

} // namespace ctran::utils
