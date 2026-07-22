// Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and
// proprietary.

#pragma once

#include <cstddef>
#include <vector>

// CudaWrap.h pulls in <cuda.h> (or the HIP shims on AMD) so the CU* handle
// types below resolve on both platforms, and provides the FB_CUPFN wrappers.
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevMemType.h" // CuMemSegment, enumerateCuMemSegments
#include "comms/utils/commSpecs.h"

namespace ctran::utils {

// A standalone NVSwitch multicast object bound over a local buffer's physical
// backing segments, plus its mapped multicast VA. Self-contained: import()
// enumerates the buffer's VMM segments and RETAINS ITS OWN allocation handles
// (via enumerateCuMemSegments), so this object neither depends on nor borrows
// from CtranIpc -- it owns everything it binds and releases it all at teardown,
// in any order relative to other owners of the same backing allocation.
//
// This class encapsulates ONLY the LOCAL (per-rank) CUDA operations + RAII
// teardown; the collective exchange of the multicast object handle across the
// NVL team is driven by the caller (e.g. CtranMapper::setupMulticastOverlay),
// which uses the CtranIpc export/importShareableHandle helpers to move the
// handle this class produces (createRoot) / consumes (adoptImported).
//
// NOTE: not internally locked; the caller serializes access.
class CtranMulticast {
 public:
  CtranMulticast(int nvlLocalRank, int nLocalRanks, int cudaDev);
  ~CtranMulticast();

  CtranMulticast(const CtranMulticast&) = delete;
  CtranMulticast& operator=(const CtranMulticast&) = delete;
  CtranMulticast(CtranMulticast&&) = delete;
  CtranMulticast& operator=(CtranMulticast&&) = delete;

  // Runtime support gate (no arch/compute-capability conditional): the
  // cuMulticast* driver entry points are present AND the device reports
  // CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED. Mirrors PyTorch
  // deviceSupportsMulticast and prims isMultimemSupported.
  static bool isSupported(int cudaDev);

  // Multicast-object allocation granularity for the device+team; mcSize and
  // every segment size/offset must be a multiple of it.
  static commResult_t granularity(int cudaDev, int nLocalRanks, size_t& gran);

  // NVL-team root only: create the multicast object (mcSize bytes, agreed
  // handle type) and return the raw object handle for the caller to
  // export+broadcast.
  commResult_t createRoot(
      size_t mcSize,
      CUmemAllocationHandleType handleType,
      CUmemGenericAllocationHandle& outHandle);

  // Non-root: adopt the object handle the caller imported from the root.
  void adoptImported(CUmemGenericAllocationHandle handle);

  // Every rank: enumerate the physical VMM segments backing [dptr, dptr+len)
  // and RETAIN this object's own allocation handle for each (via
  // enumerateCuMemSegments; each released at teardown). Must be a cuMem/VMM
  // allocation. Call once before addDeviceAndBind(); importedSize() and
  // segmentsAlignedTo() are valid afterwards.
  commResult_t import(const void* dptr, size_t len);

  // Sum of the imported segments' sizes -- the multicast object size this rank
  // requires (== root's createRoot mcSize for a symmetric team). Valid after
  // import().
  size_t importedSize() const {
    return totalSize_;
  }

  // True iff every imported segment size is a multiple of gran (a bind
  // precondition). Valid after import().
  bool segmentsAlignedTo(size_t gran) const;

  // Every rank: add the local device to the object, then bind each imported
  // backing segment at its running multicast offset (address order). Requires a
  // prior import() and createRoot()/adoptImported().
  commResult_t addDeviceAndBind();

  // Every rank: reserve + map + grant device access to one contiguous
  // multicast VA spanning mcSize; getMulticastPtr() is valid afterwards.
  commResult_t mapVA(size_t mcSize, size_t gran);

  // The multicast device VA (nullptr until mapVA succeeds).
  void* getMulticastPtr() const {
    return reinterpret_cast<void*>(mcVA_);
  }
  size_t getMulticastSize() const {
    return mcSize_;
  }

 private:
  const int nvlLocalRank_{-1};
  const int nLocalRanks_{0};
  const int cudaDev_{-1};
  CUdevice cuDev_{0}; // driver device handle for this cudaDev_

  CUmemGenericAllocationHandle mcHandle_{0}; // the multicast object
  CUdeviceptr mcVA_{0}; // mapped multicast VA (0 until mapVA)
  size_t mcSize_{0};

  // Per-segment (offset, size) recorded at bind time so teardown can unbind
  // each segment. cuMulticastUnbind errors are IGNORED at teardown (the backing
  // buffer may already be freed by the user), so ordering is a belt not the
  // guarantee.
  struct BoundSeg {
    size_t offset{0};
    size_t size{0};
  };
  std::vector<BoundSeg> bound_;

  // Physical VMM segments backing the local buffer, in address order. Each
  // allocation handle is RETAINED by this object in import() and released
  // (cuMemRelease) at teardown -- making the object self-owning. totalSize_ =
  // sum of the segment sizes.
  std::vector<CuMemSegment> segments_;
  size_t totalSize_{0};
};

} // namespace ctran::utils
