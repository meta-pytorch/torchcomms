// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <sys/syscall.h>
#include <unistd.h>

#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevMemType.h"
#include "comms/utils/commSpecs.h"

namespace ctran::utils {

static std::atomic<size_t> activeIpcMemCount = 0;
static std::atomic<size_t> activeIpcRemMemCount = 0;

size_t getActiveIpcMemCount() {
  return activeIpcMemCount.load();
}

size_t getActiveIpcRemMemCount() {
  return activeIpcRemMemCount.load();
}

static inline commResult_t importIpcFd(int pid, int fd, int* newFd) {
#ifndef IS_CTRAN_IPC_SUPPORTED
  return commInvalidUsage;
#else
  int pidfd = 0;
  FB_SYSCHECKVAL(syscall(SYS_pidfd_open, pid, 0), "pidfd_open", pidfd);

  int newfd_ = 0;
  FB_SYSCHECKVAL(syscall(SYS_pidfd_getfd, pidfd, fd, 0), "pidfd_getfd", newfd_);
  *newFd = newfd_;
  return commSuccess;
#endif
}

ctran::utils::CtranIpcMem::CtranIpcMem(
    const size_t size,
    const int cudaDev,
    const struct CommLogData* logMetaData,
    const char* desc,
    const DevMemType memType,
    const CUmemAllocationHandleType cuMemHandleType)
    : cudaDev_(cudaDev),
      mode_(CtranIpcMem::Mode::ALLOC),
      logMetaData_(logMetaData),
      desc_(desc),
      memType_(memType),
      cuMemHandleType_(cuMemHandleType) {
  if (!CtranIpcSupport()) {
    FB_COMMCHECKTHROW_EX_NOCOMM(commInternalError);
  }
  FB_COMMCHECKTHROW_EX_NOCOMM(this->alloc(size));
};

ctran::utils::CtranIpcMem::CtranIpcMem(const int cudaDev, const char* desc)
    : cudaDev_(cudaDev), mode_(CtranIpcMem::Mode::LOAD), desc_(desc) {
  if (!CtranIpcSupport()) {
    FB_COMMCHECKTHROW_EX_NOCOMM(commInternalError);
  }
  // Empty instance for loading existing memory range via load()
};

ctran::utils::CtranIpcMem::~CtranIpcMem() {
  if (pbase_) {
    FB_COMMCHECKIGNORE(this->free());
  }

  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

static inline CUmemAllocationHandleType getCuMemExportHandleType(
    CUmemAllocationHandleType cuMemHandleType) {
  CUmemAllocationHandleType exportHandleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#if CUDART_VERSION >= 12040
  if (NCCL_CTRAN_NVL_FABRIC_ENABLE &&
      (cuMemHandleType & CU_MEM_HANDLE_TYPE_FABRIC)) {
    exportHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
  }
#endif
  return exportHandleType;
}

commResult_t ctran::utils::CtranIpcMem::ipcExport(CtranIpcDesc& ipcDesc) {
  if (allocHandles_.size() > CTRAN_IPC_INLINE_SEGMENTS) {
    CLOGF(
        ERR,
        "CTRAN-IPC: tried to export CtranIpcMem backed by too many physical memory allocations.");
    return commInternalError;
  }

  // Export handle
  CUmemAllocationHandleType exportHandleType =
      getCuMemExportHandleType(cuMemHandleType_);

  ipcDesc.memType = memType_;
  ipcDesc.cuMemHandleType = exportHandleType;

  for (int i = 0; i < allocHandles_.size(); i++) {
    // Reuse handle if already exported
    if (!sharedHandlesInitialized_[i]) {
      if (memType_ == DevMemType::kCumem) {
        bool isCumemFabric = false;
#if CUDART_VERSION >= 12040
        isCumemFabric = (exportHandleType == CU_MEM_HANDLE_TYPE_FABRIC);
#endif
        if (isCumemFabric) {
          FB_CUCHECK(cuMemExportToShareableHandle(
              &sharedHandles_[i].handle,
              allocHandles_[i],
              exportHandleType,
              0));
        } else {
          FB_CUCHECK(cuMemExportToShareableHandle(
              &sharedHandles_[i].fd,
              allocHandles_[i],
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
              0));
        }
      } else if (memType_ == DevMemType::kCudaMalloc) {
        void* p = (void*)pbase_;
        FB_CUDACHECK(cudaIpcGetMemHandle(&sharedHandles_[i].cudaIpcHandle, p));
      } else {
        FB_ERRORRETURN(
            commInternalError,
            "CTRAN-IPC: unsupported memory type {}",
            devMemTypeStr(ipcDesc.memType));
      }
    }

    sharedHandlesInitialized_[i] = true;
    ipcDesc.segments[i].sharedHandle = sharedHandles_[i];
    ipcDesc.segments[i].range = segmentRanges_[i];
  }

  ipcDesc.numSegments = allocHandles_.size();
  ipcDesc.pid = getpid();
  ipcDesc.base = this->getBase();
  ipcDesc.range = this->getRange();

  CLOGF_TRACE(
      ALLOC,
      "CTRAN-IPC: exported mem [{}] to ipcDesc: {}",
      this->toString(),
      ipcDesc.toString());

  return commSuccess;
}

commResult_t ctran::utils::CtranIpcMem::alloc(const size_t size) {
  void* p = nullptr;
  allocHandles_.emplace_back();
  sharedHandles_.emplace_back();
  sharedHandlesInitialized_.emplace_back(false);

  if (memType_ == DevMemType::kCumem) {
    FB_COMMCHECK(
        ctran::utils::commCuMemAlloc(
            &p,
            &allocHandles_[0],
            cuMemHandleType_,
            size,
            logMetaData_,
            desc_));
    FB_CUCHECK(cuMemGetAddressRange(&pbase_, &range_, (CUdeviceptr)p));
    segmentRanges_.emplace_back(range_);
  } else if (memType_ == DevMemType::kCudaMalloc) {
    FB_CUDACHECK(cudaMalloc(&p, size));
    pbase_ = (CUdeviceptr)p;
    range_ = size;
    segmentRanges_.emplace_back(range_);
  } else {
    FB_ERRORRETURN(
        commInternalError,
        "CTRAN-IPC: unsupported memory type {}",
        devMemTypeStr(memType_));
  }

  activeIpcMemCount++;
  CLOGF_TRACE(
      ALLOC,
      "CTRAN-IPC: allocated mem [{}], total IPC mem {}",
      this->toString(),
      activeIpcMemCount.load());

  return commSuccess;
}

commResult_t ctran::utils::CtranIpcMem::tryLoad(
    const void* ptr,
    std::size_t len,
    bool& supported,
    bool shouldSupportCudaMalloc) {
  // Should never call load from an instance with allocation mode
  if (this->mode_ != CtranIpcMem::Mode::LOAD) {
    CLOGF(
        ERR,
        "CTRAN-IPC: try to load a memory range to an instance with allocation mode. It indicates a COMM internal bug.");
    return commInternalError;
  }

  // A load instance should manage only one memory range at lifetime
  if (this->pbase_) {
    CLOGF(
        ERR,
        "CTRAN-IPC: CtranIpcMem already manages an existing memory range: {}. It indicates a COMM internal bug.",
        this->toString().c_str());
    return commInternalError;
  }

  supported = false;
  FB_COMMCHECK(getDevMemType(ptr, cudaDev_, memType_));
  if (memType_ == DevMemType::kCumem) {
    FB_COMMCHECK(this->tryLoadCuMem(ptr, len, supported));
  } else if (memType_ == DevMemType::kCudaMalloc) {
    if (!shouldSupportCudaMalloc) {
      supported = false;
      return commSuccess;
    }
    FB_COMMCHECK(this->tryLoadCudaMallocMem(ptr, len, supported));
  } else {
    CLOGF_TRACE(
        ALLOC,
        "CTRAN-IPC: failed to load device memory {}, len={}, memType={}",
        ptr,
        len,
        devMemTypeStr(memType_));
    return commSuccess;
  }

  activeIpcMemCount++;
  CLOGF_SUBSYS(
      DBG,
      ALLOC,
      "CTRAN-IPC: loaded mem [{}], total IPC mem {}",
      this->toString().c_str(),
      activeIpcMemCount.load());
  return commSuccess;
}

inline commResult_t ctran::utils::CtranIpcMem::tryLoadCuMem(
    const void* ptr,
    std::size_t len,
    bool& supported) {
  // Handle type decision in ctran:
  // - getCuMemAllocHandleType(): Queries system-supported handle types. Used
  // when
  //   allocating ctran internal buffers (ALLOC mode). Always includes POSIX.
  // - getCuMemExportHandleType(): Determines the export handle type based on
  // CVAR
  //   enable flag (NCCL_CTRAN_NVL_FABRIC_ENABLE) and the allocation's handle
  //   type. Used in LOAD mode to validate that user-provided memory can be
  //   exported.

  // temp linear loop through physical allocations, could be faster to get from
  // pytorch level
  size_t cur_offset = 0;
  while (cur_offset < len) {
    const void* cur_ptr = (char*)ptr + cur_offset;
    allocHandles_.emplace_back();
    sharedHandles_.emplace_back();
    sharedHandlesInitialized_.emplace_back(false);
    FB_CUCHECK(cuMemRetainAllocationHandle(
        &allocHandles_.back(), const_cast<void*>(cur_ptr)));

    size_t cur_range;
    CUdeviceptr cur_pbase;
    FB_CUCHECK(
        cuMemGetAddressRange(&cur_pbase, &cur_range, (CUdeviceptr)cur_ptr));

    CUmemAllocationProp prop;
    FB_CUCHECK(
        cuMemGetAllocationPropertiesFromHandle(&prop, allocHandles_.back()));

    // Set cuMemHandleType_ from first segment's allocation properties
    if (cuMemHandleType_ == CU_MEM_HANDLE_TYPE_NONE) {
      cuMemHandleType_ = ctran::utils::getCuMemHandleTypeFromProp(prop);
    }

    // Validate allocation is supported for IPC export. Reject if:
    // 1. NONE handle type (no shareable handle)
    // 2. ctran-incompatible type (e.g., FABRIC-only when FABRIC disabled)
    // 3. no gpuDirectRDMACapable
    // 4. mixed handle types across segments
    CUmemAllocationHandleType segmentHandleType =
        ctran::utils::getCuMemHandleTypeFromProp(prop);
    CUmemAllocationHandleType supportedExportType =
        getCuMemExportHandleType(cuMemHandleType_);
    bool isUnsupportedType = (cuMemHandleType_ == CU_MEM_HANDLE_TYPE_NONE) ||
        ((cuMemHandleType_ & supportedExportType) == 0) ||
        (prop.allocFlags.gpuDirectRDMACapable != 1) ||
        (segmentHandleType != cuMemHandleType_);

    if (isUnsupportedType) {
      CLOGF(
          ERR,
          "CTRAN-IPC: [pbase {:x} range {}] associated with [ptr {} len {}] "
          "has unsupported allocation properties for IPC export: "
          "handleType = {} ({}), supportedExportType = {} ({}), gpuDirectRDMACapable = {}, "
          "segmentHandleType = {} ({})",
          cur_pbase,
          cur_range,
          (void*)ptr,
          len,
          cuMemHandleTypeStr(cuMemHandleType_),
          static_cast<int>(cuMemHandleType_),
          cuMemHandleTypeStr(supportedExportType),
          static_cast<int>(supportedExportType),
          prop.allocFlags.gpuDirectRDMACapable,
          cuMemHandleTypeStr(segmentHandleType),
          static_cast<int>(segmentHandleType));
      return commInvalidUsage;
    }

    if (cur_offset == 0) {
      pbase_ = cur_pbase;
    }
    segmentRanges_.emplace_back(cur_range);
    range_ += cur_range;

    CUdeviceptr cur_end = ctran::utils::addDevicePtr(cur_pbase, cur_range);
    cur_offset = (size_t)ctran::utils::subDevicePtr(cur_end, ptr);
  }
  supported = true;
  return commSuccess;
}

inline commResult_t CtranIpcMem::tryLoadCudaMallocMem(
    const void* ptr,
    std::size_t len,
    bool& supported) {
  void* p = const_cast<void*>(ptr);
  allocHandles_.emplace_back();
  sharedHandles_.emplace_back();
  sharedHandlesInitialized_.emplace_back(false);
  FB_CUDACHECK(cudaIpcGetMemHandle(&sharedHandles_.back().cudaIpcHandle, p));
  pbase_ = (CUdeviceptr)p;
  range_ = len;
  segmentRanges_.emplace_back(range_);
  supported = true;
  return commSuccess;
}

commResult_t ctran::utils::CtranIpcMem::free() {
  // In case context has already been destroyed by PyTorch, all resources should
  // already be released. Thus, safe to skip here.
  CUcontext pctx;
  FB_CUCHECK(cuCtxGetCurrent(&pctx));
  if (pctx == nullptr) {
    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "CTRAN-IPC: cuda context has already been destroyed. Skip free");
    return commSuccess;
  }

  if (memType_ == DevMemType::kCumem) {
    FB_COMMCHECK(this->freeCuMem());
  } else if (memType_ == DevMemType::kCudaMalloc) {
    FB_COMMCHECK(this->freeCudaMallocMem());
  } else {
    FB_ERRORRETURN(
        commInternalError,
        "CTRAN-IPC: unsupported memory type {}",
        devMemTypeStr(memType_));
  }

  activeIpcMemCount--;
  CLOGF_TRACE(
      ALLOC,
      "CTRAN-IPC: freed mem [{}], total IPC mem {}",
      this->toString(),
      activeIpcMemCount.load());

  // Mark as freed to avoid double free in destructor
  pbase_ = 0;
  return commSuccess;
}

inline commResult_t CtranIpcMem::freeCuMem() {
  for (int i = 0; i < allocHandles_.size(); i++) {
    if (mode_ == CtranIpcMem::Mode::LOAD) {
      // An explicit release is required in case of CtranIpcMem::Mode::LOAD
      // since tryLoad increments reference count to the handle via
      // cuMemRetainAllocationHandle.
      FB_CUCHECK(cuMemRelease(allocHandles_[i]));
    }
    if (memType_ == DevMemType::kCumem &&
        cuMemHandleType_ == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
        sharedHandlesInitialized_[i]) {
      FB_SYSCHECK(close(sharedHandles_[i].fd), "close sharedHandle");
    }
  }

  if (mode_ == CtranIpcMem::Mode::ALLOC) {
    FB_COMMCHECK(
        ctran::utils::commCuMemFree(
            reinterpret_cast<void*>(pbase_), logMetaData_));
  }

  return commSuccess;
}

inline commResult_t CtranIpcMem::freeCudaMallocMem() {
  if (mode_ == CtranIpcMem::Mode::ALLOC) {
    void* p = (void*)pbase_;
    if (p != nullptr) {
      FB_CUDACHECK(cudaFree(p));
    }
  }

  return commSuccess;
}

ctran::utils::CtranIpcRemMem::CtranIpcRemMem(
    const CtranIpcDesc& ipcDesc,
    const int cudaDev,
    const struct CommLogData* logMetaData,
    const char* desc)
    : cudaDev_(cudaDev),
      logMetaData_(logMetaData),
      desc_(desc),
      memType_(ipcDesc.memType),
      cuMemHandleType_(ipcDesc.cuMemHandleType) {
  if (!CtranIpcSupport()) {
    FB_COMMCHECKTHROW_EX_NOCOMM(commInternalError);
  }
  FB_COMMCHECKTHROW_EX_NOCOMM(this->import(ipcDesc));
};

ctran::utils::CtranIpcRemMem::~CtranIpcRemMem() {
  if (pbase_) {
    FB_COMMCHECKIGNORE(release());
  }
  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

commResult_t ctran::utils::CtranIpcRemMem::import(const CtranIpcDesc& ipcDesc) {
  range_ = ipcDesc.range;
  remPid_ = ipcDesc.pid;

  remHandles_.resize(ipcDesc.numSegments);
  segmentRanges_.resize(ipcDesc.numSegments);
  allocHandles_.resize(ipcDesc.numSegments);

  for (int i = 0; i < ipcDesc.numSegments; i++) {
    remHandles_[i] = ipcDesc.segments[i].sharedHandle;
    segmentRanges_[i] = ipcDesc.segments[i].range;
  }

  if (ipcDesc.memType == DevMemType::kCumem) {
    FB_COMMCHECK(importCuMem(ipcDesc));
  } else if (ipcDesc.memType == DevMemType::kCudaMalloc) {
    FB_COMMCHECK(importCudaMallocMem(ipcDesc));
  } else {
    FB_ERRORRETURN(
        commInternalError,
        "CTRAN-IPC: unsupported memory type {}",
        devMemTypeStr(memType_));
  }

  activeIpcRemMemCount++;
  CLOGF_SUBSYS(
      DBG,
      ALLOC,
      "CTRAN-IPC: imported remote mem [{}], total IPC remMem {}",
      this->toString().c_str(),
      activeIpcRemMemCount.load());

  return commSuccess;
}

commResult_t CtranIpcRemMem::importCuMem(const CtranIpcDesc& ipcDesc) {
  size_t offset = 0;
  FB_CUCHECK(cuMemAddressReserve(
      &pbase_, range_, /* alignment */ 0, /* addr */ 0, /* flags */ 0));

  for (int i = 0; i < remHandles_.size(); i++) {
    CtranIpcHandle importedHandle = remHandles_[i];
    if (cuMemHandleType_ == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      FB_COMMCHECK(importIpcFd(
          ipcDesc.pid,
          remHandles_[i].fd,
          reinterpret_cast<int*>(&importedHandle.fd)));
      FB_CUCHECK(cuMemImportFromShareableHandle(
          &allocHandles_[i],
          reinterpret_cast<void*>(importedHandle.fd),
          cuMemHandleType_));
#if CUDART_VERSION >= 12040
    } else if (cuMemHandleType_ == CU_MEM_HANDLE_TYPE_FABRIC) {
      FB_CUCHECK(cuMemImportFromShareableHandle(
          &allocHandles_[i], (void*)&importedHandle.handle, cuMemHandleType_));
#endif
    } else {
      FB_ERRORRETURN(
          commInternalError,
          "CTRAN-IPC: unsupported import handle type {}",
          cuMemHandleType_);
    }

    // map to a local addr range
    FB_CUCHECK(cuMemMap(
        ctran::utils::addDevicePtr(pbase_, offset),
        segmentRanges_[i],
        /* offset */ 0,
        allocHandles_[i],
        /* flags */ 0));
    offset += segmentRanges_[i];

    if (cuMemHandleType_ == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // Close the imported fd after mapping
      FB_SYSCHECK(close(importedHandle.fd), "close importedHandle");
    }
  }

  // Allow access by the local GPU
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev_;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  FB_CUCHECK(cuMemSetAccess(pbase_, range_, &accessDesc, 1));

  return commSuccess;
}

commResult_t CtranIpcRemMem::importCudaMallocMem(const CtranIpcDesc& ipcDesc) {
  if (ipcDesc.numSegments != 1) {
    CLOGF(
        ERR,
        "CTRAN-IPC: Number of segments is expected to be 1, but got {}",
        ipcDesc.numSegments);
    return commInternalError;
  }

  void* p = nullptr;
  FB_CUDACHECK(cudaIpcOpenMemHandle(
      &p,
      ipcDesc.segments[0].sharedHandle.cudaIpcHandle,
      cudaIpcMemLazyEnablePeerAccess));

  pbase_ = (CUdeviceptr)p;
  return commSuccess;
}

commResult_t ctran::utils::CtranIpcRemMem::release() {
  // In case context has already been destroyed by PyTorch, all resources should
  // already be released. Thus, safe to skip here.
  CUcontext pctx;
  FB_CUCHECK(cuCtxGetCurrent(&pctx));
  if (pctx == nullptr) {
    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "CTRAN-IPC: cuda context has already been destroyed. Skip release");
    return commSuccess;
  }

  if (memType_ == DevMemType::kCumem) {
    // Release handle and memory
    FB_CUCHECK(cuMemUnmap(pbase_, range_));
    for (CUmemGenericAllocationHandle allocHandle : allocHandles_) {
      FB_CUCHECK(cuMemRelease(allocHandle));
    }
    FB_CUCHECK(cuMemAddressFree(pbase_, range_));
  } else if (memType_ == DevMemType::kCudaMalloc) {
    FB_CUDACHECK(cudaIpcCloseMemHandle((void*)(pbase_)));
  } else {
    FB_ERRORRETURN(
        commInternalError,
        "CTRAN-IPC: unsupported memory type {}",
        devMemTypeStr(memType_));
  }

  activeIpcRemMemCount--;
  CLOGF_TRACE(
      ALLOC,
      "CTRAN-IPC: free remote cumem range {}, {}, total IPC remMem {}",
      this->getBase(),
      range_,
      activeIpcRemMemCount.load());

  // Mark as released to avoid double free in destructor
  pbase_ = 0;
  return commSuccess;
}
} // namespace ctran::utils
