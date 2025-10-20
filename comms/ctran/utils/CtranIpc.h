// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IPC_H_
#define CTRAN_IPC_H_

#include <sstream>
#include <vector>

#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevMemType.h"
#include "comms/utils/commSpecs.h"

// TODO: remove this once we have a more portable way for CTRAN IPC
#if (defined(SYS_pidfd_open) && defined(SYS_pidfd_getfd))
#define IS_CTRAN_IPC_SUPPORTED
#endif

struct CommLogData;

namespace ctran::utils {

static inline bool CtranIpcSupport() {
#ifndef IS_CTRAN_IPC_SUPPORTED
  return false;
#endif
#if defined(__HIP_PLATFORM_AMD__)
  // TODO: Revisit here after deciding cuMemSys support of CTRAN for HIP.
  return true;
#else
#if CUDART_VERSION < 11030
  CLOGF(
      WARN, "CTRAN-IPC: CTran IPC memory support requires CUDA 11.3 or later");
  return false;
#endif
  return ctran::utils::getCuMemSysSupported();
#endif
}

typedef union {
  uint64_t fd; // hold a CUmemGenericAllocationHandle for UDS fd support
#if CUDART_VERSION >= 12040
  CUmemFabricHandle handle;
#else
  // Dummy handle to make sure the size is the same as CUmemFabricHandle.
  struct {
    unsigned char data[CU_IPC_HANDLE_SIZE];
  } handle;
#endif
  cudaIpcMemHandle_t cudaIpcHandle;
} CtranIpcHandle;

struct CtranIpcSegDesc {
  CtranIpcHandle sharedHandle{0};
  size_t range{0};
};

static inline void printHandle(
    std::stringstream& ss,
    const DevMemType memType,
    const CUmemAllocationHandleType& cuMemHandleType,
    const CtranIpcHandle& handle) {
  bool isCumemFabric = false;
#if CUDART_VERSION >= 12040
  isCumemFabric = cuMemHandleType & CU_MEM_HANDLE_TYPE_FABRIC;
#endif
  if (memType == DevMemType::kCumem) {
    if (isCumemFabric) {
      ss << "fabric handle: ";
      for (int j = 0; j < CU_IPC_HANDLE_SIZE; ++j) {
        ss << std::hex << static_cast<int>(handle.handle.data[j]);
      }
    } else {
      ss << "posix fd handle 0x" << std::hex << handle.fd;
    }
  } else if (memType == DevMemType::kCudaMalloc) {
    ss << "ipc handle: ";
    for (int j = 0; j < CU_IPC_HANDLE_SIZE; ++j) {
      ss << "0x" << std::hex << static_cast<int>(handle.handle.data[j]);
    }
  } else {
    ss << "unsupported device memory type " << devMemTypeStr(memType);
  }
}

// maxNumHandles is needed so that CtranIpcDesc objects are constant
// size for communication. It is set to 2 to reduce ControlMsg size, since we
// temporarily disable disjoint segments support for NVL peers.
// FIXME: this is a temporary hack; we should support arbitrary size via dynamic
// allocated CtranIpcSegDesc list.
#define CTRAN_IPC_INLINE_SEGMENTS 2

struct CtranIpcDesc {
  int numSegments{0};
  void* base{nullptr};
  size_t range{0};
  // pass local pid for peer to import sharedHandle as file descriptor
  int pid{0};
  DevMemType memType{DevMemType::kHostUnregistered};
  CUmemAllocationHandleType cuMemHandleType{CU_MEM_HANDLE_TYPE_NONE};

  // Preallocated payload for multi-segment IPC memory.
  CtranIpcSegDesc segments[CTRAN_IPC_INLINE_SEGMENTS] = {};

  std::string toString() const {
    std::stringstream ss;
    ss << "base: " << base << ", range: " << std::dec << range
       << ", pid: " << pid;
    ss << ", segments: [";
    for (int i = 0; i < CTRAN_IPC_INLINE_SEGMENTS; ++i) {
      if (i != 0) {
        ss << ", ";
      }
      printHandle(ss, memType, cuMemHandleType, segments[i].sharedHandle);
      ss << " range: " << std::dec << segments[i].range;
    }
    ss << "]";
    return ss.str();
  }
};

class CtranIpcMem {
  enum Mode {
    ALLOC,
    LOAD,
  };

 public:
  // Allocation mode constructor to internally allocate IPC-exportable memory
  // Input arguments:
  //   - size: the size of the memory range to be allocated
  //   - cudaDev: the CUDA device to allocate the memory range to
  //   - logMetaData: logMetaData of the communicator for memory logging
  //                  purposes
  //   - desc: description of the usage of the memory range to be allocated
  //   - memType: the device memory type. Support kCumem and kCudaMalloc.
  //   - cuMemHandleType: the type of the handle to be used for importing and
  //   exporting the memory. Only used when memType is kCumem.
  CtranIpcMem(
      const size_t size,
      const int cudaDev,
      const struct CommLogData* logMetaData,
      const char* desc,
      const DevMemType memType = DevMemType::kCumem,
      const CUmemAllocationHandleType cuMemHandleType =
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);

  // Load mode constructor to manage an existing memory range loaded via load()
  // Input arguments:
  //   - cudaDev: the CUDA device of the memory range to be loaded
  //   - desc: description of the usage of the memory range to be loaded
  CtranIpcMem(const int cudaDev, const char* desc);

  // Release the memory range and associated handles if not yet freed explicitly
  ~CtranIpcMem();

  // Try to load an existing memory range into this IPC-exportable memory.
  // Loading may fail if the memory type is not supported
  // Input arguments:
  //   - ptr: the address of the memory range to be loaded
  //   - len: the size of the memory range to be loaded
  //   - shouldSupoortCudaMalloc: whether to support loading cudaMalloc-ed
  //   memory
  // Output arguments:
  //   - supported: whether the memory range is supported
  commResult_t tryLoad(
      const void* ptr,
      std::size_t len,
      bool& supported,
      // TODO: AMD only supports cudaMalloc buffer. We need to make it
      // work on AMD
      bool shouldSupportCudaMalloc = false);

  // Export the memory range to a CtranIpcDesc for remote process importing
  // Input arguments:
  //   - ipcDesc: the information to be filled with the memory range exporting.
  //              Caller is responsible for sending the descriptor to the remote
  //              process.
  commResult_t ipcExport(CtranIpcDesc& ipcDesc);

  // Free the memory range and associated handles.
  // Note: recommending call free explicitly to check potential cuda failure.
  // Otherwise the destructor will still free it but any failure will be
  // ignored.
  commResult_t free();

  // Return the base address of the memory range
  inline void* getBase() {
    return reinterpret_cast<void*>(pbase_);
  }

  // Return the range of the memory range (it may be greater or equal to the
  // user specified size)
  inline size_t getRange() {
    return range_;
  }

  inline const char* getDesc() {
    return desc_;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "pbase: 0x" << std::hex << pbase_ << std::dec << ", range: " << range_
       << ", cudaDev: " << cudaDev_
       << ", mode: " << (mode_ == ALLOC ? "ALLOC" : "LOAD")
       << ", desc_: " << desc_;
    ss << ", allocHandles: [";
    for (size_t i = 0; i < allocHandles_.size(); ++i) {
      if (i != 0) {
        ss << ", ";
      }
      ss << "0x" << std::hex << allocHandles_[i];
    }
    ss << "]";
    ss << ", sharedHandles: [";
    for (size_t i = 0; i < sharedHandles_.size(); ++i) {
      if (i != 0) {
        ss << ", ";
      }
      printHandle(ss, memType_, cuMemHandleType_, sharedHandles_[i]);
    }
    ss << "]";
    return ss.str();
  }

 private:
  commResult_t alloc(const size_t size);

  inline commResult_t freeCuMem();
  inline commResult_t freeCudaMallocMem();

  inline commResult_t
  tryLoadCuMem(const void* ptr, std::size_t len, bool& supported);
  inline commResult_t
  tryLoadCudaMallocMem(const void* ptr, std::size_t len, bool& supported);

  const int cudaDev_{-1};
  size_t range_{0};
  std::vector<size_t> segmentRanges_;
  CUdeviceptr pbase_{0};
  std::vector<CUmemGenericAllocationHandle> allocHandles_;
  std::vector<CtranIpcHandle> sharedHandles_;
  std::vector<bool> sharedHandlesInitialized_;
  const Mode mode_; // Decided based on which constructor is used

  // Description fields
  const struct CommLogData* logMetaData_{nullptr};
  const char* desc_{nullptr};
  // Type of the memory  to be used for importing and exporting.
  DevMemType memType_{DevMemType::kCudaMalloc};
  CUmemAllocationHandleType cuMemHandleType_{CU_MEM_HANDLE_TYPE_NONE};
};

class CtranIpcRemMem {
 public:
  // Importing constructor to import a remote memory range described by ipcDesc
  // Input arguments:
  //   - ipcDesc: the information describing the remote memory range to be
  //              imported. It is expected to be sent from the remote process
  //              that owns the memory range.
  //   - cudaDev: the CUDA device to import the memory range to
  //   - logMetaData: logMetaData of the communicator for memory logging
  //                  purposes
  //   - desc: description of the usage of the memory range to be imported
  CtranIpcRemMem(
      const CtranIpcDesc& ipcDesc,
      const int cudaDev,
      const struct CommLogData* logMetaData,
      const char* desc);

  // Release the memory range and associated handles
  ~CtranIpcRemMem();

  // Release the remote memory range and associated handles.
  // Note: recommending call release explicitly to check potential cuda failure.
  // Otherwise the destructor will still free it but any failure will be
  // ignored.
  commResult_t release();

  // Return the base address of the memory range
  inline void* getBase() {
    return reinterpret_cast<void*>(pbase_);
  }

  // Return the range of the memory range (it may be greater or equal to the
  // user specified size)
  inline size_t getRange() {
    return range_;
  }

  inline const char* getDesc() {
    return desc_;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "pbase: 0x" << std::hex << pbase_ << std::dec << ", range: " << range_
       << ", cudaDev: " << cudaDev_ << ", remPid: " << remPid_
       << ", desc_: " << desc_;
    ss << ", allocHandles: [";
    for (size_t i = 0; i < allocHandles_.size(); ++i) {
      if (i != 0) {
        ss << ", ";
      }
      ss << "0x" << std::hex << allocHandles_[i];
    }
    ss << "]";
    ss << ", remHandles: [";
    for (size_t i = 0; i < remHandles_.size(); ++i) {
      if (i != 0) {
        ss << ", ";
      }
      printHandle(ss, memType_, cuMemHandleType_, remHandles_[i]);
    }
    ss << "]";
    return ss.str();
  }

 private:
  commResult_t import(const CtranIpcDesc& ipcDesc);
  commResult_t importCuMem(const CtranIpcDesc& ipcDesc);
  commResult_t importCudaMallocMem(const CtranIpcDesc& ipcDesc);

  const int cudaDev_;
  size_t range_{0};
  std::vector<size_t> segmentRanges_;
  CUdeviceptr pbase_{0};
  std::vector<CtranIpcHandle> remHandles_;
  int remPid_{0};
  std::vector<CUmemGenericAllocationHandle> allocHandles_;

  // Description fields
  const struct CommLogData* logMetaData_{nullptr};
  const char* desc_{nullptr};
  const DevMemType memType_{DevMemType::kHostUnregistered};
  const CUmemAllocationHandleType cuMemHandleType_{CU_MEM_HANDLE_TYPE_NONE};
};

// Return the number of active IPC memory objects and IPC remote memory objects.
// Used to check resource leak in UT.
size_t getActiveIpcMemCount();
size_t getActiveIpcRemMemCount();

} // namespace ctran::utils

#endif
