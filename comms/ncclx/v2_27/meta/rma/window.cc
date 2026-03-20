// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>

#include <folly/ScopeGuard.h>

#include "checks.h"
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/CtranWin.h"
#include "meta/wrapper/MetaFactory.h"

#include "nccl.h"
#include "ncclWin.h"

ncclResult_t CheckCommAndReturn(ncclComm_t comm) {
  if (!ncclGetCuMemSysSupported()) {
    FB_ERRORRETURN(ncclInternalError, "ncclWin requires CUMEM support.");
  }

  if (!ctranInitialized(comm->ctranComm_.get())) {
    FB_ERRORRETURN(ncclInternalError, "ncclWin requires Ctran support.");
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(
        ncclInternalError, "Communicator does not have statex initialized.");
  }
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclWinAllocate,
    size_t size,
    ncclComm_t comm,
    void** baseptr,
    ncclWindow_t* win,
    const ncclx::Hints& hints);
ncclResult_t ncclWinAllocate(
    size_t size,
    ncclComm_t comm,
    void** baseptr,
    ncclWindow_t* win,
    const ncclx::Hints& hints) {
  NCCLCHECK(CheckCommAndReturn(comm));
  ncclWin* win_ = new ncclWin();
  win_->comm = comm;

  auto guard = folly::makeGuard([win_] { delete win_; });
  NCCLCHECK(metaCommToNccl(
      ctran::ctranWinAllocate(
          size,
          comm->ctranComm_.get(),
          baseptr,
          &win_->ctranWindow,
          ncclToMetaComm(hints))));

  // Create empty ncclWindow as handle and register mapping
  ncclWindow_t handle = new ncclWindow();
  ncclWinMap().insert(handle, win_);
  *win = handle;
  guard.dismiss();
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclWinRegister,
    const void* baseptr,
    const size_t size,
    ncclComm_t comm,
    ncclWindow_t* win,
    const ncclx::Hints& hints);
ncclResult_t ncclWinRegister(
    const void* baseptr,
    const size_t size,
    ncclComm_t comm,
    ncclWindow_t* win,
    const ncclx::Hints& hints) {
  NCCLCHECK(CheckCommAndReturn(comm));
  if (baseptr == nullptr) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid baseptr to create shared buffer in ncclWinRegister.");
  }

  ncclWin* win_ = new ncclWin();
  win_->comm = comm;

  auto guard = folly::makeGuard([win_] { delete win_; });
  NCCLCHECK(metaCommToNccl(
      ctran::ctranWinRegister(
          baseptr,
          size,
          comm->ctranComm_.get(),
          &win_->ctranWindow,
          ncclToMetaComm(hints))));

  // Create empty ncclWindow as handle and register mapping
  ncclWindow_t handle = new ncclWindow();
  ncclWinMap().insert(handle, win_);
  *win = handle;
  guard.dismiss();
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclWinSharedQuery,
    int rank,
    ncclComm_t comm,
    ncclWindow_t win,
    void** addr);
ncclResult_t
ncclWinSharedQuery(int rank, ncclComm_t comm, ncclWindow_t win, void** addr) {
  ncclWin* w = ncclWinMap().find(win);
  if (!comm || !win || !w || comm != w->comm) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid parameter(s) to query shared buffer in ncclWinSharedQuery: comm {}, win {}",
        (void*)comm,
        (void*)win);
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(ncclInternalError, "Empty communicator statex.");
  }

  NCCLCHECK(
      metaCommToNccl(ctran::ctranWinSharedQuery(rank, w->ctranWindow, addr)));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclWinFree, ncclComm_t comm, ncclWindow_t win);
ncclResult_t ncclWinFree(ncclComm_t comm, ncclWindow_t win) {
  ncclWin* ncclWinPtr = ncclWinMap().find(win);
  if (!comm || !win || !ncclWinPtr || comm != ncclWinPtr->comm) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid parameter(s) to free window: comm {}, win {}",
        (void*)comm,
        (void*)win);
  }

  auto statex = comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(ncclInternalError, "Empty communicator statex.");
  }

  // Remove from map first, then cleanup resources
  ncclWinMap().erase(win);

  // Guard ensures cleanup happens on both success and failure paths
  auto guard = folly::makeGuard([win, ncclWinPtr] {
    delete ncclWinPtr;
    delete win;
  });

  NCCLCHECK(metaCommToNccl(ctran::ctranWinFree(ncclWinPtr->ctranWindow)));
  return ncclSuccess;
}

NCCL_API(
    ncclResult_t,
    ncclWinGetAttributes,
    int rank,
    ncclWindow_t win,
    ncclWinAttr_t* attr);
ncclResult_t
ncclWinGetAttributes(int rank, ncclWindow_t win, ncclWinAttr_t* attr) {
  ncclWin* ncclWinPtr = ncclWinMap().find(win);
  if (!win || !ncclWinPtr || !attr) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid parameter(s) in ncclWinGetAttributes: win {}, attr {}",
        (void*)win,
        (void*)attr);
  }

  auto statex = ncclWinPtr->comm->ctranComm_->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(ncclInternalError, "Empty communicator statex.");
  }

  if (rank < 0 || rank >= statex->nRanks()) {
    FB_ERRORRETURN(
        ncclInvalidUsage,
        "Invalid rank {} in ncclWinGetAttributes: must be in range [0, {})",
        rank,
        statex->nRanks());
  }

  auto newAttr = new ncclWinAttr();
  auto guard = folly::makeGuard([newAttr] { delete newAttr; });
  auto nvlEnabled = ncclWinPtr->ctranWindow->nvlEnabled(rank);
  if (nvlEnabled) {
    newAttr->accessType = ncclWinAccessType::ncclWinAccessUnified;
  } else {
    newAttr->accessType = ncclWinAccessType::ncclWinAccessSeparate;
  }
  *attr = newAttr;
  guard.dismiss();
  return ncclSuccess;
}

#if defined(ENABLE_PIPES)
#include <cuda_runtime_api.h>

#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/window/DeviceWindow.cuh"
#include "comms/pipes/window/HostWindow.h"

NCCL_API(
    ncclResult_t,
    ncclWinCreateDeviceWin,
    ncclWindow_t win,
    int signal_count,
    int counter_count,
    int barrier_count,
    void** outDevicePtr);
ncclResult_t ncclWinCreateDeviceWin(
    ncclWindow_t win,
    int signal_count,
    int counter_count,
    int barrier_count,
    void** outDevicePtr) {
  // Creates a DeviceWindow in device memory from the ctran window underlying
  // the given ncclWindow_t. This is a COLLECTIVE operation on first call —
  // all ranks must call together because get_device_win() does an allGather.
  //
  // Subsequent calls on the same window return cached results (config ignored).
  //
  // The returned void* is a device pointer to comms::pipes::DeviceWindow.
  // The caller must free it via ncclWinDestroyDeviceWin().
  if (win == nullptr || outDevicePtr == nullptr) {
    return ncclInvalidArgument;
  }

  ncclWin* nw = ncclWinMap().find(win);
  if (nw == nullptr || nw->ctranWindow == nullptr) {
    return ncclInternalError;
  }

  // Build WindowConfig from parameters.
  comms::pipes::WindowConfig config{
      .peerSignalCount = static_cast<std::size_t>(std::max(signal_count, 0)),
      .peerCounterCount = static_cast<std::size_t>(std::max(counter_count, 0)),
      .barrierCount = static_cast<std::size_t>(std::max(barrier_count, 0)),
  };

  // Populate DeviceWindow on host stack. get_device_win() fills in transport
  // handles, remote buffer descriptors, and signal pointers.
  comms::pipes::DeviceWindow host_dev_win{};
  auto result = nw->ctranWindow->get_device_win(&host_dev_win, config);
  if (result != commSuccess) {
    WARN("ncclWinCreateDeviceWin: get_device_win failed with error %d", result);
    return ncclInternalError;
  }

  // Allocate device memory for DeviceWindow.
  // NOTE: Uses raw cudaMalloc; the caller frees via ncclWinDestroyDeviceWin()
  // or cuda_api->free() (which wraps cudaFree — compatible).
  comms::pipes::DeviceWindow* dev_ptr = nullptr;
  cudaError_t cuda_err = cudaMalloc(
      reinterpret_cast<void**>(&dev_ptr), sizeof(comms::pipes::DeviceWindow));
  if (cuda_err != cudaSuccess) {
    WARN(
        "ncclWinCreateDeviceWin: cudaMalloc failed: %s",
        cudaGetErrorString(cuda_err));
    return ncclInternalError;
  }

  // Copy populated DeviceWindow from host to device.
  // dev_ptr is non-null here (cudaMalloc succeeded above).
  // NOLINTNEXTLINE(facebook-hte-NullableDereference,facebook-security-vulnerable-memcpy)
  cuda_err = cudaMemcpy(
      dev_ptr,
      &host_dev_win,
      sizeof(comms::pipes::DeviceWindow),
      cudaMemcpyHostToDevice);
  if (cuda_err != cudaSuccess) {
    WARN(
        "ncclWinCreateDeviceWin: cudaMemcpy failed: %s",
        cudaGetErrorString(cuda_err));
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check,facebook-hte-NullableDereference)
    cudaFree(dev_ptr);
    return ncclInternalError;
  }

  *outDevicePtr = dev_ptr;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclWinDestroyDeviceWin, void* devicePtr);
ncclResult_t ncclWinDestroyDeviceWin(void* devicePtr) {
  // Frees device memory allocated by ncclWinCreateDeviceWin.
  if (devicePtr == nullptr) {
    return ncclSuccess;
  }
  cudaError_t err = cudaFree(devicePtr);
  return (err == cudaSuccess) ? ncclSuccess : ncclInternalError;
}
NCCL_API(
    ncclResult_t,
    ncclWinLocalRegisterBuffer,
    ncclComm_t comm,
    void* ptr,
    size_t size,
    uint32_t* outLkey);
ncclResult_t ncclWinLocalRegisterBuffer(
    ncclComm_t comm,
    void* ptr,
    size_t size,
    uint32_t* outLkey) {
  if (comm == nullptr || ptr == nullptr || outLkey == nullptr) {
    return ncclInvalidArgument;
  }

  if (!ctranInitialized(comm->ctranComm_.get())) {
    WARN("ncclWinLocalRegisterBuffer: ctran not initialized");
    return ncclInternalError;
  }

  auto* mpt = comm->ctranComm_->multiPeerTransport_.get();
  if (mpt == nullptr) {
    WARN(
        "ncclWinLocalRegisterBuffer: MultiPeerTransport not initialized. "
        "Set NCCL_CTRAN_USE_PIPES=1");
    return ncclInternalError;
  }

  try {
    auto ibgdaBuf = mpt->localRegisterIbgdaBuffer(ptr, size);
    *outLkey = ibgdaBuf.lkey.value;
    return ncclSuccess;
  } catch (const std::exception& e) {
    WARN("ncclWinLocalRegisterBuffer: %s", e.what());
    return ncclInternalError;
  }
}

NCCL_API(
    ncclResult_t,
    ncclWinLocalDeregisterBuffer,
    ncclComm_t comm,
    void* ptr);
ncclResult_t ncclWinLocalDeregisterBuffer(ncclComm_t comm, void* ptr) {
  if (comm == nullptr || ptr == nullptr) {
    return ncclInvalidArgument;
  }

  if (!ctranInitialized(comm->ctranComm_.get())) {
    WARN("ncclWinLocalDeregisterBuffer: ctran not initialized");
    return ncclInternalError;
  }

  auto* mpt = comm->ctranComm_->multiPeerTransport_.get();
  if (mpt == nullptr) {
    WARN(
        "ncclWinLocalDeregisterBuffer: MultiPeerTransport not initialized. "
        "Set NCCL_CTRAN_USE_PIPES=1");
    return ncclInternalError;
  }

  try {
    mpt->localDeregisterIbgdaBuffer(ptr);
    return ncclSuccess;
  } catch (const std::exception& e) {
    WARN("ncclWinLocalDeregisterBuffer: %s", e.what());
    return ncclInternalError;
  }
}
#endif // ENABLE_PIPES
