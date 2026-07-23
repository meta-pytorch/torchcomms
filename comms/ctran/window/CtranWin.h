// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <folly/Synchronized.h>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/CtranMulticast.h"
#include "comms/ctran/utils/DevMemType.h"
#include "comms/ctran/window/Types.h"
#if defined(ENABLE_PRIMS)
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#endif

#if defined(ENABLE_PRIMS)
namespace comms::prims {
class DeviceWindow;
class HostWindow;
struct WindowConfig;
} // namespace comms::prims
#endif

class CtranPersistentRequest;

namespace ctran {
struct CtranWin {
  // TODO: remove the communicator from the window allocation.
  // We will need Ctran instead of CtranComm to allocate the window. Current
  // implementation still uses CtranComm for:
  // 1. the communicator's logMetaData for memory logging purposes.
  // 2. the communicator's ctran mapper for network registration.
  // 3. the communicator's bootstrap for intra node bootstrap all gather.
  CtranComm* comm;

  // Remote window info (addr, rkey, dataBytes) for all peers in this window
  std::vector<window::RemWinInfo> remWinInfo;

  // This rank's local data buffer size in bytes
  size_t dataBytes{0};
  // Signal buffer size in number of uint64_t elements per rank
  size_t signalSize{0};
  // The ctran mapper handles for caching the allocated buffer segment
  void* baseSegHdl{nullptr};
  // The ctran mapper handles for caching the allocated buffer registration
  void* baseRegHdl{nullptr};
  // The ctran mapper handles for caching the data segment
  void* dataSegHdl{nullptr};
  // The ctran mapper handles for caching the data registration
  void* dataRegHdl{nullptr};
  // WinCache/AVL handle for this window's cached data range; used to erase the
  // entry directly at free (avoids range lookup resolving to a different
  // overlapping window).
  void* winCacheHdl{nullptr};
  // Scoped registration owning the user-provided data buffer's local
  // registration ref (SW refcnt only). dataRegHdl aliases its borrowed
  // RegElem* for the export ctrl path.
  ScopedRegHdl dataScopedReg;
  // Scoped RAII owners of this rank's imported NVL peer registrations for the
  // user data buffer (sized nLocalRanks, empty for self / non-NVL peers).
  // Released SW-only in free(); replaces the old per-peer deregRemReg loop.
  std::vector<ctran::ScopedIpcRegHdl> dataScopedIpcRegHdls;
  // The base pointer of allocated buffer by the window
  void* winBasePtr{nullptr};
  // The pointer of the data buffer of this window
  void* winDataPtr{nullptr};
  // The pointer of the signal buffer of this window
  uint64_t* winSignalPtr{nullptr};
  // Stores signal values for waiting, used to track progress
  std::deque<std::atomic<uint64_t>> waitSignalVal{};

  // Stores signal values for sent signals, used to track progress
  std::deque<std::atomic<uint64_t>> signalVal{};

  CtranWin(
      CtranComm* comm,
      size_t dataSize,
      DevMemType bufType = DevMemType::kCumem);
  ~CtranWin();

  inline uint64_t updateOpCount(
      const int rank,
      const window::OpCountType type = window::OpCountType::kWinScope) {
    const auto key = std::make_pair(rank, type);
    auto locked = opCountMap_.wlock();
    auto opCount = 0;

    auto it = locked->find(key);
    if (it == locked->end()) {
      // tracked after first update, starting from value 1
      locked->insert(std::make_pair(key, 1));
    } else {
      opCount = it->second;
      it->second++;
    }
    return opCount;
  }

  inline uint64_t ctranNextWaitSignalVal(int peer) {
    FB_CHECKTHROW_EX_NOCOMM(
        peer < signalSize,
        "peer rank {} exceed window signal buffer size {}",
        peer,
        signalSize);
    return waitSignalVal[peer].fetch_add(1, std::memory_order_relaxed);
  }

  inline uint64_t ctranNextSignalVal(int peer) {
    FB_CHECKTHROW_EX_NOCOMM(
        peer < signalSize,
        "peer rank {} exceed window signal buffer size {}",
        peer,
        signalSize);
    return signalVal[peer].fetch_add(1, std::memory_order_relaxed);
  }

  commResult_t allocate(void* userBufPtr = nullptr);
  commResult_t exchange();

#if defined(ENABLE_PRIMS)
  // COLLECTIVE on first call: all ranks must call this together.
  // Prerequisite: allocate() and exchange() must have been called first.
  // Registers the window data buffer with pipes' MultiPeerTransport for
  // IBGDA/NVL access and populates the device-side window struct.
  // Subsequent calls return the cached result (config is ignored).
  //
  // @param devWin  Output: populated device-side window handle.
  // @param config  WindowConfig controlling signal/counter/barrier allocation.
  commResult_t getDeviceWin(
      comms::prims::DeviceWindow* devWin,
      const comms::prims::WindowConfig& config);

  // Returns the pipes HostWindow pointer for this window.
  // The caller does not take ownership.
  // Returns nullptr if pipes device window is not initialized.
  comms::prims::HostWindow* getPipesHostWindow() const {
    return hostWindow_.get();
  }
#endif

  commResult_t free(bool skipBarrier = false);

  bool nvlEnabled(int rank) const;

  // Get data size for specific rank
  inline size_t getDataSize(int rank) const {
    if (rank >= 0 && rank < static_cast<int>(remWinInfo.size())) {
      return remWinInfo[rank].dataBytes;
    }
    return 0; // invalid rank
  }

  inline bool isGpuMem() const {
    return bufType_ == DevMemType::kCudaMalloc ||
        bufType_ == DevMemType::kCumem;
  }

  inline bool isAtomicCapable() const {
    return atomicCapable_;
  }

  inline void setAtomicCapable(bool val) {
    atomicCapable_ = val;
  }

  inline void setIpcOnly(bool val) {
    ipcOnly_ = val;
  }

  inline bool isIpcOnly() const {
    return ipcOnly_;
  }

  inline void setEnableSignal(bool val) {
    enableSignal_ = val;
  }

  inline bool isSignalEnabled() const {
    return enableSignal_;
  }

  inline void setSymmetric(bool val) {
    symmetric_ = val;
  }

  inline bool isSymmetric() const {
    return symmetric_;
  }

  // Opt-in (win_register_multicast): set up a standalone NVL CE-multicast
  // object over the window's data buffer during exchange(), so ctwin AllGather
  // fans out via a single NVSwitch write. Only takes effect on a symmetric,
  // cuMem-backed window on multicast-capable HW; unicast fallback otherwise.
  inline void setMulticast(bool val) {
    multicast_ = val;
  }

  inline bool isMulticast() const {
    return multicast_;
  }

  // Multicast write base for a user pointer in this window's data buffer, or
  // std::nullopt when there is no multicast object (unicast). Computed from the
  // multicast object alone -- no CtranIpc involvement -- so ctwin AllGather
  // fans out via a single NVSwitch write.
  inline std::optional<void*> multicastWriteBase(const void* userPtr) const {
    return mc_ ? mc_->writeBase(userPtr) : std::nullopt;
  }

  inline uint64_t id() const {
    return id_;
  }

  // Check whether persistent allgather (allgatherP) is supported.
  // Returns true if ctran is initialized and all peers have configured
  // backends. Static variant allows checking before a window is created.
  static bool allGatherPSupported(CtranComm* comm);
  bool allGatherPSupported() const {
    return allGatherPSupported(comm);
  }

  // Window-based persistent-request cache, keyed by <byteOffset, byteLen,
  // stream> of the recvbuf sub-range. Creates the request via `factory` on a
  // miss, otherwise reuses the cached one; returns nullptr if the factory
  // fails. The request is per-stream, so two collectives sharing the same key
  // are always serialized.
  //
  // LIFETIME CONTRACT: the returned request is WINDOW-owned and freed by the
  // window's free(). The caller must ensure every collective that uses a
  // returned request has completed (e.g. cudaStreamSynchronize) before freeing
  // the window -- freeing while a ctwin collective is still in flight is
  // undefined behavior.
  CtranPersistentRequest* getOrCreatePersistentRequest(
      size_t offset,
      size_t len,
      cudaStream_t stream,
      const std::function<CtranPersistentRequest*()>& factory);

  // Number of cached ctwin persistent requests (test/introspection helper).
  size_t numPersistentRequests() const;

 private:
  DevMemType bufType_{DevMemType::kCumem};
  // whether allocate window data buffer or provided by users
  bool allocDataBuf_{true};
  // Whether this window's data buffer meets the alignment requirements
  // for IB/NVLink atomic operations (8-byte aligned address and size).
  bool atomicCapable_{false};
  // When true, registration exchanges only intra-node NVL/CUDA-IPC handles
  // and skips the IB rkey exchange.
  bool ipcOnly_{false};
  // When false, the window carries no signal buffer: signal-buffer allocation
  // and the signal-related control exchange are skipped, and signal RMA ops are
  // rejected. Used by data-only collective windows (e.g. window-based
  // allgather) that never issue signals.
  bool enableSignal_{true};
  // When true, every rank registers a buffer of identical size at an identical
  // offset from the window base, so a peer address can be computed as
  // peerBase + (buf - localBase). Records the upstream NCCL_WIN_COLL_SYMMETRIC
  // hint; consumed by a later window-based allgather. Cached only for now.
  bool symmetric_{false};
  // Records the win_register_multicast hint; see setMulticast(). Consumed in
  // exchange() to set up the NVL CE-multicast object.
  bool multicast_{false};
  // The standalone NVL CE-multicast object for this window's data buffer, set
  // in exchange() and read via multicastWriteBase(). Self-owning; released with
  // the window. null unless win_register_multicast engaged.
  std::shared_ptr<ctran::utils::CtranMulticast> mc_;
  // Per-comm unique id assigned at exchange() (see CtranComm::assignWindowId).
  uint64_t id_{0};
  // rank: window::OpCountType as key
  folly::Synchronized<
      std::unordered_map<std::pair<int, window::OpCountType>, uint64_t>>
      opCountMap_;
  // Actual size allocated for the total buffer per rank in this window
  size_t range_{0};
  // Window-based persistent-request cache; see getOrCreatePersistentRequest.
  // Key is <byteOffset, byteLen, stream> of the recvbuf sub-range.
  folly::Synchronized<std::map<
      std::tuple<size_t, size_t, cudaStream_t>,
      CtranPersistentRequest*>>
      persistentReqs_;

#if defined(ENABLE_PRIMS)
  std::unique_ptr<comms::prims::HostWindow> hostWindow_;
#endif
};

commResult_t ctranWinAllocate(
    size_t size,
    CtranComm* comm,
    void** baseptr,
    CtranWin** win,
    const meta::comms::Hints& hints = meta::comms::Hints());

commResult_t ctranWinRegister(
    const void* baseptr,
    size_t size,
    CtranComm* comm,
    CtranWin** win,
    const meta::comms::Hints& hints = meta::comms::Hints());

commResult_t ctranWinSharedQuery(int rank, CtranWin* win, void** addr);

commResult_t ctranWinFree(CtranWin* win);

} // namespace ctran
