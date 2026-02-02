// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <string>
#include <unordered_map>

#include <folly/Synchronized.h>
#include "comms/ctran/regcache/IpcRegCacheBase.h"
#include "comms/ctran/utils/Checks.h"

namespace ctran {

// Class to manage IPC-based remote memory registrations for a single
// communicator. Currently handles NVL (NVLink) remote memory imports from peer
// processes. This class caches imported registrations to enable reuse across
// multiple collective operations without re-importing the same memory.
class IpcRegCache {
 public:
  // Register memory to be used for NVL operations. If registration fails
  // because memory type is not supported, commSuccess shall be returned and the
  // ipcRegElem will be set to nullptr. Mapper is expected to still handle the
  // case with alternative path.
  // Input arguments:
  //   - buf: the local buffer to be registered to network for direct RDMA
  //   access
  //   - len: the length of the local buffer
  //   - cudaDev: the cuda device id of the local buffer
  // Output arguments:
  //   - ipcRegElem: the ipcRegElem of the local buffer that stores the
  //                registration handle.
  static commResult_t regMem(
      const void* buf,
      const size_t len,
      const int cudaDev,
      void** ipcRegElem,
      bool shouldSupportCudaMalloc = false);

  // Deregister memory to be used for NVL operations.
  // Input arguments:
  //   - ipcRegElem: the ipcRegElem of the local buffer that stores the
  //                registration handle.
  static void deregMem(void* ipcRegElem);

  // Release the exported memory on remote rank.
  // Input arguments:
  //   - ipcRegElem: local registration
  // Output arguments:
  //   - ipcRelease: the IpcRelease struct to be populated and sent to remote
  //                 rank.
  static void remReleaseMem(
      void* ipcRegElem,
      ctran::regcache::IpcRelease& ipcRelease);

  IpcRegCache();
  ~IpcRegCache();

  // Initialize the cache with CUDA device and logging metadata.
  // Must be called before using importMem.
  void init(int cudaDev, const struct CommLogData* logMetaData);

  // Import a remote NVL memory registration from IPC descriptor.
  // The imported memory is cached for reuse across multiple operations.
  // Requires init() to be called first to set cudaDev and logMetaData.
  // Input arguments:
  //   - peerId: Id of the peer, which should be unique per process instance
  //   - ipcDesc: the remote memory IPC descriptor
  // Output arguments:
  //   - buf: the local buffer mapped to the imported remote memory
  //   - remKey: the remoteAccessKey (rkey) of the remote buffer registration
  commResult_t importMem(
      const std::string& peerId,
      const ctran::regcache::IpcDesc& ipcDesc,
      void** buf,
      struct ctran::regcache::IpcRemHandle* remKey);

  // Export local NVL memory registration for sharing with remote peers.
  // Input arguments:
  //   - buf: local buffer to export
  //   - ipcRegElem: local IPC registration element
  // Output arguments:
  //   - ipcDesc: IPC descriptor to be populated and sent to remote peer
  inline commResult_t exportMem(
      const void* buf,
      void* ipcRegElem,
      ctran::regcache::IpcDesc& ipcDesc) {
    if (ipcRegElem == nullptr) {
      CLOGF(ERR, "CTRAN-REGCACHE: ipcRegElem is nullptr in exportMem");
      return commInvalidArgument;
    }
    auto reg = reinterpret_cast<ctran::regcache::IpcRegElem*>(ipcRegElem);

    // Fill IPC descriptor content
    auto ipcMem = reg->ipcMem.wlock();
    FB_COMMCHECK(ipcMem->ipcExport(ipcDesc.desc));
    ipcDesc.offset = reinterpret_cast<size_t>(buf) -
        reinterpret_cast<size_t>(ipcMem->getBase());
    return commSuccess;
  }

  // Release a specific remote registration for a given peer and base
  // pointer.
  commResult_t releaseRemReg(const std::string& peerId, void* basePtr);

  // Get the number of existing remote registrations for a given peer
  size_t getNumRemReg(const std::string& peerId) const;

  // Release all remote registrations.
  // Called during destruction to clean up any remaining cached registrations.
  void clearAllRemReg();

 private:
  // Internal implementation for importing and caching remote NVL memory.
  commResult_t importRemMemImpl(
      const std::string& peerId,
      const ctran::utils::CtranIpcDesc& ipcDesc,
      void** mappedBase);

  // Cache of imported IPC remote registrations
  // Key: peer name -> (remote base pointer -> ctran::regcache::IpcRemRegElem)
  // This cache enables reuse of imported registrations across multiple
  // collective operations without re-importing the same memory.
  using IpcRemRegMap = std::unordered_map<
      std::string, // peer name
      std::unordered_map<
          uint64_t, // remote base pointer
          std::unique_ptr<ctran::regcache::IpcRemRegElem>>>;
  folly::Synchronized<IpcRemRegMap> ipcRemRegMap_;

  int cudaDev_;
  const struct CommLogData* logMetaData_;
};

} // namespace ctran
