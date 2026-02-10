// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

#include <folly/Hash.h>
#include <folly/SocketAddress.h>
#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <folly/io/async/ScopedEventBaseThread.h>

#include "comms/ctran/bootstrap/AsyncSocket.h"
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
  // IpcRegCache is a singleton shared by all communicators and will be
  // destroyed at program exit.
  ~IpcRegCache();

  static std::shared_ptr<IpcRegCache> getInstance();

  // Initialize the cache (starts AsyncSocket server).
  // Must be called before using importMem.
  void init();

  // Import a remote NVL memory registration from IPC descriptor.
  // The imported memory is cached for reuse across multiple operations.
  // Requires init() to be called first.
  // Input arguments:
  //   - peerId: Id of the peer, which should be unique per process instance
  //   - ipcDesc: the remote memory IPC descriptor
  //   - cudaDev: the CUDA device to import the memory to
  //   - logMetaData: (optional) logging metadata from the communicator
  // Output arguments:
  //   - buf: the local buffer mapped to the imported remote memory
  //   - remKey: the remoteAccessKey (rkey) of the remote buffer registration
  commResult_t importMem(
      const std::string& peerId,
      const ctran::regcache::IpcDesc& ipcDesc,
      int cudaDev,
      void** buf,
      struct ctran::regcache::IpcRemHandle* remKey,
      const struct CommLogData* logMetaData = nullptr);

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
    ipcDesc.uid = reg->uid;
    return commSuccess;
  }

  // Release a specific remote registration for a given peer.
  commResult_t
  releaseRemReg(const std::string& peerId, void* basePtr, uint32_t uid);

  // Get the number of existing remote registrations for a given peer
  size_t getNumRemReg(const std::string& peerId) const;

  // Release all remote registrations.
  // Called during destruction to clean up any remaining cached registrations.
  void clearAllRemReg();

  inline folly::SocketAddress getServerAddr() const {
    return serverAddr_;
  }

  // Set peer's IPC server address by peer ID (gPid).
  commResult_t setPeerIpcServerAddr(
      const std::string& peerId,
      const folly::SocketAddress& addr);

  // Get peer's IPC server address by peer ID (gPid).
  // Output argument:
  //   - addr: the socket address for the peer with the given gPid.
  commResult_t getPeerIpcServerAddr(
      const std::string& peerId,
      folly::SocketAddress& addr) const;

  // Notify remote peers to release their imported NVL memory.
  // Output argument:
  //   - reqCb: IpcReqCb that the caller can track for completion.
  //            Caller must ensure reqCb remains valid until completed.
  commResult_t notifyRemoteIpcRelease(
      const std::string& myId,
      const folly::SocketAddress& peerAddr,
      regcache::IpcRegElem* ipcRegElem,
      regcache::IpcReqCb* reqCb);

  // Notify remote peer to import our exported NVL memory.
  // The peer will call importMem upon receiving this request.
  // Input arguments:
  //   - peerAddr: the address of the remote peer
  //   - ipcDesc: the IPC descriptor to send to the peer
  // Output argument:
  //   - reqCb: IpcReqCb that the caller can track for completion.
  //            Caller must ensure reqCb remains valid until completed.
  commResult_t notifyRemoteIpcExport(
      const std::string& myId,
      const folly::SocketAddress& peerAddr,
      const regcache::IpcDesc& ipcDesc,
      regcache::IpcReqCb* reqCb);

 private:
  // Internal implementation for importing and caching remote NVL memory.
  commResult_t importRemMemImpl(
      const std::string& peerId,
      const ctran::regcache::IpcDesc& ipcDesc,
      int cudaDev,
      const struct CommLogData* logMetaData,
      void** mappedBase);

  // Initialize the AsyncSocket infrastructure for this rank.
  // Must be called once per rank before notifyRemoteIpcRelease can be used.
  // Input arguments:
  //   - bindAddr: the address to bind the server socket
  commResult_t initAsyncSocket();

  // Stop the AsyncSocket infrastructure and wait for cleanup.
  void stopAsyncSocket();

  // Cache of imported IPC remote registrations
  // Key: peer name -> ((remote base pointer, uid) ->
  // ctran::regcache::IpcRemRegElem) This cache enables reuse of imported
  // registrations across multiple collective operations without re-importing
  // the same memory.
  using IpcRemRegKey = std::pair<uint64_t, uint32_t>; // (base pointer, uid)
  using IpcRemRegMap = std::unordered_map<
      std::string, // peer name
      std::unordered_map<
          IpcRemRegKey,
          std::unique_ptr<ctran::regcache::IpcRemRegElem>,
          folly::Hash>>;
  folly::Synchronized<IpcRemRegMap> ipcRemRegMap_;

  // Flag for one-time initialization
  std::once_flag initFlag_;

  // Member variables for AsyncSocket-based remote release mechanism
  std::unique_ptr<folly::ScopedEventBaseThread> asyncSocketEvbThread_;
  std::unique_ptr<ctran::bootstrap::AsyncServerSocket> asyncServerSocket_;
  folly::SocketAddress serverAddr_;

  // Peer IPC server addresses for async socket communication, keyed by gPid.
  // Protected by Synchronized for concurrent access from multiple
  // communicators.
  folly::Synchronized<folly::F14FastMap<std::string, folly::SocketAddress>>
      peerIpcServerAddrs_;

  // Monotonically increasing unique ID counter for IPC registrations
  static std::atomic<uint32_t> nextUniqueId_;
};

} // namespace ctran
