// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IB_H_
#define CTRAN_IB_H_

#include <memory>

#include <folly/SocketAddress.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/backends/ib/CtranIbImpl.h"
#include "comms/ctran/backends/ib/CtranIbLocalVc.h"
#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/ctran/bootstrap/AbortableSocket.h"
#include "comms/ctran/bootstrap/ISocketFactory.h"
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/utils/Abort.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"

class CtranIb;
class CtranIbVirtualConn;
commResult_t checkEpochLock(CtranIb* ctranIb);

/**
 * CtranIB class to be used by algorithms and ctranMapper.
 */
class CtranIb {
 public:
  enum class BootstrapMode { kDefaultServer, kSpecifiedServer, kExternal };

  // Creates local IB resources for a given communicator including obtaining the
  // singleton PD and context, and creating a per-communicator Completion Queue
  // (CQ). It also launches a listen thread to accept remote connection. The
  // remote connection will happen when the remote peer issues the first message
  // to the local rank.
  // Input arguments:
  //   - comm: the Ctran communicator
  //   - ctrlMgr: the ctranCtrlManager that manages control message callback
  //              functions registered by other modules. Passed to VC for
  //              calling callback when receiving control message.
  //   - enableLocalFlush: whether to support local flush. If not specified, use
  //              default config based on cuda arch.
  CtranIb(
      CtranComm* comm,
      CtranCtrlManager* ctrlMgr,
      std::optional<bool> enableLocalFlush = std::nullopt,
      std::shared_ptr<ctran::bootstrap::ISocketFactory> socketFactory =
          nullptr);

  // Creates local IB resources without pre-existing communicator. It is used
  // for use cases directly control the local transport (see CtranEx). In
  // addition, it supports three types of bootstrap mode as defined below.
  // Input arguments:
  //   - rank: the rank of the calling process. Used to manage peer-to-peer
  //           connection in local connection cache.
  //   - cudaDev: the cuda device of the calling process. Used to find the
  //              mapping NIC
  //   - commHash: for logging only.
  //   - commDesc: for logging only.
  //   - ctrlMgr: same as in comm-based CtranIb constructor.
  //   - enableLocalFlush: whether to support local flush.
  //   - bootstrapMode: defines the needed bootstrap mode. If kDefaultServer,
  //                    it launches internal listen thread which binds and
  //                    listens to the default server address and port as
  //                    defined by NCCL_SOCKET_IFNAME; if kSpecifiedServer, the
  //                    internal listen thread would bind and listen to the
  //                    specified qpServerAddr; if kExternal, it omits internal
  //                    bootstrap management, and let callsite explicitly
  //                    controls peer-to-peer virtual connections (see
  //                    connectVc).
  CtranIb(
      int rank,
      int cudaDev,
      uint64_t commHash,
      const std::string& commDesc,
      CtranCtrlManager* ctrlMgr,
      bool enableLocalFlush,
      const BootstrapMode bootstrapMode = BootstrapMode::kDefaultServer,
      std::optional<const SocketServerAddr*> qpServerAddr = std::nullopt,
      std::shared_ptr<Abort> abortCtrl =
          ::ctran::utils::createAbort(/*enabled=*/false),
      std::shared_ptr<ctran::bootstrap::ISocketFactory> socketFactory =
          nullptr);

  ~CtranIb();

  // Lock the entire CtranIb instance for exclusive access from the current
  // thread to internal backend and resources. Any concurrent lock request will
  // be blocked till current thread unlocks. However, if a thread locks the same
  // CtranIb instance twice without unlocking, commInternalError is returned
  // since it is an invalid usage.
  // Calling epochLock() is required before any critical path access
  // (isendCtrlMsg|irecvCtrlMsg|iput|notify|checkNotify|waitNotify|iflush|progress)
  // in CtranIb when NCCL_CTRAN_IB_EPOCH_LOCK_ENABLE is set to true.
  //
  // See CtranIbEpochRAII for convenient RAII class to guard CtranIb epoch lock.
  commResult_t epochLock();

  // Try to lock the entire CtranIb instance. Return commInProgress if the lock
  // has been held by another thread. However, if a thread locks the same
  // CtranIb instance twice without unlocking, commInternalError is returned
  // since it is an invalid usage. It is likely used only for testing.
  //
  // See CtranIbEpochRAII for convenient RAII class to guard CtranIb epoch lock.
  commResult_t epochTryLock();

  // Unlock the entire CtranIb instance.
  commResult_t epochUnlock();

  static bool isEnabled();

  // Register control message callback function if have any.
  // Input arguments:
  //   - ctrlMgr: the ctranCtrlManager to manage the control message
  void regCtrlCb(std::unique_ptr<CtranCtrlManager>& ctrlMgr);

  // Register memory to be used for IB operations.
  // Input arguments:
  //   - buf: the local buffer to be registered to network for direct RDMA
  //          access
  //   - len: the length of the local buffer
  //   - cudaDev: the cuda device id of the local buffer
  // Output arguments:
  //   - ibRegElem: the ibRegElem of the local buffer that stores the
  //                registration handle.
  static commResult_t regMem(
      const void* buf,
      const size_t len,
      const int cudaDev,
      void** ibRegElem);

  // Deregister memory to be used for IB operations.
  // Input arguments:
  //   - ibRegElem: the ibRegElem of the local buffer that stores the
  //                registration handle.
  static commResult_t deregMem(void* ibRegElem);

  // Get the CtranIbRemoteAccessKey struct from an IB registration handle
  // Input arguments:
  //   - ibRegElem: the ibRegElem of the local buffer that stores the
  //                registration handle.
  static CtranIbRemoteAccessKey getRemoteAccessKey(void* ibRegElem) {
    CtranIbRemoteAccessKey key;
    key.nKeys = NCCL_CTRAN_IB_DEVICES_PER_RANK;
    ctran::ib::getRemoteKeysImpl(ibRegElem, key.rkeys);
    return key;
  }

  // Progress the per-communicator CQ.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t progress() {
    return progressInternal<PerfConfig>();
  }

  // Export a local memory registration for remote rank to import.
  // Input arguments:
  //   - buf: the local buffer to be exported.
  //   - ibRegElem: local registration of the to-be-exported buffer
  // Output arguments:
  //   - msg: the reference to the control message to be sent to remote rank.
  //          Contents filled at return.
  inline commResult_t
  exportMem(const void* buf, void* ibRegElem, ControlMsg& msg) {
    return exportMemImpl(buf, ibRegElem, msg);
  }

  // Import a remote memory registration for local rank to use.
  // Input arguments:
  //   - buf: the remote buffer address.
  //   - key: the remoteAccessKey (rkey) of the remote buffer.
  //   - rank: the rank of the remote peer in the current communicator
  //   - msg: the reference to the control message received from remote rank.
  inline commResult_t
  importMem(void** buf, CtranIbRemoteAccessKey* key, const ControlMsg& msg) {
    return importMemImpl(buf, key, msg);
  }

  // Send control message packet over the IB connection.
  // Input arguments:
  //   - type: the control message type to be sent
  //   - payload: pointer to the payload of the control message to be sent
  //   - size: size of the payload of the control message to be sent
  //   - peerRank: the rank to send the control message to
  //   - peerServerAddr: the ip/port/hostname info of the qp server to connect
  //   to
  // Output arguments:
  //   - req: the request object to track the progress of the control message
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t isendCtrlMsg(
      const int type,
      const void* payload,
      const size_t size,
      int peerRank,
      CtranIbRequest& req,
      std::optional<const SocketServerAddr*> peerServerAddr = std::nullopt) {
    return isendCtrlMsgImpl<PerfConfig>(
        type, payload, size, peerRank, peerServerAddr, req);
  }

  // Receive control message packet over the IB connection.
  // Input arguments:
  //   - payload: pointer to the payload buffer to receive the control message
  //   - size: size of the payload buffer
  //   - peerRank: the rank to receive the control message from
  //   - peerServerAddr: the ip/port/hostname info of the remote qp server
  // Output arguments:
  //   - req: the request object to track the progress of the control message
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t irecvCtrlMsg(
      void* payload,
      const size_t size,
      int peerRank,
      CtranIbRequest& req,
      std::optional<const SocketServerAddr*> peerServerAddr = std::nullopt) {
    return irecvCtrlMsgImpl<PerfConfig>(
        payload, size, peerRank, peerServerAddr, req);
  }

  // RDMA put data from local sbuf to a dbuf in remote rank over
  // the established IB connection.
  // Input arguments:
  //   - sbuf: local buffer to put data from
  //   - dbuf: virtual address of the remote buffer to receive data. It is
  //           exchanged via isendCtrl|irecvCtrl called by the algorithm
  //           layer.
  //   - len: length of data
  //   - peerRank: the rank of the remote peer in the current communicator
  //   - ibRegElem: the ibRegElem of the local sbuf
  //   - remoteAccessKey: the remoteAccessKey of dbuf. It is exchanged via
  //                      isendCtrl|irecvCtrl called by the algorithm layer.
  //   - notify: whether to notify the remote peer when the RDMA PUT has
  //             finished and data has arrived in the remote dbuf.
  // Output arguments:
  //   - req (optional): the request object to track the progress of the iput.
  //          If nullptr is passed in, do not signal local completion and no
  //          request is tracked.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iput(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int peerRank,
      void* ibRegElem,
      CtranIbRemoteAccessKey remoteAccessKey,
      bool notify,
      CtranIbConfig* config,
      CtranIbRequest* req,
      bool fast = false) {
    return iputImpl<PerfConfig>(
        sbuf,
        dbuf,
        len,
        peerRank,
        ibRegElem,
        remoteAccessKey,
        notify,
        config,
        req,
        fast);
  }

  // Batch RDMA put operations to efficiently transfer multiple data chunks
  // from local buffers to remote buffers in a single peer rank over the
  // established IB connection. This is an optimized version of multiple iput
  // calls that can reduce the overhead of individual RDMA operations. This
  // will fallback to individual iput calls if it is not possible to transfer
  // multiple iputs in a single batch. To deliver efficiency gains, all the
  // put messages will have to be sent over a single QP.
  // Input arguments:
  //   - puts: vector of PutIbMsg structures containing the details of each
  //           RDMA put operation (source buffer, destination buffer, length,
  //           registration handles, remote access keys, etc.)
  //   - peerRank: the rank of the remote peer in the current communicator
  // Output arguments:
  //   - return: commResult_t indicating success or failure of the batch
  //   operation
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iputBatch(
      const std::vector<PutIbMsg>& puts,
      int peerRank) {
    return iputBatchImpl<PerfConfig>(puts, peerRank);
  }

  // RDMA get data from remote sbuf to a local dbuf over
  // the established IB connection.
  // Input arguments:
  //   - sbuf: virtual address of the remote buffer to get data. It is
  //           exchanged via isendCtrl|irecvCtrl called by the algorithm
  //           layer.
  //   - dbuf: local buffer to receive data
  //   - len: length of data
  //   - peerRank: the rank of the remote peer in the current communicator
  //   - ibRegElem: the ibRegElem of the local dbuf
  //   - remoteAccessKey: the remoteAccessKey of sbuf. It is exchanged via
  //                      isendCtrl|irecvCtrl called by the algorithm layer.
  // Output arguments:
  //   - req (optional): the request object to track the progress of the iget.
  //          If nullptr is passed in, do not signal local completion and no
  //          request is tracked.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iget(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int peerRank,
      void* ibRegElem,
      CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbConfig* config,
      CtranIbRequest* req,
      bool fast = false) {
    return igetImpl<PerfConfig>(
        sbuf,
        dbuf,
        len,
        peerRank,
        ibRegElem,
        remoteAccessKey,
        config,
        req,
        fast);
  }

  // Input arguments:
  //   - sbuf: local address to store the fetched remote value
  //   - dbuf: remote address to perform the atomic operation on
  //   - addVal: value to add
  //   - peerRank: the rank of the remote peer in the current communicator
  //   - ibRegElem: the ibRegElem of the local sbuf
  //   - remoteAccessKey: the remoteAccessKey of dbuf. It is exchanged via
  //                      isendCtrl|irecvCtrl called by the algorithm layer.
  // Output arguments:
  //   - req : the request object to track the progress of the atomic operation
  // Atomicity:
  //   - Atomicity is guaranteed for concurrent accesses to the same destination
  //     buffer on the same peerRank. Concurrent accesses may occur on different
  //     local ranks, or on the same local rank in different communicators.
  //   - Ordered execution of multiple fetchAndAdd operations is guaranteed if
  //     they are issued from the same local rank to the same peerRank in the
  //     same communicator. No ordering guarantee between operations from
  //     different local ranks, or from the same local rank in different
  //     communicators.
  inline commResult_t ifetchAndAdd(
      const void* sbuf,
      void* dbuf,
      uint64_t addVal,
      int peerRank,
      void* ibRegElem,
      struct CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbRequest* req) {
    return ifetchAndAddImpl(
        sbuf, dbuf, addVal, peerRank, ibRegElem, remoteAccessKey, req);
  }

  // Input arguments:
  //   - dbuf: remote address to perform the atomic operation on
  //   - val: value to set
  //   - peerRank: the rank of the remote peer in the current communicator
  //   - remoteAccessKey: the remoteAccessKey of dbuf. It is exchanged via
  //                      isendCtrl|irecvCtrl called by the algorithm layer.
  // Output arguments:
  //   - req : the request object to track the progress of the atomic operation
  // Atomicity:
  //   - Atomic write a 64-bit value to a remote buffer.
  inline commResult_t iatomicSet(
      void* dbuf,
      uint64_t val,
      int peerRank,
      struct CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbRequest* req) {
    return iatomicSetImpl(dbuf, val, peerRank, remoteAccessKey, req);
  }

  // A local RDMA read from the local receive buffer as a PCI-e flush to ensure
  // the received data from network is visible for all GPU threads.
  //
  // NOTE: Data will eventually arrive GPU memory and be visible to GPU threads
  // even without the flush, but we don't have control when it is done. It can
  // be risky if the gap between network completion and GPU access is very
  // short. The iflush forces the data to arrive GPU memory. It is recommended
  // to use it before asking GPU to access the data.
  // Input arguments:
  //   - dbuf: the local buffer to flush
  //   - localRegHdl: the local registration handle of the local buffer
  // Output arguments:
  //   - req: the request object to track the progress of the flush.
  commResult_t
  iflush(const void* dbuf, const void* localRegHdl, CtranIbRequest* req);

  // Notify the remote peer via a zero-byte RDMA_WRITE_WITH_IMM over the
  // established IB connection without extra data transfer.
  // Input arguments:
  //   - peerRank: the rank of the remote peer to notify
  // Output arguments:
  //   - req (optional): the request object to track the progress of the
  //           notification. If nullptr is passed in, do not signal
  //           local completion and no request is tracked.
  inline commResult_t notify(int peerRank, CtranIbRequest* req) {
    return notifyImpl(peerRank, req);
  }

  // Check whether the remote rank has finished the outstanding iput
  // Input arguments:
  //   - peerRank: the rank of the remote peer in the current communicator that
  //   has
  //           issued iput to the local rank.
  // Output arguments:
  //   - notify: whether the remote peer has finished the outstanding iput.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t checkNotify(int peerRank, bool* notify) {
    return checkNotifyImpl<PerfConfig>(peerRank, notify);
  }

  // Wait until the remote rank has finished the outstanding iput
  // Input arguments:
  //   - peerRank: the rank of the remote peer in the current communicator
  //   that has issued iput to the local rank.
  //   - notifyCnt: the number of notifies to wait for.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitNotify(int peerRank, int notifyCnt = 1) {
    return waitNotifyImpl<PerfConfig>(peerRank, notifyCnt);
  }

  std::string getIbDevName(int device = 0) const;

  int getIbDevPort(int device = 0) const;

  using CtranIbVcConfig_t =
      std::tuple<size_t, int, enum NCCL_CTRAN_IB_VC_MODE, int>;
  commResult_t getVcConfig(int peer, CtranIbVcConfig_t& config);

  // Release CtranIb backend qps and cq state
  commResult_t releaseRemoteTransStates(bool fromDestructor = false);

  // Init CtranIb backend cq state
  commResult_t initRemoteTransStates();

  // Pre-connect specified peer ranks
  commResult_t preConnect(const std::unordered_set<int>& peerRanks);

  CtranComm* comm{nullptr};

  // Setup virtual connection with a connected peer.
  // If serverRank is given, it indicates the calling rank behaves as client and
  // the other side behaves as server, so that we can switch the socket
  // send/recv to match.
  // Input arguments:
  // - sock: socket connection established by callsite. Used for internal
  //         business card exchange
  // - isServer: whether the local rank is isServer
  // - peerRank: the peer rank of remote client or server
  commResult_t connectVc(
      std::unique_ptr<ctran::bootstrap::ISocket> sock,
      const bool isServer,
      const int peerRank);

  // APIs to get the connection identifier for a given rank, and establish
  // the connection to remote VC.
  std::string getLocalVcIdentifier(const int peerRank);
  commResult_t connectVcDirect(
      const std::string& remoteVcIdentifier,
      const int peerRank);

  // Get the virtual connection for a given peer.
  // If the peer is not yet connected, return nullptr.
  // For a returned VC, it is guaranteed to be ready to use.
  // NOTE: callsite must be careful on taking the IB epoch
  // lock when using VC directly. For a given CtranIb instance,
  // the individial VC lock  (vc->mutex) should never be mix-used with IB epoch
  // lock.
  // Input arguments:
  // - peerRank: the peer rank
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline std::shared_ptr<CtranIbVirtualConn> getVc(int peerRank) {
    return getVcImpl<PerfConfig>(peerRank);
  }

  // Return the listen address of the listen socket.
  folly::Expected<folly::SocketAddress, int> getListenSocketListenAddr() {
    return listenSocket->getListenAddress();
  }

 private:
  friend class CtranIbRequest;
  void init(
      CtranComm* comm,
      int rank,
      int cudaDev,
      uint64_t commHash,
      const std::string& commDesc,
      CtranCtrlManager* ctrlMgr,
      bool enableLocalFlush,
      const BootstrapMode bootstrapMode = BootstrapMode::kDefaultServer,
      std::optional<const SocketServerAddr*> qpServerAddr = std::nullopt,
      std::shared_ptr<Abort> abortCtrl =
          ::ctran::utils::createAbort(/*enabled=*/false),
      std::shared_ptr<ctran::bootstrap::ISocketFactory> socketFactory =
          nullptr);

  void bootstrapStart(std::optional<const SocketServerAddr*> qpServerAddr);
  static void bootstrapAccept(CtranIb* ib);
  commResult_t bootstrapConnect(
      int peerRank,
      std::optional<const SocketServerAddr*> peerAddr = std::nullopt);

  // Create a virtual connection for a given peer.
  // Expect used only in bootstrapAccept() and bootstrapConnect().
  std::shared_ptr<CtranIbVirtualConn> createVc(int peerRank);

  // Update and query vcStateMaps.
  // It is thread-safe to be called by server and client threads concurrently.
  // NOTE: vc lock doesn't protect qpToRankMap, since it is for all peers.
  commResult_t updateVcState(std::shared_ptr<CtranIbVirtualConn> vc, int rank);
  commResult_t queryQpToRank(uint32_t qpn, int& rank);

  commResult_t setPgToTrafficClassMap();

  uint32_t getPgToTrafficClassValue() const;

  const char* ibv_wc_status_str(enum ibverbx::ibv_wc_status status);

  // check where the peer is connected
  inline bool isPeerConnected(int peerRank) const {
    return (
        connectedPeerMap.size() > peerRank && connectedPeerMap.at(peerRank));
  }

  inline bool canTransfer(CtranIbVirtualConn* vc) {
    CTRAN_IB_PER_OBJ_LOCK_GUARD(vc->mutex, { return vc->canTransferData(); });
  }

  inline commResult_t checkValidPeer(int peerRank) {
    if (peerRank < 0 || (comm && peerRank >= comm->statex_->nRanks())) {
      CLOGF(
          ERR,
          "invalid peerRank ({}) < 0 or >= nRanks {}",
          peerRank,
          comm ? comm->statex_->nRanks() : -1);
      return commInternalError;
    }
    return commSuccess;
  }

  inline commResult_t checkValidVc(
      std::shared_ptr<CtranIbVirtualConn>& vc,
      int peerRank) {
    if (vc == nullptr) {
      CLOGF(
          ERR,
          "No valid VirtualConnection (VC) found for peerRank {}",
          peerRank);
      return commInternalError;
    }
    return commSuccess;
  }

  inline commResult_t checkAndInsertQpToVcMap(
      folly::F14FastMap<QpUniqueId, std::shared_ptr<CtranIbVirtualConn>>& map,
      QpUniqueId& qpId,
      std::shared_ptr<CtranIbVirtualConn>& vc) {
    if (map.find(qpId) != map.end()) {
      CLOGF(
          ERR,
          "CTRAN-IB: QP {} on device {} already exists in pimpl {} commHash {:x}, commDesc {}. It likely indicates a COMM bug.",
          qpId.first,
          qpId.second,
          (void*)this,
          commHash,
          commDesc);
      return commInternalError;
    }
    map.emplace(qpId, vc);
    return commSuccess;
  }

  // Get a virtual connection for a given peer or QP.
  // If the peer is not yet connected, return nullptr.
  // For a returned VC, it is guaranteed to be ready to use.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline std::shared_ptr<CtranIbVirtualConn> getVcImpl(int peerRank) {
    if (PerfConfig::skipVcConnectionCheck ||
        (vcStateMapsPtr && isPeerConnected(peerRank))) {
      return vcStateMapsPtr->rankToVcMap.at(peerRank);
    } else {
      auto locked = vcStateMaps.rlock();

      auto it = locked->rankToVcMap.find(peerRank);
      if (it == locked->rankToVcMap.end()) {
        return nullptr;
      }

      return it->second;
    }
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline std::shared_ptr<CtranIbVirtualConn> getVcByQp(QpUniqueId qpUniqueId) {
    // lambda function to get VC from a given pointer
    auto getVcFrom = [&](const VcStateMaps* maybeLockedVcStateMapsPtr)
        -> std::shared_ptr<CtranIbVirtualConn> {
      auto it = maybeLockedVcStateMapsPtr->qpToVcMap.find(qpUniqueId);
      // VC should be already created and added to vcStateMaps. If not found, it
      // likely indicates a NCCL bug, e.g., not using CtranIb epoch lock
      if (it == maybeLockedVcStateMapsPtr->qpToVcMap.end()) {
        CLOGF(
            ERR,
            "CTRAN-IB: Received unknown QP number {} on IB device {} in pimpl {} commHash {:x}, commDesc {}. Known QPs: {}. It likely indicates a COMM bug.",
            qpUniqueId.first,
            qpUniqueId.second,
            (void*)this,
            commHash,
            commDesc,
            f14FastMapToStr(maybeLockedVcStateMapsPtr->qpToVcMap));
        return nullptr;
      }
      return it->second;
    };

    auto vcstatemaps = PerfConfig::skipVcConnectionCheck
        ? vcStateMapsPtr
        : &(*vcStateMaps.rlock());
    return getVcFrom(vcstatemaps);
  }

  // Check whether vc is ready and no outstanding pending ops
  inline bool addToPendingOpsIfRequired(
      std::optional<const int> type,
      void* payload,
      const size_t size,
      int peerRank,
      CtranIbRequest& req,
      bool isSend,
      CtranIbVirtualConn* vc) {
    bool pending = false;
    CTRAN_IB_PER_OBJ_LOCK_GUARD(pendingOpsMutex, {
      auto it = rankToPendingOpsMap.find(peerRank);

      // If VC is not established or there are pending ops, put the op into
      // pending queue, because we must issue all pending ops before we can
      // issue any ctrlMsg directly, otherwise ctrlMsg may mismatch.
      if (!vc || it != rankToPendingOpsMap.end()) {
        PendingOp::PendingOpType opType = isSend
            ? PendingOp::PendingOpType::ISEND_CTRL
            : PendingOp::PendingOpType::IRECV_CTRL;
        auto pendingOp = std::make_unique<PendingOp>(
            opType, type, payload, size, peerRank, req);
        CLOGF_TRACE(COLL, "Enqueue pendingOp [{}]", pendingOp->toString());

        if (it == rankToPendingOpsMap.end()) {
          // create a new entry for the peer if it does not exist and VC is not
          // established
          rankToPendingOpsMap[peerRank].push_back(std::move(pendingOp));
        } else {
          it->second.push_back(std::move(pendingOp));
        }
        pending = true;
      }
    });
    return pending;
  }

  inline commResult_t progressPendingOps(void) {
    std::unordered_map<int, std::vector<std::unique_ptr<PendingOp>>>
        readyToIssueOps;

    CTRAN_IB_PER_OBJ_LOCK_GUARD(pendingOpsMutex, {
      if (rankToPendingOpsMap.empty()) {
        return commSuccess;
      }

      // Try to find ready-to-issue pending ops for each peer
      // NOTE: we move ctrlMsg issue out of the critical section of pendingOps
      // to avoid potential deadlock with lock to vc. We expect pendingOps only
      // before initial connection, thus the extra move should not cause perf
      // overhead.
      for (auto it = rankToPendingOpsMap.begin();
           it != rankToPendingOpsMap.end();) {
        int peerRank = it->first;
        std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl(peerRank);
        // If VC has established, move the pendingOps list to readyToIssueOps;
        // otherwise, skip and move to next peer.
        if (vc) {
          for (auto& op : it->second) {
            readyToIssueOps[peerRank].push_back(std::move(op));
          }
          // Remove the peer entry once all pending ops are issued.
          // We should no longer have any pending ops for this peer, since new
          // entry would be created only when vc is not established. See
          // addToPendingOpsIfRequired().
          it = rankToPendingOpsMap.erase(it);
        } else {
          it++;
        }
      }
    });

    // Post ready-to-issue pending ops
    for (auto& [peerRank, pendingOps] : readyToIssueOps) {
      std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl(peerRank);
      CTRAN_IB_PER_OBJ_LOCK_GUARD(vc->mutex, {
        for (auto& op : pendingOps) {
          if (op->opType == PendingOp::PendingOpType::ISEND_CTRL) {
            FB_CHECKABORT(
                op->type.has_value(),
                "Expect a pending ISEND_CTRL has a valid type but is not specified. It indicates a COMM internal bug")
            FB_COMMCHECK(vc->isendCtrlMsg(
                op->type.value(), op->payload, op->size, op->req));
          } else {
            FB_COMMCHECK(vc->irecvCtrlMsg(op->payload, op->size, op->req));
          }
        }
      });
    }

    return commSuccess;
  }

  inline commResult_t
  exportMemImpl(const void* buf, void* ibRegElem, ControlMsg& msg) {
    msg.setType(ControlMsgType::IB_EXPORT_MEM);
    msg.ibExp.remoteAddr = reinterpret_cast<uint64_t>(buf);
    msg.ibExp.nKeys = NCCL_CTRAN_IB_DEVICES_PER_RANK;
    ctran::ib::getRemoteKeysImpl(ibRegElem, msg.ibExp.rkeys);

    return commSuccess;
  }

  inline commResult_t importMemImpl(
      void** buf,
      CtranIbRemoteAccessKey* key,
      const ControlMsg& msg) {
    (*buf) = reinterpret_cast<void*>(msg.ibExp.remoteAddr);
    for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
      key->rkeys[device] = msg.ibExp.rkeys[device];
    }
    key->nKeys = msg.ibExp.nKeys;
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t isendCtrlMsgImpl(
      const int type,
      const void* payload,
      const size_t size,
      int peerRank,
      std::optional<const SocketServerAddr*> peerServerAddr,
      CtranIbRequest& req) {
    FB_COMMCHECK(checkEpochLock(this));
    FB_COMMCHECK(checkValidPeer(peerRank));

    std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl<PerfConfig>(peerRank);

    // nullptr VC indicates not yet established connection; try to connect.
    // For smaller peerRank, wait peerRank connects to local listenThread.
    // The ctrlMsg will be enqueued to pendingOps and sent out when polling
    // progress.
    if (!PerfConfig::skipVcConnectionCheck && rank < peerRank &&
        vc == nullptr) {
      FB_COMMCHECK(bootstrapConnect(peerRank, peerServerAddr));
      // Get VC again after connection is established
      vc = getVcImpl<PerfConfig>(peerRank);
    }

    // Skip vc and pendingOps check if pre-connected
    if (PerfConfig::skipVcConnectionCheck || isPeerConnected(peerRank) ||
        !addToPendingOpsIfRequired(
            type,
            // FIXME: need refactor to keep the constness if possible
            (void*)payload,
            size,
            peerRank,
            req,
            true /*isSend*/,
            vc.get())) {
      CTRAN_IB_PER_OBJ_LOCK_GUARD(vc->mutex, {
        FB_COMMCHECK(vc->isendCtrlMsg(type, payload, size, req));
      });
    }
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t irecvCtrlMsgImpl(
      void* payload,
      const size_t size,
      int peerRank,
      std::optional<const SocketServerAddr*> peerServerAddr,
      CtranIbRequest& req) {
    FB_COMMCHECK(checkEpochLock(this));
    FB_COMMCHECK(checkValidPeer(peerRank));

    std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl<PerfConfig>(peerRank);

    // nullptr VC indicates not yet established connection; try to connect.
    // For smaller peerRank, wait peerRank connects to local listenThread.
    // The ctrlMsg will be enqueued to pendingOps and sent out when polling
    // progress.
    if (!PerfConfig::skipVcConnectionCheck && rank < peerRank &&
        vc == nullptr) {
      FB_COMMCHECK(bootstrapConnect(peerRank, peerServerAddr));
      // Get VC again after connection is established
      vc = getVcImpl<PerfConfig>(peerRank);
    }

    // Skip vc and pendingOps check if pre-connected
    if (PerfConfig::skipVcConnectionCheck || isPeerConnected(peerRank) ||
        !addToPendingOpsIfRequired(
            std::nullopt,
            payload,
            size,
            peerRank,
            req,
            false /*isSend*/,
            vc.get())) {
      CTRAN_IB_PER_OBJ_LOCK_GUARD(vc->mutex, {
        FB_COMMCHECK(vc->irecvCtrlMsg<PerfConfig>(payload, size, req));
      });
    }

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iputBatchImpl(
      const std::vector<PutIbMsg>& puts,
      int peerRank) {
    FB_COMMCHECK(checkEpochLock(this));

    std::shared_ptr<CtranIbVirtualConn> vc = getVc<PerfConfig>(peerRank);
    FB_COMMCHECK(checkValidVc(vc, peerRank));

    CTRAN_IB_PER_OBJ_LOCK_GUARD(
        vc->mutex, { FB_COMMCHECK(vc->iputBatch<PerfConfig>(puts)); });

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iputImpl(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int peerRank,
      void* ibRegElem,
      CtranIbRemoteAccessKey remoteAccessKey,
      bool notify,
      CtranIbConfig* config,
      CtranIbRequest* req,
      bool fast) {
    FB_COMMCHECK(checkEpochLock(this));

    std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl<PerfConfig>(peerRank);
    FB_COMMCHECK(checkValidVc(vc, peerRank));

    CTRAN_IB_PER_OBJ_LOCK_GUARD(vc->mutex, {
      FB_COMMCHECK(vc->iput<PerfConfig>(
          sbuf,
          dbuf,
          len,
          ibRegElem,
          remoteAccessKey,
          notify,
          config,
          req,
          fast));
    });

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t igetImpl(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int peerRank,
      void* ibRegElem,
      CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbConfig* config,
      CtranIbRequest* req,
      bool fast) {
    FB_COMMCHECK(checkEpochLock(this));

    std::shared_ptr<CtranIbVirtualConn> vc = getVc<PerfConfig>(peerRank);
    FB_COMMCHECK(checkValidVc(vc, peerRank));

    CTRAN_IB_PER_OBJ_LOCK_GUARD(vc->mutex, {
      FB_COMMCHECK(vc->iget<PerfConfig>(
          sbuf, dbuf, len, ibRegElem, remoteAccessKey, config, req, fast));
    });

    return commSuccess;
  }

  inline commResult_t ifetchAndAddImpl(
      const void* sbuf,
      void* dbuf,
      uint64_t addVal,
      int peerRank,
      void* ibRegElem,
      struct CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbRequest* req) {
    std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl(peerRank);
    FB_COMMCHECK(checkValidVc(vc, peerRank));
    CTRAN_IB_PER_OBJ_LOCK_GUARD(vc->mutex, {
      FB_COMMCHECK(vc->ifetchAndAdd(
          sbuf, dbuf, addVal, ibRegElem, remoteAccessKey, req));
    });
    return commSuccess;
  }

  inline commResult_t iatomicSetImpl(
      void* dbuf,
      uint64_t val,
      int peerRank,
      struct CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbRequest* req) {
    std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl(peerRank);
    FB_COMMCHECK(checkValidVc(vc, peerRank));
    CTRAN_IB_PER_OBJ_LOCK_GUARD(vc->mutex, {
      FB_COMMCHECK(vc->iatomicSet(dbuf, val, remoteAccessKey, req));
    });
    return commSuccess;
  }

  inline commResult_t notifyImpl(int peerRank, CtranIbRequest* req) {
    FB_COMMCHECK(checkEpochLock(this));

    std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl(peerRank);
    FB_COMMCHECK(checkValidVc(vc, peerRank));

    // Waits till the VC can transfer data before posting a put
    // Note this may happen only when too many no-signal puts have been
    // scheduled but not yet finished by VC, which rarely happens.
    while (!this->canTransfer(vc.get())) {
      FB_COMMCHECK(this->progressInternal());
    }

    CTRAN_IB_PER_OBJ_LOCK_GUARD(vc->mutex, { FB_COMMCHECK(vc->notify(req)); });

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t checkNotifyImpl(int peerRank, bool* notify) {
    FB_COMMCHECK(checkEpochLock(this));
    FB_COMMCHECK(checkValidPeer(peerRank));

    FB_COMMCHECK(this->progressInternal<PerfConfig>());

    std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl<PerfConfig>(peerRank);

    // VC may not be established yet, e.g., if the connection is established by
    // listenThread. Return and check again next time.
    if (vc) {
      CTRAN_IB_PER_OBJ_LOCK_GUARD(
          vc->mutex, { FB_COMMCHECK(vc->checkNotify(notify)); });
    }
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t waitNotifyImpl(int peerRank, int notifyCnt) {
    FB_COMMCHECK(checkEpochLock(this));
    FB_COMMCHECK(checkValidPeer(peerRank));

    std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl<PerfConfig>(peerRank);
    // VC may not be established yet, e.g., if the connection is established
    // by listenThread. Continue check till it is established.
    if (!PerfConfig::skipVcConnectionCheck) {
      while (vc == nullptr && !abortCtrl_->Test()) {
        vc = getVcImpl<PerfConfig>(peerRank);
      }
    }

    while (notifyCnt != 0 && !abortCtrl_->Test()) {
      FB_COMMCHECK(this->progressInternal<PerfConfig>());

      CTRAN_IB_PER_OBJ_LOCK_GUARD(
          vc->mutex, { FB_COMMCHECK(vc->checkNotifies(notifyCnt)); });
    }

    if (abortCtrl_->Test()) {
      // TODO(T238821628): re-evaluate error code
      throw ctran::utils::Exception("comm aborted", commRemoteError);
    }

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t progressInternal() {
    FB_COMMCHECK(checkEpochLock(this));
    /* complete as many requests as possible */
    while (1) {
      bool continueWhileLoop = false;
      for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
        ibverbx::ibv_wc wc;
        int count;

        CTRAN_IB_PER_OBJ_LOCK_GUARD(cqMutex, {
          auto maybeWcsVector = devices[device].ibvCq->pollCq(1);
          if (maybeWcsVector.hasError()) {
            CLOGF(
                WARN,
                "Call to pollCq() on device {} failed with error {}",
                device,
                maybeWcsVector.error().errStr);
            return commSystemError;
          }
          count = maybeWcsVector->size();
          if (count > 0) {
            wc = maybeWcsVector->at(0);
          }
        });

        if (count == 0) {
          continue;
        } else {
          continueWhileLoop = true;
        }

        if (enableLocalFlush) {
          CTRAN_IB_PER_OBJ_LOCK_GUARD(localVcMutex, {
            auto& vc = localVc;
            // First check if it is a local flush CQE
            if (wc.qp_num == vc->qpNum(device)) {
              CQE_ERROR_CHECK(wc, rank, "localFlush");
              FB_COMMCHECK(vc->processCqe(wc.opcode));
              continue;
            }
          });
        }

        // Next search from remote VCs.
        // Expect received CQE should be with a registered rank and established
        // VC
        std::shared_ptr<CtranIbVirtualConn> vc =
            getVcByQp<PerfConfig>(std::make_pair(wc.qp_num, device));
        if (vc == nullptr) {
          CLOGF(
              ERR,
              "No valid VirtualConnection (VC) found for qpn {}",
              wc.qp_num);
          return commInternalError;
        }
        CQE_ERROR_CHECK(wc, vc->peerRank, "remote");

        /* wc.wr_id is valid even if the poll_cq returned an error; use it
         * to gather information about the error */
        CTRAN_IB_PER_OBJ_LOCK_GUARD(vc->mutex, {
          FB_COMMCHECK(vc->processCqe<PerfConfig>(
              wc.opcode, wc.qp_num, wc.imm_data, wc.wr_id, device));
        });
      }
      if (!continueWhileLoop) {
        break;
      }
    }

    if (!PerfConfig::skipVcConnectionCheck) {
      FB_COMMCHECK(this->progressPendingOps());
    }
    return commSuccess;
  }

  // Use a global mutex to protect the entire mapper instance,
  // so we can get rid of per-object locks in critical path
  std::mutex epochMutex;
  std::mutex cqMutex;

  int maxCqe{-1};
  std::vector<CtranIbDevice> devices; // IB device info
  std::vector<ibverbx::IbvCq> cqs;
  int cudaDev{-1};
  int rank{-1};
  uint64_t commHash{0};
  std::string commDesc;
  CommLogData ncclLogData;
  bool enableLocalFlush{true};
  BootstrapMode bootstrapMode{BootstrapMode::kDefaultServer};

  std::shared_ptr<ctran::bootstrap::ISocketFactory> socketFactory_;

  // bitmap to indicate whether a peer is connected.
  // note that only one thread access it (e.g., GPE thread) or the epoch lock
  // needs to be acquired.
  std::vector<bool> connectedPeerMap;

  std::unique_ptr<ctran::bootstrap::IServerSocket> listenSocket;
  std::vector<sockaddr_storage> allListenSocketAddrs{};
  std::thread listenThread;

  // Virtual connection status for all peers.
  // NOTE: when using a VC, additional per-VC lock is required.
  struct VcStateMaps {
    folly::F14FastMap<QpUniqueId, std::shared_ptr<CtranIbVirtualConn>>
        qpToVcMap;
    folly::F14FastMap<int, std::shared_ptr<CtranIbVirtualConn>> rankToVcMap;
  };
  // VCs that are created but not yet connected
  std::unordered_map<int, std::shared_ptr<CtranIbVirtualConn>> pendingVcs_;

  folly::Synchronized<VcStateMaps> vcStateMaps;
  // Lock-free struct to be used in eager connect mode which gurantees
  // single-threaded access
  VcStateMaps* vcStateMapsPtr{nullptr};

  // Local virtual connection for local flush.
  std::unique_ptr<::ctran::ib::LocalVirtualConn> localVc;
  std::mutex localVcMutex;

  folly::F14FastMap<int, std::deque<std::unique_ptr<PendingOp>>>
      rankToPendingOpsMap;
  std::mutex pendingOpsMutex;

  CtranCtrlManager* ctrlMgr{nullptr};

  std::unordered_map<std::string, uint32_t> pgToTrafficClassMap_;

  std::shared_ptr<::ctran::utils::Abort> abortCtrl_{nullptr};
};

// Convenient RAII class to guard CtranIb epoch lock.
class CtranIbEpochRAII {
 public:
  // Allow set ctranIb to nullptr to skip lock. It can be used when lock is
  // needed only for selected cases.
  explicit CtranIbEpochRAII(CtranIb* ctranIb) : ctranIb_(ctranIb) {
    if (ctranIb_ != nullptr) {
      FB_COMMCHECKTHROW(ctranIb_->epochLock());
    }
  }

  ~CtranIbEpochRAII() {
    if (ctranIb_ != nullptr) {
      FB_COMMCHECKIGNORE(ctranIb_->epochUnlock());
    }
  }

 private:
  CtranIb* ctranIb_{nullptr};
};

#endif
