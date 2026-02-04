// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/utils/cvars/nccl_cvars.h"

commResult_t ctran::IpcRegCache::regMem(
    const void* buf,
    const size_t len,
    const int cudaDev,
    void** ipcRegElem,
    bool shouldSupportCudaMalloc) {
  auto reg = new ctran::regcache::IpcRegElem(buf, len, cudaDev);
  bool supported = false;

  FB_COMMCHECK(reg->tryLoad(supported, shouldSupportCudaMalloc));
  if (supported) {
    CLOGF_TRACE(
        COLL, "CTRAN-REGCACHE: Registered IPC memory {}", reg->toString());

    *ipcRegElem = reinterpret_cast<void*>(reg);
  } else {
    // Return nullptr to indicate unsupported memory type
    delete reg;
    *ipcRegElem = nullptr;
  }

  return commSuccess;
}

void ctran::IpcRegCache::deregMem(void* ipcRegElem) {
  auto reg = reinterpret_cast<ctran::regcache::IpcRegElem*>(ipcRegElem);

  CLOGF_TRACE(
      COLL, "CTRAN-REGCACHE: Deregistered IPC memory {}", reg->toString());
  // Memory handle release in ~CtranIpcMem()
  delete reg;
}

void ctran::IpcRegCache::remReleaseMem(
    void* ipcRegElem,
    ctran::regcache::IpcRelease& ipcRelease) {
  auto reg = reinterpret_cast<ctran::regcache::IpcRegElem*>(ipcRegElem);
  ipcRelease.base = reg->ipcMem.rlock()->getBase();
}

ctran::IpcRegCache::IpcRegCache() : cudaDev_(0), logMetaData_(nullptr) {}

void ctran::IpcRegCache::init(
    int cudaDev,
    const struct CommLogData* logMetaData) {
  cudaDev_ = cudaDev;
  logMetaData_ = logMetaData;
  initAsyncSocket();
}

ctran::IpcRegCache::~IpcRegCache() {
  stopAsyncSocket();
  clearAllRemReg();
}

commResult_t ctran::IpcRegCache::importMem(
    const std::string& peerId,
    const ctran::regcache::IpcDesc& ipcDesc,
    void** buf,
    struct ctran::regcache::IpcRemHandle* remKey) {
  void* basePtr = nullptr;
  FB_COMMCHECK(importRemMemImpl(peerId, ipcDesc.desc, &basePtr));

  // import from baseAddr of a remote segment, return buf at offset from
  // baseAddr
  *buf = reinterpret_cast<char*>(basePtr) + ipcDesc.offset;
  remKey->peerId = peerId;
  remKey->basePtr = ipcDesc.desc.base;
  CLOGF_TRACE(
      COLL,
      "CTRAN-REGCACHE: Imported NVL remote mem from peer {}: buf {} (base {} offset {})",
      peerId,
      (void*)*buf,
      (void*)basePtr,
      ipcDesc.offset);
  return commSuccess;
}

commResult_t ctran::IpcRegCache::importRemMemImpl(
    const std::string& peerId,
    const ctran::utils::CtranIpcDesc& ipcDesc,
    void** mappedBase) {
  auto lockedMap = ipcRemRegMap_.wlock();
  uint64_t base = reinterpret_cast<uint64_t>(ipcDesc.base);

  // Check if already mapped
  auto peerIt = lockedMap->find(peerId);
  if (peerIt != lockedMap->end()) {
    auto baseIt = peerIt->second.find(base);
    if (baseIt != peerIt->second.end()) {
      *mappedBase = baseIt->second->ipcRemMem.getBase();
      return commSuccess;
    }
  }

  std::unique_ptr<ctran::regcache::IpcRemRegElem> reg = nullptr;
  try {
    reg = std::make_unique<ctran::regcache::IpcRemRegElem>(
        ipcDesc, cudaDev_, logMetaData_);
  } catch (std::exception& e) {
    CLOGF(
        WARN,
        "CTRAN-REGCACHE: failed to import IPC remote registration from peer {} ipcDesc {}, error {}",
        peerId,
        ipcDesc.toString(),
        e.what());
    return ErrorStackTraceUtil::log(commInternalError);
  }

  CLOGF_TRACE(
      COLL,
      "CTRAN-REGCACHE: cache IPC remote registration peer:base=<{}:{}> {}",
      peerId,
      reinterpret_cast<void*>(ipcDesc.base),
      reg->toString());

  *mappedBase = reg->ipcRemMem.getBase();
  (*lockedMap)[peerId][base] = std::move(reg);

  return commSuccess;
}

commResult_t ctran::IpcRegCache::releaseRemReg(
    const std::string& peerId,
    void* basePtr) {
  auto lockedMap = ipcRemRegMap_.wlock();
  uint64_t base = reinterpret_cast<uint64_t>(basePtr);

  if (lockedMap->find(peerId) == lockedMap->end() ||
      (*lockedMap)[peerId].find(base) == (*lockedMap)[peerId].end()) {
    CLOGF(
        ERR,
        "CTRAN-REGCACHE: Unknown IPC remote memory registration from peer {} base {}",
        peerId,
        basePtr);
    return ErrorStackTraceUtil::log(commInternalError);
  }

  CLOGF_TRACE(
      COLL,
      "CTRAN-REGCACHE: remove IPC remote registration from cache peer:base=<{}:{}> : {}",
      peerId,
      basePtr,
      (*lockedMap)[peerId][base]->toString());

  try {
    (*lockedMap)[peerId].erase(base);
  } catch (std::exception& e) {
    CLOGF(
        WARN,
        "CTRAN-REGCACHE: failed to remove IPC remote registration from cache peer:base=<{}:{}>, error {}",
        peerId,
        basePtr,
        e.what());
    return ErrorStackTraceUtil::log(commInternalError);
  }

  return commSuccess;
}

void ctran::IpcRegCache::clearAllRemReg() {
  auto lockedMap = ipcRemRegMap_.wlock();

  for (auto& [peerId, regs] : *lockedMap) {
    CLOGF_TRACE(
        INIT,
        "CTRAN-REGCACHE: clear all {} cached IPC remote registrations from peer {}",
        regs.size(),
        peerId);
  }

  // Memory and handle will be released in ~CtranIpcRemMem()
  lockedMap->clear();
}

size_t ctran::IpcRegCache::getNumRemReg(const std::string& peerId) const {
  auto lockedMap = ipcRemRegMap_.rlock();

  auto it = lockedMap->find(peerId);
  if (it != lockedMap->end()) {
    return it->second.size();
  }
  return 0;
}

commResult_t ctran::IpcRegCache::notifyRemoteIpcRelease(
    const std::string& myId,
    const folly::SocketAddress& peerAddr,
    ctran::regcache::IpcRegElem* ipcRegElem,
    ctran::regcache::IpcReqCb* reqCb) {
  // Check if AsyncSocket is initialized
  if (!asyncSocketEvbThread_ || !asyncServerSocket_) {
    CLOGF(
        WARN,
        "CTRAN-REGCACHE: AsyncSocket not initialized, skipping remote release");
    return commInternalError;
  }

  // Initialize the request
  reqCb->req =
      ctran::regcache::IpcReq(ctran::regcache::IpcReqType::kRelease, myId);
  reqCb->completed.store(false);
  remReleaseMem(ipcRegElem, reqCb->req.release);

  CLOGF_TRACE(
      COLL,
      "CTRAN-REGCACHE: Sending IPC_RELEASE to peerAddr {}: {}",
      peerAddr.describe(),
      reqCb->req.toString());

  // Send the whole IpcReq via AsyncClientSocket
  // The peer checks IpcReqType to determine which callback to invoke
  ctran::bootstrap::AsyncClientSocket::send(
      *asyncSocketEvbThread_->getEventBase(),
      peerAddr,
      &reqCb->req,
      sizeof(ctran::regcache::IpcReq),
      [reqCb, peerAddr](const folly::AsyncSocketException* err) {
        if (err != nullptr) {
          CLOGF(
              WARN,
              "CTRAN-REGCACHE: Failed to send IpcReq to peerAddr {}: {}",
              peerAddr.describe(),
              err->what());
        }
        // Mark as completed regardless of success/failure
        reqCb->completed.store(true);
      });

  return commSuccess;
}

commResult_t ctran::IpcRegCache::notifyRemoteIpcExport(
    const std::string& myId,
    const folly::SocketAddress& peerAddr,
    const ctran::regcache::IpcDesc& ipcDesc,
    ctran::regcache::IpcReqCb* reqCb) {
  // Check if AsyncSocket is initialized
  if (!asyncSocketEvbThread_ || !asyncServerSocket_) {
    CLOGF(
        WARN,
        "CTRAN-REGCACHE: AsyncSocket not initialized, skipping remote export");
    return commInternalError;
  }

  // Initialize the request
  reqCb->req =
      ctran::regcache::IpcReq(ctran::regcache::IpcReqType::kDesc, myId);
  reqCb->completed.store(false);
  reqCb->req.desc = ipcDesc;

  CLOGF_TRACE(
      COLL,
      "CTRAN-REGCACHE: Sending IPC_DESC to peerAddr {}: {}",
      peerAddr.describe(),
      reqCb->req.toString());

  // Send the whole IpcReq via AsyncClientSocket
  // The peer checks IpcReqType and calls importMem for kDesc
  ctran::bootstrap::AsyncClientSocket::send(
      *asyncSocketEvbThread_->getEventBase(),
      peerAddr,
      &reqCb->req,
      sizeof(ctran::regcache::IpcReq),
      [reqCb, peerAddr](const folly::AsyncSocketException* err) {
        if (err != nullptr) {
          CLOGF(
              WARN,
              "CTRAN-REGCACHE: Failed to send IpcReq (DESC) to peerAddr {}: {}",
              peerAddr.describe(),
              err->what());
        }
        // Mark as completed regardless of success/failure
        reqCb->completed.store(true);
      });

  return commSuccess;
}

commResult_t ctran::IpcRegCache::initAsyncSocket() {
  // Create the event base thread for async socket operations
  asyncSocketEvbThread_ = std::make_unique<folly::ScopedEventBaseThread>();

  // Create and start the async server socket
  asyncServerSocket_ = std::make_unique<ctran::bootstrap::AsyncServerSocket>(
      *asyncSocketEvbThread_->getEventBase());

  // Start the server with a callback that handles IPC requests
  // The peer sends the whole IpcReq, and we check the type to dispatch
  auto serverAddrFuture = asyncServerSocket_->start(
      folly::SocketAddress("::1", 0),
      sizeof(ctran::regcache::IpcReq),
      [this](std::unique_ptr<folly::IOBuf> buf) {
        // Extract the IpcReq from the received buffer
        ctran::regcache::IpcReq ipcReq;
        std::memcpy(&ipcReq, buf->data(), sizeof(ipcReq));

        // Dispatch based on request type
        switch (ipcReq.type) {
          case ctran::regcache::IpcReqType::kRelease: {
            // Handle release request - release the imported NVL memory
            std::string peerId = ipcReq.getPeerId();

            CLOGF_TRACE(
                COLL,
                "CTRAN-REGCACHE: Received IPC_RELEASE from peer {}, base {}",
                peerId,
                ipcReq.release.base);

            FB_COMMCHECKIGNORE(releaseRemReg(peerId, ipcReq.release.base));
            break;
          }
          case ctran::regcache::IpcReqType::kDesc: {
            // Handle descriptor request - import the remote memory
            std::string peerId = ipcReq.getPeerId();
            void* buf = nullptr;
            ctran::regcache::IpcRemHandle remKey;

            CLOGF_TRACE(
                COLL,
                "CTRAN-REGCACHE: Received IPC_DESC from peer {}: {}",
                peerId,
                ipcReq.desc.toString());

            FB_COMMCHECKIGNORE(importMem(peerId, ipcReq.desc, &buf, &remKey));
            break;
          }
        }
      });

  // Get the server address
  serverAddr_ = std::move(serverAddrFuture).get();

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-REGCACHE: AsyncSocket server started at {}",
      serverAddr_.describe());

  return commSuccess;
}

void ctran::IpcRegCache::stopAsyncSocket() {
  if (asyncServerSocket_) {
    auto fut = asyncServerSocket_->stop();
    std::move(fut).get();
    asyncServerSocket_.reset();
    CLOGF_SUBSYS(INFO, INIT, "CTRAN-REGCACHE: AsyncSocket server stopped");
  }

  if (asyncSocketEvbThread_) {
    asyncSocketEvbThread_.reset();
  }
}
