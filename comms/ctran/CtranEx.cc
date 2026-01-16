// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranEx.h"

#include <cstdint>
#include <vector>

#include <fmt/core.h>
#include <folly/ScopeGuard.h>

#include "comms/ctran/CtranExImpl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/utils/logger/LogUtils.h"

#include "comms/ctran/utils/LogInit.h"

namespace ctran {
std::string CtranExHostInfo::toString() const {
  return fmt::format(
      "[port={}, ipv6={}, hostname={}, ifName={}]",
      port,
      ipv6,
      hostname,
      ifName);
}

namespace {
std::once_flag initOnceFlag;
} // namespace

void initEnvCtranEx() {
  std::call_once(initOnceFlag, [] {
    ncclCvarInit();
    ctran::logging::initCtranLogging();
  });
}

CtranExImpl::CtranExImpl(int rank, int cudaDev, const std::string& desc)
    : rank(rank), cudaDev(cudaDev), desc(desc) {
  ctrlMgr = std::make_unique<CtranCtrlManager>();
};

void CtranExImpl::initialize(
    std::optional<const CtranExHostInfo*> hostInfo,
    const std::vector<CtranExBackend>& backends) {
  auto bootstrapMode = CtranIb::BootstrapMode::kExternal;
  SocketServerAddr serverAddr;
  std::optional<const SocketServerAddr*> serverAddrOpt = std::nullopt;
  // If hostInfo is sepicified, launch ctranIb internal bootstrap server with
  // it; otherwise assuming callsite will take care bootstrap explictly.

  // FIXME: now we convert user facing CtranExHostInfo to internal
  // SocketServerAddr, then CtranIb internally converts SocketServerAddr to
  // folly::SocketAddress + ifname. The two-step convertion is redundant, and
  // we'd directly pass in the final format to CtranIb.
  if (hostInfo.has_value()) {
    auto info = hostInfo.value();
    serverAddr.port = info->port;
    serverAddr.ipv6 = info->ipv6;
    serverAddr.hostname = info->hostname;
    serverAddr.ifName = info->ifName;
    bootstrapMode = CtranIb::BootstrapMode::kSpecifiedServer;
    serverAddrOpt = &serverAddr;
  }

  for (auto backend : backends) {
    switch (backend) {
      case CtranExBackend::kCtranIbBackend:
        try {
          ctranIb = std::make_unique<CtranIb>(
              rank,
              cudaDev,
              0,
              desc,
              ctrlMgr.get(),
              true, /* enableLocalFlush */
              bootstrapMode,
              serverAddrOpt);
        } catch (const std::bad_alloc& e) {
          CLOGF(WARN, "CTRAN-EX: IB backend not enabled");
          throw e;
        }
        break;
      default:
        CLOGF(WARN, "CTRAN-EX: Unknown backend {}", backend);
        throw ctran::utils::Exception(
            fmt::format("Unknown backend {}", backend), commInvalidArgument);
    };
  }
}

CtranEx::CtranEx(
    const int rank,
    const int cudaDev,
    const CtranExHostInfo& hostInfo,
    const std::vector<CtranExBackend> backends,
    const std::string& desc) {
  // Ensure NCCL logging / cvar to be initialized even without communicator
  initEnvCtranEx();

  auto impl = new CtranExImpl(rank, cudaDev, desc);
  auto guard = folly::makeGuard([impl] { delete impl; });

  impl->initialize(&hostInfo, backends);
  impl_ = reinterpret_cast<void*>(impl);
  guard.dismiss();

  std::vector<std::string> backendStrs;
  backendsToStr(backends, backendStrs);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-EX: Initialized CtranEx transport {} with rank {}, cudaDev {}, hostInfo {}, backend [{}], desc {}",
      (void*)this,
      rank,
      cudaDev,
      hostInfo.toString(),
      vecToStr(backendStrs),
      desc);
}

CtranEx::~CtranEx() {
  delete reinterpret_cast<CtranExImpl*>(impl_);
  CLOGF_SUBSYS(
      INFO, INIT, "CTRAN-EX: Destroyed CtranEx transport {}", (void*)this);
}

bool CtranEx::isInitialized() {
  CtranExImpl* impl = reinterpret_cast<CtranExImpl*>(impl_);
  return impl && impl->ctranIb;
}

#define GET_VALID_IMPL(impl)                               \
  do {                                                     \
    if (!isInitialized()) {                                \
      CLOGF(ERR, "CTRAN-EX: instance is not initialized"); \
      return commInvalidUsage;                             \
    }                                                      \
    impl = reinterpret_cast<CtranExImpl*>(impl_);          \
  } while (0);

#define CHECK_VALID_PEER(impl, peerRank)              \
  do {                                                \
    if (impl->rank == peerRank) {                     \
      CLOGF(                                          \
          ERR,                                        \
          "CTRAN-EX: invalid peerRank {} == rank {}", \
          peerRank,                                   \
          impl->rank);                                \
      return commInvalidUsage;                        \
    }                                                 \
  } while (0);

commResult_t
CtranEx::regMem(const void* ptr, const size_t size, void** regHdl) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);

  FB_COMMCHECK(impl->ctranIb->regMem(ptr, size, impl->cudaDev, regHdl));

  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-EX: regMem ptr {} size {} to CtranEx transport {}, return regHdl {}",
      ptr,
      size,
      (void*)this,
      *regHdl);
  return commSuccess;
}

commResult_t CtranEx::deregMem(void* regHdl) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);

  FB_COMMCHECK(impl->ctranIb->deregMem(regHdl));

  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-EX: deregMem regHdl {} from CtranEx transport {}",
      regHdl,
      (void*)this);
  return commSuccess;
}

commResult_t CtranEx::isendCtrl(
    const void* buf,
    const size_t size,
    const void* bufRegHdl,
    const int peerRank,
    const CtranExHostInfo& peerHostInfo,
    CtranExRequest** req) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);
  CHECK_VALID_PEER(impl, peerRank);

  const SocketServerAddr peerAddr = {
      .port = peerHostInfo.port,
      .ipv6 = peerHostInfo.ipv6,
      .hostname = peerHostInfo.hostname,
      .ifName = peerHostInfo.ifName,
  };

  auto* reqPtr = new CtranExRequest();
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(reqPtr->impl_);
  reqImpl->initialize(CtranExRequestImpl::SEND_CTRL, impl->ctranIb.get());

  CtranIbEpochRAII epochRAII(impl->ctranIb.get());
  // Fill in ctrl msg
  FB_COMMCHECK(
      CtranIb::exportMem(
          buf, const_cast<void*>(bufRegHdl), reqImpl->sendCtrl.msg));

  FB_COMMCHECK(impl->ctranIb->isendCtrlMsg(
      reqImpl->sendCtrl.msg.type,
      &reqImpl->sendCtrl.msg,
      sizeof(ControlMsg),
      peerRank,
      reqImpl->ibReq,
      &peerAddr));
  *req = reqPtr;

  CLOGF_TRACE(
      COLL,
      "isendCtrl peerRank {}, peerHostInfo {}, bufRegHdl {} from CtranEx transport {}, return req {}",
      peerRank,
      peerHostInfo.toString(),
      bufRegHdl,
      (void*)this,
      (void*)reqPtr);

  return commSuccess;
}

commResult_t CtranEx::irecvCtrl(
    const int peerRank,
    const CtranExHostInfo& peerHostInfo,
    void** peerRemoteBuf,
    uint32_t* peerRemoteKey,
    CtranExRequest** req) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);
  CHECK_VALID_PEER(impl, peerRank);

  const SocketServerAddr peerAddr = {
      .port = peerHostInfo.port,
      .ipv6 = peerHostInfo.ipv6,
      .hostname = peerHostInfo.hostname,
      .ifName = peerHostInfo.ifName,
  };

  // Cache user provided rKey/rBuf pointers, will be updated upon request
  // completion.
  auto* reqPtr = new CtranExRequest();
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(reqPtr->impl_);
  reqImpl->initialize(CtranExRequestImpl::RECV_CTRL, impl->ctranIb.get());
  reqImpl->recvCtrl.rKey = peerRemoteKey;
  reqImpl->recvCtrl.rBuf = peerRemoteBuf;

  CtranIbEpochRAII epochRAII(impl->ctranIb.get());
  FB_COMMCHECK(impl->ctranIb->irecvCtrlMsg(
      &reqImpl->recvCtrl.msg,
      sizeof(ControlMsg),
      peerRank,
      reqImpl->ibReq,
      &peerAddr));
  *req = reqPtr;

  CLOGF_TRACE(
      COLL,
      "irecvCtrl peerRank {}, peerHostInfo {}, peerRemoteBuf {}, peerRemoteKey {} from CtranEx transport {}, return req {}",
      peerRank,
      peerHostInfo.toString(),
      (void*)peerRemoteBuf,
      (void*)peerRemoteKey,
      (void*)this,
      (void*)reqPtr);
  return commSuccess;
}

commResult_t CtranEx::isendCtrl(
    const int peerRank,
    const CtranExHostInfo& peerHostInfo,
    CtranExRequest** req) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);
  CHECK_VALID_PEER(impl, peerRank);

  const SocketServerAddr peerAddr = {
      .port = peerHostInfo.port,
      .ipv6 = peerHostInfo.ipv6,
      .hostname = peerHostInfo.hostname,
      .ifName = peerHostInfo.ifName,
  };

  auto* reqPtr = new CtranExRequest();
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(reqPtr->impl_);
  reqImpl->initialize(CtranExRequestImpl::SEND_SYNC_CTRL, impl->ctranIb.get());
  reqImpl->sendSyncCtrl.msg.setType(ControlMsgType::SYNC);

  CtranIbEpochRAII epochRAII(impl->ctranIb.get());
  FB_COMMCHECK(impl->ctranIb->isendCtrlMsg(
      reqImpl->sendSyncCtrl.msg.type,
      &reqImpl->sendSyncCtrl.msg,
      sizeof(ControlMsg),
      peerRank,
      reqImpl->ibReq,
      &peerAddr));
  *req = reqPtr;

  CLOGF_TRACE(
      COLL,
      "isendCtrl peerRank {}, peerHostInfo {}, from CtranEx transport {}, return req {}",
      peerRank,
      peerHostInfo.toString(),
      (void*)this,
      (void*)reqPtr);
  return commSuccess;
}

commResult_t CtranEx::irecvCtrl(
    const int peerRank,
    const CtranExHostInfo& peerHostInfo,
    CtranExRequest** req) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);
  CHECK_VALID_PEER(impl, peerRank);

  const SocketServerAddr peerAddr = {
      .port = peerHostInfo.port,
      .ipv6 = peerHostInfo.ipv6,
      .hostname = peerHostInfo.hostname,
      .ifName = peerHostInfo.ifName,
  };

  auto* reqPtr = new CtranExRequest();
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(reqPtr->impl_);
  reqImpl->initialize(CtranExRequestImpl::RECV_SYNC_CTRL, impl->ctranIb.get());
  reqImpl->recvSyncCtrl.msg.setType(ControlMsgType::SYNC);

  CtranIbEpochRAII epochRAII(impl->ctranIb.get());
  FB_COMMCHECK(impl->ctranIb->irecvCtrlMsg(
      &reqImpl->recvSyncCtrl.msg,
      sizeof(ControlMsg),
      peerRank,
      reqImpl->ibReq,
      &peerAddr));
  *req = reqPtr;

  CLOGF_TRACE(
      COLL,
      "irecvCtrl(SYNC) peerRank {}, peerHostInfo {}, return req {}",
      peerRank,
      peerHostInfo.toString(),
      (void*)*req);
  return commSuccess;
}

commResult_t CtranEx::isendCtrl(const int peerRank, CtranExRequest** req) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);
  CHECK_VALID_PEER(impl, peerRank);

  auto* reqPtr = new CtranExRequest();
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(reqPtr->impl_);
  reqImpl->initialize(CtranExRequestImpl::SEND_SYNC_CTRL, impl->ctranIb.get());
  reqImpl->sendSyncCtrl.msg.setType(ControlMsgType::SYNC);

  CtranIbEpochRAII epochRAII(impl->ctranIb.get());
  FB_COMMCHECK(impl->ctranIb->isendCtrlMsg(
      reqImpl->sendSyncCtrl.msg.type,
      &reqImpl->sendSyncCtrl.msg,
      sizeof(ControlMsg),
      peerRank,
      reqImpl->ibReq));
  *req = reqPtr;

  CLOGF_TRACE(
      COLL,
      "isendCtrl(SYNC) peerRank {}, return req {}",
      peerRank,
      (void*)*req);
  return commSuccess;
}

commResult_t CtranEx::irecvCtrl(const int peerRank, CtranExRequest** req) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);
  CHECK_VALID_PEER(impl, peerRank);

  auto* reqPtr = new CtranExRequest();
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(reqPtr->impl_);
  reqImpl->initialize(CtranExRequestImpl::RECV_SYNC_CTRL, impl->ctranIb.get());
  reqImpl->recvSyncCtrl.msg.setType(ControlMsgType::SYNC);

  CtranIbEpochRAII epochRAII(impl->ctranIb.get());
  FB_COMMCHECK(impl->ctranIb->irecvCtrlMsg(
      &reqImpl->recvSyncCtrl.msg,
      sizeof(ControlMsg),
      peerRank,
      reqImpl->ibReq));
  *req = reqPtr;

  CLOGF_TRACE(
      COLL,
      "irecvCtrl(SYNC) peerRank {}, return req {}",
      peerRank,
      (void*)*req);
  return commSuccess;
}

commResult_t CtranEx::iput(
    const void* localBuf,
    const std::size_t len,
    void* localRegHdl,
    const int peerRank,
    void* peerRemoteBuf,
    uint32_t peerRemoteKey,
    bool notify,
    CtranExRequest** req) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);
  CHECK_VALID_PEER(impl, peerRank);

  auto* reqPtr = new CtranExRequest();
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(reqPtr->impl_);
  reqImpl->initialize(CtranExRequestImpl::PUT, impl->ctranIb.get());

  // FIXME: need peerRemoteKey to be vector. For now, just use rkey of first
  // device
  CtranIbEpochRAII epochRAII(impl->ctranIb.get());
  CtranIbRemoteAccessKey remoteAccessKey{};
  remoteAccessKey.rkeys[0] = peerRemoteKey;
  FB_COMMCHECK(impl->ctranIb->iput(
      localBuf,
      peerRemoteBuf,
      len,
      peerRank,
      localRegHdl,
      remoteAccessKey,
      notify,
      nullptr,
      &reqImpl->ibReq));
  *req = reqPtr;

  CLOGF_TRACE(
      COLL,
      "iput localBuf {} len {} localRegHdl {} peerRank {} peerRemoteBuf {} peerRemoteKey {} notify {} from CtranEx transport {}, return req {}",
      localBuf,
      len,
      localRegHdl,
      peerRank,
      peerRemoteBuf,
      peerRemoteKey,
      notify,
      (void*)this,
      (void*)reqPtr);
  return commSuccess;
}

commResult_t
CtranEx::iflush(const void* localBuf, void* localRegHdl, CtranExRequest** req) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);

  auto* reqPtr = new CtranExRequest();
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(reqPtr->impl_);
  reqImpl->initialize(CtranExRequestImpl::FLUSH, impl->ctranIb.get());

  CtranIbEpochRAII epochRAII(impl->ctranIb.get());
  FB_COMMCHECK(impl->ctranIb->iflush(localBuf, localRegHdl, &reqImpl->ibReq));
  *req = reqPtr;

  CLOGF_TRACE(COLL, "iflush, return req {}", (void*)*req);
  return commSuccess;
}

commResult_t CtranEx::checkNotify(int peerRank, bool& done) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);
  CHECK_VALID_PEER(impl, peerRank);

  CtranIbEpochRAII epochRAII(impl->ctranIb.get());
  FB_COMMCHECK(impl->ctranIb->checkNotify(peerRank, &done));

  CLOGF_TRACE(
      COLL,
      "checkNotify peerRank {} from CtranEx transport {}, return done {}",
      peerRank,
      (void*)this,
      done);
  return commSuccess;
}

commResult_t CtranEx::waitNotify(int peerRank) {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);
  CHECK_VALID_PEER(impl, peerRank);

  CtranIbEpochRAII epochRAII(impl->ctranIb.get());
  FB_COMMCHECK(impl->ctranIb->waitNotify(peerRank));

  CLOGF_TRACE(
      COLL,
      "waitNotify peerRank {} from CtranEx transport {}",
      peerRank,
      (void*)this);
  return commSuccess;
}

commResult_t CtranEx::releaseRemoteTransStates() {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);

  // No need explicit lock here, as it's already locked inside
  FB_COMMCHECK(impl->ctranIb->releaseRemoteTransStates());

  CLOGF_TRACE(
      COLL,
      "CtranEx transport {} released remote transport states",
      (void*)this);
  return commSuccess;
}

commResult_t CtranEx::initRemoteTransStates() {
  CtranExImpl* impl = nullptr;
  GET_VALID_IMPL(impl);

  // No need explicit lock here, as it's already locked inside
  FB_COMMCHECK(impl->ctranIb->initRemoteTransStates());

  CLOGF_TRACE(
      COLL,
      "CtranEx transport {} initialized remote transport states",
      (void*)this);
  return commSuccess;
}

} // namespace ctran
