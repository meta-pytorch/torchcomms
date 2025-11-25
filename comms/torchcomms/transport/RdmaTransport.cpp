// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "RdmaTransport.h"

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/LogInit.h"

namespace {

constexpr std::chrono::microseconds kProgressInterval{0};
constexpr int kDummyRank = 0;
constexpr int kDummyDevice = 0;

std::once_flag initOnceFlag;
void initEnvironment() {
  std::call_once(initOnceFlag, [] {
    ncclCvarInit();
    ctran::logging::initCtranLogging();
    ctran::utils::commCudaLibraryInit();
  });
}

} // namespace

namespace torch::comms {

RdmaMemory::RdmaMemory(const void* buf, size_t len, int cudaDev)
    : buf_(buf), len_(len), cudaDev_(cudaDev) {
  initEnvironment();
  FB_COMMCHECKTHROW(CtranIb::regMem(buf_, len_, cudaDev_, &regHdl_));
  remoteKey_ = CtranIb::getRemoteAccessKey(regHdl_).toString();
}

RdmaMemory::~RdmaMemory() {
  if (remoteKey_.size()) {
    FB_COMMCHECKTHROW(CtranIb::deregMem(regHdl_));
  }
}

bool RdmaMemory::contains(const void* buf, size_t len) const {
  return (buf_ <= buf) && (((uint8_t*)buf + len) <= ((uint8_t*)buf_ + len_));
}

struct RdmaTransport::Work {
  enum class Type { Write, Read, WaitForWrite };
  Type type{Type::Write};
  CtranIbRequest ibReq;
  folly::Promise<commResult_t> promise;
};

RdmaTransport::RdmaTransport(int cudaDev, folly::EventBase* evb)
    : cudaDev_(cudaDev), evb_(evb) {
  initEnvironment();
  // Create IB Instance
  ib_ = std::make_unique<CtranIb>(
      kDummyRank,
      cudaDev,
      -1 /* commHash */,
      "RDMA-Transport",
      nullptr /* ctrlManager */,
      true /* enableLocalFlush */,
      CtranIb::BootstrapMode::kExternal);

  if (evb_) {
    // Optionally create progress timeout; skip it if the transport is never
    // used for remote RDMA
    progressTimeout_ =
        folly::AsyncTimeout::make(*evb_, [this]() noexcept { progress(); });
  }
}

RdmaTransport::~RdmaTransport() {}

namespace {
std::once_flag queryRdmaSupportOnceFlag;
bool rdmaSupport = false;
bool queryRdmaSupport() {
  std::call_once(queryRdmaSupportOnceFlag, [] {
    XLOG(INFO) << "Querying RdmaTransport support";
    try {
      auto ib = std::make_unique<CtranIb>(
          kDummyRank,
          kDummyDevice,
          -1 /* commHash */,
          "Query-RDMA-Support",
          nullptr /* ctrlManager */,
          true /* enableLocalFlush */,
          CtranIb::BootstrapMode::kExternal);
    } catch (const std::exception& e) {
      XLOG(WARN)
          << "RdmaTransport is not supported. Failed to create CtranIb instance: "
          << e.what();
      rdmaSupport = false;
      return;
    }
    XLOG(INFO) << "RdmaTransport is supported";
    rdmaSupport = true;
  });
  return rdmaSupport;
}
} // namespace

bool RdmaTransport::supported() {
  initEnvironment();
  return queryRdmaSupport();
}

std::string RdmaTransport::bind() {
  return ib_->getLocalVcIdentifier(kDummyRank);
}

commResult_t RdmaTransport::connect(const std::string& peerId) {
  FB_COMMCHECK(ib_->connectVcDirect(peerId, kDummyRank));
  return commSuccess;
}

bool RdmaTransport::connected() const {
  return ib_->getVc(kDummyRank) != nullptr;
}

folly::SemiFuture<commResult_t> RdmaTransport::write(
    RdmaMemory::View localBuffer,
    RdmaRemoteBuffer remoteBuffer,
    bool notify) {
  CHECK_THROW(connected(), std::runtime_error);
  CHECK_THROW(evb_, std::runtime_error);

  CHECK(cudaDev_ == localBuffer->getDevice());

  auto ibRemoteKey = CtranIbRemoteAccessKey::fromString(remoteBuffer.accessKey);
  auto work = std::make_unique<Work>();
  work->type = Work::Type::Write;
  auto sf = work->promise.getSemiFuture();

  CtranIbEpochRAII epochRAII(ib_.get());
  FB_COMMCHECK(ib_->iput(
      localBuffer.data(),
      remoteBuffer.ptr,
      localBuffer.size(),
      kDummyRank,
      localBuffer->localKey(),
      ibRemoteKey,
      notify,
      nullptr,
      &work->ibReq,
      false));

  // Add work to pending list and schedule progress
  auto pendingWorks = pendingWorks_.wlock();
  pendingWorks->emplace_back(std::move(work));
  evb_->runInEventBaseThread([this]() { progress(); });

  return sf;
}

folly::SemiFuture<commResult_t> RdmaTransport::waitForWrite() {
  CHECK_THROW(connected(), std::runtime_error);
  CHECK_THROW(evb_, std::runtime_error);

  auto work = std::make_unique<Work>();
  work->type = Work::Type::WaitForWrite;
  auto sf = work->promise.getSemiFuture();

  // Add work to pending list and schedule progress
  auto pendingWorks = pendingWorks_.wlock();
  pendingWorks->emplace_back(std::move(work));
  evb_->runInEventBaseThread([this]() { progress(); });

  return sf;
}

folly::SemiFuture<commResult_t> RdmaTransport::read(
    RdmaMemory::MutableView& localBuffer,
    const RdmaRemoteBuffer& remoteBuffer) {
  CHECK_THROW(connected(), std::runtime_error);
  CHECK_THROW(evb_, std::runtime_error);

  CHECK(cudaDev_ == localBuffer->getDevice());

  auto ibRemoteKey = CtranIbRemoteAccessKey::fromString(remoteBuffer.accessKey);
  auto work = std::make_unique<Work>();
  work->type = Work::Type::Read;
  auto sf = work->promise.getSemiFuture();

  CtranIbEpochRAII epochRAII(ib_.get());
  FB_COMMCHECK(ib_->iget(
      remoteBuffer.ptr,
      localBuffer.mutable_data(),
      localBuffer.size(),
      kDummyRank,
      localBuffer->localKey(),
      ibRemoteKey,
      nullptr,
      &work->ibReq,
      false));

  // Add work to pending list and schedule progress
  auto pendingWorks = pendingWorks_.wlock();
  pendingWorks->emplace_back(std::move(work));
  evb_->runInEventBaseThread([this]() { progress(); });

  return sf;
}

void RdmaTransport::progress() {
  CtranIbEpochRAII epochRAII(ib_.get());
  auto res = ib_->progress();
  bool hasError = false;
  if (res != commSuccess && res != commInProgress) {
    hasError = true;
    LOG(ERROR) << "IB progress failed";
  }

  auto pendingWorks = pendingWorks_.wlock();
  for (auto it = pendingWorks->begin(); it != pendingWorks->end();) {
    if (hasError) {
      (*it)->promise.setValue(res);
      it = pendingWorks->erase(it);
      continue;
    }

    if (((*it)->type == Work::Type::Write || (*it)->type == Work::Type::Read) &&
        (*it)->ibReq.isComplete()) {
      (*it)->promise.setValue(hasError ? res : commSuccess);
      it = pendingWorks->erase(it);
      continue;
    }

    if ((*it)->type == Work::Type::WaitForWrite) {
      bool done = false;
      auto waitRes = ib_->checkNotify(kDummyRank, &done);
      if (waitRes != commSuccess || done) {
        (*it)->promise.setValue(waitRes);
        it = pendingWorks->erase(it);
        continue;
      }
    }

    // Increment the iterator
    ++it;
  }

  // Schedule progress if there are more pending works
  if (pendingWorks->size()) {
    progressTimeout_->scheduleTimeoutHighRes(kProgressInterval);
  }
}

} // namespace torch::comms
