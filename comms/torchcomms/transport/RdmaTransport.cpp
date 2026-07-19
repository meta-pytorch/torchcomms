// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "RdmaTransport.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <folly/logging/xlog.h>
#include <folly/synchronization/CallOnce.h>

#include <fmt/core.h>
#include "comms/ctran/backends/ib/BootstrapExternal.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/utils/checks.h"
#include "comms/utils/commSpecs.h"

namespace {

constexpr std::chrono::microseconds kProgressInterval{0};
constexpr int kDummyRank = 0;
constexpr int kDummyDevice = 0;

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
folly::once_flag initOnceFlag;
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
folly::once_flag runtimeInitOnceFlag;
void initRuntimeEnvironment() {
  folly::call_once(runtimeInitOnceFlag, [] {
    ctran::logging::initCtranLogging();
    ctran::utils::commCudaLibraryInit();
  });
}
void initEnvironment() {
  folly::call_once(initOnceFlag, [] {
    ncclCvarInit();
    initRuntimeEnvironment();
  });
}
void initSupportEnvironment() {
  // re-read cvars so failed first probe does not poison later probes
  ncclCvarInit();
  initRuntimeEnvironment();
}

// Gates verbose RDMA registration logs via env var
// TORCHCOMMS_RDMA_ENABLE_REG_VERBOSE_LOG. Disabled when unset/empty or set to a
// falsy value (case-insensitive: "0", "false", "off", "no"); any other value
// enables it.
bool rdmaRegVerboseLogEnabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("TORCHCOMMS_RDMA_ENABLE_REG_VERBOSE_LOG");
    if (env == nullptr || env[0] == '\0') {
      return false;
    }
    std::string value(env);
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char ch) {
          return static_cast<char>(std::tolower(ch));
        });
    return value != "0" && value != "false" && value != "off" && value != "no";
  }();
  return enabled;
}
} // namespace

// Logs a verbose RDMA registration message (prefixed with [RDMA]) only when
// TORCHCOMMS_RDMA_ENABLE_REG_VERBOSE_LOG is enabled. fmtStr must be a string
// literal; trailing args are the fmt arguments.
#define REG_VERBOSE_LOG(fmtStr, ...)                            \
  do {                                                          \
    if (rdmaRegVerboseLogEnabled()) {                           \
      XLOGF(INFO, "[RDMA] " fmtStr __VA_OPT__(, ) __VA_ARGS__); \
    }                                                           \
  } while (0)

extern "C" int RdmaRegTensor(void* addr, size_t len) {
  initEnvironment();
  const auto regCache = ctran::RegCache::getInstance();
  REG_VERBOSE_LOG(
      "RdmaRegTensor regcache={} addr={} len={}",
      fmt::ptr(regCache.get()),
      addr,
      len);
  return static_cast<int>(regCache->globalRegister(
      addr, len, /*forceReg=*/false, /*ncclManaged=*/false, /*deviceId=*/-1));
}

extern "C" int RdmaDeregTensor(void* addr, size_t len) {
  initEnvironment();
  const auto regCache = ctran::RegCache::getInstance();
  REG_VERBOSE_LOG(
      "RdmaDeregTensor regcache={} addr={} len={}",
      fmt::ptr(regCache.get()),
      addr,
      len);
  return static_cast<int>(regCache->globalDeregister(
      addr, len, /*skipRemRelease=*/false, /*deviceId=*/-1));
}

namespace torch::comms {

RdmaMemory::RdmaMemory(const void* buf, size_t len, int cudaDev)
    : buf_(buf), len_(len), cudaDev_(cudaDev) {
  initEnvironment();
  // Hold a shared_ptr to ensure RegCache lifetime while RdmaMemory is in
  // scope
  regCache_ = ctran::RegCache::getInstance();

  regHdl_ = regCache_->searchIbRegHandle(buf_, len_, cudaDev_);
  if (regHdl_ != nullptr) {
    // Cache HIT: reuse the existing cached registration. This RdmaMemory does
    // not own it (no dynamic handle), so the dtor will not deregister it.
    cacheReg_ = true;
  } else {
    // Cache MISS: register dynamically with an IB-only backend set — an
    // isolated registration that is NOT cached and NOT reused (searchRegElem
    // skips dynamic RegElems), so a transient per-buffer registration never
    // pollutes the shared segment cache or force-frees a segment another
    // buffer still needs. This RdmaMemory owns it (deregistered in the dtor).
    std::vector<bool> backends(CommBackend::NUM_BACKENDS, false);
    backends[CommBackend::IB] = true;
    ctran::regcache::RegElem* dynHdl = nullptr;
    FB_COMMCHECKTHROW(
        regCache_->regDynamic(buf_, len_, cudaDev_, backends, &dynHdl));
    dynRegHdl_ = dynHdl;
    regHdl_ = dynHdl->ibRegElem;
  }
  remoteKey_ = CtranIb::getRemoteAccessKey(regHdl_).toString();
  REG_VERBOSE_LOG(
      "RdmaMemory regcache={} buf={} len={} cudaDev={} cacheReg_={} regHdl={} dynRegHdl={} remoteKey={}",
      fmt::ptr(regCache_.get()),
      buf_,
      len_,
      cudaDev_,
      cacheReg_,
      fmt::ptr(regHdl_),
      fmt::ptr(dynRegHdl_),
      remoteKey_);
}

RdmaMemory::RdmaMemory(RdmaMemory&& other) noexcept
    : buf_(other.buf_),
      len_(other.len_),
      cudaDev_(other.cudaDev_),
      regHdl_(other.regHdl_),
      dynRegHdl_(other.dynRegHdl_),
      remoteKey_(std::move(other.remoteKey_)),
      cacheReg_(other.cacheReg_),
      regCache_(std::move(other.regCache_)) {
  // Properly invalidate the moved-from object to prevent double-free
  // and ensure the object is in a valid but unspecified state
  other.buf_ = nullptr;
  other.len_ = 0;
  other.cudaDev_ = -1;
  other.regHdl_ = nullptr;
  other.dynRegHdl_ = nullptr;
  other.cacheReg_ = false;
  // Note: remoteKey_ is already moved, leaving other.remoteKey_ empty
}

RdmaMemory::~RdmaMemory() noexcept {
  // Ownership is keyed on dynRegHdl_: the HIT path has dynRegHdl_ == nullptr,
  // so this is a no-op there and only the owned dynamic registration is freed.
  if (dynRegHdl_ != nullptr) {
    FB_COMMCHECKIGNORE(regCache_->deregDynamic(
        static_cast<ctran::regcache::RegElem*>(dynRegHdl_)));
    dynRegHdl_ = nullptr;
    regHdl_ = nullptr;
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

  // Mock context for this work (type from setMockForTest)
  RdmaTransport::MockContext mockContext;

  // Timeout tracking for write operations (production and mock)
  std::optional<std::chrono::milliseconds> timeout;
  // Only valid and set when timeout is set.
  std::chrono::steady_clock::time_point creationTime;
};

RdmaTransport::RdmaTransport(
    int cudaDev,
    folly::EventBase* evb,
    std::optional<int> maxNumCqe,
    std::optional<int> maxNumNic)
    : cudaDev_(cudaDev), evb_(evb) {
  initEnvironment();
  // Create IB Instance
  ib_ = std::make_unique<CtranIb>(
      kDummyRank,
      cudaDev,
      -1 /* commHash */,
      "RDMA-Transport",
      true /* enableLocalFlush */,
      CtranIb::BootstrapMode::kExternal,
      std::nullopt /* qpServerAddr */,
      ::ctran::utils::createAbort(/*enabled=*/false),
      nullptr /* socketFactory */,
      maxNumCqe,
      maxNumNic);

  if (evb_) {
    // Optionally create progress timeout; skip it if the transport is never
    // used for remote RDMA
    progressTimeout_ =
        folly::AsyncTimeout::make(*evb_, [this]() noexcept { progress(); });
  }
}

RdmaTransport::~RdmaTransport() {
  // Run cleanup on the EventBase thread to safely cancel the timeout
  // and prevent progress() from racing with destruction.
  if (evb_) {
    evb_->runImmediatelyOrRunInEventBaseThreadAndWait([this]() {
      if (progressTimeout_) {
        progressTimeout_->cancelTimeout();
      }
      auto pendingWorks = pendingWorks_.wlock();
      auto numPending = pendingWorks->size();
      if (numPending > 0) {
        XLOGF(
            WARN,
            "~RdmaTransport: draining {} pending works with commUserAbort",
            numPending);
      }
      for (auto it = pendingWorks->begin(); it != pendingWorks->end();) {
        (*it)->promise.setValue(commUserAbort);
        it = pendingWorks->erase(it);
      }
    });
  }
}

namespace {
bool queryRdmaSupport() {
  XLOG(INFO) << "Querying RdmaTransport support";
  try {
    auto ib = std::make_unique<CtranIb>(
        kDummyRank,
        kDummyDevice,
        -1 /* commHash */,
        "Query-RDMA-Support",
        true /* enableLocalFlush */,
        CtranIb::BootstrapMode::kExternal);
  } catch (const std::exception& e) {
    XLOG(WARN)
        << "RdmaTransport is not supported. Failed to create CtranIb instance: "
        << e.what();
    return false;
  }
  XLOG(INFO) << "RdmaTransport is supported";
  return true;
}
} // namespace

bool RdmaTransport::supported() {
  initSupportEnvironment();
  return queryRdmaSupport();
}

std::string RdmaTransport::bind() {
  return ib_->externalBootstrap()->getLocalVcId(kDummyRank);
}

commResult_t RdmaTransport::connect(const std::string& peerId) {
  FB_COMMCHECK(ib_->externalBootstrap()->connectVc(peerId, kDummyRank));
  return commSuccess;
}

bool RdmaTransport::connected() const {
  return ib_->getVc(kDummyRank) != nullptr;
}

int RdmaTransport::getMaxCqe() const {
  return ib_->getMaxCqe();
}

int RdmaTransport::getNumNics() const {
  return ib_->getNumNics();
}

folly::SemiFuture<commResult_t> RdmaTransport::write(
    RdmaMemory::View localBuffer,
    const RdmaRemoteBuffer& remoteBuffer,
    bool notify,
    std::optional<std::chrono::milliseconds> timeout) {
  auto currentMockType = mockContext_.rlock()->type;

  // Skip connected check when mock is enabled for testing
  if (currentMockType == MockType::None) {
    CHECK_THROW(connected(), std::runtime_error);
  }
  CHECK_THROW(evb_, std::runtime_error);
  CHECK_THROW(localBuffer.size() <= remoteBuffer.len, std::runtime_error);

  CHECK_EQ(cudaDev_, localBuffer->getDevice());

  auto work = std::make_unique<Work>();
  work->type = Work::Type::Write;
  work->mockContext.type = currentMockType;
  auto sf = work->promise.getSemiFuture();

  if (currentMockType == MockType::None) {
    // Production path - perform actual IB operations
    auto ibRemoteKey =
        CtranIbRemoteAccessKey::fromString(remoteBuffer.accessKey);
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
    // Capture timeout for production write timeout
    if (timeout.has_value()) {
      work->timeout = timeout;
      work->creationTime = std::chrono::steady_clock::now();
    }
  } else if (currentMockType == MockType::Timeout) {
    // Mock timeout path - capture timeout if provided
    if (timeout.has_value()) {
      work->timeout = timeout;
      work->creationTime = std::chrono::steady_clock::now();
    }
  }

  // Add work to pending list and schedule progress
  auto pendingWorks = pendingWorks_.wlock();
  pendingWorks->emplace_back(std::move(work));
  evb_->runInEventBaseThread([this]() { progress(); });

  // NOLINTNEXTLINE(clang-diagnostic-nrvo)
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

  CHECK_EQ(cudaDev_, localBuffer->getDevice());

  auto work = std::make_unique<Work>();
  work->type = Work::Type::Read;
  auto sf = work->promise.getSemiFuture();

  auto ibRemoteKey = CtranIbRemoteAccessKey::fromString(remoteBuffer.accessKey);
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

  // NOLINTNEXTLINE(clang-diagnostic-nrvo)
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
    // Mock failure always takes precedence — return commInternalError
    // regardless of IB state
    auto& work = *it;
    if (work->mockContext.type == MockType::Failure) {
      work->promise.setValue(commInternalError);
      it = pendingWorks->erase(it);
      continue;
    }

    // Mock timeout: no IB operation was issued, so skip IB checks.
    // Only check the timeout parameter (if any); otherwise keep waiting.
    if (work->mockContext.type == MockType::Timeout) {
      if (work->timeout.has_value()) {
        auto elapsed = std::chrono::steady_clock::now() - work->creationTime;
        if (elapsed >= work->timeout.value()) {
          work->promise.setValue(commTimeout);
          it = pendingWorks->erase(it);
          continue;
        }
      }
      ++it;
      continue;
    }

    if (hasError) {
      (*it)->promise.setValue(res);
      it = pendingWorks->erase(it);
      continue;
    }

    // Check IB completion
    if (((*it)->type == Work::Type::Write || (*it)->type == Work::Type::Read) &&
        (*it)->ibReq.isComplete()) {
      (*it)->promise.setValue(commSuccess);
      it = pendingWorks->erase(it);
      continue;
    }

    // Check write timeout (production path only — mock timeout handled above)
    if ((*it)->type == Work::Type::Write && (*it)->timeout.has_value()) {
      auto elapsed = std::chrono::steady_clock::now() - (*it)->creationTime;
      if (elapsed >= (*it)->timeout.value()) {
        (*it)->promise.setValue(commTimeout);
        it = pendingWorks->erase(it);
        continue;
      }
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

void RdmaTransport::setMockForTest(MockContext context) {
  *mockContext_.wlock() = context;
}

// Deprecated no-op: cleanup is handled by the destructor.
// TODO: Remove after upper layer removes calling abort().
void RdmaTransport::abort() {
  XLOG(DBG) << "abort() called (no-op, cleanup deferred to destructor)";
}

} // namespace torch::comms
