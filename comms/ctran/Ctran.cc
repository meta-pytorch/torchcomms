// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <memory>
#include <optional>

#include <cuda_runtime.h>

#include <folly/container/F14Set.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/CtranPipes.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/PersistentCleanup.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/LogInit.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

// Import "commGroupDepth" from CommGroupUtils.h
#include "comms/ctran/utils/CommGroupUtils.h"

#if defined(ENABLE_PRIMS)
#include "comms/prims/trace/PipesTrace.h"
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"
#include "comms/prims/transport/MultiPeerTransport.h"
#endif // defined(ENABLE_PRIMS)

Ctran::Ctran(
    CtranComm* comm,
    std::unique_ptr<ctran::IProfilerReporter> reporter,
    std::unique_ptr<ctran::IGpeProfilerReporter> gpeReporter)
    : comm_(comm) {
  ctran::logging::initCtranLogging();

  // Profiler is constructed first so it can be passed into the mapper
  // (and through to CtranTcpDm) -- CtranTcpDm registers its profiler
  // hooks during construction, removing the need for a separate
  // post-construction registerProfilerHooks() round-trip via the mapper.
  if (comm->config_.enableProfiler) {
    profiler = std::make_unique<ctran::Profiler>(comm, std::move(reporter));
  }

  mapper = std::make_unique<CtranMapper>(comm_, profiler.get());
  gpe = std::make_unique<CtranGpe>(
      comm->statex_->cudaDev(), comm_, std::move(gpeReporter));

  algo = std::make_unique<CtranAlgo>(comm, this);
}

Ctran::~Ctran() {
  if (mapper) {
    // Tell mapper to avoid any further communication at buffer deregistration;
    mapper->setAtDestruction();
  }
}

commResult_t Ctran::commRegister(void* buff, size_t size, void** handle) {
  commResult_t res = commSuccess;

  if (!this->mapper) {
    CLOGF(ERR, "Ctran mapper is not initialized, skip commRegister");
    return commInternalError;
  } else if (NCCL_CTRAN_REGISTER != NCCL_CTRAN_REGISTER::none) {
    return this->mapper->regMem(buff, size, handle);
  }

  return res;
}

commResult_t Ctran::commDeregister(void* handle) {
  commResult_t res = commSuccess;

  if (!this->mapper) {
    CLOGF(ERR, "Ctran mapper is not initialized, skip commDeregister");
    return commInternalError;
  } else if (NCCL_CTRAN_REGISTER != NCCL_CTRAN_REGISTER::none) {
    return this->mapper->deregMem(handle);
  }

  return res;
}

bool Ctran::isInitialized() const {
  return mapper && gpe && algo;
}

void Ctran::updateOpCount() {
  if (commGroupDepth == 0) {
    // Increase after submitted a single op. Grouped-op uses same opCount and
    // increase at groupEnd.
    // NOTE: when calling from NCCL, NCCL side manages group depth and calls
    // ctranGroupEndHook to submit the grouped op. commGroupDepth should be
    // always zero in this path.
    (*comm_->opCount_)++;
    // Also increase ctran-only opCount if opCount_ is shared with baseline, so
    // we can track the calls to Ctran collectives
    if (!comm_->useNativeOpCount()) {
      comm_->updateCtranOpCount();
    }
  }
}

uint64_t Ctran::getOpCount() const {
  return *comm_->opCount_;
}

uint64_t Ctran::getCtranOpCount() const {
  return comm_->getCtranOpCount();
}

#if defined(ENABLE_PRIMS)
comms::prims::Transport* CtranComm::getMultiPeerTransportsPtr() const {
  if (!multiPeerTransport_) {
    return nullptr;
  }
  return multiPeerTransport_->get_device_handle().transports.data();
}
#else
comms::prims::Transport* CtranComm::getMultiPeerTransportsPtr() const {
  return nullptr;
}
#endif // defined(ENABLE_PRIMS)

std::optional<meta::comms::colltrace::AlgoStatDump> CtranComm::dumpAlgoStats()
    const {
  if (!algoStats_) {
    return std::nullopt;
  }
  return algoStats_->dump();
}

void CtranComm::recordAlgoStats(
    const std::string& opName,
    const std::string& algoName,
    const size_t msgSize) {
  if (algoStats_) {
    algoStats_->record(opName, algoName, msgSize);
  }
}

void CtranComm::registerPersistentCleanup(
    std::shared_ptr<PersistentCleanup> cleanup) {
  persistentCleanups_.wlock()->insert(std::move(cleanup));
}

void CtranComm::unregisterPersistentCleanup(
    const std::shared_ptr<PersistentCleanup>& cleanup) {
  persistentCleanups_.wlock()->erase(cleanup);
}

void CtranComm::drainPersistentCleanups() {
  folly::F14FastSet<std::shared_ptr<PersistentCleanup>> local;
  persistentCleanups_.withWLock([&local](auto& set) { local.swap(set); });
  for (const auto& cleanup : local) {
    cleanup->run();
  }
}

commResult_t ctranInit(
    CtranComm* comm,
    std::unique_ptr<ctran::IProfilerReporter> reporter,
    std::unique_ptr<ctran::IGpeProfilerReporter> gpeReporter) {
  NcclScubaEvent initEvent(&comm->logMetaData_);
  initEvent.lapAndRecord("CtranInit START");
  try {
    comm->ctran_ = std::make_shared<Ctran>(
        comm, std::move(reporter), std::move(gpeReporter));
  } catch (std::exception& e) {
    CLOGF(ERR, "Ctran initialization failed: {}", e.what());
    return commInternalError;
  }

  for (const auto& opt : NCCL_COLLTRACE) {
    if (opt == "algostat") {
      comm->algoStats_ = meta::comms::colltrace::AlgoStats::getOrCreate(
          comm->statex_->commHash(), comm->statex_->commDesc());
      break;
    }
  }

  auto res = ctranInitializePipes(comm);
  if (res != commSuccess) {
    return res;
  }

  initEvent.lapAndRecord("CtranInit COMPLETE");
  return commSuccess;
}

bool ctranInitialized(CtranComm* comm) {
  // comm->finalizeCalled used to prevent double finalization but we don't need
  // it in ctran as we use cpp style with smart pointers
  return comm && comm->ctran_ && comm->ctran_->isInitialized();
}

commResult_t CtranComm::finalize() {
  // TODO: placeholder, to add completion wait logic
  return commSuccess;
}

CtranComm::CtranComm(std::shared_ptr<Abort> abort, ctranConfig commConfig)
    : config_(commConfig), abort_(abort) {
  asyncErr_ =
      std::make_shared<AsyncError>(NCCL_CTRAN_ABORT_ON_ERROR, "CtranComm");
  if (!abort_) {
    throw ctran::utils::Exception("abort must not be empty", commInternalError);
  }
  // Default points to internal opCount
  opCount_ = &ctranOpCount_;
}

void CtranComm::destroy() {
  cudagraphDeferredCleanup.runAll();

  // All smart pointers are automatically de-initialized, but we want to
  // ensure they do so in a specific order. Therefore, we manually handle
  // their de-initialization here.
#if defined(ENABLE_PRIMS)
  pipesTrace_.reset();
  // Must be destroyed before ctran_ (which owns SharedResource staging
  // buffers used as external data buffers) and before bootstrap_ (since
  // multiPeerTransport_ holds a non-owning reference to it).
  multiPeerTransport_.reset();
#endif // defined(ENABLE_PRIMS)
  // Release every outstanding persistent request's pooled pipeSync + scoped
  // registration before ctran_.reset() (which triggers CtranGpe::terminate()'s
  // pool-drain spin-wait). Runs each cleanup token at most once; tokens already
  // run by an eager free / graph-destroy callback no-op here.
  drainPersistentCleanups();
  ctran_.reset();
  bootstrap_.reset();
  colltraceNew_.reset();
  statex_.reset();
  // NOTE: memCache needs to be destroyed after transportProxy_ to release
  // all buffers
  memCache_.reset();

  this->logMetaData_.commDesc.clear();
  this->logMetaData_.commDesc.shrink_to_fit();
}

CtranComm::~CtranComm() {
  this->destroy();
}

CtranComm::CtranComm(CtranComm&&) = default;
CtranComm& CtranComm::operator=(CtranComm&&) = default;

commResult_t ctranFinalize(CtranComm* comm) {
  if (comm) {
    return comm->finalize();
  }
  return commSuccess;
}

namespace ctran {

commResult_t globalRegisterWithPtr(
    void* buff,
    size_t size,
    bool forceReg,
    bool ncclManaged) {
  if (NCCL_CTRAN_REGISTER == NCCL_CTRAN_REGISTER::none) {
    // ctran registration is disabled, no-op
    return commSuccess;
  }

  auto regCache = RegCache::getInstance();
  if (!regCache) {
    CLOGF(ERR, "globalRegisterWithPtr: RegCache not available");
    return commInternalError;
  }

  return regCache->globalRegister(buff, size, forceReg, ncclManaged);
}

commResult_t
globalDeregisterWithPtr(void* buff, size_t size, bool skipRemRelease) {
  if (NCCL_CTRAN_REGISTER == NCCL_CTRAN_REGISTER::none) {
    // ctran registration is disabled, no-op
    return commSuccess;
  }

  auto regCache = RegCache::getInstance();
  if (!regCache) {
    CLOGF(ERR, "globalDeregisterWithPtr: RegCache not available");
    return commInternalError;
  }

  return regCache->globalDeregister(buff, size, skipRemRelease);
}

commResult_t registerAll() {
  return RegCache::regAll();
}

commResult_t deregisterAll() {
  return RegCache::deregAll();
}

} // namespace ctran
