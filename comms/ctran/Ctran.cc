// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <memory>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/LogInit.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

// Import "commGroupDepth" from CommGroupUtils.h
#include "comms/ctran/utils/CommGroupUtils.h"

Ctran::Ctran(CtranComm* comm) : comm_(comm) {
  ctran::logging::initCtranLogging();

  mapper = std::make_unique<CtranMapper>(comm_);
  gpe = std::make_unique<CtranGpe>(comm->statex_->cudaDev(), comm_);

  algo = std::make_unique<CtranAlgo>(comm, this);

  if (NCCL_CTRAN_TRANSPORT_PROFILER) {
    profiler = std::make_unique<ctran::Profiler>(comm);
  }
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

commResult_t ctranInit(CtranComm* comm) {
  NcclScubaEvent initEvent(&comm->logMetaData_);
  initEvent.lapAndRecord("CtranInit START");
  try {
    comm->ctran_ = std::make_shared<Ctran>(comm);
  } catch (std::exception& e) {
    CLOGF(ERR, "Ctran initialization failed: {}", e.what());
    return commInternalError;
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

commResult_t ctranFinalize(CtranComm* comm) {
  if (comm) {
    return comm->finalize();
  }
  return commSuccess;
}

namespace ctran {

commResult_t globalRegisterWithPtr(void* buff, size_t size, bool forceReg) {
  if (NCCL_CTRAN_REGISTER == NCCL_CTRAN_REGISTER::none) {
    // ctran registration is disabled, no-op
    return commSuccess;
  }

  auto regCache = RegCache::getInstance();
  if (!regCache) {
    CLOGF(ERR, "globalRegisterWithPtr: RegCache not available");
    return commInternalError;
  }

  return regCache->globalRegister(buff, size, forceReg);
}

commResult_t globalDeregisterWithPtr(void* buff, size_t size) {
  if (NCCL_CTRAN_REGISTER == NCCL_CTRAN_REGISTER::none) {
    // ctran registration is disabled, no-op
    return commSuccess;
  }

  auto regCache = RegCache::getInstance();
  if (!regCache) {
    CLOGF(ERR, "globalDeregisterWithPtr: RegCache not available");
    return commInternalError;
  }

  return regCache->globalDeregister(buff, size);
}

} // namespace ctran
