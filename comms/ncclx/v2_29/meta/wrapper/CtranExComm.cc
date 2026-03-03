// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "checks.h"
#include "comm.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranEx.h"
#include "comms/utils/checks.h"
#include "comms/utils/logger/LogUtils.h"

#include "meta/wrapper/CtranExComm.h"
#include "meta/wrapper/MetaFactory.h"

namespace ctran {

#define CHECK_VALID_COMM()                                                                                                                       \
  do {                                                                                                                                           \
    if (!isInitialized()) {                                                                                                                      \
      CLOGF(                                                                                                                                     \
          ERR,                                                                                                                                   \
          "CTRAN-EX: instance is not initialized with a valid nccl communicator or the communicator internal CTran backend is not initialized"); \
      return ncclInvalidUsage;                                                                                                                   \
    }                                                                                                                                            \
  } while (0);

CtranExComm::CtranExComm(const ncclComm_t comm, const std::string& commDesc) {
  comm_ = NCCL_COMM_NULL;

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.commDesc = commDesc.c_str();
  std::vector<int> globalRanks =
      comm->ctranComm_->statex_->commRanksToWorldRanksRef();
  config.splitGroupRanks = globalRanks.data();
  config.splitGroupSize = comm->ctranComm_->statex_->nRanks();
  config.blocking = 1; // Ensure communicator is fully created upon return
  // Enable lazy features to avoid allocating extra resources from baseline NCCL
  config.lazyConnect = 1;
  config.lazySetupChannels = 1;

  // if parent comm is non-blocking, ncclCommSplit will be non-blocking
  // as well which would lead to undefined behavior. Adding a throw here to
  // warn user instead of silently failing.
  if (comm->config.blocking == 0) {
    CLOGF(
        ERR,
        "CTRAN-EX: parent communicator {} is non-blocking, which will cause CtranExComm commSplit to fail.",
        comm->config.commDesc);
    throw std::runtime_error("CTRAN-EX: parent communicator is non-blocking");
  }

  // Duplicate the communicator
  // TODO: completely skip non-Ctran resource initialization;
  auto res = ncclCommSplit(
      comm, 0, comm->ctranComm_->statex_->rank(), &comm_, &config);
  if (res == ncclInProgress) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-EX: ncclCommSplit from {}, new comm {} commDesc {} is in progress. We will wait for it to finish.",
        (void*)comm,
        (void*)comm_,
        commDesc);
    while (res == ncclInProgress) {
      // TODO: BUG: ncclCommGetAsyncError(comm_, &res) should be called within
      // the loop
      std::this_thread::yield();
    }
  } else if (res != ncclSuccess) {
    CLOGF(
        ERR,
        "CTRAN-EX: failed to create a new communicator from {}, new comm {} commDesc {}",
        (void*)comm,
        (void*)comm_,
        commDesc);
    throw std::runtime_error("CTRAN-EX: failed to create a new communicator");
  }
}

CtranExComm::~CtranExComm() {
  if (comm_ && comm_ != NCCL_COMM_NULL) {
    if (comm_->ctranComm_ &&
        comm_->ctranComm_->getAsyncResult() != commSuccess) {
      FB_COMMCHECKTHROW(ncclToMetaComm(ncclCommAbort(comm_)));
    } else {
      FB_COMMCHECKTHROW(ncclToMetaComm(ncclCommDestroy(comm_)));
    }
  }
}

bool CtranExComm::isInitialized() const {
  return comm_ && comm_ != NCCL_COMM_NULL &&
      ctranInitialized(comm_->ctranComm_.get());
}

ncclResult_t CtranExComm::regMem(
    const void* ptr,
    const size_t size,
    void** segHdl,
    bool forceRegister) {
  CHECK_VALID_COMM();

  CtranMapper* mapper = comm_->ctranComm_->ctran_->mapper.get();
  return metaCommToNccl(mapper->regMem(ptr, size, segHdl, forceRegister));
}

ncclResult_t CtranExComm::deregMem(void* segHdl) {
  CHECK_VALID_COMM();

  CtranMapper* mapper = comm_->ctranComm_->ctran_->mapper.get();
  return metaCommToNccl(mapper->deregMem(segHdl));
}

bool CtranExComm::supportBroadcast() const {
  CHECK_VALID_COMM();
  auto algo = NCCL_BROADCAST_ALGO;
  if (algo == NCCL_BROADCAST_ALGO::orig) {
    algo = NCCL_BROADCAST_ALGO::ctran;
  }
  return ctranBroadcastSupport(
      comm_->ctranComm_.get(), algo, CtranMapperBackend::IB);
}

ncclResult_t CtranExComm::broadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    CtranExRequest** req) {
  CHECK_VALID_COMM();
  auto ctranComm = comm_->ctranComm_.get();

  // Restrict to use only IB backend
  auto algo = NCCL_BROADCAST_ALGO;
  if (algo == NCCL_BROADCAST_ALGO::orig) {
    algo = NCCL_BROADCAST_ALGO::ctran;
  }
  if (!ctranBroadcastSupport(ctranComm, algo, CtranMapperBackend::IB)) {
    CLOGF(
        ERR,
        "CTRAN-EX: the specified communicator does not support broadcast with only IB backend.");
    return ncclInvalidUsage;
  }

  auto* reqPtr = new CtranExRequest();
  auto reqImpl = reinterpret_cast<CtranExRequestImpl*>(reqPtr->impl_);
  reqImpl->initialize(CtranExRequestImpl::BCAST, ctranComm);

  // TODO: only tree supports host mem for now. Add support for direct.
  NCCLCHECK(metaCommToNccl(ctranComm->ctran_->algo->broadcastBinomialTree(
      sendbuff,
      recvbuff,
      count,
      ncclToMetaComm(datatype),
      root,
      reqImpl->bcast_complete)));

  *req = reqPtr;
  return ncclSuccess;
}

std::string CtranExComm::getAsyncErrorString() const {
  auto ctranComm = comm_->ctranComm_.get();
  auto e = ctranComm->getAsyncException();
  // TODO: we don't return richer Exception nor Error class since we are going
  // to deprecate CtranExComm in favor of MCCL. We should return an error
  // structure rather than string so that user may access individual fields for
  // custom logging.
  return e.what();
}
} // namespace ctran
