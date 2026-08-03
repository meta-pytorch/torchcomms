// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdexcept>
#include "meta/wrapper/NcclCommCtran.h"
#include "meta/wrapper/NcclCommLogData.h"

#include "comm.h"
#include "comms/ctran/algos/AllToAll/AllToAllPHintUtils.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/ctran/window/WinHintUtils.h"
#include "comms/utils/checks.h"
#include "comms/utils/commSpecs.h"
#include "meta/NcclxConfig.h" // @manual
#include "meta/commstate/FactoryCommStateX.h"
#include "meta/ctran-integration/BaselineBootstrap.h"
#include "meta/wrapper/MetaFactory.h"
#include "meta/wrapper/NcclCommCollTrace.h"

using namespace ctran;

#define NCCLCHECK_COMM(call) NCCLCHECK(metaCommToNccl(call))

meta::comms::Hints ncclToMetaComm(const ncclx::Hints& hints) {
  meta::comms::Hints ret;
  std::string v;
  for (const auto& k : meta::comms::hints::AllToAllPHintUtils::keys()) {
    FB_COMMCHECKTHROW(ncclToMetaComm(hints.get(k, v)));
    FB_COMMCHECKTHROW(ret.set(k, v));
  }
  for (const auto& k : meta::comms::hints::WinHintUtils::keys()) {
    FB_COMMCHECKTHROW(ncclToMetaComm(hints.get(k, v)));
    FB_COMMCHECKTHROW(ret.set(k, v));
  }
  return ret;
}

namespace {

ctranConfig makeCtranConfigFrom(ncclComm* comm) {
  struct ctranConfig tconfig = {
      .blocking = comm->config.blocking,
      .commDesc = NCCLX_CONFIG_FIELD(comm->config, commDesc),
  };
  if (comm->config.ncclxConfig != nullptr) {
    const auto* ncclxCfg =
        static_cast<ncclx::Config*>(comm->config.ncclxConfig);
    if (ncclxCfg->pipesNvlChunkSize.has_value()) {
      tconfig.pipesConfig.nvlChunkSize =
          static_cast<int64_t>(ncclxCfg->pipesNvlChunkSize.value());
    }
    tconfig.pipesConfig.ibLazyConnect = ncclxCfg->deviceIbLazyConnect;
    if (ncclxCfg->pipesIbgdaDataBufferSize.has_value()) {
      tconfig.pipesConfig.ibgdaDataBufferSize =
          static_cast<int64_t>(ncclxCfg->pipesIbgdaDataBufferSize.value());
    }
  }
  return tconfig;
}

commResult_t setCtranCommBase(ncclComm* ncclCommVal) {
  if (!ncclCommVal) {
    return commInvalidArgument;
  }
  meta::comms::ncclx::ncclCommCtran(ncclCommVal) =
      std::make_unique<CtranComm>();

  const auto tconfig = makeCtranConfigFrom(ncclCommVal);
  meta::comms::ncclx::ncclCommCtran(ncclCommVal)->config_ = tconfig;
  meta::comms::ncclx::ncclCommCtran(ncclCommVal)->opCount_ =
      &ncclCommVal->opCount;
  meta::comms::ncclx::ncclCommCtran(ncclCommVal)->logMetaData_ =
      ncclCommLogData(ncclCommVal);
  meta::comms::ncclx::ncclCommCtran(ncclCommVal)->runtimeConn_ =
      ncclCommVal->runtimeConn;

  return commSuccess;
}

} // namespace

ncclResult_t createCtranComm(ncclComm* comm) {
  NCCLCHECK_COMM(setCtranCommBase(comm));

  if (NCCL_USE_MEM_CACHE) {
    meta::comms::ncclx::ncclCommCtran(comm)->memCache_ =
        ncclx::memory::memCacheAllocator::getInstance();
  }

  meta::comms::ncclx::ncclCommCtran(comm)->bootstrap_ =
      std::make_unique<ncclx::BaselineBootstrap>(comm);

  NCCLCHECK(
      ncclx::initCommStateXFromNcclComm(
          comm, meta::comms::ncclx::ncclCommCtran(comm).get()));

  meta::comms::ncclx::ncclCommCtran(comm)->colltraceNew_ =
      meta::comms::ncclx::ncclCommNewCollTrace(comm);

  NCCLCHECK_COMM(ctranInit(meta::comms::ncclx::ncclCommCtran(comm).get()));

  return ncclSuccess;
}

ncclResult_t destroyCtranComm(ncclComm* comm) {
  if (!comm || !meta::comms::ncclx::ncclCommCtran(comm)) {
    return ncclSuccess;
  }
  NCCLCHECK_COMM(ctranFinalize(meta::comms::ncclx::ncclCommCtran(comm).get()));
  try {
    meta::comms::ncclx::ncclCommCtran(comm)->destroy();
    meta::comms::ncclx::ncclCommCtran(comm).reset();
  } catch (const std::exception& e) {
    CLOGF(ERR, "CtranComm destruction failed: {}", e.what());
    return ncclInternalError;
  }
  return ncclSuccess;
}
