// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdexcept>

#include "BaselineBootstrap.h"
#include "MetaFactory.h"
#include "comm.h"
#include "comms/ctran/algos/AllToAll/AllToAllPHintUtils.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicHintUtils.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/WinHintUtils.h"
#include "comms/utils/checks.h"
#include "comms/utils/commSpecs.h"

using namespace ctran;

ncclResult_t metaCommToNccl(commResult_t result) {
  switch (result) {
    case commSuccess:
      return ncclSuccess;
    case commUnhandledCudaError:
      return ncclUnhandledCudaError;
    case commSystemError:
      return ncclSystemError;
    case commInternalError:
      return ncclInternalError;
    case commInvalidArgument:
      return ncclInvalidArgument;
    case commInvalidUsage:
      return ncclInvalidUsage;
    case commRemoteError:
      return ncclRemoteError;
    case commInProgress:
      return ncclInProgress;
    case commNumResults:
      return ncclNumResults;
    default:
      throw std::runtime_error(
          std::string("commToNccl: unimplemented comm Result ") +
          std::to_string(result));
  }
}

commResult_t ncclToMetaComm(ncclResult_t result) {
  switch (result) {
    case ncclSuccess:
      return commSuccess;
    case ncclUnhandledCudaError:
      return commUnhandledCudaError;
    case ncclSystemError:
      return commSystemError;
    case ncclInternalError:
      return commInternalError;
    case ncclInvalidArgument:
      return commInvalidArgument;
    case ncclInvalidUsage:
      return commInvalidUsage;
    case ncclRemoteError:
      return commRemoteError;
    case ncclInProgress:
      return commInProgress;
    case ncclNumResults:
      return commNumResults;
    default:
      throw std::runtime_error(
          std::string("ncclToComm: unimplemented nccl Result ") +
          std::to_string(result));
  }
}

meta::comms::Hints ncclToMetaComm(const ncclx::Hints& hints) {
  // TODO: consolidate ncclx::Hints and meta::comms::Hints. This would require
  // changing the existing NCCLX APIs that use ncclx::Hints.
  meta::comms::Hints ret;
  std::string v;
  for (const auto& k : meta::comms::hints::AllToAllvDynamicHintUtils::keys()) {
    FB_COMMCHECKTHROW(ncclToMetaComm(hints.get(k, v)));
    FB_COMMCHECKTHROW(ret.set(k, v));
  }
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

commDataType_t ncclToMetaComm(ncclDataType_t dataType) {
  switch (dataType) {
    case ncclInt8:
      return commInt8;
    case ncclUint8:
      return commUint8;
    case ncclInt32:
      return commInt32;
    case ncclUint32:
      return commUint32;
    case ncclInt64:
      return commInt64;
    case ncclUint64:
      return commUint64;
    case ncclFloat16:
      return commFloat16;
    case ncclFloat32:
      return commFloat32;
    case ncclFloat64:
      return commFloat64;
    case ncclBfloat16:
      return commBfloat16;
    case ncclFloat8e4m3:
      return commFloat8e4m3;
    case ncclFloat8e5m2:
      return commFloat8e5m2;
    default:
      throw std::runtime_error(
          std::string("ncclToComm: unimplemented nccl DataType") +
          std::to_string(dataType));
  }
}

commResult_t initNcclCommCtran(ncclComm* ncclComm) {
  auto ctranComm = std::make_unique<CtranComm>();
  ctranComm->opCount_ = &ncclComm->opCount;
  ctranComm->bootstrap_ = std::make_unique<rcclx::BaselineBootstrap>(ncclComm);
  ctranComm->statex_ =
      createCtranCommStateXFromNcclComm(ncclComm, ctranComm.get());
  // TODO: init CtranComm newCollTrace
  FB_COMMCHECK(ctranInit(ctranComm.get()));

  // TODO: add RCCL config to configure all gather algo
  ctranComm->config_.ncclAllGatherAlgo = "undefined";
  FB_COMMCHECK(ctranConfigCommAlgoOverride(ctranComm.get()));

  // Ensure Ctran has been initialized correctly
  FB_CHECKTHROW(ctranInitialized(ctranComm.get()), "Ctran not initialized");
  ncclComm->ctranComm_ = std::move(ctranComm);
  return commSuccess;
}

std::unique_ptr<ncclx::CommStateX> createCtranCommStateXFromNcclComm(
    ncclComm* ncclComm,
    CtranComm* ctranComm) {
  FB_CHECKABORT(ncclComm->rankToNode, "rankToNode is nullptr");
  FB_CHECKABORT(ncclComm->localRankToRank, "localRankToRank is nullptr");

  auto commStateX = std::make_unique<ncclx::CommStateX>(
      ncclComm->rank,
      ncclComm->nRanks,
      ncclComm->cudaDev,
      ncclComm->cudaArch,
      ncclComm->busId,
      ncclComm->commHash,
      std::vector<ncclx::RankTopology>(), /* rankTopologies */
      std::vector<int>() /* commRanksToWorldRanks */);

  if (NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::nolocal) {
    commStateX->initRankTopologyNolocal();
  } else if (NCCL_COMM_STATE_DEBUG_TOPO == NCCL_COMM_STATE_DEBUG_TOPO::vnode) {
    FB_CHECKABORT(
        ncclComm->nRanks >= NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS,
        "CommStateX: NCCL_COMM_STATE_DEBUG_TOPO::vnode initialize failed because number of available ranks (%d) is less than nLocalRanks per vnode (NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS=%d).",
        ncclComm->nRanks,
        NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS);
    commStateX->initRankTopologyVnode(
        NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS);
  } else {
    commStateX->initRankStatesTopology(ctranComm->bootstrap_.get());
  }

  INFO(
      NCCL_INIT | NCCL_GRAPH,
      "CommStateX: initialize from ncclComm with %s",
      ncclx::topoNameMap[NCCL_COMM_STATE_DEBUG_TOPO].c_str());

  return commStateX;
}
