// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdexcept>

#include "comm.h"
#include "comms/ctran/algos/AllToAll/AllToAllPHintUtils.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicHintUtils.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/WinHintUtils.h"
#include "comms/utils/commSpecs.h"
#include "meta/wrapper/MetaFactory.h"

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

// Convert ncclResult_t to ctranResult_t
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

ncclRedOp_t metaCommToNccl(commRedOp_t op) {
  switch (op) {
    case commSum:
      return ncclSum;
    case commProd:
      return ncclProd;
    case commMax:
      return ncclMax;
    case commMin:
      return ncclMin;
    case commAvg:
      return ncclAvg;
    case commNumOps:
      return ncclNumOps;
    case commMaxRedOp:
      return ncclMaxRedOp;
    default:
      throw std::runtime_error(
          std::string("commToNccl: unimplemented comm RedOp ") +
          std::to_string(op));
  }
}

commRedOp_t ncclToMetaComm(ncclRedOp_t op) {
  switch (op) {
    case ncclSum:
      return commSum;
    case ncclProd:
      return commProd;
    case ncclMax:
      return commMax;
    case ncclMin:
      return commMin;
    case ncclAvg:
      return commAvg;
    case ncclNumOps:
      return commNumOps;
    case ncclMaxRedOp:
      return commMaxRedOp;
    default:
      throw std::runtime_error(
          std::string("ncclToComm: unimplemented nccl RedOp ") +
          std::to_string(op));
  }
}

commCmpOp_t ncclToMetaComm(ncclCmpOp_t op) {
  switch (op) {
    case ncclCmpEQ:
      return commCmpEQ;
    case ncclCmpGE:
      return commCmpGE;
    case ncclCmpLE:
      return commCmpLE;
    default:
      throw std::runtime_error(
          std::string("ncclToComm: unimplemented nccl CmpOp ") +
          std::to_string(op));
  }
}

ncclDataType_t metaCommToNccl(commDataType_t datatype) {
  switch (datatype) {
    case commInt8:
      return ncclInt8;
    case commUint8:
      return ncclUint8;
    case commInt32:
      return ncclInt32;
    case commUint32:
      return ncclUint32;
    case commInt64:
      return ncclInt64;
    case commUint64:
      return ncclUint64;
    case commFloat16:
      return ncclFloat16;
    case commFloat32:
      return ncclFloat32;
    case commFloat64:
      return ncclFloat64;
    case commBfloat16:
      return ncclBfloat16;
    case commFloat8e4m3:
      return ncclFloat8e4m3;
    case commFloat8e5m2:
      return ncclFloat8e5m2;
    default:
      throw std::runtime_error(
          std::string("commToNccl: unimplemented comm DataType") +
          std::to_string(datatype));
  }
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

meta::comms::Hints ncclToMetaComm(const ncclx::Hints& hints) {
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

ctranConfig makeCtranConfigFrom(ncclComm* comm) {
  struct ctranConfig tconfig = {
      .blocking = comm->config.blocking,
      .commDesc = comm->config.commDesc,
      .ncclAllGatherAlgo = comm->config.ncclAllGatherAlgo,
  };
  return tconfig;
}

// TODO: remove this factory method once we have proper CtranComm initialization
// Initialize all fields except Ctran. Since Ctran/Bootstra/Colltrace requires
// stateX and other fields to be initialized beforehand, we split its
// initialization into two parts:
// 1. Pre-initialization to enable Ctran/Bootstra/Colltrace initialization.
// 2. Final initialization (final-init) to set up the remaining fields.
commResult_t setCtranCommBase(ncclComm* ncclCommVal) {
  if (!ncclCommVal) {
    return commInvalidArgument;
  }

  // can not call make_unique with a private constructor
  // CtranComm has provate constructor for sagety reasons for now
  // no one should use CtranComm constructor untill refactoring is finished
  // TODO: move to make_unique after finish refactoring and defining a proper
  // constructor
  ncclCommVal->ctranComm_ =
      std::unique_ptr<CtranComm>(std::move(new CtranComm()));

  const auto tconfig = makeCtranConfigFrom(ncclCommVal);
  ncclCommVal->ctranComm_->config_ = tconfig;
  ncclCommVal->ctranComm_->opCount_ = &ncclCommVal->opCount;
  ncclCommVal->ctranComm_->logMetaData_ = ncclCommVal->logMetaData;
  ncclCommVal->ctranComm_->runtimeConn_ = ncclCommVal->runtimeConn;

  return commSuccess;
}
