// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/utils/commSpecs.h"
#include "nccl.h"

// Convert ncclx::Hints to meta::comms::Hints
meta::comms::Hints ncclToMetaComm(const ncclx::Hints& datatype);

inline ncclResult_t metaCommToNccl(commResult_t result) {
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
inline commResult_t ncclToMetaComm(ncclResult_t result) {
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

inline ncclRedOp_t metaCommToNccl(commRedOp_t op) {
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

inline commRedOp_t ncclToMetaComm(ncclRedOp_t op) {
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

inline commCmpOp_t ncclToMetaComm(ncclCmpOp_t op) {
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

inline ncclDataType_t metaCommToNccl(commDataType_t datatype) {
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

inline commDataType_t ncclToMetaComm(ncclDataType_t dataType) {
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

// Those are temorarly functions to initialized ctranComm from ncclComm
// TODO: remove this factory methods once we have proper CtranComm
// initialization
ctranConfig makeCtranConfigFrom(ncclComm* comm);
commResult_t setCtranCommBase(ncclComm* ncclCommVal);

// Public accessor returning the CtranComm* hanging off an ncclComm, or
// nullptr if it isn't initialized (e.g. NCCL_CTRAN_ENABLE not set).
CtranComm* getCtranCommFromNcclComm(ncclComm* ncclComm);
