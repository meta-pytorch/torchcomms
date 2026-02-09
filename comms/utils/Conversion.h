// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

#include "comms/utils/commSpecs.h"

namespace meta::comms {

constexpr const char* commCodeToName(const commResult_t code) {
  switch (code) {
    case commSuccess:
      return "commSuccess";
    case commUnhandledCudaError:
      return "commUnhandledCudaError";
    case commSystemError:
      return "commSystemError";
    case commInternalError:
      return "commInternalError";
    case commInvalidArgument:
      return "commInvalidArgument";
    case commInvalidUsage:
      return "commInvalidUsage";
    case commRemoteError:
      return "commRemoteError";
    case commInProgress:
      return "commInProgress";
    case commTimeout:
      return "commTimeout";
    case commUserAbort:
      return "commUserAbort";
    case commNumResults:
      return "commNumResults";
    default:
      return "unknown result code";
  }
}

// String conversion functions for CommPattern
std::string_view commPatternToString(CommPattern pattern);
CommPattern stringToCommPattern(std::string_view str);

// String conversion functions for CommFunc
std::string_view commFuncToString(CommFunc func);
CommFunc stringToCommFunc(std::string_view str);

// String conversion functions for CommRedOp
std::string_view commRedOpToString(commRedOp_t op);
commRedOp_t stringToCommRedOp(std::string_view str);

// String conversion functions for CommAlgo
std::string_view commAlgoToString(CommAlgo algo);
CommAlgo stringToCommAlgo(std::string_view str);

// String conversion functions for CommProtocol
std::string_view commProtocolToString(CommProtocol protocol);
CommProtocol stringToCommProtocol(std::string_view str);

constexpr const char* commCodeToString(commResult_t code) {
  switch (code) {
    case commSuccess:
      return "no error";
    case commUnhandledCudaError:
      return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case commSystemError:
      return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case commInternalError:
      return "internal error - please report this issue to the NCCL developers";
    case commInvalidArgument:
      return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case commInvalidUsage:
      return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case commRemoteError:
      return "remote process exited or there was a network error";
    case commInProgress:
      return "NCCL operation in progress";
    case commTimeout:
      return "operation timed out";
    case commUserAbort:
      return "operation aborted by user";
    case commNumResults:
      return "numericall error";
    default:
      return "unknown result code";
  }
}

constexpr std::string_view getCommsDatatypeStr(commDataType_t type) {
  switch (type) {
    case commInt8:
      return "commInt8";
    case commUint8:
      return "commUint8";
    case commInt32:
      return "commInt32";
    case commUint32:
      return "commUint32";
    case commInt64:
      return "commInt64";
    case commUint64:
      return "commUint64";
    case commFloat16:
      return "commFloat16";
    case commFloat32:
      return "commFloat32";
    case commFloat64:
      return "commFloat64";
    case commBfloat16:
      return "commBfloat16";
    case commFloat8e4m3:
      return "commFloat8e4m3";
    case commFloat8e5m2:
      return "commFloat8e5m2";
    default:
      return "Unknown type";
  }
}

constexpr commDataType_t stringToCommsDatatype(std::string_view str) {
  if (str == "commInt8") {
    return commInt8;
  }
  if (str == "commUint8") {
    return commUint8;
  }
  if (str == "commInt32") {
    return commInt32;
  }
  if (str == "commUint32") {
    return commUint32;
  }
  if (str == "commInt64") {
    return commInt64;
  }
  if (str == "commUint64") {
    return commUint64;
  }
  if (str == "commFloat16") {
    return commFloat16;
  }
  if (str == "commFloat32") {
    return commFloat32;
  }
  if (str == "commFloat64") {
    return commFloat64;
  }
  if (str == "commBfloat16") {
    return commBfloat16;
  }
  if (str == "commFloat8e4m3") {
    return commFloat8e4m3;
  }
  if (str == "commFloat8e5m2") {
    return commFloat8e5m2;
  }
  return commNumTypes; // Return an invalid type if not found
}

} // namespace meta::comms
