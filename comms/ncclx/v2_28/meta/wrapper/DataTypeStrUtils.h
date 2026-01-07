// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "nccl.h"

inline std::string getDatatypeStr(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
      return "ncclInt8";
    case ncclUint8:
      return "ncclUint8";
    case ncclInt32:
      return "ncclInt32";
    case ncclUint32:
      return "ncclUint32";
    case ncclInt64:
      return "ncclInt64";
    case ncclUint64:
      return "ncclUint64";
    case ncclFloat16:
      return "ncclFloat16";
    case ncclFloat32:
      return "ncclFloat32";
    case ncclFloat64:
      return "ncclFloat64";
    case ncclBfloat16:
      return "ncclBfloat16";
    default:
      return "Unknown type";
  }
}

inline std::string getRedOpStr(ncclRedOp_t op) {
  switch (op) {
    case ncclSum:
      return "ncclSum";
    case ncclProd:
      return "ncclProd";
    case ncclMax:
      return "ncclMax";
    case ncclMin:
      return "ncclMin";
    case ncclAvg:
      return "ncclAvg";
    default:
      return "Unknown op";
  }
}
