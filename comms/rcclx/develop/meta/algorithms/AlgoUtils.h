// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/commSpecs.h"
#include "nccl.h"

namespace meta::comms {

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

} // namespace meta::comms
