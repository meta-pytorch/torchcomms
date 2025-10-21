// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAll/AllToAllvDynamicHintUtils.h"
#include "comms/ctran/hints/HintUtils.h"
#include "comms/ctran/utils/Checks.h"

namespace meta::comms::hints {

namespace {
const std::string kNcclxAlltoallvDynamicSendBuffsContig =
    "ncclx_alltoallv_dynamic_sendbuffs_contig";
const std::string kNcclxAlltoallvDynamicRecvBuffsContig =
    "ncclx_alltoallv_dynamic_recvbuffs_contig";
const std::string kNcclxAlltoallvDynamicSendBuffsLocation =
    "ncclx_alltoallv_dynamic_sendbuffs_location";
const std::string kNcclxAlltoallvDynamicSendcountsLocation =
    "ncclx_alltoallv_dynamic_sendcounts_location";
const std::string kNcclxAlltoallvDynamicRecvBuffsLocation =
    "ncclx_alltoallv_dynamic_recvbuffs_location";
const std::string kNcclxAlltoallvDynamicMaxSendcountsLocation =
    "ncclx_alltoallv_dynamic_max_sendcounts_location";
const std::string kNcclxAlltoallvDynamicMaxRecvcountsLocation =
    "ncclx_alltoallv_dynamic_max_recvcounts_location";
const std::string kNcclxAlltoallvDynamicActualRecvcountsLocation =
    "ncclx_alltoallv_dynamic_actual_recvcounts_location";
} // namespace

void AllToAllvDynamicHintUtils::init(kvType& kv) {
  kv[kNcclxAlltoallvDynamicSendBuffsContig] = "false";
  kv[kNcclxAlltoallvDynamicRecvBuffsContig] = "false";
  kv[kNcclxAlltoallvDynamicSendBuffsLocation] = "auto";
  kv[kNcclxAlltoallvDynamicSendcountsLocation] = "auto";
  kv[kNcclxAlltoallvDynamicRecvBuffsLocation] = "auto";
  kv[kNcclxAlltoallvDynamicMaxSendcountsLocation] = "auto";
  kv[kNcclxAlltoallvDynamicMaxRecvcountsLocation] = "auto";
  kv[kNcclxAlltoallvDynamicActualRecvcountsLocation] = "auto";
}

commResult_t AllToAllvDynamicHintUtils::set(
    const std::string& key,
    const std::string& val,
    kvType& kv) {
  std::string b;

  if (key == kNcclxAlltoallvDynamicSendBuffsContig ||
      key == kNcclxAlltoallvDynamicRecvBuffsContig) {
    FB_COMMCHECK(HintUtils::clean_bool_string(val, b));
    kv[key] = b;
  } else if (
      key == kNcclxAlltoallvDynamicSendBuffsLocation ||
      key == kNcclxAlltoallvDynamicSendcountsLocation ||
      key == kNcclxAlltoallvDynamicRecvBuffsLocation ||
      key == kNcclxAlltoallvDynamicMaxSendcountsLocation ||
      key == kNcclxAlltoallvDynamicMaxRecvcountsLocation ||
      key == kNcclxAlltoallvDynamicActualRecvcountsLocation) {
    FB_COMMCHECK(HintUtils::clean_location_string(val, b));
    kv[key] = b;
  } else {
    return commInvalidArgument;
  }

  return commSuccess;
}

const std::vector<std::string>& AllToAllvDynamicHintUtils::keys() {
  static std::vector<std::string> kKeys = {
      kNcclxAlltoallvDynamicSendBuffsContig,
      kNcclxAlltoallvDynamicRecvBuffsContig,
      kNcclxAlltoallvDynamicSendBuffsLocation,
      kNcclxAlltoallvDynamicSendcountsLocation,
      kNcclxAlltoallvDynamicRecvBuffsLocation,
      kNcclxAlltoallvDynamicMaxSendcountsLocation,
      kNcclxAlltoallvDynamicMaxRecvcountsLocation,
      kNcclxAlltoallvDynamicActualRecvcountsLocation,
  };
  return kKeys;
}

} // namespace meta::comms::hints
