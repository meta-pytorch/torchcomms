// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAll/AllToAllPHintUtils.h"

#include "comms/ctran/hints/HintUtils.h"
#include "comms/ctran/utils/Checks.h"

namespace meta::comms::hints {

namespace {
const std::string kNcclxAlltoallpSkipCtrlMsgExchange =
    "ncclx_alltoallp_skip_ctrl_msg_exchange";
} // namespace

void AllToAllPHintUtils::init(kvType& kv) {
  kv[kNcclxAlltoallpSkipCtrlMsgExchange] = "false";
}

commResult_t AllToAllPHintUtils::set(
    const std::string& key,
    const std::string& val,
    kvType& kv) {
  std::string b;

  if (key == kNcclxAlltoallpSkipCtrlMsgExchange) {
    FB_COMMCHECK(HintUtils::clean_bool_string(val, b));
    kv[key] = b;
  } else {
    return commInvalidArgument;
  }

  return commSuccess;
}

const std::vector<std::string>& AllToAllPHintUtils::keys() {
  static std::vector<std::string> kKeys = {
      kNcclxAlltoallpSkipCtrlMsgExchange,
  };
  return kKeys;
}

} // namespace meta::comms::hints
