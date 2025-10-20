// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/hints/Hints.h"

#include "comms/ctran/algos/AllToAll/AllToAllPHintUtils.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicHintUtils.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/WinHintUtils.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms {

using meta::comms::hints::AllToAllPHintUtils;
using meta::comms::hints::AllToAllvDynamicHintUtils;
using meta::comms::hints::WinHintUtils;

Hints::Hints() {
  AllToAllvDynamicHintUtils::init(this->kv);
  AllToAllPHintUtils::init(this->kv);
  WinHintUtils::init(this->kv);
}

// TODO: Hints may not need individual submodules, consider simplification
commResult_t Hints::set(const std::string& key, const std::string& val) {
  if (key.starts_with("ncclx_alltoallv_dynamic")) {
    FB_COMMCHECK(AllToAllvDynamicHintUtils::set(key, val, this->kv));
    return commSuccess;
  } else if (key.starts_with("ncclx_alltoallp")) {
    FB_COMMCHECK(AllToAllPHintUtils::set(key, val, this->kv));
    return commSuccess;
  } else if (key.starts_with(("window"))) {
    FB_COMMCHECK(WinHintUtils::set(key, val, this->kv));
    return commSuccess;
  } else {
    return commInvalidArgument;
  }
}

commResult_t Hints::get(const std::string& key, std::string& val) const {
  auto iter = this->kv.find(key);
  if (iter != this->kv.end()) {
    val = iter->second;
    return commSuccess;
  } else {
    return commInvalidArgument;
  }
}

} // namespace meta::comms
