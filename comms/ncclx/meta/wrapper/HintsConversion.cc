// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string>

#include "comms/ctran/algos/AllToAll/AllToAllPHintUtils.h"
#include "comms/ctran/window/WinHintUtils.h"
#include "meta/NcclxChecks.h"
#include "meta/wrapper/MetaFactory.h"

meta::comms::Hints ncclToMetaComm(const ncclx::Hints& hints) {
  meta::comms::Hints ret;
  std::string v;
  for (const auto& k : meta::comms::hints::AllToAllPHintUtils::keys()) {
    NCCLX_COMMCHECKTHROW(ncclToMetaComm(hints.get(k, v)));
    NCCLX_COMMCHECKTHROW(ret.set(k, v));
  }
  for (const auto& k : meta::comms::hints::WinHintUtils::keys()) {
    NCCLX_COMMCHECKTHROW(ncclToMetaComm(hints.get(k, v)));
    NCCLX_COMMCHECKTHROW(ret.set(k, v));
  }
  return ret;
}
