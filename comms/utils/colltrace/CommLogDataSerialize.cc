// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CommLogDataSerialize.h"

// We need to define in folly namespace, so intentionally not using namespace
// here.

folly::dynamic folly::DynamicConstructor<CommLogData>::construct(
    const CommLogData& m) {
  folly::dynamic result = folly::dynamic::object();

  result["commId"] = m.commId;
  result["commHash"] = m.commHash;
  result["commDesc"] = m.commDesc;
  result["rank"] = m.rank;
  result["nRanks"] = m.nRanks;

  return result;
}

CommLogData folly::DynamicConverter<CommLogData>::convert(
    const folly::dynamic& d) {
  CommLogData result;

  result.commId = d["commId"].asInt();
  result.commHash = d["commHash"].asInt();
  result.commDesc = d["commDesc"].asString();
  result.rank = d["rank"].asInt();
  result.nRanks = d["nRanks"].asInt();

  return result;
}
