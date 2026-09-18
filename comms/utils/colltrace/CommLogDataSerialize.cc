// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CommLogDataSerialize.h"

namespace meta::comms::colltrace {

folly::dynamic commLogDataToDynamic(const CommLogData& m) {
  folly::dynamic result = folly::dynamic::object();

  result["commId"] = m.commId;
  result["commHash"] = m.commHash;
  result["commDesc"] = m.commDesc;
  result["rank"] = m.rank;
  result["nRanks"] = m.nRanks;

  return result;
}

CommLogData commLogDataFromDynamic(const folly::dynamic& d) {
  CommLogData result;

  result.commId = d["commId"].asInt();
  result.commHash = d["commHash"].asInt();
  result.commDesc = d["commDesc"].asString();
  result.rank = d["rank"].asInt();
  result.nRanks = d["nRanks"].asInt();

  return result;
}

} // namespace meta::comms::colltrace

folly::dynamic folly::DynamicConstructor<CommLogData>::construct(
    const CommLogData& m) {
  return meta::comms::colltrace::commLogDataToDynamic(m);
}

CommLogData folly::DynamicConverter<CommLogData>::convert(
    const folly::dynamic& d) {
  return meta::comms::colltrace::commLogDataFromDynamic(d);
}
