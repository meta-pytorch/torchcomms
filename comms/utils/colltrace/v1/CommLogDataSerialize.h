// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/DynamicConverter.h>
#include <folly/dynamic.h>

#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

folly::dynamic commLogDataToDynamic(const CommLogData& metadata);
CommLogData commLogDataFromDynamic(const folly::dynamic& dynamic);

} // namespace meta::comms::colltrace

namespace folly {

// Defines dynamic constructor for ICollMetadata so we can use it during
// serialization.
template <>
struct DynamicConstructor<CommLogData> {
  static dynamic construct(const CommLogData& m);
};

template <>
struct DynamicConverter<CommLogData> {
  static CommLogData convert(const dynamic& d);
};

} // namespace folly
