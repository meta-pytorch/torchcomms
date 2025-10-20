// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ALLTOALLV_DYNAMIC_HINTS_H_
#define ALLTOALLV_DYNAMIC_HINTS_H_

#include <string>
#include <unordered_map>

#include "comms/utils/commSpecs.h"

namespace meta::comms::hints {

typedef std::unordered_map<std::string, std::string> kvType;

class AllToAllvDynamicHintUtils {
 public:
  static void init(kvType& kv);
  static commResult_t
  set(const std::string& key, const std::string& val, kvType& kv);
  static const std::vector<std::string>& keys();
};

} // namespace meta::comms::hints

#endif /* ALLTOALLV_DYNAMIC_HINTS_H_ */
