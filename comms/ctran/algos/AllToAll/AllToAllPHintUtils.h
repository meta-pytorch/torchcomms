// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <string>
#include <unordered_map>

#include "comms/utils/commSpecs.h"

namespace meta::comms::hints {

typedef std::unordered_map<std::string, std::string> kvType;

class AllToAllPHintUtils {
 public:
  static void init(kvType& kv);
  static commResult_t
  set(const std::string& key, const std::string& val, kvType& kv);

  static const std::vector<std::string>& keys();
};

} // namespace meta::comms::hints
