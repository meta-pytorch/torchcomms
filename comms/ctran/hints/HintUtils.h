// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

#include "comms/utils/commSpecs.h"

namespace meta::comms::hints {

class HintUtils {
 public:
  static commResult_t clean_bool_string(const std::string& s, std::string& b);
  static commResult_t clean_location_string(
      const std::string& s,
      std::string& b);
};

} // namespace meta::comms::hints
