// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/hints/HintUtils.h"

#include <cctype>
#include <string>

#include "comms/utils/commSpecs.h"

namespace meta::comms::hints {

commResult_t HintUtils::clean_bool_string(
    const std::string& s_,
    std::string& b) {
  std::string s;
  for (auto x : s_) {
    s += std::tolower(x);
  }

  if (s == "y" || s == "yes" || s == "true" || s == "t" || s == "1") {
    b = "true";
  } else if (s == "n" || s == "no" || s == "false" || s == "f" || s == "0") {
    b = "false";
  } else {
    return commInvalidArgument;
  }

  return commSuccess;
}

commResult_t HintUtils::clean_location_string(
    const std::string& s_,
    std::string& b) {
  std::string s;
  for (auto x : s_) {
    s += std::tolower(x);
  }

  if (s == "cpu" || s == "gpu" || s == "auto") {
    b = s;
  } else {
    return commInvalidArgument;
  }

  return commSuccess;
}

} // namespace meta::comms::hints
