// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <string>

#include "comms/utils/commSpecs.h"

class ErrorStackTraceUtil {
 public:
  static commResult_t log(commResult_t result);

  // Useful if we detect an error in a function that does not return
  // ncclResult_t.
  static void logErrorMessage(std::string errorMessage);
};
