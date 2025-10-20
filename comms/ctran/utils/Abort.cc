// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/Abort.h"

namespace ctran::utils {

std::shared_ptr<Abort> createAbort(bool enabled) {
  if (enabled) {
    return std::make_shared<Abort>(/*enabled=*/true);
  } else {
    static const std::shared_ptr<Abort> o =
        std::make_unique<Abort>(/*enabled=*/false);
    return o;
  }
}

} // namespace ctran::utils
