// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <spdlog/tweakme.h>
#include <spdlog/version.h>

#if SPDLOG_VERSION < 11502
#error "spdlog 1.15.2 or newer is required"
#endif

#if !defined(SPDLOG_FMT_EXTERNAL) && !defined(SPDLOG_USE_STD_FORMAT)
#error "spdlog must not use bundled fmt"
#endif

int main() {
  return 0;
}
