// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <spdlog/tweakme.h>
#include <spdlog/version.h>

#if SPDLOG_VERSION < 11502
#error "spdlog 1.15.2 or newer is required"
#endif

#if defined(UNIFLOW_EXPECT_SPDLOG_EXTERNAL_FMT)
#if !defined(SPDLOG_FMT_EXTERNAL) || defined(SPDLOG_USE_STD_FORMAT)
#error "spdlog must use external fmt only"
#endif
#elif defined(UNIFLOW_EXPECT_SPDLOG_STD_FORMAT)
#if !defined(SPDLOG_USE_STD_FORMAT) || defined(SPDLOG_FMT_EXTERNAL)
#error "spdlog must use std::format only"
#endif
#elif !defined(SPDLOG_FMT_EXTERNAL) && !defined(SPDLOG_USE_STD_FORMAT)
#error "spdlog must not use bundled fmt"
#endif

int main() {
  return 0;
}
