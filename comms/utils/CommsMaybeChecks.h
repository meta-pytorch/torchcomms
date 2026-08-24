// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/logger/SpdlogLogger.h"

#define EXPECT_CHECK_ALWAYS_RETURN(cmd)                                        \
  {                                                                            \
    const auto res = cmd;                                                      \
    if (res.hasError()) {                                                      \
      COMMS_LOG(ERR, "Call for {} failed with {}", #cmd, res.error().message); \
      return folly::makeUnexpected(std::move(res.error()));                    \
    }                                                                          \
    return res;                                                                \
  }

#define EXPECT_CHECK(cmd)                                                      \
  {                                                                            \
    const auto res = cmd;                                                      \
    if (res.hasError()) {                                                      \
      COMMS_LOG(ERR, "Call for {} failed with {}", #cmd, res.error().message); \
      return folly::makeUnexpected(std::move(res.error()));                    \
    }                                                                          \
  }

#define EXPECT_CHECK_LOG_FIRST_N(n, cmd)                                    \
  {                                                                         \
    const auto res = cmd;                                                   \
    if (res.hasError()) {                                                   \
      COMMS_LOG_STREAM_FIRST_N(ERR, n)                                      \
          << "Call for " << #cmd << " failed with " << res.error().message; \
      return folly::makeUnexpected(std::move(res.error()));                 \
    }                                                                       \
  }

#define EXPECT_CHECK_CONTINUE_LOG_FIRST_N(cmd, n)                           \
  {                                                                         \
    const auto res = cmd;                                                   \
    if (res.hasError()) {                                                   \
      COMMS_LOG_STREAM_FIRST_N(ERR, n)                                      \
          << "Call for " << #cmd << " failed with " << res.error().message; \
      continue;                                                             \
    }                                                                       \
  }

#define EXPECT_CHECK_RES(res)                               \
  {                                                         \
    if (res.hasError()) {                                   \
      return folly::makeUnexpected(std::move(res.error())); \
    }                                                       \
  }
