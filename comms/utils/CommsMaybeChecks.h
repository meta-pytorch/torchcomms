// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/format.h>
#include <folly/logging/xlog.h>

#define EXPECT_CHECK_ALWAYS_RETURN(cmd)                             \
  {                                                                 \
    const auto res = cmd;                                           \
    if (res.hasError()) {                                           \
      XLOG(ERR) << fmt::format(                                     \
          "Call for {} failed with {}", #cmd, res.error().message); \
      return folly::makeUnexpected(std::move(res.error()));         \
    }                                                               \
    return res;                                                     \
  }

#define EXPECT_CHECK(cmd)                                           \
  {                                                                 \
    const auto res = cmd;                                           \
    if (res.hasError()) {                                           \
      XLOG(ERR) << fmt::format(                                     \
          "Call for {} failed with {}", #cmd, res.error().message); \
      return folly::makeUnexpected(std::move(res.error()));         \
    }                                                               \
  }

#define EXPECT_CHECK_LOG_FIRST_N(n, cmd)                            \
  {                                                                 \
    const auto res = cmd;                                           \
    if (res.hasError()) {                                           \
      XLOG_FIRST_N(ERR, n) << fmt::format(                          \
          "Call for {} failed with {}", #cmd, res.error().message); \
      return folly::makeUnexpected(std::move(res.error()));         \
    }                                                               \
  }

#define EXPECT_CHECK_CONTINUE_LOG_FIRST_N(cmd, n)                   \
  {                                                                 \
    const auto res = cmd;                                           \
    if (res.hasError()) {                                           \
      XLOG_FIRST_N(ERR, n) << fmt::format(                          \
          "Call for {} failed with {}", #cmd, res.error().message); \
      continue;                                                     \
    }                                                               \
  }

#define EXPECT_CHECK_RES(res)                               \
  {                                                         \
    if (res.hasError()) {                                   \
      return folly::makeUnexpected(std::move(res.error())); \
    }                                                       \
  }
