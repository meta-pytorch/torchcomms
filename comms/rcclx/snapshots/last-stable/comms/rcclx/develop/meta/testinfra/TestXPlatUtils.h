// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gtest/gtest.h>
#include <sys/statvfs.h>

#define EXPECT_CHECK_TEST(cmd)       \
  {                                  \
    auto ret = cmd;                  \
    if (ret.hasError()) {            \
      FAIL() << ret.error().message; \
    }                                \
  }

#define CUDACHECK_TEST(cmd)                  \
  do {                                       \
    cudaError_t e = cmd;                     \
    if (e != cudaSuccess) {                  \
      printf(                                \
          "Failed: Cuda error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          cudaGetErrorString(e));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)
