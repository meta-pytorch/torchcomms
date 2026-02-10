// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gtest/gtest.h>
#include <sys/statvfs.h>

#include "bootstrap.h" // @manual
#include "comm.h" // @manual
#include "nccl.h" // @manual
#include "param.h" // @manual
#include "transport.h" // @manual

#define NCCLCHECK_TEST(cmd)                  \
  do {                                       \
    ncclResult_t r = cmd;                    \
    if (r != ncclSuccess) {                  \
      printf(                                \
          "Failed, NCCL error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          ncclGetErrorString(r));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

// TODO: remove after ncclResult_t is fully migrated to commResult_t
#define NCCLCHECKTHROW_TEST(cmd)                                        \
  do {                                                                  \
    ncclResult_t r = cmd;                                               \
    if (r != ncclSuccess) {                                             \
      throw std::runtime_error(                                         \
          std::string("Failed, NCCL error: ") + ncclGetErrorString(r)); \
    }                                                                   \
  } while (0)
