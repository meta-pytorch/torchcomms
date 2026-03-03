// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

#define CUDACHECK(cmd)                          \
  do {                                          \
    cudaError_t err = cmd;                      \
    if (err != cudaSuccess) {                   \
      fprintf(                                  \
          stderr,                               \
          "Failed: CUDA error %s:%d '%s'\n",    \
          __FILE__,                             \
          __LINE__,                             \
          cudaGetErrorString(err));             \
      throw std::runtime_error("CUDA error\n"); \
    }                                           \
  } while (0)

#define NCCLCHECK(cmd)                          \
  do {                                          \
    ncclResult_t res = cmd;                     \
    if (res != ncclSuccess) {                   \
      fprintf(                                  \
          stderr,                               \
          "Failed, NCCL error %s:%d '%s'\n",    \
          __FILE__,                             \
          __LINE__,                             \
          ncclGetErrorString(res));             \
      throw std::runtime_error("NCCL error\n"); \
    }                                           \
  } while (0)
