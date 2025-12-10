// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <stdexcept>
#include <string>

#define PIPES_CUDA_CHECK(EXPR)                      \
  do {                                              \
    const cudaError_t err = EXPR;                   \
    if (err == cudaSuccess) {                       \
      break;                                        \
    }                                               \
    std::string error_message;                      \
    error_message.append(__FILE__);                 \
    error_message.append(":");                      \
    error_message.append(std::to_string(__LINE__)); \
    error_message.append(" CUDA error: ");          \
    error_message.append(cudaGetErrorString(err));  \
    throw std::runtime_error(error_message);        \
  } while (0)

#define PIPES_KERNEL_LAUNCH_CHECK() PIPES_CUDA_CHECK(cudaGetLastError())
