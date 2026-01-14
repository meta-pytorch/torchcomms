/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdint.h>
#include <string_view>
#include "comms/utils/cvars/nccl_cvars.h"

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();
const char* ncclGetEnvStr(std::string_view name);

#define ncclGetEnv(name)                                                                                                                     \
  (([]() -> const char* {                                                                                                                    \
    static_assert(                                                                                                                           \
        ncclx::isCvarRegistered(name),                                                                                                       \
        "Unregistered CVAR \"" name                                                                                                          \
        "\". Please add this CVAR to \"fbcode/comms/utils/cvars/nccl_cvars.yaml\", regenerate the nccl_cvars.(h|cc) files, and recompile."); \
    return ncclGetEnvStr(name);                                                                                                              \
  })())

void ncclLoadParam(
    char const* env,
    int64_t deftVal,
    int64_t uninitialized,
    int64_t* cache);

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache);

#define NCCL_PARAM(name, env, deftVal) \
  static_assert(ncclx::isCvarRegistered("NCCL_" env), \
      "Unregistered CVAR \"NCCL_" env "\". Please add this CVAR to \"fbcode/comms/utils/cvars/nccl_cvars.yaml\", regenerate the nccl_cvars.(h|cc) files, and recompile."); \
  int64_t ncclParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized; \
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false)) { \
      ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache); \
    } \
    return cache; \
  }

void initNcclLogger();

#endif
