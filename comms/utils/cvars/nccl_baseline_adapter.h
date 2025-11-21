// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef NCCL_BASELINE_ADAPTER_H_INCLUDED
#define NCCL_BASELINE_ADAPTER_H_INCLUDED

#include <pthread.h>
#include <pwd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

namespace nccl_baseline_adapter {

/**
 *  @brief Load a numeric/integer-based CVAR.
 */
void ncclLoadParam(
    char const* env,
    int64_t deftVal,
    int64_t uninitialized,
    int64_t* cache);

/**
 * Load a string-based CVAR.
 */
const char* ncclGetEnvImpl(const char* name);
} // namespace nccl_baseline_adapter

#endif // NCCL_BASELINE_ADAPTER_H_INCLUDED
