// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// NCCLX-only debug/thread helpers layered on top of the pristine upstream NCCL
// debug.h. These have no upstream equivalent, so they are hoisted here and
// pulled in by a single trailing include at the bottom of debug.h, keeping the
// forked upstream file closer to pristine. The NCCL_NAMED_THREAD_START* macros
// expand INFO(...), which debug.h defines before this header is included.
//
// Do not include this header directly; include "debug.h".

#include <string_view>

// This header is a trailing include of debug.h and relies on INFO(...) /
// NCCL_INIT being defined there. Fail loudly if it is pulled in on its own
// instead of emitting a confusing macro-expansion error.
#ifndef INFO
#error "Include debug.h, not meta/checks/MetaDebug.h directly"
#endif

void ncclSetMyThreadLoggingName(std::string_view name);

#define NCCL_NAMED_THREAD_START(threadName)       \
  do {                                            \
    ncclSetMyThreadLoggingName(threadName);       \
    INFO(                                         \
        NCCL_INIT,                                \
        "[NCCL THREAD] Starting %s thread at %s", \
        threadName,                               \
        __func__);                                \
  } while (0)

#define NCCL_NAMED_THREAD_START_EXT(threadName, rank, commHash, commDesc)              \
  do {                                                                                 \
    ncclSetMyThreadLoggingName(threadName);                                            \
    INFO(                                                                              \
        NCCL_INIT,                                                                     \
        "[NCCL THREAD] Starting %s thread for rank %d commHash %lx commDesc %s at %s", \
        threadName,                                                                    \
        rank,                                                                          \
        commHash,                                                                      \
        commDesc.c_str(),                                                              \
        __func__);                                                                     \
  } while (0)
