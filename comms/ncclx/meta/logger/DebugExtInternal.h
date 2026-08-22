// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdarg>
#include <mutex>

#include "nccl_common.h"

// Internal seam between the forked upstream debug.cc -- which owns the debug
// state and its lazy initialization -- and the hoisted Meta logging entry
// points in comms/ncclx/meta/logger/DebugExt.cc. Both translation units are
// linked into the same NCCLX library, so these symbols resolve directly.
//
// This is NOT part of any public NCCL API; only debug.cc and DebugExt.cc
// should include it.

// Guards ncclLastError and the lazy ncclDebugInit() call below.
extern std::mutex ncclDebugMutex;

// Parses NCCL_DEBUG* environment variables and populates the debug level,
// subsystem mask, timestamp settings and debug file. Must be called with
// ncclDebugMutex held.
void ncclDebugInit();

// Records the last WARN/ERROR message into ncclLastError, whose fixed-size
// storage is private to debug.cc. Callers must hold ncclDebugMutex.
void ncclDebugSaveLastError(const char* fmt, va_list vargs);

// Shared terminal logging sink, defined in DebugExt.cc and used by both the
// forked upstream ncclDebugLog (debug.cc) and the Meta ncclMetaDebugLog
// (DebugExt.cc): formats the printf message and emits one line through the
// common NCCLX logger. Keeping the single copy here is what lets the forked
// debug.cc stay a thin shim over the exported ncclDebugLog symbol.
// The caller owns `vargs` (must va_start and va_end it); the sink va_copy's it.
void ncclMetaEmitLog(
    ncclDebugLogLevel level,
    const char* file,
    int line,
    const char* func,
    const char* fmt,
    va_list vargs);
