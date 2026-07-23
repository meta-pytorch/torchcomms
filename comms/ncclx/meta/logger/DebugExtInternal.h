// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdarg>
#include <mutex>

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
