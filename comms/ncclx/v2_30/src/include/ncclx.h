// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// NCCLX-only public C++ API.
//
// This header holds the NCCLX extensions to the public NCCL API that are not
// part of upstream NVIDIA NCCL. It is included at the tail of the generated
// nccl.h (behind IS_NCCLX) so every consumer of nccl.h sees these
// declarations, while keeping the forked upstream nccl.h.in free of the NCCLX
// API surface. It depends on the NCCL types (ncclComm_t, ncclResult_t, ...)
// declared earlier in nccl.h and must only be reached through nccl.h.
#ifndef NCCL_H_
#error "ncclx.h must be included through nccl.h, not directly."
#endif

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#define NCCL_COMM_DUMP
#define NCCL_COMM_DUMP_ALL

/* Dump NCCL current internal state for a given communicator in a key-value store format.
 * define outside extern "C"{} to pass C++ template */
ncclResult_t  ncclCommDump(ncclComm_t comm, std::unordered_map<std::string, std::string>& map);

/* Dump NCCL current internal state for all the communicators.
 * The returned map is in the format {commHash: {key: value}} where
 * {key: value} is the result of ncclCommDump in the communicator with hash commHash.
 * hints: key-value map of options. Supported hints:
 *   "comm_dump::requestFields" — semicolon-separated list of field names to include.
 *   "comm_dump::flush" — "1" to flush ring buffers before dumping.
 *   Empty map (default) dumps all fields without flushing.
 */
ncclResult_t ncclCommDumpAll(std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& map,
    const std::unordered_map<std::string, std::string>& hints = {});

// NCCL_HAS_DUMP_ALGO_STAT controls whether dumpAlgoStat() is available.
// To disable (e.g., when using a shim with a
// different ncclComm layout), compile with -DNCCL_HAS_DUMP_ALGO_STAT=0.
#if !defined(NCCL_HAS_DUMP_ALGO_STAT)
#define NCCL_HAS_DUMP_ALGO_STAT
#elif NCCL_HAS_DUMP_ALGO_STAT == 0
#undef NCCL_HAS_DUMP_ALGO_STAT
#endif

#ifdef NCCL_HAS_DUMP_ALGO_STAT
namespace ncclx::colltrace {

// Dump collective algorithm statistics for a communicator.
// Output map format: collective name -> algorithm name -> call count.
// Requires NCCL_COLLTRACE=algostat to be enabled.
// Clears and populates the output map. Empty if algostat not enabled or comm is null.
void dumpAlgoStat(ncclComm_t comm, std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>& map);

} // namespace ncclx::colltrace
#endif // NCCL_HAS_DUMP_ALGO_STAT
