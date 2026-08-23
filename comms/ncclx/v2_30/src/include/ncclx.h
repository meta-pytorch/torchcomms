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
#include <vector>

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

/* Snapshot of a communicator's collective tuning model.
 *
 * The bandwidths/latencies tables are the raw model built once at communicator
 * init by ncclTopoTuneModel(): algorithm bandwidth in GB/s (0 means the
 * (function, algorithm, protocol) combination is disabled or unavailable,
 * including NCCL_ALGO/NCCL_PROTO masks) and base latency in microseconds.
 * They are uncorrected: the per-size correction factors that
 * ncclTopoGetAlgoTime() applies are NOT reflected in them.
 *
 * bestBySize[f][s] is the evaluated model at messageSizes[s] bytes: the
 * algorithm/protocol/channel/thread selection a real collective of that size
 * would run with, and its predicted execution time in microseconds with all
 * correction factors applied. It is produced by the same selection path real
 * calls take (including an external tuner plugin when one is loaded), with
 * numPipeOps = 1 and unregistered buffers (regBuff = 0). algorithm = -1 and
 * timeUs = -1 mean no combination is available. Within an interval where the
 * selection does not change, predicted time is affine in nBytes, so linear
 * interpolation between adjacent entries reproduces the model.
 *
 * Not modeled: the fp8 ring relegation for comms larger than 8 ranks (a
 * precision preference, not a time estimate), and per-call op aggregation
 * (numPipeOps > 1 scales only the latency term).
 */
#define NCCL_COLL_TUNING_VERSION 1
#define NCCL_TUNING_MAX_FUNCTIONS 8
#define NCCL_TUNING_MAX_ALGORITHMS 16
#define NCCL_TUNING_MAX_PROTOCOLS 8
#define NCCL_TUNING_SIZE_POINTS 31
#define NCCL_TUNING_NAME_LEN 16 /* incl. NUL; longest today is "ReduceScatter" */

typedef struct {
  int8_t algorithm;  /* index into algorithmNames, -1 if none available */
  int8_t protocol;   /* index into protocolNames */
  int16_t nChannels;
  int16_t nThreads;
  float timeUs;
} ncclCollTuningEntry;

typedef struct {
  int version; /* NCCL_COLL_TUNING_VERSION */

  int nRanks, nNodes, nChannels;
  int minCompCap, maxCompCap;

  int numFunctions, numAlgorithms, numProtocols;
  char functionNames[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_NAME_LEN];
  char algorithmNames[NCCL_TUNING_MAX_ALGORITHMS][NCCL_TUNING_NAME_LEN];
  char protocolNames[NCCL_TUNING_MAX_PROTOCOLS][NCCL_TUNING_NAME_LEN];

  float bandwidths[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_MAX_ALGORITHMS][NCCL_TUNING_MAX_PROTOCOLS];
  float latencies[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_MAX_ALGORITHMS][NCCL_TUNING_MAX_PROTOCOLS];

  uint64_t messageSizes[NCCL_TUNING_SIZE_POINTS]; /* messageSizes[s] = 1 << s */
  ncclCollTuningEntry bestBySize[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_SIZE_POINTS];
} ncclCollTuning;

/* Fill `tuning` from the communicator's init-time tuning state. Read-only
 * with respect to the communicator and callable from any thread once the
 * communicator is initialized. */
ncclResult_t ncclQueryCollTuning(ncclComm_t comm, ncclCollTuning* tuning);

namespace ncclx::colltrace {

inline constexpr uint64_t kInvalidReplayId = UINT64_MAX;

enum class LifecycleEventType : uint8_t {
  Enqueue,
  Start,
  End,
};

struct LifecycleEvent {
  uint64_t replayId{kInvalidReplayId};
  uint64_t commId{0};
  // Stable one-based submission identity. Zero is reserved for no record.
  // Matches getLatestCollTraceCollectiveId().
  uint64_t collId{0};
  // One-based per-execution identity. Differs from collId for graph replays.
  uint64_t executionCollId{0};
  LifecycleEventType eventType{LifecycleEventType::Enqueue};
  double timestamp{0};
};

// Returns the process-local communicator identity used by lifecycle events.
// Returns ncclInvalidArgument for a null communicator or ncclInvalidUsage if
// the lifecycle feed is disabled.
ncclResult_t getCollTraceCommId(ncclComm_t comm, uint64_t& commId);

// Returns the latest one-based collective ID submitted on this communicator.
// Writes zero and returns ncclSuccess if no collective has been submitted on
// this communicator. Returns ncclInvalidArgument for a null communicator or
// ncclInvalidUsage if the lifecycle feed is disabled.
ncclResult_t getLatestCollTraceCollectiveId(ncclComm_t comm, uint64_t& collId);

// Destructively drains lifecycle events across all lifecycle-enabled
// communicators in this process. This call blocks until every registered
// colltrace instance finishes a flush. Events are ordered by timestamp.
ncclResult_t drainUnreadLifecycleEvents(std::vector<LifecycleEvent>& events);

} // namespace ncclx::colltrace

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
