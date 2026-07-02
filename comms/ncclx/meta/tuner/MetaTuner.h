// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef NCCLX_META_TUNER_META_TUNER_H_
#define NCCLX_META_TUNER_META_TUNER_H_

#include <cstddef>
#include <string>
#include <vector>

#include "meta/tuner/Int64Range.h"
#include "nccl_tuner.h"

struct ncclComm;

namespace ncclx::tuner {

// One tuning rule parsed from a CSV row or JSON object. A rule overrides NCCL
// core algorithm/protocol selection (and optionally nChannels / chunkSize) for
// collectives whose attributes AND-match every populated field. The
// bytesPerRank, nNodes and nLocalRanks fields are Int64Range matchers (an
// interval, an exact value, or the "*" wildcard that matches any value);
// numPipeOps / regBuff are exact-or-(-1) int matches; chunkSize == 0 means "no
// override". bytesPerRank matches nBytes / nRanks (nRanks = nNodes *
// nLocalRanks, taken from the comm). For AllGather/ReduceScatter the tuner
// nBytes is the rank-scaled total, so nBytes / nRanks is the per-rank shard;
// for AllReduce nBytes is the full buffer, so it is buffer / nRanks (see
// matchesCollective).
struct TuningConfig {
  ncclFunc_t collType;
  Int64Range bytesPerRank;
  int algorithm;
  int protocol;
  int nChannels;
  Int64Range nNodes;
  Int64Range nLocalRanks;
  int numPipeOps;
  int regBuff;
  size_t chunkSize;

  std::string toString() const;
};

// Returns true when NCCLX_TUNER_CONFIG_FILE is set to a non-empty value, i.e.
// the built-in tuner should be wired into comm->tuner as a fallback when no
// external NCCL_TUNER_PLUGIN is present.
bool metaTunerEnabled();

// Statically-linked tuner function table. Core takes the address of this
// (it is NOT dlopen-discovered) and assigns it to comm->tuner. Field order
// matches this version's ncclTuner_t: name, init, getCollInfo, finalize, and
// (on tuner API v6 only) getChunkSize.
extern const ncclTuner_t kMetaTuner;

// Built-in CSV/JSON tuner entry point for NCCL's tuner-load path.
//
// When NCCLX_TUNER_CONFIG_FILE is set, this assigns comm->tuner to the
// statically linked built-in tuner (kMetaTuner), logs, and returns true so
// the caller skips the external NCCL_TUNER_PLUGIN dlopen path entirely (the
// built-in tuner takes precedence). The cvar is process-global and read
// identically by every comm, so no plugin-load status machinery is needed.
// comm->tunerPluginLoaded is intentionally left 0 so ncclTunerPluginUnload
// skips dlclose on the static struct. Returns false (no-op) when the cvar is
// empty, leaving comm->tuner untouched (zero regression).
bool tryLoadMetaTuner(struct ncclComm* comm);

} // namespace ncclx::tuner

#endif // NCCLX_META_TUNER_META_TUNER_H_
