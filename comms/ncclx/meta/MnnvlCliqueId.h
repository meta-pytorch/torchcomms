// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include "nccl.h"

namespace ncclx {

// Derive the MNNVL clique id for this rank from NCCL_MNNVL_CLIQUE_SIZE, as
// `globalRank / NCCL_MNNVL_CLIQUE_SIZE`.
//
// This soft-partitions the NVLink domain on a rank-derived boundary instead of
// the hardware-reported one, so the reduction graph is a function of the job
// config rather than of which racks the scheduler happened to allocate. That is
// what makes numerics bitwise-reproducible across restarts. See
// meta/baseline_modification_docs/mnnvl_numeric_determinism.md.
//
// Rank and world size come from the process environment (RANK / WORLD_SIZE),
// not from comm state, so every communicator in the job derives the same
// partition.
//
// Fatal if NCCL_MNNVL_CLIQUE_SIZE <= 0, if NCCL_MNNVL_CLIQUE_ID is also set, if
// RANK or WORLD_SIZE is unset, or if WORLD_SIZE is not a multiple of
// NCCL_MNNVL_CLIQUE_SIZE.
ncclResult_t assignMnnvlCliqueIdBasedOnCliqueSize(int* cliqueId);

} // namespace ncclx
