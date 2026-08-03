// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

#include "comms/utils/cvars/nccl_cvars.h"

// Seam between the forked upstream NCCL sources and the generated NCCLX CVAR
// registry (`comms/utils/cvars/nccl_cvars.h`).
//
// The forked tree is re-reconciled against pristine NCCL on every upstream
// rebase, so every direct reference to a generated CVAR global is a conflict
// site that has to be re-applied by hand. Routing those reads through this
// header keeps the registry — and the CVAR-typed enums — out of the forked
// sources entirely.
//
// Accessors name the DECISION the fork makes rather than the raw knob, so a
// forked file never has to spell a CVAR enum type.
namespace meta::comms::ncclx {

inline bool mnnvlDeterministicCollectiveEnabled() {
  return NCCL_MNNVL_DETERMINISTIC_COLLECTIVE_ENABLE;
}

inline int mnnvlCliqueSize() {
  return NCCL_MNNVL_CLIQUE_SIZE;
}

inline bool mnnvlTrunkDisabled() {
  return NCCL_MNNVL_TRUNK_DISABLE;
}

inline const std::string& topoFilePath() {
  return NCCL_TOPO_FILE_PATH;
}

inline bool topoBondV229Enabled() {
  return NCCL_TOPO_BOND_V229;
}

inline bool lazySetupChannelsEnabled() {
  return NCCL_LAZY_SETUP_CHANNELS;
}

inline bool slabAllocatorEnabled() {
  return NCCL_MEM_USE_SLAB_ALLOCATOR;
}

inline bool firstCommAsWorldEnabled() {
  return NCCL_FIRST_COMM_AS_WORLD;
}

inline bool commRegisterLogEnabled() {
  return NCCL_COMM_REGISTER_LOG_ENABLE;
}

// Whether a bootstrap address was supplied through the NCCL_COMM_ID knob, the
// fallback when the caller passes no initialized `ncclUniqueId`.
inline bool commIdIsSet() {
  return !NCCL_COMM_ID.empty();
}

inline bool commAbortScopeIsComm() {
  return NCCL_COMM_ABORT_SCOPE == NCCL_COMM_ABORT_SCOPE::comm;
}

inline bool commAbortScopeIsNone() {
  return NCCL_COMM_ABORT_SCOPE == NCCL_COMM_ABORT_SCOPE::none;
}

inline bool commAbortScopeIsJob() {
  return NCCL_COMM_ABORT_SCOPE == NCCL_COMM_ABORT_SCOPE::job;
}

} // namespace meta::comms::ncclx
