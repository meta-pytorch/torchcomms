// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "comms/utils/cvars/nccl_cvars.h"
#include "nccl.h" // @manual

namespace ncclx {

class Config {
 public:
  // Default constructor — uses in-class initializers.
  Config() = default;

  // Parsing constructor — populates fields from an ncclConfig_t using
  // flat fields (old format), hints (new format), and env defaults.
  // Throws std::invalid_argument on conflict or validation error.
  explicit Config(const ncclConfig_t* config);

  // NCCLX-specific config fields (canonical storage).
  // New fields should be added here, NOT to ncclConfig_t.
  // When adding a new field, also add its key to knownHintKeys below.
  std::string commDesc = "undefined";
  std::vector<int> splitGroupRanks;
  bool fastInitMode = false;

  bool useCtran = false;
  bool usePatAvg = false;
  bool noLocal = false;

  // When true, windows registered on this comm exchange only intra-node NVL/IPC
  // handles and defer the IB rkey exchange. Mutable via commSetConfig.
  bool winRegisterIpcOnly = false;

  // When false, windows registered on this comm skip signal-buffer allocation
  // and the signal control exchange; signal RMA ops are then rejected. Mutable
  // via commSetConfig.
  bool winRegisterEnableSignal = true;

  // When true, all ranks register a buffer of identical size at an identical
  // offset from the window base (upstream NCCL_WIN_COLL_SYMMETRIC semantics),
  // so a peer address is peerBase + (buf - localBase). Mutable via
  // commSetConfig.
  bool winRegisterSymmetric = false;

  enum NCCL_SENDRECV_ALGO sendrecvAlgo = NCCL_SENDRECV_ALGO::orig;
  enum NCCL_ALLGATHER_ALGO allgatherAlgo = NCCL_ALLGATHER_ALGO::orig;
  enum NCCL_ALLREDUCE_ALGO allreduceAlgo = NCCL_ALLREDUCE_ALGO::orig;
  enum NCCL_ALLTOALL_ALGO alltoallAlgo = NCCL_ALLTOALL_ALGO::orig;
  enum NCCL_ALLTOALLV_ALGO alltoallvAlgo = NCCL_ALLTOALLV_ALGO::orig;
  enum NCCL_RMA_ALGO rmaAlgo = NCCL_RMA_ALGO::orig;

  // Per-communicator MultiPeerTransport (pipes) NVL config overrides.
  // When set, override the corresponding CVARs for this communicator.
  std::optional<size_t> pipesNvlChunkSize;
  std::optional<size_t> pipesIbgdaDataBufferSize;
  int vCliqueSize = 0;

  // Per-communicator buffer size override (Simple protocol).
  // When set, overrides the global NCCL_BUFFSIZE for this communicator.
  // Only supported with splitShare=0.
  std::optional<int> ncclBuffSize;

  // Per-communicator IB transport config overrides.
  std::optional<int> ibSplitDataOnQps;
  std::optional<int> ibQpsPerConnection;

  // Defer per-peer IBGDA state to first use (hint > CVAR).
  bool ibLazyConnect = false;
  // Update mutable hint fields (algo config only).  Rejects immutable keys.
  ncclResult_t update(const ncclx::Hints* hints);
};

// Hint keys corresponding to Config fields above.  Used by
// Hints::set() to warn on unrecognized keys (typo detection).
inline const std::vector<std::string>& knownHintKeys() {
  static const std::vector<std::string> keys = {
      "commDesc",
      "splitGroupRanks",
      "fastInitMode",
      "useCtran",
      "usePatAvg",
      "noLocal",
      "sendrecvAlgo",
      "allgatherAlgo",
      "allreduceAlgo",
      "pipesIbgdaDataBufferSize",
      "alltoallAlgo",
      "alltoallvAlgo",
      "rmaAlgo",
      "pipesNvlChunkSize",
      "vCliqueSize",
      "ncclBuffSize",
      "ibSplitDataOnQps",
      "ibQpsPerConnection",
      "ibLazyConnect",
      "win_register_ipc_only",
      "win_register_enable_signal",
      "win_register_symmetric",
  };
  return keys;
}

// Algo hint keys that are safe to update on a live communicator.
inline const std::vector<std::string>& mutableHintKeys() {
  static const std::vector<std::string> keys = {
      "sendrecvAlgo",
      "allgatherAlgo",
      "allreduceAlgo",
      "alltoallAlgo",
      "alltoallvAlgo",
      "rmaAlgo",
      "win_register_ipc_only",
      "win_register_enable_signal",
      "win_register_symmetric",
  };
  return keys;
}

} // namespace ncclx

// Convenience macro: access an NCCLX-specific field from the canonical
// ncclx::Config stored inside an ncclConfig_t.
// Usage: NCCLX_CONFIG_FIELD(comm->config, commDesc)
#define NCCLX_CONFIG_FIELD(cfg, field) \
  (static_cast<ncclx::Config*>((cfg).ncclxConfig)->field)

// C-style wrapper around the ncclx::Config parsing constructor.
// Most NCCL code is C-based, so this function translates C++
// exceptions into ncclResult_t error codes for the C callers.
// Stores the result in config->ncclxConfig.  Must be called
// exactly once per config.
// TODO: Move into ncclx namespace as ncclx::parseCommConfig and update callers.
ncclResult_t ncclxParseCommConfig(ncclConfig_t* config);

// Log all specified ncclConfig_t and resolved ncclx::Config fields for a
// communicator.  Call after comm creation when commHash is available.
void ncclxLogCommConfig(ncclComm_t comm);
