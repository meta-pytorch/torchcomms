// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "nccl_tuner.h" // @manual -- provides NCCL_NUM_ALGORITHMS

#include "comms/utils/colltrace/AlgoStats.h"
#include "comms/utils/colltrace/CollTraceInterface.h"

struct ncclKernelCommAndChannels;

// Opaque per-communicator NCCLX state.
//
// This handle collects NCCLX-only members that would otherwise be woven
// directly into the forked upstream `ncclComm` struct, keeping that struct
// close to pristine NCCL. NCCLX code reaches the state through the single
// `ncclComm::ncclxExt` pointer (including this header and dereferencing
// `comm->ncclxExt`), while the forked `comm.h` only forward-declares the type.
//
// `ncclComm` is raw-allocated (calloc) and raw-freed, so no constructor or
// destructor runs on it; this handle is instead created explicitly right after
// the comm is allocated and destroyed in the NCCLX comm-free hook, giving it a
// lifetime that exactly matches the communicator.
struct ncclxCommExt {
  // Ctran per-communicator control; populated from the parsed ncclx::Config at
  // communicator init and gates creation of the per-comm CtranComm.
  bool useCtran{false};

  // Disable local transports (P2P and SHM); forces NET for all connections.
  // Populated from the parsed ncclx::Config at communicator init.
  bool noLocal{false};

  // CollTrace: per-communicator collective tracing (verbose/trace modes).
  std::shared_ptr<meta::comms::colltrace::ICollTrace> newCollTrace;
  // AlgoStats: per-communicator algorithm-selection stats (algostat mode).
  std::shared_ptr<meta::comms::colltrace::AlgoStats> algoStats;

  // ---- Lazy Channel Setup (NCCL_LAZY_SETUP_CHANNELS) per-communicator state
  // -- Relocated off the forked upstream ncclComm/ncclKernelPlanner to keep
  // those structs pristine; the feature lives in
  // meta/transport/transportConnect.*.
  //
  // if channels can/will be setup lazily for this communicator
  bool lazySetupChannels{false};
  // number of channels that are initialized and ready for use
  int nChannelsReady{0};
  // number of channels that are connected for each algorithm
  std::array<int, NCCL_NUM_ALGORITHMS> algoConnectedChannels{0};
  // metadata to be used for initializing channels lazily if enabled
  std::optional<struct ncclKernelCommAndChannels*> devCommAndChans{
      std::nullopt};
  // cached ring info, used to set up channels lazily when needed
  std::optional<std::vector<int>> rings{std::nullopt};
  // high-water mark of channels that need to be initialized (across plans)
  int nMaxChannelsNeedInit{0};
  // high-water mark of channels each algorithm needs to connect (across plans)
  std::array<int, NCCL_NUM_ALGORITHMS> algoMaxChannelsNeedConnect{0};

  // Forces the deterministic PAT algorithm with ncclDevPatSumPostDiv for
  // ncclAvg ReduceScatter on supported datatypes. Populated from the parsed
  // ncclx::Config at communicator init.
  bool usePatAvg{false};
};
