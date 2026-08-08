// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/utils/colltrace/AlgoStats.h"
#include "comms/utils/colltrace/CollTraceInterface.h"

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
};
