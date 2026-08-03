// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

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
};
