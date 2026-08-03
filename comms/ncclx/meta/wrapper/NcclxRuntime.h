// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace meta::comms::ncclx {

// Initialize the NCCLX loggers for the "comms.ncclx" and "meta" logging
// contexts. Unlike `ncclxInitRuntime`, this is NOT once-guarded, so callers
// that reset logging state (e.g. tests that null the debug file and re-init)
// can call it repeatedly. Also invoked by `ncclxInitRuntime` as the final
// step of the one-time bootstrap.
void ncclxInitLogger();

// One-time NCCLX runtime bootstrap, hoisted out of the forked upstream
// `param.cc` so that file stays close to pristine NCCL. Runs exactly once per
// process, in this fixed order:
//   1. Folly runtime init
//   2. NCCLX CVAR init
//   3. `loadEnvFiles` -- the upstream `.nccl.conf` / `/etc/nccl.conf` loader.
//      It is kept in the forked file (it is pristine NCCL behavior) and passed
//      in so only the NCCLX-only bootstrap lives here.
//   4. NCCLX logger init
//
// `loadEnvFiles` must be non-null. Subsequent calls are no-ops (the bootstrap
// is guarded by an internal `std::once_flag`).
void ncclxInitRuntime(void (*loadEnvFiles)());

} // namespace meta::comms::ncclx
