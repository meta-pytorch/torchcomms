// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <memory>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/transport/p2p/P2pRegistrationHandle.h"

namespace uniflow {

class CudaDriverApi;

/// The kernel boot and PID namespace of the calling process.
Result<P2pProcessScope> currentProcessScope();

/// VMM (hipMem) sharing for the P2P transport; built for AMD only.
///
/// A VMM segment is exported as one POSIX fd per physical chunk it spans. The
/// importer duplicates each fd with pidfd_getfd, which needs ptrace-attach
/// permission over the exporter and a shared PID namespace, and maps the
/// chunks back to back into one reservation. Callers keep the owning device
/// current around every call. Retained chunk handles are never released, even
/// when the export fails afterwards, so on runtimes whose retain adds a
/// reference each such chunk's physical memory stays allocated until process
/// exit.
class P2pVmm {
 public:
  P2pVmm(int deviceId, std::shared_ptr<CudaDriverApi> driver);

  /// True when @p ptr is VMM-mapped memory. Safe for any pointer: it never
  /// retains the allocation, which crashes for hipMalloc memory.
  bool isVmm(const void* ptr) const;

  /// Export the chunks covering [ptr, ptr + len). Fails before retaining any
  /// chunk when a chunk is not VMM, the chunks are not VA-contiguous, there
  /// are more than kP2pMaxVmmChunks, or holding one fd per chunk would exceed
  /// the RLIMIT_NOFILE soft limit. Importers map the physical chunks found
  /// here, so the owner must keep them mapped at this range while the segment
  /// is registered: a later remap is not detected.
  Result<std::unique_ptr<P2pRegistrationHandle>> exportSegment(
      void* ptr,
      size_t len) const;

  /// Map a peer's exported chunks into this process with read-write access
  /// for this device and the exporter's device. Rejects a payload from another
  /// boot or PID namespace before resolving its pid.
  Result<std::unique_ptr<P2pRemoteRegistrationHandle>> importSegment(
      const P2pRegistrationHandle::VmmPayload& payload) const;

 private:
  int deviceId_{-1};
  std::shared_ptr<CudaDriverApi> driver_;
};

} // namespace uniflow
