/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "hipipc_record.h"

#define HIPIPC_WORKSPACE_EXPORT __attribute__((visibility("default")))

namespace rcclx::relay::hipipc {

struct Config {
  int device{-1};
  int peerDevice{-1};
  std::vector<int> helperDevices;
  size_t capacity{kMaxCapacityBytes};
};

class HIPIPC_WORKSPACE_EXPORT UnsupportedAllocation
    : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

namespace detail {
struct WorkspaceState;
}

// Linux x86_64, little-endian, 64-byte HIP IPC handles. Descriptors travel over
// a trusted live TP2 control channel. They carry identity, never ownership.
// Native state owns mappings, flags, rings and retained producer storage.
// Owners must retain storage through every alias: no resize/free/rebind while
// registered. Producer writes and relay reads/writes must be stream-ordered.
// Callers provide device BF16 buffers with at least bytes accessible; outputs
// are 16-byte aligned, while staged sources may be unaligned. Borrowed eager
// inputs/outputs survive GPU completion; Torch adapters record their use on the
// bound stream to prevent allocator reuse elsewhere.
// Each output is distinct from its own input. Ordered output-to-input chains
// are valid; callers must not overwrite another concurrently used input.
// Canonical argument records borrow pointers: copying them retains no storage.
// Kernel enqueue is not completion. Captured users additionally require stable
// storage until future replay is disabled and their registrations retire.
class HIPIPC_WORKSPACE_EXPORT Workspace {
 public:
  static std::unique_ptr<Workspace> create(const Config& config);
  ~Workspace();
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  std::vector<uint8_t> localDescriptor() const;
  void connect(const std::vector<uint8_t>& peer);
  // Both ranks call once, then exchange success before submitting any payload.
  // Probe binds EXEC, reads peer canaries and warms the zero-vector handshake.
  // Stream types remain opaque so this header needs no HIP SDK.
  void probe(void* stream);
  void
  allReduceStaged(const void* input, void* output, size_t bytes, void* stream);
  // UnsupportedAllocation declines legacy IPC eligibility without poisoning.
  RegistrationId registerInput(const void* input, size_t bytes, Owner owner);
  std::vector<uint8_t> describe(RegistrationId id) const;
  void bindPeer(RegistrationId id, const std::vector<uint8_t>& peer);
  void
  allReduceDirect(RegistrationId id, void* output, size_t bytes, void* stream);

  // Control path: stop new submissions/future replays on BOTH ranks, drain,
  // exchange "stopped", closeImports(batch), exchange "closed", then retire.
  // The batch uses each rank's local IDs for the same agreed logical
  // operations. Imports shared by surviving aliases stay open; all base owners
  // stay retained until the last alias retires. Dispatch pauses between
  // closeImports/retire. retire/close attest that the caller received the
  // peer's closed acknowledgment. Never release exported storage merely because
  // GPUs stopped: HIP requires hipIpcCloseMemHandle in the importing context
  // BEFORE exporter hipFree. Invoke retire/close without caller/bridge locks
  // needed by Owner deleters; native destroys detached owners outside its
  // locks.
  void drain();
  void closeImports(const std::vector<RegistrationId>& ids);
  void retire(const std::vector<RegistrationId>& ids);
  // Whole-workspace variant, including registrations. Idempotent when closed.
  void closeImports();
  void close();
  bool poisoned() const;
  // Control-channel failure/unknown peer quiescence: no waits or frees.
  void poison() noexcept;

 private:
  explicit Workspace(std::shared_ptr<detail::WorkspaceState> state);
  std::shared_ptr<detail::WorkspaceState> state_;
};
} // namespace rcclx::relay::hipipc

extern "C" HIPIPC_WORKSPACE_EXPORT uint32_t hipipc_relay_abi_version();
