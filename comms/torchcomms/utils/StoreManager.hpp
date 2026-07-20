// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <string>

namespace torch::comms {

// Create a PrefixStore wrapping a root TCPStore, using `prefix` to
// namespace all keys.  This avoids key collisions when multiple
// communicators share the same underlying store (e.g. the torchrun
// agent store).
c10::intrusive_ptr<c10d::Store> createPrefixStore(
    const std::string& prefix,
    std::chrono::milliseconds timeout);

// Open an independent client connection to the same store server that
// `bootstrapStore` talks to, wrapped in a PrefixStore with `prefix`.
// The connection is a distinct TCPStore object (its own socket and lock),
// so a blocking rendezvous over the result never head-of-line-blocks (or is
// blocked by) other communicators/backends sharing the parent store.  No
// per-comm server is created, so this works for subgroups whose rank 0 is not
// global rank 0: the job's existing (rank 0) server serves every comm.  The
// caller must keep `bootstrapStore` alive until all ranks have returned.
c10::intrusive_ptr<c10d::Store> dupPrefixStore(
    const std::string& prefix,
    const c10::intrusive_ptr<c10d::Store>& bootstrapStore,
    std::chrono::milliseconds timeout);

} // namespace torch::comms
