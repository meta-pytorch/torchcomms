// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "comms/prims/core/SignalState.cuh"
#include "comms/prims/transport/nvl/MultimemNvlTransportDevice.cuh"

namespace comms::prims::test {

// Each launch is one warp -> one ThreadGroup. The leader performs the
// multimem PTX store; the remaining lanes sync alongside it. Callers must
// cudaStreamSynchronize (or cudaDeviceSynchronize) before observing effects
// on the host.

// user_signal(signalId) <- value on this rank; broadcasts to every peer's
// local backing via multimem.st.
void launchSetUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream = nullptr);

// internal_signal(signalId) <- value; identical shape, different span.
void launchSetInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream = nullptr);

// user_signal(signalId) += value via multimem.red.add.
void launchAddUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream = nullptr);

// internal_signal(signalId) += value via multimem.red.add.
void launchAddInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream = nullptr);

// Wait via wait_signal_until until user_signal(signalId) satisfies (op
// expected); then read the local user signal state and write it to *out.
void launchWaitAndReadUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out,
    cudaStream_t stream = nullptr);

// Same as above, but through wait_internal_signal_until +
// read_internal_signal.
void launchWaitAndReadInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out,
    cudaStream_t stream = nullptr);

// Reads user + internal signals (no wait). out[0] receives read_signal(userId),
// out[1] receives read_internal_signal(internalId). Used to prove the two
// spans are isolated: touching one is not observable through the other.
void launchReadUserAndInternal(
    MultimemNvlTransportDevice transport,
    uint64_t userId,
    uint64_t internalId,
    uint64_t* out,
    cudaStream_t stream = nullptr);

} // namespace comms::prims::test
