// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace comms::prims::test {

/// Runs make_progress_geometry<protocol::LL> against a synthetic channel layout
/// and writes the resulting `chunkPayload` to `out_d`.
///
/// The layout is built inside the .cu from these scalars so this header stays
/// free of transport includes. Only the fields make_progress_geometry reads are
/// populated: `numChannels` (bounds the group id), `pipelineDepth` and
/// `perChannelBufferSize` (which together give perBlockSlotWire).
///
/// Note the deliberate absence of a CopyOp or element-type parameter: chunk
/// sizing is a property of the protocol alone, which is what makes the sender's
/// and receiver's view of the chunk boundary identical even though the tree's
/// send lane uses Memcpy and its recv lane uses IbReduceCopy<T>. Keying the
/// alignment on the CopyOp would let the two sides disagree and deadlock.
void launch_ll_chunk_payload(
    std::size_t perChannelBufferSize,
    int pipelineDepth,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    std::size_t* out_d);

/// Same, for the blocking path's calcGeometry. The two functions duplicate the
/// chunk-sizing arithmetic, so both need the alignment and both need covering.
void launch_ll_blocking_chunk_payload(
    std::size_t perChannelBufferSize,
    int pipelineDepth,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    std::size_t* out_d);

} // namespace comms::prims::test
