// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstddef>

#include "comms/prims/tests/ProgressGeometryTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::prims {
namespace {

// MCCL_CHANNEL_BUFFER_SIZE's default. Chunk sizing is
// perChannelBufferSize / pipelineDepth, then converted to payload bytes by
// LL's max_payload (wire / 8 * 4), so both inputs shape the result.
constexpr std::size_t kChannelBuffer = 4ull * 1024 * 1024;

// The widest element the fused AllReduce instantiates (double, int64_t). LL's
// payload quantum is only kData = 4, so a chunk that is a multiple of 4 but not
// of 8 splits one of these elements across the chunk boundary:
// LLImpl::unpack_reduce's wide-T branch truncates `nbytes / sizeof(T)` and
// drops the trailing half element, and the next chunk's dst lands 4 bytes off a
// natural 8-byte boundary.
// Spelled as sizeof(double) rather than reusing the transport's
// kMaxReducedTypeBytes, so the expectation is derived from the dtype the test
// is reasoning about instead of mirroring the constant under test.
constexpr std::size_t kWidestElem = sizeof(double);

// Any non-zero payload works: chunkPayload depends only on the layout and
// maxSignalBytes. Zero would trap in make_progress_geometry.
constexpr std::size_t kNbytes = 1ull * 1024 * 1024;

// The resumable and blocking paths duplicate the chunk-sizing arithmetic
// (make_progress_geometry vs calcGeometry), so every case runs against both.
using Launcher =
    void (*)(std::size_t, int, std::size_t, std::size_t, std::size_t*);

struct Path {
  const char* name;
  Launcher launch;
};

constexpr Path kPaths[] = {
    {"progress", &test::launch_ll_chunk_payload},
    {"blocking", &test::launch_ll_blocking_chunk_payload},
};

std::size_t llChunkPayload(
    Launcher launch,
    int pipelineDepth,
    std::size_t maxSignalBytes,
    std::size_t perChannelBufferSize = kChannelBuffer) {
  DeviceBuffer out(sizeof(std::size_t));
  launch(
      perChannelBufferSize,
      pipelineDepth,
      kNbytes,
      maxSignalBytes,
      static_cast<std::size_t*>(out.get()));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::size_t chunkPayload = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &chunkPayload, out.get(), sizeof(chunkPayload), cudaMemcpyDeviceToHost));
  return chunkPayload;
}

// Trigger 1: MCCL_ALLREDUCE_IB_SIGNAL_BYTES (benchmark_allreduce.sh
// --signal-bytes) is unvalidated, so any 4-mod-8 value reaches the geometry.
TEST(ProgressGeometryTest, SignalBytesChunkHoldsWholeWidestElements) {
  for (const Path& path : kPaths) {
    for (const std::size_t signalBytes : {std::size_t{12}, std::size_t{20}}) {
      const std::size_t chunkPayload =
          llChunkPayload(path.launch, /*pipelineDepth=*/8, signalBytes);
      EXPECT_EQ(chunkPayload % kWidestElem, 0u)
          << path.name << " signalBytes=" << signalBytes
          << " chunkPayload=" << chunkPayload;
    }
  }
}

// Trigger 2: even with signal bytes untouched, perBlockSlotPayload itself is
// 4-mod-8 at these pipeline depths, because max_payload halves an odd
// wire-byte count. benchmark_allreduce.sh exposes this as --pipeline-depths.
TEST(ProgressGeometryTest, PipelineDepthChunkHoldsWholeWidestElements) {
  for (const Path& path : kPaths) {
    for (const int pipelineDepth : {5, 6, 13, 14}) {
      const std::size_t chunkPayload =
          llChunkPayload(path.launch, pipelineDepth, /*maxSignalBytes=*/0);
      EXPECT_EQ(chunkPayload % kWidestElem, 0u)
          << path.name << " pipelineDepth=" << pipelineDepth
          << " chunkPayload=" << chunkPayload;
    }
  }
}

// Guards against over-tightening: the default configuration is already aligned
// and must keep its full chunk size, not get rounded down to something smaller.
TEST(ProgressGeometryTest, DefaultConfigurationChunkIsUnchanged) {
  for (const Path& path : kPaths) {
    const std::size_t chunkPayload =
        llChunkPayload(path.launch, /*pipelineDepth=*/8, /*maxSignalBytes=*/0);
    EXPECT_EQ(chunkPayload, 262144u) << path.name;
  }
}

// The cvar is documented as a MAXIMUM ("Maximum bytes per signaled sub-chunk"),
// so a request below one element cannot be honoured exactly: rounding down to a
// whole element yields zero. Clamp to the finest granularity that still holds a
// whole element -- one packet -- rather than falling back to the whole slot,
// which would overshoot the requested maximum by five orders of magnitude.
// Rounding *up* instead would also violate the documented maximum.
TEST(ProgressGeometryTest, SubElementSignalBytesClampsToOnePacket) {
  // lcm(LL's kData = 4, sizeof(double) = 8) == 8.
  constexpr std::size_t kLlChunkAlign = kWidestElem;
  for (const Path& path : kPaths) {
    const std::size_t chunkPayload =
        llChunkPayload(path.launch, /*pipelineDepth=*/8, /*maxSignalBytes=*/4);
    EXPECT_EQ(chunkPayload, kLlChunkAlign)
        << path.name << " chunkPayload=" << chunkPayload;
  }
}

} // namespace
} // namespace comms::prims
