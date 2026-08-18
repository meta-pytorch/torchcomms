// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/benchmarks/CopyOpReduceBench.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/utils/CudaRAII.h"

namespace comms::prims::benchmark {
namespace {

constexpr int kBlocks = 1;
// fp32 packed into 16-byte vectors.
constexpr int kElemsPerVec = 4;

/*
 * (threads, vectors-per-thread) points the sweep instantiates.
 *
 * vpt is the memory-level-parallelism knob. The unfused and pipelined shapes
 * keep 2 * vpt sixteen-byte loads live, costing 8 * vpt registers against the
 * 65536 / threads budget __launch_bounds__ grants -- 102 registers at 640
 * threads, so vpt=12 (96) is the last point expected to fit.
 */
#define COPY_OP_REDUCE_CONFIGS(X) \
  X(640, 6)                       \
  X(640, 12)                      \
  X(1024, 6)

// Nominal bytes of HBM traffic per payload byte, per shape. "Nominal" because
// a store may cost more than one byte if the line is not already resident;
// separating that from a store-port limit is what WriteOnly is for.
__host__ __device__ inline float shape_multiplier(CopyOpReduceShape shape) {
  switch (shape) {
    // Stores only the final tile, so write traffic is ~0.
    case CopyOpReduceShape::ReadOnly:
      return 2.0f;
    // One hoisted load, then stores only.
    case CopyOpReduceShape::WriteOnly:
      return 1.0f;
    // One load + one store per payload byte.
    case CopyOpReduceShape::Copy:
      return 2.0f;
    default:
      return 3.0f;
  }
}

// ---------------------------------------------------------------- shapes

template <int kThreads, int kVpt>
__device__ __forceinline__ void body_unfused(
    float* out,
    const float* a,
    const float* b,
    std::size_t nbytes,
    ThreadGroup& group) {
  constexpr int kTileElems = kThreads * kVpt * kElemsPerVec;
  using Policy = TileReduceStaged<float, SumOp, kTileElems, kThreads>;
  Policy::recv(
      reinterpret_cast<char*>(out),
      reinterpret_cast<const char*>(a),
      nbytes,
      group,
      0,
      reinterpret_cast<const char*>(b));
}

/*
 * Production shape. Full tiles use the UNMASKED tile_load overload, matching
 * tileReduceCopy in AllReduceIbRingImpl.cuh; only the tail passes `valid`. An
 * earlier revision of this benchmark passed `valid` on every tile, which made
 * the fused numbers unusable on sm_103a.
 */
template <int kThreads, int kVpt>
__device__ __forceinline__ void body_fused(
    float* out,
    const float* a,
    const float* b,
    std::size_t nbytes,
    ThreadGroup& group) {
  constexpr int kTileElems = kThreads * kVpt * kElemsPerVec;
  const std::size_t nelems = nbytes / sizeof(float);
  const std::size_t nFull = nelems / kTileElems;
  const std::size_t rem = nelems % kTileElems;
  for (std::size_t t = 0; t < nFull; t++) {
    auto acc = tile_load<float, kTileElems, kThreads>(a, t, group);
    tile_load_accumulate<float, SumOp, kTileElems, kThreads>(acc, b, t, group);
    tile_store<float, kTileElems, kThreads>(out, t, acc, group);
  }
  if (rem > 0) {
    auto acc = tile_load<float, kTileElems, kThreads>(a, nFull, group, rem);
    tile_load_accumulate<float, SumOp, kTileElems, kThreads>(
        acc, b, nFull, group, rem);
    tile_store<float, kTileElems, kThreads>(out, nFull, acc, group, rem);
  }
}

// Loads only; stores just the final tile so the loads cannot be eliminated.
template <int kThreads, int kVpt>
__device__ __forceinline__ void body_read_only(
    float* out,
    const float* a,
    const float* b,
    std::size_t nbytes,
    ThreadGroup& group) {
  constexpr int kTileElems = kThreads * kVpt * kElemsPerVec;
  const std::size_t nelems = nbytes / sizeof(float);
  const std::size_t nFull = nelems / kTileElems;
  for (std::size_t t = 0; t < nFull; t++) {
    auto acc = tile_load<float, kTileElems, kThreads>(a, t, group);
    tile_load_accumulate<float, SumOp, kTileElems, kThreads>(acc, b, t, group);
    if (t + 1 == nFull) {
      tile_store<float, kTileElems, kThreads>(out, t, acc, group);
    }
  }
}

/*
 * Stores only. One tile is loaded before the loop and restored to every tile
 * position, so the load amortises to nothing and what remains is the store
 * path in isolation. Addresses differ per iteration, so the store cannot be
 * hoisted.
 */
template <int kThreads, int kVpt>
__device__ __forceinline__ void body_write_only(
    float* out,
    const float* a,
    const float* /*b*/,
    std::size_t nbytes,
    ThreadGroup& group) {
  constexpr int kTileElems = kThreads * kVpt * kElemsPerVec;
  const std::size_t nelems = nbytes / sizeof(float);
  const std::size_t nFull = nelems / kTileElems;
  if (nFull == 0) {
    return;
  }
  auto acc = tile_load<float, kTileElems, kThreads>(a, 0, group);
  for (std::size_t t = 0; t < nFull; t++) {
    tile_store<float, kTileElems, kThreads>(out, t, acc, group);
  }
}

/*
 * Plain copy: one load, one store, no reduce. Its 1:1 load:store ratio is more
 * store-heavy than the reduce shape's 2:1, so under the store-port model it
 * should be SLOWER per byte despite doing less work.
 */
template <int kThreads, int kVpt>
__device__ __forceinline__ void body_copy(
    float* out,
    const float* a,
    const float* /*b*/,
    std::size_t nbytes,
    ThreadGroup& group) {
  constexpr int kTileElems = kThreads * kVpt * kElemsPerVec;
  const std::size_t nelems = nbytes / sizeof(float);
  const std::size_t nFull = nelems / kTileElems;
  for (std::size_t t = 0; t < nFull; t++) {
    auto tile = tile_load<float, kTileElems, kThreads>(a, t, group);
    tile_store<float, kTileElems, kThreads>(out, t, tile, group);
  }
}

// Fused with the next tile's loads issued before the current tile's store.
template <int kThreads, int kVpt>
__device__ __forceinline__ void body_pipelined(
    float* out,
    const float* a,
    const float* b,
    std::size_t nbytes,
    ThreadGroup& group) {
  constexpr int kTileElems = kThreads * kVpt * kElemsPerVec;
  const std::size_t nelems = nbytes / sizeof(float);
  const std::size_t nFull = nelems / kTileElems;
  if (nFull == 0) {
    return;
  }
  auto acc = tile_load<float, kTileElems, kThreads>(a, 0, group);
  tile_load_accumulate<float, SumOp, kTileElems, kThreads>(acc, b, 0, group);
  for (std::size_t t = 0; t + 1 < nFull; t++) {
    auto nxt = tile_load<float, kTileElems, kThreads>(a, t + 1, group);
    tile_store<float, kTileElems, kThreads>(out, t, acc, group);
    tile_load_accumulate<float, SumOp, kTileElems, kThreads>(
        nxt, b, t + 1, group);
    acc = nxt;
  }
  tile_store<float, kTileElems, kThreads>(out, nFull - 1, acc, group);
}

// ---------------------------------------------------------------- kernel

template <CopyOpReduceShape kShape, int kThreads, int kVpt>
__global__ __launch_bounds__(kThreads, 1) void roofline_kernel(
    float* out,
    const float* a,
    const float* b,
    std::size_t nbytes,
    int repeats,
    unsigned long long* cycles) {
  auto group = make_block_group();
  const bool timer = threadIdx.x == 0;
  group.sync();
  const unsigned long long begin = timer ? clock64() : 0ULL;

  for (int r = 0; r < repeats; r++) {
    if constexpr (kShape == CopyOpReduceShape::Unfused) {
      body_unfused<kThreads, kVpt>(out, a, b, nbytes, group);
    } else if constexpr (kShape == CopyOpReduceShape::Fused) {
      body_fused<kThreads, kVpt>(out, a, b, nbytes, group);
    } else if constexpr (kShape == CopyOpReduceShape::ReadOnly) {
      body_read_only<kThreads, kVpt>(out, a, b, nbytes, group);
    } else if constexpr (kShape == CopyOpReduceShape::WriteOnly) {
      body_write_only<kThreads, kVpt>(out, a, b, nbytes, group);
    } else if constexpr (kShape == CopyOpReduceShape::Copy) {
      body_copy<kThreads, kVpt>(out, a, b, nbytes, group);
    } else {
      body_pipelined<kThreads, kVpt>(out, a, b, nbytes, group);
    }
  }

  group.sync();
  if (timer) {
    *cycles = clock64() - begin;
  }
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void launch(
    CopyOpReduceShape shape,
    int threads,
    int vpt,
    float* out,
    const float* a,
    const float* b,
    std::size_t nbytes,
    int repeats,
    unsigned long long* cycles) {
#define COPY_OP_REDUCE_SHAPE_CASE(SHAPE, T, V)                \
  if (shape == CopyOpReduceShape::SHAPE) {                    \
    roofline_kernel<CopyOpReduceShape::SHAPE, T, V>           \
        <<<kBlocks, T>>>(out, a, b, nbytes, repeats, cycles); \
    return;                                                   \
  }
#define COPY_OP_REDUCE_DISPATCH(T, V)          \
  if (threads == (T) && vpt == (V)) {          \
    COPY_OP_REDUCE_SHAPE_CASE(Unfused, T, V)   \
    COPY_OP_REDUCE_SHAPE_CASE(Fused, T, V)     \
    COPY_OP_REDUCE_SHAPE_CASE(ReadOnly, T, V)  \
    COPY_OP_REDUCE_SHAPE_CASE(WriteOnly, T, V) \
    COPY_OP_REDUCE_SHAPE_CASE(Copy, T, V)      \
    COPY_OP_REDUCE_SHAPE_CASE(Pipelined, T, V) \
  }
  COPY_OP_REDUCE_CONFIGS(COPY_OP_REDUCE_DISPATCH)
#undef COPY_OP_REDUCE_DISPATCH
#undef COPY_OP_REDUCE_SHAPE_CASE
  throw std::invalid_argument(
      "unsupported (threads, vpt): " + std::to_string(threads) + ", " +
      std::to_string(vpt));
}

} // namespace

CopyOpReduceTiming runCopyOpReduceBenchmark(
    CopyOpReduceShape shape,
    std::size_t nbytes,
    int iterations,
    int threads,
    int vpt,
    int repeats) {
  const int tileBytes =
      threads * vpt * kElemsPerVec * static_cast<int>(sizeof(float));
  if (nbytes < static_cast<std::size_t>(tileBytes)) {
    throw std::invalid_argument(
        "nbytes " + std::to_string(nbytes) + " is smaller than one tile (" +
        std::to_string(tileBytes) + "); the kernel would measure nothing");
  }

  meta::comms::DeviceBuffer a(nbytes);
  meta::comms::DeviceBuffer b(nbytes);
  meta::comms::DeviceBuffer out(nbytes);
  meta::comms::DeviceBuffer cycleBuf(sizeof(unsigned long long));
  check_cuda(cudaMemset(a.get(), 1, nbytes), "initialize a");
  check_cuda(cudaMemset(b.get(), 2, nbytes), "initialize b");

  auto* outPtr = static_cast<float*>(out.get());
  const auto* aPtr = static_cast<const float*>(a.get());
  const auto* bPtr = static_cast<const float*>(b.get());
  auto* cyclePtr = static_cast<unsigned long long*>(cycleBuf.get());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  check_cuda(cudaEventCreate(&start), "create start event");
  check_cuda(cudaEventCreate(&stop), "create stop event");

  launch(shape, threads, vpt, outPtr, aPtr, bPtr, nbytes, repeats, cyclePtr);
  check_cuda(cudaGetLastError(), "warmup launch");
  check_cuda(cudaDeviceSynchronize(), "warmup synchronize");

  check_cuda(cudaEventRecord(start), "record start");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    launch(shape, threads, vpt, outPtr, aPtr, bPtr, nbytes, repeats, cyclePtr);
    check_cuda(cudaGetLastError(), "benchmark launch");
  }
  check_cuda(cudaEventRecord(stop), "record stop");
  check_cuda(cudaEventSynchronize(stop), "benchmark synchronize");

  float elapsedMs = 0;
  check_cuda(cudaEventElapsedTime(&elapsedMs, start, stop), "measure elapsed");
  check_cuda(cudaEventDestroy(stop), "destroy stop event");
  check_cuda(cudaEventDestroy(start), "destroy start event");

  unsigned long long cycles = 0;
  check_cuda(
      cudaMemcpy(&cycles, cyclePtr, sizeof(cycles), cudaMemcpyDeviceToHost),
      "read cycle counter");

  const float timeUs = elapsedMs * 1000.0f / static_cast<float>(iterations);
  // One launch touches nbytes * repeats of payload.
  const double payloadBytes =
      static_cast<double>(nbytes) * static_cast<double>(repeats);
  const float payloadGBps =
      static_cast<float>(payloadBytes / 1000.0 / static_cast<double>(timeUs));
  const float memoryGBps = shape_multiplier(shape) * payloadGBps;
  // Measured, not assumed: SM-issued bytes divided by SM cycles for one launch.
  const float bytesPerClock = cycles == 0
      ? 0.0f
      : static_cast<float>(
            payloadBytes * shape_multiplier(shape) /
            static_cast<double>(cycles));

  return CopyOpReduceTiming{
      .timeUs = timeUs,
      .payloadGBps = payloadGBps,
      .memoryGBps = memoryGBps,
      .bytesPerClock = bytesPerClock,
      .cycles = cycles,
  };
}

} // namespace comms::prims::benchmark
