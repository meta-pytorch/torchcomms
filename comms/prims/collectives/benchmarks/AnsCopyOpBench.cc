// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Standalone single-GPU microbenchmark for the raw ANS CopyOp APIs
// (`AnsCompress::send()` / `AnsCompress::recv()` in
// `comms/prims/core/CopyOp.cuh`) — no transport, no inter-rank
// communication. Extracted from AllToAllvTileBenchmark.cc so the ANS
// CopyOp primitive can be enabled and benchmarked independently of the
// AllToAllv-tile collective.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda.h> // driver API: green contexts for SM partitioning

#include "comms/common/CudaWrap.h"
#include "comms/prims/collectives/benchmarks/AnsCopyOpBench.cuh"
#include "comms/prims/core/Checks.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::prims::benchmark {
namespace {

// Local placeholder for the ANS stats struct. The standalone microbench
// measures the produced compressed size directly per iteration rather than
// via the device-global stat counters, so this stays {0, 0} (the ratio
// branch guarded by `compressed_bytes > 0` is intentionally inert here).
struct LocalCompStats {
  uint64_t uncompressed_bytes = 0;
  uint64_t compressed_bytes = 0;
};

// Driver-API error check (runtime calls use PIPES_CUDA_CHECK).
#define PIPES_CU_CHECK(expr)                                           \
  do {                                                                 \
    CUresult _res = (expr);                                            \
    if (_res != CUDA_SUCCESS) {                                        \
      const char* _msg = nullptr;                                      \
      cuGetErrorString(_res, &_msg);                                   \
      XLOG(FATAL) << "[AnsCopyOpStandalone] CUDA driver error: " #expr \
                  << " -> " << (_msg ? _msg : "unknown");              \
    }                                                                  \
  } while (0)

// A stream confined to a green context spanning >= `numSms` SMs of the
// current device. Launching the bench kernels on this stream restricts
// them to that SM partition, so a fixed grid runs at a controlled
// blocks/SM ratio (e.g. 512 blocks on 32 SMs => 16 blocks/SM). `numSms
// <= 0` returns an empty handle (stream == nullptr) = the default stream
// over the whole GPU. The realized SM count may round up to the hardware
// split granularity.
struct GreenCtxStream {
  CUgreenCtx ctx = nullptr;
  cudaStream_t stream = nullptr;
};

GreenCtxStream makeGreenCtxStream(int numSms) {
  GreenCtxStream out;
  if (numSms <= 0) {
    return out;
  }
  int ordinal = 0;
  PIPES_CUDA_CHECK(cudaGetDevice(&ordinal));
  CUdevice dev = 0;
  PIPES_CU_CHECK(cuDeviceGet(&dev, ordinal));

  CUdevResource sm{};
  const CUresult smRes =
      cuDeviceGetDevResource(dev, &sm, CU_DEV_RESOURCE_TYPE_SM);
  if (smRes != CUDA_SUCCESS) {
    const char* nm = nullptr;
    cuGetErrorName(smRes, &nm);
    XLOG(FATAL)
        << "[AnsCopyOpStandalone] green-context SM partitioning unavailable "
        << "(cuDeviceGetDevResource -> " << (nm ? nm : "?") << "). "
        << "PIPES_COPYOP_BENCH_NUM_SMS requires CUDA driver >= 12.4 (R550); "
        << "this host's driver is older. Re-run without "
        << "PIPES_COPYOP_BENCH_NUM_SMS (whole GPU) or use a newer-driver host.";
  }
  CUdevResource group{};
  CUdevResource remaining{};
  unsigned int nGroups = 1;
  PIPES_CU_CHECK(cuDevSmResourceSplitByCount(
      &group,
      &nGroups,
      &sm,
      &remaining,
      /*useFlags=*/0,
      /*minCount=*/static_cast<unsigned int>(numSms)));

  CUdevResourceDesc desc{};
  PIPES_CU_CHECK(cuDevResourceGenerateDesc(&desc, &group, 1));
  PIPES_CU_CHECK(
      cuGreenCtxCreate(&out.ctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));

  CUstream cuStream = nullptr;
  PIPES_CU_CHECK(cuGreenCtxStreamCreate(
      &cuStream, out.ctx, CU_STREAM_NON_BLOCKING, /*priority=*/0));
  out.stream = reinterpret_cast<cudaStream_t>(cuStream);
  return out;
}

std::string format_bytes(std::size_t bytes) {
  if (bytes >= 1024UL * 1024 * 1024) {
    return std::to_string(bytes / (1024UL * 1024 * 1024)) + "GB";
  }
  if (bytes >= 1024 * 1024) {
    return std::to_string(bytes / (1024 * 1024)) + "MB";
  }
  if (bytes >= 1024) {
    return std::to_string(bytes / 1024) + "KB";
  }
  return std::to_string(bytes) + "B";
}

class AnsCopyOpBenchFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    const char* localRankEnv = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (localRankEnv) {
      localRank = std::atoi(localRankEnv);
    }
    PIPES_CUDA_CHECK(cudaSetDevice(localRank));
  }
};

TEST_F(AnsCopyOpBenchFixture, AnsCopyOpStandalone) {
  // Single-GPU microbenchmark for the raw `AnsCompress::send()` /
  // `AnsCompress::recv()` APIs in `CopyOp.cuh` — no transport, no
  // inter-rank communication. Only rank 0 drives the launches; the
  // other ranks return immediately so the run wall-time isn't
  // dominated by per-rank serialisation. There are no in-test
  // `bootstrap->barrierAll()` calls because each rank's behaviour is
  // independent.
  if (globalRank != 0) {
    return;
  }

  // Defaults requested by the bench owner: 20 warmup + 200 timed
  // iterations per (sparsity, size) cell, driven from inside the
  // kernel so we measure pure ANS (de)compression time and amortise
  // the per-launch overhead.
  constexpr int kCopyOpWarmupIter = 20;
  constexpr int kCopyOpMeasureIter = 200;
  // Launch geometry. Both overridable at runtime via env vars (no
  // rebuild) for quick sweeps:
  //
  //   PIPES_COPYOP_BENCH_BLOCKS=128 PIPES_COPYOP_BENCH_THREADS=512 \
  //       buck2 run ... -- --gtest_filter='*AnsCopyOpStandalone*'
  //
  // `threads` must be one of {32, 64, 128, 256, 512} (the NumWarps
  // ∈ {1, 2, 4, 8, 16} values explicitly instantiated in
  // `AnsCopyOpBench.cu`); `min_blocks_per_sm` must be one of {1, 2,
  // 3, 4} (the second `__launch_bounds__` argument). Other values
  // are rejected at launch.
  auto env_int = [](const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : fallback;
  };
  const int kCopyOpBlocks = env_int("PIPES_COPYOP_BENCH_BLOCKS", 256);
  const int kCopyOpThreads = env_int("PIPES_COPYOP_BENCH_THREADS", 256);
  const int kCopyOpMinBlocksPerSM =
      env_int("PIPES_COPYOP_BENCH_MIN_BLOCKS_PER_SM", 1);
  // 0 (default) = whole GPU; >0 = confine the kernel to a green context
  // of this many SMs (rounds up to the hw split granularity).
  const int kCopyOpNumSms = env_int("PIPES_COPYOP_BENCH_NUM_SMS", 0);
  const AnsCopyOpBenchKernels bench_kernels =
      pick_ans_copy_op_bench_kernels(kCopyOpThreads, kCopyOpMinBlocksPerSM);
  if (kCopyOpBlocks <= 0 || bench_kernels.compress_kernel == nullptr) {
    GTEST_FAIL() << "Invalid PIPES_COPYOP_BENCH_BLOCKS=" << kCopyOpBlocks
                 << " / PIPES_COPYOP_BENCH_THREADS=" << kCopyOpThreads
                 << " / PIPES_COPYOP_BENCH_MIN_BLOCKS_PER_SM="
                 << kCopyOpMinBlocksPerSM
                 << " (threads must be one of 32, 64, 128, 256, 512; "
                 << "min_blocks_per_sm must be one of 1, 2, 3, 4; "
                 << "blocks must be positive)";
  }

  const std::vector<std::size_t> sizes = {
      256ULL * 1024,
      512ULL * 1024,
      1024ULL * 1024,
      2ULL * 1024 * 1024,
      4ULL * 1024 * 1024,
  };

  // Allocate buffers once at the largest size and reuse them across
  // the sparsity × size sweep. Layout: [block 0 region | block 1
  // region | ...]. The compress kernel writes into `d_staging` and
  // the decompress kernel reads from the same buffer, so the
  // decompress measurement always sees a freshly-produced layout
  // from the immediately preceding compress measurement (matches
  // the ratio reported below).
  const std::size_t maxInputBytes = sizes.back();
  const std::size_t maxStagingStride =
      ans_copy_op_bench_staging_stride(maxInputBytes);

  DeviceBuffer d_input(static_cast<std::size_t>(kCopyOpBlocks) * maxInputBytes);
  DeviceBuffer d_staging(
      static_cast<std::size_t>(kCopyOpBlocks) * maxStagingStride);
  DeviceBuffer d_output(
      static_cast<std::size_t>(kCopyOpBlocks) * maxInputBytes);

  // Host-side incompressible (under ANS) pattern — same SplitMix64
  // hash used by `IbSweepCompressed` so the two benches see
  // identical ANS ratios at matched sparsity. Each byte is a hash
  // of its position so the value distribution within any
  // `kAnsMaxUncompBytes` (256 KiB) window is uniform over [0..255].
  // The pattern is tiled into every block's input region; the
  // leading `sparsityPct%` of each block's region is then
  // `cudaMemset`-zeroed to produce the requested sparsity.
  constexpr std::size_t kPatternBytes = 1ULL * 1024 * 1024;
  std::vector<uint8_t> host_pattern(kPatternBytes);
  for (std::size_t i = 0; i < kPatternBytes; ++i) {
    uint64_t x = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL +
        0xDEADBEEFCAFEBABEULL;
    x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
    x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    x = x ^ (x >> 33);
    host_pattern[i] = static_cast<uint8_t>(x & 0xff);
  }

  const GreenCtxStream gctx = makeGreenCtxStream(kCopyOpNumSms);
  const cudaStream_t benchStream =
      gctx.stream; // nullptr => default stream (whole GPU)

  XLOG(INFO) << "\n=== ANS CopyOp Standalone Microbenchmark "
             << "(single GPU) ===\n"
             << "PIPES_COPYOP_BENCH_BLOCKS=" << kCopyOpBlocks
             << " PIPES_COPYOP_BENCH_THREADS=" << kCopyOpThreads
             << " PIPES_COPYOP_BENCH_MIN_BLOCKS_PER_SM="
             << kCopyOpMinBlocksPerSM
             << " PIPES_COPYOP_BENCH_NUM_SMS=" << kCopyOpNumSms
             << (kCopyOpNumSms > 0 ? " (green-ctx SM-limited)" : " (full GPU)")
             << "\n"
             << kCopyOpBlocks << " threadblocks × " << kCopyOpThreads
             << " threads/block, " << kCopyOpWarmupIter << " warmup + "
             << kCopyOpMeasureIter << " timed iterations per cell\n"
             << "Sweep: sizes {256KB, 512KB, 1MB, 2MB, 4MB} × "
             << "sparsity 0%→100% step 10%\n"
             << "BW = aggregate uncompressed bytes / measured kernel "
             << "time (sum over all " << kCopyOpBlocks << " blocks)\n";

  for (int sparsityPct = 0; sparsityPct <= 100; sparsityPct += 10) {
    XLOG(INFO) << "\n--- Sparsity " << sparsityPct
               << "% (zero-byte prefix of each block's input region) ---";
    printf(
        "%-10s  %14s  %16s  %10s  %11s\n",
        "Size",
        "CompBW(GB/s)",
        "DecompBW(GB/s)",
        "compRatio",
        "Correctness");
    printf(
        "%-10s  %14s  %16s  %10s  %11s\n",
        "--------",
        "--------------",
        "----------------",
        "----------",
        "-----------");

    for (std::size_t inputBytes : sizes) {
      // Local non-const copies so we can take addresses for the
      // `void* ka[]` kernel argument array (cudaLaunchKernel needs
      // non-const argument pointers). The range-for variable
      // `inputBytes` and the `const std::size_t stagingStride` below
      // would otherwise be const-qualified, making `&...` a
      // `const std::size_t*` that doesn't implicitly convert to
      // `void*`.
      std::size_t inputBytesArg = inputBytes;
      std::size_t stagingStride = ans_copy_op_bench_staging_stride(inputBytes);
      std::size_t stagingStrideArg = stagingStride;

      // Re-initialise every block's input region for this
      // (sparsity, inputBytes) cell. We do this per-cell rather than
      // once because the sparsity overlay differs between rows.
      for (int b = 0; b < kCopyOpBlocks; ++b) {
        const std::size_t blockStart = static_cast<std::size_t>(b) * inputBytes;
        std::size_t off = 0;
        while (off < inputBytes) {
          const std::size_t copy = std::min(kPatternBytes, inputBytes - off);
          PIPES_CUDA_CHECK(cudaMemcpy(
              static_cast<char*>(d_input.get()) + blockStart + off,
              host_pattern.data(),
              copy,
              cudaMemcpyHostToDevice));
          off += copy;
        }
        const std::size_t zeroBytes =
            (inputBytes * static_cast<std::size_t>(sparsityPct)) / 100;
        if (zeroBytes > 0) {
          PIPES_CUDA_CHECK(cudaMemset(
              static_cast<char*>(d_input.get()) + blockStart, 0, zeroBytes));
        }
      }

      // ---------------------------------------------------------------
      // Compression bandwidth measurement.
      // The kernel itself loops `iters` times so the kernel-event
      // window captures only ANS work plus one launch's worth of
      // setup overhead, amortised across `kCopyOpMeasureIter`
      // iterations.
      //
      // Per-launch synchronous error checks (cudaGetLastError +
      // cudaDeviceSynchronize) sit on EACH launch so a kernel-side
      // illegal memory access surfaces immediately at the failing
      // (sparsity, size, phase) cell instead of getting deferred to
      // scope-exit's `cudaFree`.
      // ---------------------------------------------------------------
      auto sync_check = [&](const char* phase) {
        const cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
          XLOG(FATAL) << "[AnsCopyOpStandalone] launch error in " << phase
                      << " sparsity=" << sparsityPct
                      << "% size=" << format_bytes(inputBytes)
                      << " blocks=" << kCopyOpBlocks
                      << " threads=" << kCopyOpThreads
                      << " stagingStride=" << stagingStride
                      << " err=" << cudaGetErrorString(launchErr);
        }
        const cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess) {
          XLOG(FATAL) << "[AnsCopyOpStandalone] async error in " << phase
                      << " sparsity=" << sparsityPct
                      << "% size=" << format_bytes(inputBytes)
                      << " blocks=" << kCopyOpBlocks
                      << " threads=" << kCopyOpThreads
                      << " stagingStride=" << stagingStride
                      << " err=" << cudaGetErrorString(syncErr);
        }
      };

      char* in_ptr = static_cast<char*>(d_input.get());
      char* st_ptr = static_cast<char*>(d_staging.get());
      auto launch_compress = [&](int& iters_ref) {
        void* ka[] = {
            &in_ptr,
            &st_ptr,
            &inputBytesArg,
            &stagingStrideArg,
            &iters_ref,
        };
        return comms::common::launchKernel(
            reinterpret_cast<void*>(bench_kernels.compress_kernel),
            dim3(kCopyOpBlocks),
            dim3(kCopyOpThreads),
            ka,
            /*stream=*/benchStream,
            /*clusterDim=*/std::nullopt);
      };

      int warmupIters = kCopyOpWarmupIter;
      PIPES_CUDA_CHECK(launch_compress(warmupIters));
      sync_check("compress-warmup");

      // NOTE: Intentionally NOT calling
      // `fetch_and_reset_ans_compress_stats()` here. That function
      // lives in the production `:alltoallv_tile_compressed_obj` TU
      // and references `g_pipes_ans_total_*_bytes` `__device__`
      // globals defined in `CopyOp.cuh`. Because the bench's
      // `:ans_copy_op_bench_obj` is its OWN device-link unit (with
      // its own nvcompdx dlink), it ALSO emits its own copies of
      // those globals — the runtime then has two duplicate symbols,
      // `cudaMemcpyFromSymbol(g_pipes_ans_total_uncomp_bytes)` fails
      // with "invalid device symbol", and CUDA's "first error
      // sticks" semantics make every subsequent `cudaLaunchKernel`
      // fail with the same error. We always report `n/a` for the
      // bench's compRatio column instead — the bench TU has
      // `PIPES_ANS_COLLECT_STATS` off by default anyway so the
      // counters wouldn't be populated even if the symbol lookup
      // worked.
      const LocalCompStats compStats{0, 0};

      int measureIters = kCopyOpMeasureIter;
      CudaEvent c_start, c_stop;
      PIPES_CUDA_CHECK(cudaEventRecord(c_start.get(), benchStream));
      PIPES_CUDA_CHECK(launch_compress(measureIters));
      PIPES_CUDA_CHECK(cudaEventRecord(c_stop.get(), benchStream));
      sync_check("compress-timed");

      float compMs = 0;
      PIPES_CUDA_CHECK(
          cudaEventElapsedTime(&compMs, c_start.get(), c_stop.get()));
      const float compAvgMs = compMs / static_cast<float>(kCopyOpMeasureIter);
      // Aggregate uncompressed bytes processed per iteration across
      // all blocks (each block compresses its own `inputBytes`).
      const std::size_t bytesPerIter =
          static_cast<std::size_t>(kCopyOpBlocks) * inputBytes;
      const float compBwGbps = (bytesPerIter / 1e9f) / (compAvgMs / 1000.0f);

      // ---------------------------------------------------------------
      // Decompression bandwidth measurement.
      // d_staging is already populated by the timed compress run
      // above. The recv kernel just walks the existing per-block
      // layout `iters` times.
      // ---------------------------------------------------------------
      char* out_ptr = static_cast<char*>(d_output.get());
      const char* cst_ptr = static_cast<const char*>(d_staging.get());
      auto launch_decompress = [&](int& iters_ref) {
        void* ka[] = {
            &out_ptr,
            const_cast<char**>(&cst_ptr),
            &inputBytesArg,
            &stagingStrideArg,
            &iters_ref,
        };
        return comms::common::launchKernel(
            reinterpret_cast<void*>(bench_kernels.decompress_kernel),
            dim3(kCopyOpBlocks),
            dim3(kCopyOpThreads),
            ka,
            /*stream=*/benchStream,
            /*clusterDim=*/std::nullopt);
      };

      warmupIters = kCopyOpWarmupIter;
      PIPES_CUDA_CHECK(launch_decompress(warmupIters));
      sync_check("decompress-warmup");

      measureIters = kCopyOpMeasureIter;
      CudaEvent d_start, d_stop;
      PIPES_CUDA_CHECK(cudaEventRecord(d_start.get(), benchStream));
      PIPES_CUDA_CHECK(launch_decompress(measureIters));
      PIPES_CUDA_CHECK(cudaEventRecord(d_stop.get(), benchStream));
      sync_check("decompress-timed");

      float decompMs = 0;
      PIPES_CUDA_CHECK(
          cudaEventElapsedTime(&decompMs, d_start.get(), d_stop.get()));
      const float decompAvgMs =
          decompMs / static_cast<float>(kCopyOpMeasureIter);
      const float decompBwGbps =
          (bytesPerIter / 1e9f) / (decompAvgMs / 1000.0f);

      char ratioBuf[32];
      if (compStats.compressed_bytes > 0) {
        const double ratio = static_cast<double>(compStats.uncompressed_bytes) /
            static_cast<double>(compStats.compressed_bytes);
        snprintf(ratioBuf, sizeof(ratioBuf), "%.2fx", ratio);
      } else {
        // PIPES_ANS_COLLECT_STATS was compiled out — the kernels
        // still run, the ratio just isn't observable from the host.
        snprintf(ratioBuf, sizeof(ratioBuf), "n/a");
      }

      printf(
          "%-10s  %14.2f  %16.2f  %10s  ",
          format_bytes(inputBytes).c_str(),
          compBwGbps,
          decompBwGbps,
          ratioBuf);
      fflush(stdout);

      // ---------------------------------------------------------------
      // Round-trip correctness check.
      // After the timed decompress, `d_output` holds the decompressed
      // bytes that the ans_copy_op_decompress_bench_kernel just
      // produced from `d_staging` (which was populated by the timed
      // compress run from `d_input`). Compare per-block.
      // ---------------------------------------------------------------
      std::vector<uint8_t> host_input(inputBytes);
      std::vector<uint8_t> host_output(inputBytes);
      for (int b = 0; b < kCopyOpBlocks; ++b) {
        const std::size_t blockStart = static_cast<std::size_t>(b) * inputBytes;
        PIPES_CUDA_CHECK(cudaMemcpy(
            host_input.data(),
            static_cast<const char*>(d_input.get()) + blockStart,
            inputBytes,
            cudaMemcpyDeviceToHost));
        PIPES_CUDA_CHECK(cudaMemcpy(
            host_output.data(),
            static_cast<const char*>(d_output.get()) + blockStart,
            inputBytes,
            cudaMemcpyDeviceToHost));
        if (host_input != host_output) {
          std::size_t firstMismatch = 0;
          while (firstMismatch < inputBytes &&
                 host_input[firstMismatch] == host_output[firstMismatch]) {
            ++firstMismatch;
          }
          // Close the row with FAIL before fataling so the partial
          // row in the table makes clear which cell crashed.
          printf("%11s\n", "FAIL");
          fflush(stdout);
          XLOG(FATAL) << "[AnsCopyOpStandalone] round-trip mismatch sparsity="
                      << sparsityPct << "% size=" << format_bytes(inputBytes)
                      << " block=" << b << " firstMismatch=" << firstMismatch
                      << " in=" << static_cast<int>(host_input[firstMismatch])
                      << " out="
                      << static_cast<int>(host_output[firstMismatch]);
        }
      }
      printf("%11s\n", "OK");
    }
  }

  if (gctx.ctx != nullptr) {
    PIPES_CU_CHECK(cuStreamDestroy(reinterpret_cast<CUstream>(benchStream)));
    PIPES_CU_CHECK(cuGreenCtxDestroy(gctx.ctx));
  }
}

} // namespace
} // namespace comms::prims::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
