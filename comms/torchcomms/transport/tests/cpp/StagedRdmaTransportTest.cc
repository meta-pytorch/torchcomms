// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <nccl.h>

#include <folly/init/Init.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#include <folly/logging/Init.h>
#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/torchcomms/transport/StagedRdmaTransport.h"
#include "comms/torchcomms/transport/tests/cpp/busy_kernel/BusyKernel.h"
#include "comms/utils/cvars/nccl_cvars.h"

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace torch::comms;
using namespace meta::comms;

// Logs (does not abort) on NCCL error so a failing background NCCL thread
// surfaces a message instead of crashing the whole MPI test process.
#define NCCLCHECK_LOG(cmd)                                                 \
  do {                                                                     \
    const ncclResult_t ncclErr = (cmd);                                    \
    if (ncclErr != ncclSuccess) {                                          \
      std::cerr << "[NCCL] error " << ncclGetErrorString(ncclErr) << " ("  \
                << static_cast<int>(ncclErr) << ") at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                  \
    }                                                                      \
  } while (0)

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

// --- Fill mode for transfer helpers ---

namespace {

enum class FillMode { CONSTANT, POSITIONAL };

// Fill buffer with position-dependent pattern: buf[i] = (uint8_t)(i % 251).
// Prime modulus 251 prevents alignment with chunk boundaries.
void fillPositionalPattern(std::vector<uint8_t>& buf) {
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<uint8_t>(i % 251);
  }
}

// Verify positional pattern. Returns (valid, firstMismatchIdx).
std::pair<bool, size_t> verifyPositionalPattern(
    const std::vector<uint8_t>& buf) {
  for (size_t i = 0; i < buf.size(); ++i) {
    if (buf[i] != static_cast<uint8_t>(i % 251)) {
      return {false, i};
    }
  }
  return {true, 0};
}

} // namespace

// --- Construction tests (no MPI needed, run independently on each rank) ---

TEST(StagedRdmaTransportTest, ConstructAndDestroy) {
  StagedTransferConfig config;
  config.stagingBufSize = 2 * 1024 * 1024;
  StagedRdmaServerTransport server(0, nullptr, config);
  StagedRdmaClientTransport client(0, nullptr, config);
  EXPECT_EQ(server.stagingBufSize(), config.stagingBufSize);
  EXPECT_EQ(client.stagingBufSize(), config.stagingBufSize);
}

TEST(StagedRdmaTransportTest, ConstructWithConfig) {
  StagedTransferConfig config;
  config.stagingBufSize = 1024 * 1024;
  config.chunkTimeout = std::chrono::milliseconds{5000};

  StagedRdmaServerTransport server(0, nullptr, config);
  StagedRdmaClientTransport client(0, nullptr, config);
  EXPECT_EQ(server.stagingBufSize(), config.stagingBufSize);
  EXPECT_EQ(client.stagingBufSize(), config.stagingBufSize);
}

TEST(StagedRdmaTransportTest, ConstructWithEventBase) {
  StagedTransferConfig config;
  config.stagingBufSize = 8 * 1024 * 1024;
  folly::ScopedEventBaseThread evbThread("test-evb");
  StagedRdmaServerTransport server(0, evbThread.getEventBase(), config);
  StagedRdmaClientTransport client(0, evbThread.getEventBase(), config);
  EXPECT_EQ(server.stagingBufSize(), config.stagingBufSize);
  EXPECT_EQ(client.stagingBufSize(), config.stagingBufSize);
}

// --- CUDA driver-lock contention reproducer (single GPU, no MPI) ---
//
// Reproduces the intermittent cudaStreamSynchronize-after-cudaMemcpyAsync stall
// the staged-RDMA hot path is exposed to (StagedRdmaTransport.cpp send :648 and
// recv :855). Two properties of the transport are mirrored exactly:
//   1. The staging buffer is *pageable* host memory (posix_memalign, identical
//      to StagedBuffer's CPU-mode ctor at StagedRdmaTransport.cpp:215, NOT
//      cudaHostRegister'd), so cudaMemcpyAsync degrades to a synchronous,
//      driver-staged copy that cannot overlap.
//   2. Completion is forced with a per-chunk blocking cudaStreamSynchronize on
//   a
//      cudaStreamNonBlocking stream.
// A second thread hammers cudaMalloc/cudaFree -- both take the device-wide CUDA
// driver lock and cudaFree forces a context-wide sync -- modelling the PyTorch
// caching allocator / NCCL / compute threads that share the same CUDA context
// in a real trainer. Under that contention the blocking synchronize stalls
// intermittently; on the sender side that delays posting the RDMA write and
// surfaces on the peer as a recv commTimeout.

namespace {

constexpr double kSlowSyncMs = 5.0;

struct SyncLatencyStats {
  double medianMs{0.0};
  double maxMs{0.0};
  size_t slowCount{0}; // syncs slower than kSlowSyncMs
};

SyncLatencyStats summarize(std::vector<double> samples) {
  SyncLatencyStats stats;
  if (samples.empty()) {
    return stats;
  }
  stats.maxMs = *std::max_element(samples.begin(), samples.end());
  for (double s : samples) {
    if (s > kSlowSyncMs) {
      ++stats.slowCount;
    }
  }
  const size_t mid = samples.size() / 2;
  std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
  stats.medianMs = samples[mid];
  return stats;
}

// Run `iters` of the staged-RDMA send copy pattern: D2H cudaMemcpyAsync from a
// device buffer into the pageable staging buffer, then a blocking
// cudaStreamSynchronize. Returns per-iteration synchronize latency (ms).
std::vector<double> runCopySyncLoop(
    void* stagingHost,
    const void* deviceSrc,
    size_t bytes,
    cudaStream_t stream,
    int iters) {
  std::vector<double> latenciesMs;
  latenciesMs.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(
        cudaMemcpyAsync(
            stagingHost, deviceSrc, bytes, cudaMemcpyDefault, stream),
        cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const auto t1 = std::chrono::steady_clock::now();
    latenciesMs.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return latenciesMs;
}

constexpr int kContenderThreads = 4; // saturate the device-wide driver lock
constexpr size_t kContenderBytes =
    64 * 1024 * 1024; // frequent allocs => many lock-heavy cudaFree syncs

struct ContendedResult {
  SyncLatencyStats stats;
  int contenderOps{0};
};

// How the concurrent contender threads behave:
//   Synchronizing      -- cudaMalloc/cudaFree every iteration: cudaFree forces
//   a
//                         device-wide sync and both take the global driver lock
//                         (the bug trigger; what the default PyTorch caching
//                         allocator does when it grows/frees).
//   PreallocatedReuse  -- allocate one buffer up front, then reuse it (no
//                         per-iteration cudaMalloc/cudaFree, hence no
//                         device-wide sync). Models the fix:
//                         PYTORCH_CUDA_ALLOC_CONF=expandable_segments / buffer
//                         reuse, where the allocator stops churning driver
//                         allocation calls in steady state.
//   BusyKernel         -- launch a clock64-spin kernel that saturates all SMs
//   to
//                         ~100% on a dedicated stream (no cudaMalloc/cudaFree).
//                         Models a compute-bound trainer; tests whether GPU
//                         compute saturation (vs allocator-lock churn) stalls
//                         the staging copy+sync.
enum class AllocMode { Synchronizing, PreallocatedReuse, BusyKernel };

constexpr long long kBusySpinTicks = 3'000'000; // ~2-3 ms of SM spin per launch

// Run the copy+sync loop while kContenderThreads threads run the given
// contention behaviour, modelling the PyTorch caching allocator / NCCL /
// compute threads that share the CUDA context in a real trainer.
ContendedResult measureUnderContention(
    void* stagingHost,
    const void* deviceSrc,
    size_t bytes,
    cudaStream_t stream,
    int iters,
    AllocMode mode) {
  std::atomic<bool> stop{false};
  std::atomic<int> contenderOps{0};
  std::vector<std::thread> contenders;
  contenders.reserve(kContenderThreads);
  for (int t = 0; t < kContenderThreads; ++t) {
    contenders.emplace_back([&]() {
      EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
      // PreallocatedReuse: take the memory once, before measurement starts.
      void* reused = nullptr;
      if (mode == AllocMode::PreallocatedReuse) {
        EXPECT_EQ(cudaMalloc(&reused, kContenderBytes), cudaSuccess);
      }
      // BusyKernel: a dedicated stream (separate from the staging stream) the
      // SM saturation runs on, so we measure cross-stream interference.
      cudaStream_t busyStream = nullptr;
      if (mode == AllocMode::BusyKernel) {
        EXPECT_EQ(
            cudaStreamCreateWithFlags(&busyStream, cudaStreamNonBlocking),
            cudaSuccess);
      }
      int launched = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        if (mode == AllocMode::Synchronizing) {
          void* p = nullptr;
          if (cudaMalloc(&p, kContenderBytes) == cudaSuccess) {
            cudaFree(p); // device-wide sync + global lock
            contenderOps.fetch_add(1, std::memory_order_relaxed);
          }
        } else if (mode == AllocMode::BusyKernel) {
          // Keep ~8 spin kernels queued so the SMs never go idle; sync
          // periodically to bound the queue without leaving compute gaps.
          stagedrdma_test::launchBusyKernel(busyStream, kBusySpinTicks);
          contenderOps.fetch_add(1, std::memory_order_relaxed);
          if (++launched % 8 == 0) {
            cudaStreamSynchronize(busyStream);
          }
        } else {
          // No driver allocation churn -- the buffer is already held. Sleep
          // (don't spin) so this isolates the removed cudaMalloc/cudaFree churn
          // without starving the measuring thread of CPU.
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          contenderOps.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (busyStream != nullptr) {
        cudaStreamSynchronize(busyStream);
        cudaStreamDestroy(busyStream);
      }
      if (reused != nullptr) {
        cudaFree(reused);
      }
    });
  }
  // Let the contenders reach steady state before measuring.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  auto lat = runCopySyncLoop(stagingHost, deviceSrc, bytes, stream, iters);
  stop.store(true, std::memory_order_relaxed);
  for (auto& c : contenders) {
    c.join();
  }
  return {summarize(lat), contenderOps.load()};
}

// Where the staging buffer lives:
//   HostPageable -- posix_memalign host memory (StagedBuffer CPU mode,
//                   StagedRdmaTransport.cpp:215): the staging copy is D2H/H2D
//                   into pageable memory, so cudaMemcpyAsync is synchronous.
//   Device       -- cudaMalloc device memory (StagedBuffer GPU mode, :191): the
//                   staging copy is device-to-device, so cudaMemcpyAsync is a
//                   genuine async copy-engine DMA.
enum class StagingLoc { HostPageable, Device };

void* allocStaging(StagingLoc loc, size_t bytes) {
  void* p = nullptr;
  if (loc == StagingLoc::HostPageable) {
    EXPECT_EQ(posix_memalign(&p, 4096, bytes), 0);
  } else {
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
  }
  return p;
}

void freeStaging(StagingLoc loc, void* p) {
  if (loc == StagingLoc::HostPageable) {
    free(p);
  } else {
    cudaFree(p);
  }
}

const char* stagingName(StagingLoc loc) {
  return loc == StagingLoc::HostPageable ? "host-pageable" : "device";
}

// Reproducer body: per-iteration cudaMemcpyAsync (deviceSrc -> staging) +
// cudaStreamSynchronize, measured uncontended (baseline) then under 4 threads
// churning cudaMalloc/cudaFree (device-wide-sync contention).
void runReproducer(StagingLoc loc) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  constexpr size_t kStagingBytes = 4 * 1024 * 1024; // default staging chunk
  constexpr int kIters = 1000; // chunks in a multi-hundred-MB transfer

  void* deviceSrc = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceSrc, kStagingBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(deviceSrc, 0xAB, kStagingBytes), cudaSuccess);

  void* staging = allocStaging(loc, kStagingBytes);
  ASSERT_NE(staging, nullptr);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  auto baselineStats = summarize(
      runCopySyncLoop(staging, deviceSrc, kStagingBytes, stream, kIters));
  auto contended = measureUnderContention(
      staging,
      deviceSrc,
      kStagingBytes,
      stream,
      kIters,
      AllocMode::Synchronizing);
  const auto& contendedStats = contended.stats;

  std::cout << "[StagedRdmaLockContention][" << stagingName(loc)
            << "] copy+sync latency (ms): baseline median="
            << baselineStats.medianMs << " max=" << baselineStats.maxMs
            << " slow(>" << kSlowSyncMs << "ms)=" << baselineStats.slowCount
            << " | contended median=" << contendedStats.medianMs
            << " max=" << contendedStats.maxMs
            << " slow=" << contendedStats.slowCount
            << " (contender malloc/free ops=" << contended.contenderOps
            << ", maxSlowdown="
            << (baselineStats.maxMs > 0
                    ? contendedStats.maxMs / baselineStats.maxMs
                    : 0.0)
            << "x)" << std::endl;

  // Liveness guard: a true global-lock deadlock manifests as a multi-second
  // single synchronize (prod chunk timeout is 300s). Fail well below that.
  constexpr double kHangThresholdMs = 30'000.0;
  EXPECT_LT(contendedStats.maxMs, kHangThresholdMs)
      << "cudaStreamSynchronize stalled " << contendedStats.maxMs
      << " ms under allocator-lock contention -- staged-RDMA hang reproduced";

  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  freeStaging(loc, staging);
  EXPECT_EQ(cudaFree(deviceSrc), cudaSuccess);
}

// Fix body: same copy+sync loop, comparing 5 rounds of cudaMalloc/cudaFree
// churn against 5 rounds with no allocation churn (PreallocatedReuse). Removing
// the churn removes the device-wide syncs, so the copy-stream sync stops
// stalling. Models PYTORCH_CUDA_ALLOC_CONF=expandable_segments / buffer reuse.
void runReuseFix(StagingLoc loc) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  constexpr size_t kStagingBytes = 4 * 1024 * 1024;
  constexpr int kIters = 1000;
  constexpr int kRounds = 5;

  void* deviceSrc = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceSrc, kStagingBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(deviceSrc, 0xAB, kStagingBytes), cudaSuccess);

  void* staging = allocStaging(loc, kStagingBytes);
  ASSERT_NE(staging, nullptr);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  // Aggregate across rounds: a single max is noisy; slow-sync counts and the
  // worst-of-rounds max are stable signals.
  size_t churnSlowTotal = 0;
  size_t reuseSlowTotal = 0;
  double churnWorstMax = 0.0;
  double reuseWorstMax = 0.0;
  for (int r = 0; r < kRounds; ++r) {
    auto churn = measureUnderContention(
        staging,
        deviceSrc,
        kStagingBytes,
        stream,
        kIters,
        AllocMode::Synchronizing);
    auto reuse = measureUnderContention(
        staging,
        deviceSrc,
        kStagingBytes,
        stream,
        kIters,
        AllocMode::PreallocatedReuse);
    churnSlowTotal += churn.stats.slowCount;
    reuseSlowTotal += reuse.stats.slowCount;
    churnWorstMax = std::max(churnWorstMax, churn.stats.maxMs);
    reuseWorstMax = std::max(reuseWorstMax, reuse.stats.maxMs);
  }

  std::cout << "[StagedRdmaLockContention][" << stagingName(loc)
            << "] FIX (no alloc churn / reuse) over " << kRounds
            << " rounds: churn slowSyncs=" << churnSlowTotal
            << " worstMax=" << churnWorstMax
            << "ms | reuse slowSyncs=" << reuseSlowTotal
            << " worstMax=" << reuseWorstMax << "ms" << std::endl;

  EXPECT_LT(reuseWorstMax, churnWorstMax)
      << "removing allocation churn did not reduce the worst sync stall";
  EXPECT_LE(reuseSlowTotal, churnSlowTotal)
      << "removing allocation churn did not reduce slow-sync count";

  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  freeStaging(loc, staging);
  EXPECT_EQ(cudaFree(deviceSrc), cudaSuccess);
}

} // namespace

// Reproduces the stall with a host-pageable staging buffer (StagedBuffer CPU
// mode): the synchronous pageable copy is amplified by the contender's
// device-wide cudaFree syncs.
TEST(
    StagedRdmaLockContentionTest,
    SyncAfterMemcpyAsyncStallsUnderAllocatorLock) {
  runReproducer(StagingLoc::HostPageable);
}

// Same reproducer with a device (GPU) staging buffer (StagedBuffer GPU mode):
// the staging copy is D2D so cudaMemcpyAsync is genuinely async; this measures
// whether device staging alone avoids the contention stall.
TEST(
    StagedRdmaLockContentionTest,
    SyncAfterMemcpyAsyncStallsUnderAllocatorLock_DeviceStaging) {
  runReproducer(StagingLoc::Device);
}

// Confirms the resolution with a host-pageable staging buffer: removing the
// per-iteration cudaMalloc/cudaFree churn (the device-wide-sync source) removes
// the stall. Models PYTORCH_CUDA_ALLOC_CONF=expandable_segments / buffer reuse.
// (stream-ordered cudaMallocAsync/cudaFreeAsync churn was measured and did NOT
// resolve it; pinning via cudaHostRegister is only a partial mitigation.)
TEST(StagedRdmaLockContentionTest, PreallocatedReuseResolvesStall) {
  runReuseFix(StagingLoc::HostPageable);
}

// Same resolution check with a device (GPU) staging buffer.
TEST(
    StagedRdmaLockContentionTest,
    PreallocatedReuseResolvesStall_DeviceStaging) {
  runReuseFix(StagingLoc::Device);
}

// Sweeps the staging-buffer size (1KB..100GB) across host-pageable and device
// staging, measuring four conditions per size: uncontended (baseline),
// cudaMalloc/cudaFree churn (driver-lock contention), no-churn reuse (the fix),
// and a GPU busy kernel saturating all SMs to ~100% (compute contention). Emits
// pipe-delimited "SWEEP|..." lines (and "SKIP|..." for configs that do not fit
// in device memory) for offline plotting. Iteration count is scaled down for
// large buffers to bound wall time; the median is the stable signal.
//
// Run single-rank (mpirun -np 1) to avoid 2-rank GPU-memory doubling at huge
// sizes; also self-gates to local rank 0 so it is safe under any launcher.
TEST(StagedRdmaLockContentionTest, StagingSizeSweep) {
  const char* localRankEnv = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (localRankEnv != nullptr && std::string(localRankEnv) != "0") {
    GTEST_SKIP() << "sweep runs on local rank 0 only";
  }
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  struct SizeSpec {
    const char* name;
    size_t bytes;
    int iters;
  };
  const std::vector<SizeSpec> sizes = {
      {"1KB", size_t{1} << 10, 500},
      {"100KB", size_t{100} << 10, 500},
      {"1MB", size_t{1} << 20, 500},
      {"100MB", size_t{100} << 20, 100},
      {"500MB", size_t{500} << 20, 30},
      {"1GB", size_t{1} << 30, 15},
      {"10GB", size_t{10} << 30, 5},
      {"50GB", size_t{50} << 30, 3},
      {"100GB", size_t{100} << 30, 2},
  };
  constexpr size_t kHeadroom = size_t{2} << 30; // leave 2GB device headroom

  for (auto loc : {StagingLoc::HostPageable, StagingLoc::Device}) {
    for (const auto& s : sizes) {
      // Device memory needed: deviceSrc (bytes) + (device staging ? bytes : 0).
      size_t freeB = 0, totalB = 0;
      ASSERT_EQ(cudaMemGetInfo(&freeB, &totalB), cudaSuccess);
      const size_t devNeed =
          s.bytes + (loc == StagingLoc::Device ? s.bytes : 0);
      if (freeB < devNeed + kHeadroom) {
        std::cout << "SKIP|" << stagingName(loc) << "|" << s.name << "|"
                  << s.bytes << "|device_mem free=" << freeB
                  << " need=" << devNeed << std::endl;
        continue;
      }

      void* deviceSrc = nullptr;
      if (cudaMalloc(&deviceSrc, s.bytes) != cudaSuccess) {
        std::cout << "SKIP|" << stagingName(loc) << "|" << s.name << "|"
                  << s.bytes << "|deviceSrc cudaMalloc failed" << std::endl;
        continue;
      }
      ASSERT_EQ(cudaMemset(deviceSrc, 0xAB, s.bytes), cudaSuccess);

      void* staging = nullptr;
      bool ok = false;
      if (loc == StagingLoc::Device) {
        ok = (cudaMalloc(&staging, s.bytes) == cudaSuccess);
      } else {
        ok = (posix_memalign(&staging, 4096, s.bytes) == 0 && staging);
      }
      if (!ok) {
        std::cout << "SKIP|" << stagingName(loc) << "|" << s.name << "|"
                  << s.bytes << "|staging alloc failed" << std::endl;
        cudaFree(deviceSrc);
        continue;
      }

      cudaStream_t stream = nullptr;
      ASSERT_EQ(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          cudaSuccess);

      auto base = summarize(
          runCopySyncLoop(staging, deviceSrc, s.bytes, stream, s.iters));
      auto churn = measureUnderContention(
          staging,
          deviceSrc,
          s.bytes,
          stream,
          s.iters,
          AllocMode::Synchronizing);
      auto reuse = measureUnderContention(
          staging,
          deviceSrc,
          s.bytes,
          stream,
          s.iters,
          AllocMode::PreallocatedReuse);
      auto busy = measureUnderContention(
          staging, deviceSrc, s.bytes, stream, s.iters, AllocMode::BusyKernel);

      // SWEEP|loc|name|bytes|iters|base_med|base_max|churn_med|churn_max|
      //   churn_slow|reuse_med|reuse_max|reuse_slow|busy_med|busy_max|busy_slow
      std::cout << "SWEEP|" << stagingName(loc) << "|" << s.name << "|"
                << s.bytes << "|" << s.iters << "|" << base.medianMs << "|"
                << base.maxMs << "|" << churn.stats.medianMs << "|"
                << churn.stats.maxMs << "|" << churn.stats.slowCount << "|"
                << reuse.stats.medianMs << "|" << reuse.stats.maxMs << "|"
                << reuse.stats.slowCount << "|" << busy.stats.medianMs << "|"
                << busy.stats.maxMs << "|" << busy.stats.slowCount << std::endl;

      EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
      freeStaging(loc, staging);
      EXPECT_EQ(cudaFree(deviceSrc), cudaSuccess);
    }
  }
}

// Long-running CUDA driver-lock stall probe, for live tracing (gdb / nsys /
// nvidia-smi). Reproduces the cudaStreamSynchronize-after-cudaMemcpyAsync
// stall: pageable posix_memalign staging + per-iter cudaMemcpyAsync + blocking
// cudaStreamSynchronize on the main thread, while kContenderThreads hammer
// cudaMalloc/cudaFree (device-wide sync + global driver lock). Loops for
// PROBE_SECONDS (default 60) so an external profiler can attach; prints the
// pid.
TEST(StagedRdmaLockContentionTest, CudaLockStallTraceProbe) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  const char* secEnv = std::getenv("PROBE_SECONDS");
  const double kSeconds = secEnv ? std::strtod(secEnv, nullptr) : 60.0;
  constexpr size_t kStagingBytes = 4 * 1024 * 1024;

  void* deviceSrc = nullptr;
  ASSERT_EQ(cudaMalloc(&deviceSrc, kStagingBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(deviceSrc, 0xAB, kStagingBytes), cudaSuccess);
  void* staging = nullptr;
  ASSERT_EQ(posix_memalign(&staging, 4096, kStagingBytes), 0);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  std::cout << "[CudaLockStallProbe] pid=" << getpid() << " running "
            << kSeconds << "s; attach with: gdb -p " << getpid() << std::endl;

  std::atomic<bool> stop{false};
  std::atomic<long> contenderOps{0};
  std::vector<std::thread> contenders;
  contenders.reserve(kContenderThreads);
  for (int t = 0; t < kContenderThreads; ++t) {
    contenders.emplace_back([&]() {
      EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
      while (!stop.load(std::memory_order_relaxed)) {
        void* p = nullptr;
        if (cudaMalloc(&p, kContenderBytes) == cudaSuccess) {
          cudaFree(p); // device-wide sync + global driver lock
          contenderOps.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  const auto t0 = std::chrono::steady_clock::now();
  size_t iters = 0;
  size_t slow = 0;
  double maxMs = 0.0;
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
             .count() < kSeconds) {
    const auto a = std::chrono::steady_clock::now();
    cudaMemcpyAsync(
        staging, deviceSrc, kStagingBytes, cudaMemcpyDefault, stream);
    cudaStreamSynchronize(stream);
    const auto b = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(b - a).count();
    ++iters;
    if (ms > kSlowSyncMs) {
      ++slow;
    }
    maxMs = std::max(maxMs, ms);
    if (iters % 5000 == 0) {
      std::cout << "[CudaLockStallProbe] iters=" << iters << " slow(>"
                << kSlowSyncMs << "ms)=" << slow << " maxMs=" << maxMs
                << " contenderOps=" << contenderOps.load() << std::endl;
    }
  }
  stop.store(true, std::memory_order_relaxed);
  for (auto& c : contenders) {
    c.join();
  }
  std::cout << "[CudaLockStallProbe] DONE iters=" << iters << " slow=" << slow
            << " maxMs=" << maxMs << std::endl;
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  free(staging);
  EXPECT_EQ(cudaFree(deviceSrc), cudaSuccess);
}

// Copy-only stall probe: can repeated/concurrent cudaMemcpyAsync +
// cudaStreamSynchronize stall WITHOUT any cudaMalloc/cudaFree? COPY_THREADS
// workers each loop a pageable D2H copy+sync on their own stream +
// pre-allocated buffers -- no alloc/free in steady state. COPY_PINNED=1
// cudaHostRegisters the staging buffers (A/B control: a true async DMA bypasses
// the pageable bounce-buffer pool). Loops PROBE_SECONDS; prints pid + aggregate
// stall stats.
TEST(StagedRdmaLockContentionTest, CudaCopyOnlyStallProbe) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto envD = [](const char* k, double d) {
    const char* v = std::getenv(k);
    return v ? std::strtod(v, nullptr) : d;
  };
  auto envI = [](const char* k, long d) {
    const char* v = std::getenv(k);
    return v ? std::strtol(v, nullptr, 10) : d;
  };
  const double kSeconds = envD("PROBE_SECONDS", 60.0);
  const int kThreads = static_cast<int>(envI("COPY_THREADS", 8));
  const size_t kBytes =
      static_cast<size_t>(envI("COPY_STAGING_BYTES", 4 * 1024 * 1024));
  const bool kPinned = envI("COPY_PINNED", 0) != 0;

  std::cout << "[CopyOnlyProbe] pid=" << getpid() << " threads=" << kThreads
            << " bytes=" << kBytes << " pinned=" << kPinned
            << " secs=" << kSeconds << "; attach: gdb -p " << getpid()
            << std::endl;

  std::atomic<long> totalIters{0};
  std::atomic<long> totalSlow{0};
  std::atomic<long long> maxUs{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&]() {
      EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
      void* dsrc = nullptr;
      EXPECT_EQ(cudaMalloc(&dsrc, kBytes), cudaSuccess);
      cudaMemset(dsrc, 0xAB, kBytes);
      void* staging = nullptr;
      EXPECT_EQ(posix_memalign(&staging, 4096, kBytes), 0);
      if (kPinned) {
        EXPECT_EQ(
            cudaHostRegister(staging, kBytes, cudaHostRegisterDefault),
            cudaSuccess);
      }
      cudaStream_t s = nullptr;
      EXPECT_EQ(
          cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);

      const auto t0 = std::chrono::steady_clock::now();
      long it = 0;
      long slow = 0;
      double mx = 0.0;
      while (
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count() < kSeconds) {
        const auto a = std::chrono::steady_clock::now();
        cudaMemcpyAsync(staging, dsrc, kBytes, cudaMemcpyDefault, s);
        cudaStreamSynchronize(s);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - a)
                              .count();
        ++it;
        if (ms > kSlowSyncMs) {
          ++slow;
        }
        mx = std::max(mx, ms);
      }
      totalIters.fetch_add(it, std::memory_order_relaxed);
      totalSlow.fetch_add(slow, std::memory_order_relaxed);
      const auto mus = static_cast<long long>(mx * 1000);
      long long prev = maxUs.load();
      while (mus > prev && !maxUs.compare_exchange_weak(prev, mus)) {
      }
      if (kPinned) {
        cudaHostUnregister(staging);
      }
      cudaStreamDestroy(s);
      free(staging);
      cudaFree(dsrc);
    });
  }

  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
             .count() < kSeconds) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "[CopyOnlyProbe] t+"
              << static_cast<int>(std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - start)
                                      .count())
              << "s iters=" << totalIters.load() << " slow(>" << kSlowSyncMs
              << "ms)=" << totalSlow.load()
              << " maxMs=" << (maxUs.load() / 1000.0) << std::endl;
  }
  for (auto& w : workers) {
    w.join();
  }
  std::cout << "[CopyOnlyProbe] DONE pinned=" << kPinned
            << " iters=" << totalIters.load() << " slow=" << totalSlow.load()
            << " maxMs=" << (maxUs.load() / 1000.0) << std::endl;
}

// --- Distributed test fixture (MPI-based, 2 ranks) ---

class StagedRdmaTransportDistributedTest : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    if (numRanks < 2) {
      GTEST_SKIP() << "Need at least 2 ranks";
    }
    ncclCvarInit();
    ASSERT_TRUE(ibverbx::ibvInit());
  }

  // Exchange connection info between rank 0 (server) and rank 1 (client).
  // Returns the peer's connection info string.
  std::string exchangeConnInfo(const std::string& localConnInfo) {
    // Exchange lengths first
    int localLen = static_cast<int>(localConnInfo.size());
    int peerLen = 0;
    int peerRank = 1 - globalRank;
    MPI_CHECK(MPI_Sendrecv(
        &localLen,
        1,
        MPI_INT,
        peerRank,
        0,
        &peerLen,
        1,
        MPI_INT,
        peerRank,
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE));

    // Exchange actual strings
    std::string peerConnInfo(peerLen, '\0');
    MPI_CHECK(MPI_Sendrecv(
        localConnInfo.data(),
        localLen,
        MPI_CHAR,
        peerRank,
        1,
        peerConnInfo.data(),
        peerLen,
        MPI_CHAR,
        peerRank,
        1,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE));

    return peerConnInfo;
  }

  // Broadcast a size_t from rank 0 to all ranks.
  size_t broadcastSize(size_t value) {
    MPI_CHECK(MPI_Bcast(&value, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD));
    return value;
  }

  // Run a transfer with specified source/destination memory types.
  void runTransferWithMemType(bool srcOnGpu, bool dstOnGpu);

  // Run a contiguous transfer: server fills + sends, client recvs + verifies.
  // Both ranks must call this. Internally branches on globalRank.
  void runTransfer(
      size_t totalBytes,
      uint8_t fillByte,
      StagedTransferConfig config = {},
      FillMode fillMode = FillMode::CONSTANT) {
    int cudaDev = localRank;
    ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

    folly::ScopedEventBaseThread evbThread("RDMA-Worker");

    if (globalRank == 0) {
      // --- Server ---
      StagedRdmaServerTransport transport(
          cudaDev, evbThread.getEventBase(), config);
      auto connInfo = transport.setupLocalTransport();
      auto peerConnInfo = exchangeConnInfo(connInfo);
      transport.connectRemoteTransport(peerConnInfo);

      broadcastSize(totalBytes);

      // Allocate source GPU buffer and fill with pattern
      void* srcGpu = nullptr;
      ASSERT_EQ(cudaMalloc(&srcGpu, totalBytes), cudaSuccess);
      std::vector<uint8_t> hostBuf(totalBytes, fillByte);
      if (fillMode == FillMode::POSITIONAL) {
        fillPositionalPattern(hostBuf);
      }
      ASSERT_EQ(
          cudaMemcpy(
              srcGpu, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice),
          cudaSuccess);

      ScatterGatherDescriptor sgDesc;
      sgDesc.entries.push_back({srcGpu, totalBytes});
      auto result = transport.send(sgDesc).get();
      EXPECT_EQ(result, commSuccess);

      EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
    } else {
      // --- Client ---
      StagedRdmaClientTransport transport(
          cudaDev, evbThread.getEventBase(), config);
      auto connInfo = transport.setupLocalTransport();
      auto peerConnInfo = exchangeConnInfo(connInfo);
      transport.connectRemoteTransport(peerConnInfo);

      auto recvBytes = broadcastSize(0);

      // Allocate destination GPU buffer (zeroed)
      void* dstGpu = nullptr;
      ASSERT_EQ(cudaMalloc(&dstGpu, recvBytes), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstGpu, 0, recvBytes), cudaSuccess);

      ScatterGatherDescriptor sgDesc;
      sgDesc.entries.push_back({dstGpu, recvBytes});
      auto result = transport.recv(sgDesc).get();
      EXPECT_EQ(result, commSuccess);

      if (result == commSuccess) {
        std::vector<uint8_t> hostBuf(recvBytes);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(), dstGpu, recvBytes, cudaMemcpyDeviceToHost),
            cudaSuccess);

        if (fillMode == FillMode::POSITIONAL) {
          auto [valid, idx] = verifyPositionalPattern(hostBuf);
          EXPECT_TRUE(valid)
              << "Positional mismatch at byte " << idx << ": expected 0x"
              << std::hex << static_cast<int>(idx % 251) << " got 0x"
              << static_cast<int>(hostBuf[idx]);
        } else {
          for (size_t i = 0; i < recvBytes; ++i) {
            if (hostBuf[i] != fillByte) {
              ADD_FAILURE() << "Data mismatch at byte " << i << ": expected 0x"
                            << std::hex << static_cast<int>(fillByte)
                            << " got 0x" << static_cast<int>(hostBuf[i]);
              break;
            }
          }
        }
      }

      EXPECT_EQ(cudaFree(dstGpu), cudaSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
};

// --- Connection setup test ---

TEST_F(StagedRdmaTransportDistributedTest, SetupAndConnect) {
  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(cudaDev);
    auto connInfo = transport.setupLocalTransport();
    EXPECT_FALSE(connInfo.empty());
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);
  } else {
    StagedRdmaClientTransport transport(cudaDev);
    auto connInfo = transport.setupLocalTransport();
    EXPECT_FALSE(connInfo.empty());
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// --- Data transfer tests ---

TEST_F(StagedRdmaTransportDistributedTest, SingleChunkTransfer) {
  // Transfer smaller than staging buffer → single chunk
  runTransfer(8192, 0xAB);
}

TEST_F(StagedRdmaTransportDistributedTest, MultiChunkTransfer) {
  // 256KB total with 64KB staging = 4 chunks
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;
  runTransfer(256 * 1024, 0xCD, config);
}

TEST_F(StagedRdmaTransportDistributedTest, LastChunkSmaller) {
  // 100KB total with 64KB staging = 2 chunks (64KB + 36KB)
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;
  runTransfer(100 * 1024, 0xEF, config);
}

TEST_F(StagedRdmaTransportDistributedTest, ExactlyOneStagingBuffer) {
  // Transfer exactly equal to staging buffer size → single chunk, no remainder
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;
  runTransfer(64 * 1024, 0x42, config);
}

TEST_F(StagedRdmaTransportDistributedTest, LargeTransfer) {
  // 4MB transfer with default 64MB staging = single chunk but large data
  runTransfer(4 * 1024 * 1024, 0x77);
}

TEST_F(StagedRdmaTransportDistributedTest, UniquePatternIntegrity) {
  // Positional pattern catches offset bugs that constant fill cannot.
  // 200KB with 64KB staging = 4 chunks (non-aligned last chunk).
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;
  runTransfer(200 * 1024, 0, config, FillMode::POSITIONAL);
}

TEST_F(StagedRdmaTransportDistributedTest, ManyChunksTransfer) {
  // 2MB total with 32KB staging = 64 chunks.
  StagedTransferConfig config;
  config.stagingBufSize = 32 * 1024;
  runTransfer(2 * 1024 * 1024, 0, config, FillMode::POSITIONAL);
}

// --- Sequential transfers (transport reuse) ---

TEST_F(StagedRdmaTransportDistributedTest, SequentialTransfers) {
  // Two transfers on the same transport pair with different fill bytes.
  // Tests the always-auto-replenish recv WR fix.
  // Both ranks construct identical transfer specs; no broadcastSize needed.
  StagedTransferConfig config;
  config.stagingBufSize = 32 * 1024;

  struct TransferSpec {
    size_t totalBytes;
    uint8_t fillByte;
  };
  std::vector<TransferSpec> transfers = {
      {64 * 1024, 0xAA},
      {64 * 1024, 0xBB},
  };

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    // --- Server ---
    StagedRdmaServerTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    for (size_t t = 0; t < transfers.size(); ++t) {
      auto& spec = transfers[t];

      void* srcGpu = nullptr;
      ASSERT_EQ(cudaMalloc(&srcGpu, spec.totalBytes), cudaSuccess);
      std::vector<uint8_t> hostBuf(spec.totalBytes, spec.fillByte);
      ASSERT_EQ(
          cudaMemcpy(
              srcGpu, hostBuf.data(), spec.totalBytes, cudaMemcpyHostToDevice),
          cudaSuccess);

      ScatterGatherDescriptor sgDesc;
      sgDesc.entries.push_back({srcGpu, spec.totalBytes});
      auto result = transport.send(sgDesc).get();
      EXPECT_EQ(result, commSuccess) << "Transfer " << t << " failed";

      EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
    }
  } else {
    // --- Client ---
    StagedRdmaClientTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    for (size_t t = 0; t < transfers.size(); ++t) {
      auto& spec = transfers[t];

      void* dstGpu = nullptr;
      ASSERT_EQ(cudaMalloc(&dstGpu, spec.totalBytes), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstGpu, 0, spec.totalBytes), cudaSuccess);

      ScatterGatherDescriptor sgDesc;
      sgDesc.entries.push_back({dstGpu, spec.totalBytes});
      auto result = transport.recv(sgDesc).get();
      EXPECT_EQ(result, commSuccess) << "Transfer " << t << " failed";

      if (result == commSuccess) {
        std::vector<uint8_t> hostBuf(spec.totalBytes);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(),
                dstGpu,
                spec.totalBytes,
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        for (size_t i = 0; i < spec.totalBytes; ++i) {
          if (hostBuf[i] != spec.fillByte) {
            ADD_FAILURE() << "Transfer " << t << ": mismatch at byte " << i
                          << ": expected 0x" << std::hex
                          << static_cast<int>(spec.fillByte) << " got 0x"
                          << static_cast<int>(hostBuf[i]);
            break;
          }
        }
      }

      EXPECT_EQ(cudaFree(dstGpu), cudaSuccess);
    }
  }
}

// --- Parameterized buffer size tests ---

class StagedRdmaTransportBufferSizeTest
    : public StagedRdmaTransportDistributedTest,
      public ::testing::WithParamInterface<size_t> {};

TEST_P(StagedRdmaTransportBufferSizeTest, TransferWithSize) {
  const size_t totalBytes = GetParam();
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;
  runTransfer(totalBytes, 0, config, FillMode::POSITIONAL);
}

INSTANTIATE_TEST_SUITE_P(
    BufferSizes,
    StagedRdmaTransportBufferSizeTest,
    ::testing::Values(
        1, // 1 byte — minimal transfer
        1023, // sub-page non-power-of-2
        1024, // 1KB — sub-page
        4096, // page boundary
        4097, // page boundary + 1
        65535, // staging - 1
        65536, // exactly staging
        65537, // staging + 1 (forces 2 chunks)
        131072, // exactly 2 chunks
        1048576), // 16 chunks (1MB)
    [](const ::testing::TestParamInfo<size_t>& info) {
      return "Size_" + std::to_string(info.param);
    });

// --- Scatter/Gather transfer tests ---

// Server sends contiguous data, client scatters to non-contiguous GPU regions.
TEST_F(StagedRdmaTransportDistributedTest, ScatterRecvTransfer) {
  const size_t entrySize = 4096;
  const size_t numEntries = 4;
  const size_t totalBytes = entrySize * numEntries;

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(totalBytes);

    // Contiguous source with positional pattern
    void* srcGpu = nullptr;
    ASSERT_EQ(cudaMalloc(&srcGpu, totalBytes), cudaSuccess);
    std::vector<uint8_t> hostBuf(totalBytes);
    fillPositionalPattern(hostBuf);
    ASSERT_EQ(
        cudaMemcpy(srcGpu, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice),
        cudaSuccess);

    ScatterGatherDescriptor putDesc;
    putDesc.entries.push_back({srcGpu, totalBytes});
    auto result = transport.send(putDesc).get();
    EXPECT_EQ(result, commSuccess);

    EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
  } else {
    StagedRdmaClientTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(0);

    // Allocate non-contiguous destination: 4 separate GPU buffers
    std::vector<void*> dstPtrs(numEntries);
    for (size_t i = 0; i < numEntries; ++i) {
      ASSERT_EQ(cudaMalloc(&dstPtrs[i], entrySize), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstPtrs[i], 0, entrySize), cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < numEntries; ++i) {
      sgDesc.entries.push_back({dstPtrs[i], entrySize});
    }

    auto result = transport.recv(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    if (result == commSuccess) {
      for (size_t i = 0; i < numEntries; ++i) {
        std::vector<uint8_t> hostBuf(entrySize);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(), dstPtrs[i], entrySize, cudaMemcpyDeviceToHost),
            cudaSuccess);
        size_t globalOffset = i * entrySize;
        for (size_t j = 0; j < entrySize; ++j) {
          uint8_t expected = static_cast<uint8_t>((globalOffset + j) % 251);
          EXPECT_EQ(hostBuf[j], expected)
              << "Mismatch in entry " << i << " at byte " << j
              << " (global offset " << globalOffset + j << ")";
          if (hostBuf[j] != expected) {
            break;
          }
        }
      }
    }

    for (auto* ptr : dstPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Server gathers from non-contiguous GPU regions, client receives contiguously.
TEST_F(StagedRdmaTransportDistributedTest, GatherPutTransfer) {
  const size_t entrySize = 4096;
  const size_t numEntries = 4;
  const size_t totalBytes = entrySize * numEntries;

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);
    broadcastSize(totalBytes);

    // Non-contiguous source with positional pattern
    std::vector<void*> srcPtrs(numEntries);
    for (size_t i = 0; i < numEntries; ++i) {
      ASSERT_EQ(cudaMalloc(&srcPtrs[i], entrySize), cudaSuccess);
      std::vector<uint8_t> hostBuf(entrySize);
      size_t globalOffset = i * entrySize;
      for (size_t j = 0; j < entrySize; ++j) {
        hostBuf[j] = static_cast<uint8_t>((globalOffset + j) % 251);
      }
      ASSERT_EQ(
          cudaMemcpy(
              srcPtrs[i], hostBuf.data(), entrySize, cudaMemcpyHostToDevice),
          cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < numEntries; ++i) {
      sgDesc.entries.push_back({srcPtrs[i], entrySize});
    }

    auto result = transport.send(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    for (auto* ptr : srcPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  } else {
    StagedRdmaClientTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    auto recvBytes = broadcastSize(0);

    // Contiguous destination
    void* dstGpu = nullptr;
    ASSERT_EQ(cudaMalloc(&dstGpu, recvBytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(dstGpu, 0, recvBytes), cudaSuccess);

    ScatterGatherDescriptor sgDesc;
    sgDesc.entries.push_back({dstGpu, recvBytes});
    auto result = transport.recv(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    if (result == commSuccess) {
      std::vector<uint8_t> hostBuf(recvBytes);
      ASSERT_EQ(
          cudaMemcpy(hostBuf.data(), dstGpu, recvBytes, cudaMemcpyDeviceToHost),
          cudaSuccess);
      auto [valid, idx] = verifyPositionalPattern(hostBuf);
      EXPECT_TRUE(valid) << "Positional mismatch at byte " << idx
                         << ": expected 0x" << std::hex
                         << static_cast<int>(idx % 251) << " got 0x"
                         << static_cast<int>(hostBuf[idx]);
    }

    EXPECT_EQ(cudaFree(dstGpu), cudaSuccess);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Both sides use scatter/gather: server gathers, client scatters.
TEST_F(StagedRdmaTransportDistributedTest, GatherPutScatterRecv) {
  const size_t entrySize = 4096;
  const size_t numEntries = 4;
  const size_t totalBytes = entrySize * numEntries;

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);
    broadcastSize(totalBytes);

    // Non-contiguous source
    std::vector<void*> srcPtrs(numEntries);
    for (size_t i = 0; i < numEntries; ++i) {
      ASSERT_EQ(cudaMalloc(&srcPtrs[i], entrySize), cudaSuccess);
      std::vector<uint8_t> hostBuf(entrySize);
      size_t globalOffset = i * entrySize;
      for (size_t j = 0; j < entrySize; ++j) {
        hostBuf[j] = static_cast<uint8_t>((globalOffset + j) % 251);
      }
      ASSERT_EQ(
          cudaMemcpy(
              srcPtrs[i], hostBuf.data(), entrySize, cudaMemcpyHostToDevice),
          cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < numEntries; ++i) {
      sgDesc.entries.push_back({srcPtrs[i], entrySize});
    }

    auto result = transport.send(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    for (auto* ptr : srcPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  } else {
    StagedRdmaClientTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(0);

    // Non-contiguous destination
    std::vector<void*> dstPtrs(numEntries);
    for (size_t i = 0; i < numEntries; ++i) {
      ASSERT_EQ(cudaMalloc(&dstPtrs[i], entrySize), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstPtrs[i], 0, entrySize), cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < numEntries; ++i) {
      sgDesc.entries.push_back({dstPtrs[i], entrySize});
    }

    auto result = transport.recv(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    if (result == commSuccess) {
      for (size_t i = 0; i < numEntries; ++i) {
        std::vector<uint8_t> hostBuf(entrySize);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(), dstPtrs[i], entrySize, cudaMemcpyDeviceToHost),
            cudaSuccess);
        size_t globalOffset = i * entrySize;
        for (size_t j = 0; j < entrySize; ++j) {
          uint8_t expected = static_cast<uint8_t>((globalOffset + j) % 251);
          EXPECT_EQ(hostBuf[j], expected)
              << "Mismatch in entry " << i << " at byte " << j;
          if (hostBuf[j] != expected) {
            break;
          }
        }
      }
    }

    for (auto* ptr : dstPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Scatter recv with multi-chunk transfer. 256KB total, 64KB staging = 4 chunks,
// scattered across 8 non-contiguous 32KB destination buffers.
TEST_F(StagedRdmaTransportDistributedTest, ScatterRecvMultiChunk) {
  const size_t entrySize = 32 * 1024;
  const size_t numEntries = 8;
  const size_t totalBytes = entrySize * numEntries;
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(totalBytes);

    // Contiguous source with positional pattern
    void* srcGpu = nullptr;
    ASSERT_EQ(cudaMalloc(&srcGpu, totalBytes), cudaSuccess);
    std::vector<uint8_t> hostBuf(totalBytes);
    fillPositionalPattern(hostBuf);
    ASSERT_EQ(
        cudaMemcpy(srcGpu, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice),
        cudaSuccess);

    ScatterGatherDescriptor putDesc;
    putDesc.entries.push_back({srcGpu, totalBytes});
    auto result = transport.send(putDesc).get();
    EXPECT_EQ(result, commSuccess);

    EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
  } else {
    StagedRdmaClientTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(0);

    // 8 non-contiguous 32KB destination buffers
    std::vector<void*> dstPtrs(numEntries);
    for (size_t i = 0; i < numEntries; ++i) {
      ASSERT_EQ(cudaMalloc(&dstPtrs[i], entrySize), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstPtrs[i], 0, entrySize), cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < numEntries; ++i) {
      sgDesc.entries.push_back({dstPtrs[i], entrySize});
    }

    auto result = transport.recv(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    if (result == commSuccess) {
      for (size_t i = 0; i < numEntries; ++i) {
        std::vector<uint8_t> hostBuf(entrySize);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(), dstPtrs[i], entrySize, cudaMemcpyDeviceToHost),
            cudaSuccess);
        size_t globalOffset = i * entrySize;
        for (size_t j = 0; j < entrySize; ++j) {
          uint8_t expected = static_cast<uint8_t>((globalOffset + j) % 251);
          EXPECT_EQ(hostBuf[j], expected)
              << "Mismatch in entry " << i << " at byte " << j
              << " (global offset " << globalOffset + j << ")";
          if (hostBuf[j] != expected) {
            break;
          }
        }
      }
    }

    for (auto* ptr : dstPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Scatter recv with uneven entry sizes. 3 entries of 40KB, 24KB, 64KB = 128KB
// total, with 48KB staging = 3 chunks.
TEST_F(StagedRdmaTransportDistributedTest, ScatterRecvUnevenEntries) {
  const std::vector<size_t> entrySizes = {
      40 * 1024,
      24 * 1024,
      64 * 1024,
  };
  size_t totalBytes = 0;
  for (auto s : entrySizes) {
    totalBytes += s;
  }
  StagedTransferConfig config;
  config.stagingBufSize = 48 * 1024;

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(totalBytes);

    void* srcGpu = nullptr;
    ASSERT_EQ(cudaMalloc(&srcGpu, totalBytes), cudaSuccess);
    std::vector<uint8_t> hostBuf(totalBytes);
    fillPositionalPattern(hostBuf);
    ASSERT_EQ(
        cudaMemcpy(srcGpu, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice),
        cudaSuccess);

    ScatterGatherDescriptor putDesc;
    putDesc.entries.push_back({srcGpu, totalBytes});
    auto result = transport.send(putDesc).get();
    EXPECT_EQ(result, commSuccess);

    EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
  } else {
    StagedRdmaClientTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(0);

    // Allocate separate GPU buffers with varying sizes
    std::vector<void*> dstPtrs(entrySizes.size());
    for (size_t i = 0; i < entrySizes.size(); ++i) {
      ASSERT_EQ(cudaMalloc(&dstPtrs[i], entrySizes[i]), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstPtrs[i], 0, entrySizes[i]), cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < entrySizes.size(); ++i) {
      sgDesc.entries.push_back({dstPtrs[i], entrySizes[i]});
    }

    auto result = transport.recv(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    if (result == commSuccess) {
      size_t globalOffset = 0;
      for (size_t i = 0; i < entrySizes.size(); ++i) {
        std::vector<uint8_t> hostBuf(entrySizes[i]);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(),
                dstPtrs[i],
                entrySizes[i],
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        for (size_t j = 0; j < entrySizes[i]; ++j) {
          uint8_t expected = static_cast<uint8_t>((globalOffset + j) % 251);
          EXPECT_EQ(hostBuf[j], expected)
              << "Mismatch in entry " << i << " at byte " << j;
          if (hostBuf[j] != expected) {
            break;
          }
        }
        globalOffset += entrySizes[i];
      }
    }

    for (auto* ptr : dstPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// --- CPU/GPU memory type transfer tests ---

namespace {

void* allocBuffer(size_t bytes, bool onGpu, int cudaDev) {
  if (onGpu) {
    void* ptr = nullptr;
    EXPECT_EQ(cudaSetDevice(cudaDev), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
    return ptr;
  }
  void* ptr = malloc(bytes);
  EXPECT_NE(ptr, nullptr);
  return ptr;
}

void freeBuffer(void* ptr, bool onGpu) {
  if (onGpu) {
    cudaFree(ptr);
  } else {
    free(ptr);
  }
}

void copyToBuffer(void* dst, const void* src, size_t bytes, bool dstOnGpu) {
  if (dstOnGpu) {
    EXPECT_EQ(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), cudaSuccess);
  } else {
    memcpy(dst, src, bytes);
  }
}

void copyFromBuffer(void* dst, const void* src, size_t bytes, bool srcOnGpu) {
  if (srcOnGpu) {
    EXPECT_EQ(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
  } else {
    memcpy(dst, src, bytes);
  }
}

} // namespace

// Transfers 1MB with positional pattern, parameterized by source/destination
// memory type (CPU or GPU). Both ranks must call this.
void StagedRdmaTransportDistributedTest::runTransferWithMemType(
    bool srcOnGpu,
    bool dstOnGpu) {
  const size_t totalBytes = 1024 * 1024;
  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);
    broadcastSize(totalBytes);

    void* src = allocBuffer(totalBytes, srcOnGpu, cudaDev);
    std::vector<uint8_t> pattern(totalBytes);
    fillPositionalPattern(pattern);
    copyToBuffer(src, pattern.data(), totalBytes, srcOnGpu);

    ScatterGatherDescriptor sgDesc;
    sgDesc.entries.push_back({src, totalBytes});
    EXPECT_EQ(transport.send(sgDesc).get(), commSuccess);
    freeBuffer(src, srcOnGpu);
  } else {
    StagedRdmaClientTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);
    auto recvBytes = broadcastSize(0);

    void* dst = allocBuffer(recvBytes, dstOnGpu, cudaDev);
    if (dstOnGpu) {
      EXPECT_EQ(cudaMemset(dst, 0, recvBytes), cudaSuccess);
    } else {
      memset(dst, 0, recvBytes);
    }

    ScatterGatherDescriptor sgDesc;
    sgDesc.entries.push_back({dst, recvBytes});
    EXPECT_EQ(transport.recv(sgDesc).get(), commSuccess);

    std::vector<uint8_t> hostBuf(recvBytes);
    copyFromBuffer(hostBuf.data(), dst, recvBytes, dstOnGpu);
    auto [valid, idx] = verifyPositionalPattern(hostBuf);
    EXPECT_TRUE(valid) << "Mismatch at byte " << idx;
    freeBuffer(dst, dstOnGpu);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(StagedRdmaTransportDistributedTest, CpuSrcGpuDst) {
  runTransferWithMemType(/*srcOnGpu=*/false, /*dstOnGpu=*/true);
}

TEST_F(StagedRdmaTransportDistributedTest, CpuSrcCpuDst) {
  runTransferWithMemType(/*srcOnGpu=*/false, /*dstOnGpu=*/false);
}

TEST_F(StagedRdmaTransportDistributedTest, GpuSrcCpuDst) {
  runTransferWithMemType(/*srcOnGpu=*/true, /*dstOnGpu=*/false);
}

TEST_F(StagedRdmaTransportDistributedTest, GpuSrcGpuDst) {
  runTransferWithMemType(/*srcOnGpu=*/true, /*dstOnGpu=*/true);
}

// --- Multi-EventBase stream cache test ---
// Verifies that multiple transports on different EventBase threads can
// share the process-global stream cache and transfer data correctly.

TEST_F(StagedRdmaTransportDistributedTest, MultiEventBaseStreamCache) {
  const size_t totalBytes = 8192;
  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  // Two separate EventBase threads — simulates multiple handlers in one process
  folly::ScopedEventBaseThread evbThread1("RDMA-Worker-1");
  folly::ScopedEventBaseThread evbThread2("RDMA-Worker-2");

  if (globalRank == 0) {
    // First server transport on EventBase thread 1
    StagedRdmaServerTransport transport1(cudaDev, evbThread1.getEventBase());
    auto connInfo1 = transport1.setupLocalTransport();
    auto peerConnInfo1 = exchangeConnInfo(connInfo1);
    transport1.connectRemoteTransport(peerConnInfo1);

    broadcastSize(totalBytes);

    void* srcGpu = nullptr;
    ASSERT_EQ(cudaMalloc(&srcGpu, totalBytes), cudaSuccess);
    std::vector<uint8_t> hostBuf(totalBytes, 0xAA);
    ASSERT_EQ(
        cudaMemcpy(srcGpu, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice),
        cudaSuccess);

    // Send on transport1 (thread 1 — creates stream, calls cudaSetDevice)
    ScatterGatherDescriptor sgDesc;
    sgDesc.entries.push_back({srcGpu, totalBytes});
    auto result1 = transport1.send(sgDesc).get();
    EXPECT_EQ(result1, commSuccess) << "Transport 1 send failed";

    // Second server transport on EventBase thread 2 (stream cache hit —
    // without the fix, thread 2 never gets cudaSetDevice)
    StagedRdmaServerTransport transport2(cudaDev, evbThread2.getEventBase());
    auto connInfo2 = transport2.setupLocalTransport();
    auto peerConnInfo2 = exchangeConnInfo(connInfo2);
    transport2.connectRemoteTransport(peerConnInfo2);

    broadcastSize(totalBytes);

    std::fill(hostBuf.begin(), hostBuf.end(), 0xBB);
    ASSERT_EQ(
        cudaMemcpy(srcGpu, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice),
        cudaSuccess);

    sgDesc.entries.clear();
    sgDesc.entries.push_back({srcGpu, totalBytes});
    auto result2 = transport2.send(sgDesc).get();
    EXPECT_EQ(result2, commSuccess)
        << "Transport 2 send failed (stream cache hit bug)";

    EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
  } else {
    // Client side — two transports matching the server
    StagedRdmaClientTransport transport1(cudaDev, evbThread1.getEventBase());
    auto connInfo1 = transport1.setupLocalTransport();
    auto peerConnInfo1 = exchangeConnInfo(connInfo1);
    transport1.connectRemoteTransport(peerConnInfo1);

    auto recvBytes1 = broadcastSize(0);
    void* dstGpu1 = nullptr;
    ASSERT_EQ(cudaMalloc(&dstGpu1, recvBytes1), cudaSuccess);
    ASSERT_EQ(cudaMemset(dstGpu1, 0, recvBytes1), cudaSuccess);

    ScatterGatherDescriptor sgDesc1;
    sgDesc1.entries.push_back({dstGpu1, recvBytes1});
    auto result1 = transport1.recv(sgDesc1).get();
    EXPECT_EQ(result1, commSuccess) << "Transport 1 recv failed";

    if (result1 == commSuccess) {
      std::vector<uint8_t> hostBuf(recvBytes1);
      ASSERT_EQ(
          cudaMemcpy(
              hostBuf.data(), dstGpu1, recvBytes1, cudaMemcpyDeviceToHost),
          cudaSuccess);
      const std::vector<uint8_t> expected1(recvBytes1, 0xAA);
      EXPECT_EQ(hostBuf, expected1) << "Transport 1 data mismatch";
    }
    EXPECT_EQ(cudaFree(dstGpu1), cudaSuccess);

    // Second client transport on thread 2
    StagedRdmaClientTransport transport2(cudaDev, evbThread2.getEventBase());
    auto connInfo2 = transport2.setupLocalTransport();
    auto peerConnInfo2 = exchangeConnInfo(connInfo2);
    transport2.connectRemoteTransport(peerConnInfo2);

    auto recvBytes2 = broadcastSize(0);
    void* dstGpu2 = nullptr;
    ASSERT_EQ(cudaMalloc(&dstGpu2, recvBytes2), cudaSuccess);
    ASSERT_EQ(cudaMemset(dstGpu2, 0, recvBytes2), cudaSuccess);

    ScatterGatherDescriptor sgDesc2;
    sgDesc2.entries.push_back({dstGpu2, recvBytes2});
    auto result2 = transport2.recv(sgDesc2).get();
    EXPECT_EQ(result2, commSuccess)
        << "Transport 2 recv failed (stream cache hit bug)";

    if (result2 == commSuccess) {
      std::vector<uint8_t> hostBuf(recvBytes2);
      ASSERT_EQ(
          cudaMemcpy(
              hostBuf.data(), dstGpu2, recvBytes2, cudaMemcpyDeviceToHost),
          cudaSuccess);
      const std::vector<uint8_t> expected2(recvBytes2, 0xBB);
      EXPECT_EQ(hostBuf, expected2) << "Transport 2 data mismatch";
    }
    EXPECT_EQ(cudaFree(dstGpu2), cudaSuccess);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// --- Bidirectional stress test (for the stress team: reproduce the hang) ---
//
// Each of the 2 ranks runs BOTH a server and a client (4 transports, 2 logical
// connections: rank0.server->rank1.client and rank1.server->rank0.client) and
// mutually transfers STRESS_BYTES (default 30 GB) each way, looped STRESS_ITERS
// (default 100) times. Exercises the real staged-RDMA send/recv path over IB RC
// QPs to surface hangs / deadlocks / leaks. Server and client run on separate
// EventBase threads so the two directions proceed concurrently (a single
// EventBase runs each transfer to completion and would deadlock a mutual
// transfer). Buffers are allocated once and reused (no per-iteration alloc
// churn). Knobs (smoke vs full): STRESS_BYTES, STRESS_ITERS,
// STRESS_STAGING_BYTES.
TEST_F(StagedRdmaTransportDistributedTest, BidirectionalStress30GBx100) {
  auto envSize = [](const char* key, size_t def) -> size_t {
    const char* v = std::getenv(key);
    return v ? static_cast<size_t>(std::strtoull(v, nullptr, 10)) : def;
  };
  const size_t kBytes = envSize("STRESS_BYTES", size_t{30} << 30);
  const size_t kIters = envSize("STRESS_ITERS", 100);
  const size_t kStaging = envSize("STRESS_STAGING_BYTES", size_t{64} << 20);

  const int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  // Need src + dst (each kBytes) on this rank's device, plus headroom.
  size_t freeB = 0, totalB = 0;
  ASSERT_EQ(cudaMemGetInfo(&freeB, &totalB), cudaSuccess);
  if (freeB < 2 * kBytes + (size_t{2} << 30)) {
    GTEST_SKIP() << "need " << (2 * kBytes) << " B free on device " << cudaDev
                 << ", have " << freeB;
  }

  StagedTransferConfig cfg;
  cfg.stagingBufSize = kStaging; // prod uses STAGED_RDMA_BUF_SIZE = 1 MB

  // Server and client each on their own EventBase so the two directions run
  // concurrently.
  folly::ScopedEventBaseThread serverEvb("RDMA-Server");
  folly::ScopedEventBaseThread clientEvb("RDMA-Client");
  StagedRdmaServerTransport server(cudaDev, serverEvb.getEventBase(), cfg);
  StagedRdmaClientTransport client(cudaDev, clientEvb.getEventBase(), cfg);

  // Bootstrap both connections. exchangeConnInfo is a matched Sendrecv; calling
  // it twice in the same order on both ranks exchanges server then client info.
  // Per the transport bootstrap, the server connects with the REMOTE client's
  // info and the client with the REMOTE server's info.
  const auto serverInfo = server.setupLocalTransport();
  const auto clientInfo = client.setupLocalTransport();
  const auto peerServerInfo = exchangeConnInfo(serverInfo);
  const auto peerClientInfo = exchangeConnInfo(clientInfo);
  server.connectRemoteTransport(peerClientInfo);
  client.connectRemoteTransport(peerServerInfo);

  // Allocate once, reuse across iterations.
  void* srcGpu = nullptr;
  void* dstGpu = nullptr;
  ASSERT_EQ(cudaMalloc(&srcGpu, kBytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dstGpu, kBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(srcGpu, 0xCD, kBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(dstGpu, 0, kBytes), cudaSuccess);

  ScatterGatherDescriptor sendDesc;
  sendDesc.entries.push_back({srcGpu, kBytes});
  ScatterGatherDescriptor recvDesc;
  recvDesc.entries.push_back({dstGpu, kBytes});

  std::cout << "[BidirectionalStress] rank " << globalRank << " dev " << cudaDev
            << ": " << kIters << " iters x " << kBytes
            << " B each way, staging " << kStaging << " B" << std::endl;

  for (size_t i = 0; i < kIters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    // Post recv first, then send; both progress on their own EventBase threads.
    auto recvF = client.recv(recvDesc);
    auto sendF = server.send(sendDesc);
    const auto sendRes = std::move(sendF).get();
    const auto recvRes = std::move(recvF).get();
    const auto t1 = std::chrono::steady_clock::now();
    EXPECT_EQ(sendRes, commSuccess) << "send failed at iter " << i;
    EXPECT_EQ(recvRes, commSuccess) << "recv failed at iter " << i;
    if (i == 0 || (i + 1) % 10 == 0 || i + 1 == kIters) {
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      std::cout << "[BidirectionalStress] rank " << globalRank << " iter "
                << (i + 1) << "/" << kIters << " " << ms << " ms" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
  EXPECT_EQ(cudaFree(dstGpu), cudaSuccess);
  MPI_Barrier(MPI_COMM_WORLD);
}

// --- Bidirectional stress + concurrent NCCL comm init (the prod insight) ---
//
// Reproduces the field insight that staged-RDMA weight transfers stall when a
// NCCL communicator (re)initialization (ncclCommInitRankConfig) runs on the
// same GPU / CUDA context at the same time. NCCL comm init is alloc-heavy and
// takes the device-wide CUDA driver lock (channel/buffer cudaMallocs, kernel
// module load, IB setup); concurrent with the staged-RDMA D2H/H2D copy +
// per-chunk cudaStreamSynchronize it can serialize behind that lock (cf. the
// synthetic cudaMalloc/cudaFree + copy-only repros). This drives REAL NCCL
// alongside the real bidirectional transport.
//
// A background thread on each rank runs a FIXED number of NCCL rounds (NOT a
// stop flag): ncclCommInitRankConfig is COLLECTIVE, so an early exit on one
// rank would deadlock the peer's init. The collective init also self-
// synchronizes the two ranks' background threads. uniqueIds are generated on
// rank 0 and MPI_Bcast'd up front (the MPI env is THREAD_SINGLE -> no MPI off
// the main thread). Set STRESS_NCCL_ROUNDS=0 for a clean no-NCCL control.
//
// Modes (STRESS_NCCL_MODE):
//   initchurn (default): each round inits a fresh comm + 1 allreduce + destroy
//                        -> the ncclCommInitRankConfig storm from the insight.
//   allreduce:           init one comm, then loop allreduce+sync -> steady-
//                        state collective traffic (lighter-contention control).
// Knobs: STRESS_BYTES, STRESS_ITERS, STRESS_STAGING_BYTES, STRESS_NCCL_ROUNDS,
//        STRESS_NCCL_BYTES, STRESS_NCCL_MODE.
TEST_F(StagedRdmaTransportDistributedTest, BidirectionalStressWithNccl) {
  auto envSize = [](const char* key, size_t def) -> size_t {
    const char* v = std::getenv(key);
    return v ? static_cast<size_t>(std::strtoull(v, nullptr, 10)) : def;
  };
  const size_t kBytes = envSize("STRESS_BYTES", size_t{256} << 20);
  const size_t kIters = envSize("STRESS_ITERS", 50);
  const size_t kStaging = envSize("STRESS_STAGING_BYTES", size_t{4} << 20);
  const size_t kNcclRounds = envSize("STRESS_NCCL_ROUNDS", 50);
  const size_t kNcclBytes = envSize("STRESS_NCCL_BYTES", size_t{16} << 20);
  const char* modeEnv = std::getenv("STRESS_NCCL_MODE");
  const bool kInitChurn =
      (modeEnv == nullptr) || std::string(modeEnv) == "initchurn";
  // Per-round allreduce on the freshly-created comm. OFF by default: the
  // insight is ncclCommInitRankConfig itself, and NCCL's collective device
  // kernels are not loadable in this standalone unittest binary (kernel launch
  // returns "named symbol not found"). Init/destroy is the contention under
  // test; flip STRESS_NCCL_ALLREDUCE=1 only on a binary that has the kernels.
  const bool kNcclAllReduce = envSize("STRESS_NCCL_ALLREDUCE", 0) != 0;

  const int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  // src + dst (staged-RDMA) + NCCL send/recv buffers + headroom.
  size_t freeB = 0, totalB = 0;
  ASSERT_EQ(cudaMemGetInfo(&freeB, &totalB), cudaSuccess);
  if (freeB < 2 * kBytes + 2 * kNcclBytes + (size_t{2} << 30)) {
    GTEST_SKIP() << "need " << (2 * kBytes + 2 * kNcclBytes)
                 << " B free on device " << cudaDev << ", have " << freeB;
  }

  // --- NCCL uniqueId bootstrap on the MAIN thread (MPI is THREAD_SINGLE) ---
  // initchurn needs a fresh id per round (each id's bootstrap root is consumed
  // by one comm init); allreduce needs one. Generate on rank 0, broadcast all.
  const size_t kNumIds =
      (kNcclRounds == 0) ? 0 : (kInitChurn ? kNcclRounds : 1);
  std::vector<ncclUniqueId> ncclIds(kNumIds);
  if (kNumIds > 0) {
    if (globalRank == 0) {
      for (size_t i = 0; i < kNumIds; ++i) {
        NCCLCHECK_LOG(ncclGetUniqueId(&ncclIds[i]));
      }
    }
    MPI_CHECK(MPI_Bcast(
        ncclIds.data(),
        static_cast<int>(kNumIds * sizeof(ncclUniqueId)),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD));
  }

  StagedTransferConfig cfg;
  cfg.stagingBufSize = kStaging;

  folly::ScopedEventBaseThread serverEvb("RDMA-Server");
  folly::ScopedEventBaseThread clientEvb("RDMA-Client");
  StagedRdmaServerTransport server(cudaDev, serverEvb.getEventBase(), cfg);
  StagedRdmaClientTransport client(cudaDev, clientEvb.getEventBase(), cfg);

  const auto serverInfo = server.setupLocalTransport();
  const auto clientInfo = client.setupLocalTransport();
  const auto peerServerInfo = exchangeConnInfo(serverInfo);
  const auto peerClientInfo = exchangeConnInfo(clientInfo);
  server.connectRemoteTransport(peerClientInfo);
  client.connectRemoteTransport(peerServerInfo);

  void* srcGpu = nullptr;
  void* dstGpu = nullptr;
  ASSERT_EQ(cudaMalloc(&srcGpu, kBytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dstGpu, kBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(srcGpu, 0xCD, kBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(dstGpu, 0, kBytes), cudaSuccess);

  ScatterGatherDescriptor sendDesc;
  sendDesc.entries.push_back({srcGpu, kBytes});
  ScatterGatherDescriptor recvDesc;
  recvDesc.entries.push_back({dstGpu, kBytes});

  // Both ranks enter the concurrent phase together.
  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "[NcclStress] rank " << globalRank << " dev " << cudaDev << ": "
            << kIters << " iters x " << kBytes << " B each way, staging "
            << kStaging << " B, ncclRounds " << kNcclRounds << " mode "
            << (kInitChurn ? "initchurn" : "allreduce") << " ncclBytes "
            << kNcclBytes << std::endl;

  // --- Background NCCL worker: fixed kNcclRounds, no MPI, no early stop ---
  std::atomic<size_t> ncclDone{0};
  std::atomic<double> ncclMaxRoundMs{0.0};
  std::thread ncclThread;
  if (kNumIds > 0) {
    ncclThread = std::thread([&]() {
      if (cudaSetDevice(cudaDev) != cudaSuccess) {
        std::cerr << "[NcclStress] cudaSetDevice failed in worker" << std::endl;
        return;
      }
      cudaStream_t ncclStream = nullptr;
      if (cudaStreamCreateWithFlags(&ncclStream, cudaStreamNonBlocking) !=
          cudaSuccess) {
        std::cerr << "[NcclStress] stream create failed" << std::endl;
        return;
      }
      void* nsend = nullptr;
      void* nrecv = nullptr;
      if (cudaMalloc(&nsend, kNcclBytes) != cudaSuccess ||
          cudaMalloc(&nrecv, kNcclBytes) != cudaSuccess) {
        std::cerr << "[NcclStress] nccl buf malloc failed" << std::endl;
        return;
      }
      cudaMemset(nsend, 1, kNcclBytes);

      // float32 + sum is always instantiated (some ncclChar/op kernels are
      // not, which surfaces as "named symbol not found" at launch).
      const size_t nFloats = kNcclBytes / sizeof(float);
      auto oneAllReduce = [&](ncclComm_t comm) {
        if (!kNcclAllReduce) {
          return;
        }
        NCCLCHECK_LOG(ncclAllReduce(
            nsend, nrecv, nFloats, ncclFloat32, ncclSum, comm, ncclStream));
        cudaStreamSynchronize(ncclStream);
      };

      ncclComm_t persistent = nullptr;
      if (!kInitChurn) {
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        NCCLCHECK_LOG(ncclCommInitRankConfig(
            &persistent, numRanks, ncclIds[0], globalRank, &config));
      }

      for (size_t r = 0; r < kNcclRounds; ++r) {
        const auto r0 = std::chrono::steady_clock::now();
        if (kInitChurn) {
          // The insight: a fresh ncclCommInitRankConfig concurrent with the
          // staged-RDMA transfer, repeated.
          ncclComm_t comm = nullptr;
          ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
          NCCLCHECK_LOG(ncclCommInitRankConfig(
              &comm, numRanks, ncclIds[r], globalRank, &config));
          if (comm != nullptr) {
            oneAllReduce(comm);
            NCCLCHECK_LOG(ncclCommDestroy(comm));
          }
        } else {
          oneAllReduce(persistent);
        }
        const auto r1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(r1 - r0).count();
        double prev = ncclMaxRoundMs.load();
        while (ms > prev && !ncclMaxRoundMs.compare_exchange_weak(prev, ms)) {
        }
        ncclDone.fetch_add(1);
      }

      if (persistent != nullptr) {
        NCCLCHECK_LOG(ncclCommDestroy(persistent));
      }
      cudaFree(nsend);
      cudaFree(nrecv);
      cudaStreamDestroy(ncclStream);
    });
  }

  // --- Staged-RDMA transfer loop, concurrent with the NCCL worker ---
  double maxIterMs = 0.0;
  size_t slowIters = 0;
  for (size_t i = 0; i < kIters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    auto recvF = client.recv(recvDesc);
    auto sendF = server.send(sendDesc);
    const auto sendRes = std::move(sendF).get();
    const auto recvRes = std::move(recvF).get();
    const auto t1 = std::chrono::steady_clock::now();
    EXPECT_EQ(sendRes, commSuccess) << "send failed at iter " << i;
    EXPECT_EQ(recvRes, commSuccess) << "recv failed at iter " << i;
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    maxIterMs = std::max(maxIterMs, ms);
    if (ms > 1000.0) {
      ++slowIters;
    }
    if (i == 0 || (i + 1) % 10 == 0 || i + 1 == kIters || ms > 1000.0) {
      std::cout << "[NcclStress] rank " << globalRank << " iter " << (i + 1)
                << "/" << kIters << " " << ms << " ms (ncclRounds done "
                << ncclDone.load() << "/" << kNcclRounds << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (ncclThread.joinable()) {
    ncclThread.join();
  }

  std::cout << "[NcclStress] rank " << globalRank << " SUMMARY maxIterMs "
            << maxIterMs << " slowIters(>1s) " << slowIters
            << " ncclRoundsDone " << ncclDone.load() << "/" << kNcclRounds
            << " maxNcclRoundMs " << ncclMaxRoundMs.load() << std::endl;

  EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
  EXPECT_EQ(cudaFree(dstGpu), cudaSuccess);
  MPI_Barrier(MPI_COMM_WORLD);
}

// --- Lifecycle + fault-injection stress test (hunt for the hang) ---
//
// Like BidirectionalStress, but (a) creates and DESTROYS both transports (and
// their EventBase threads) every iteration to stress setup/teardown, and
// (b) injects faults to exercise the error/abort/timeout paths that are the
// likely hang vectors:
//   fault 0-2: clean bidirectional transfer
//   fault 3:   cancelPendingRecv() mid-flight (abort path)
//   fault 4:   size mismatch -- client expects one extra staging chunk that
//              never arrives -> recv hits chunkTimeout (the prod commTimeout
//              class), then the transport is torn down and recreated.
// A short chunkTimeout (default 2s) bounds each fault so the test completes
// unless something truly hangs. Run the binary in a loop under an outer
// wall-clock `timeout`: a run that exceeds it is a real hang to capture.
// Knobs: STRESS_BYTES, STRESS_ITERS, STRESS_STAGING_BYTES,
//        STRESS_CHUNK_TIMEOUT_MS, STRESS_FAULTS (0 disables fault injection).
TEST_F(StagedRdmaTransportDistributedTest, BidirectionalStressLifecycleFaults) {
  auto envSize = [](const char* key, size_t def) -> size_t {
    const char* v = std::getenv(key);
    return v ? static_cast<size_t>(std::strtoull(v, nullptr, 10)) : def;
  };
  const size_t kBytes = envSize("STRESS_BYTES", size_t{64} << 20);
  const size_t kIters = envSize("STRESS_ITERS", 50);
  const size_t kStaging = envSize("STRESS_STAGING_BYTES", size_t{4} << 20);
  const size_t kTimeoutMs = envSize("STRESS_CHUNK_TIMEOUT_MS", 2000);
  const bool kFaults = envSize("STRESS_FAULTS", 1) != 0;

  const int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  // dst is sized for the mismatch case (one extra staging chunk). Buffers are
  // reused across iters; the *transports* are what we lifecycle.
  const size_t kDstBytes = kBytes + kStaging;
  void* srcGpu = nullptr;
  void* dstGpu = nullptr;
  ASSERT_EQ(cudaMalloc(&srcGpu, kBytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dstGpu, kDstBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(srcGpu, 0xCD, kBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(dstGpu, 0, kDstBytes), cudaSuccess);

  std::cout << "[LifecycleFaults] rank " << globalRank << " dev " << cudaDev
            << ": " << kIters << " iters, " << kBytes << " B, staging "
            << kStaging << " B, chunkTimeout " << kTimeoutMs << " ms, faults "
            << kFaults << std::endl;

  for (size_t iter = 0; iter < kIters; ++iter) {
    const int fault = kFaults ? static_cast<int>(iter % 5) : 0;
    const auto t0 = std::chrono::steady_clock::now();
    commResult_t sendRes = commSuccess;
    commResult_t recvRes = commSuccess;

    { // ---- fresh transports each iteration (lifecycle) ----
      folly::ScopedEventBaseThread serverEvb("RDMA-Server");
      folly::ScopedEventBaseThread clientEvb("RDMA-Client");
      StagedTransferConfig cfg;
      cfg.stagingBufSize = kStaging;
      cfg.chunkTimeout = std::chrono::milliseconds(kTimeoutMs);
      StagedRdmaServerTransport server(cudaDev, serverEvb.getEventBase(), cfg);
      StagedRdmaClientTransport client(cudaDev, clientEvb.getEventBase(), cfg);

      const auto serverInfo = server.setupLocalTransport();
      const auto clientInfo = client.setupLocalTransport();
      const auto peerServerInfo = exchangeConnInfo(serverInfo);
      const auto peerClientInfo = exchangeConnInfo(clientInfo);
      server.connectRemoteTransport(peerClientInfo);
      client.connectRemoteTransport(peerServerInfo);

      const size_t recvBytes = (fault == 4) ? kDstBytes : kBytes;
      ScatterGatherDescriptor sendDesc;
      sendDesc.entries.push_back({srcGpu, kBytes});
      ScatterGatherDescriptor recvDesc;
      recvDesc.entries.push_back({dstGpu, recvBytes});

      auto recvF = client.recv(recvDesc);
      auto sendF = server.send(sendDesc);

      if (fault == 3) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        client.cancelPendingRecv(); // both futures must still return
      }

      sendRes = std::move(sendF).get(); // must return (no hang)
      recvRes = std::move(recvF).get(); // must return (no hang)

      if (fault <= 2) {
        EXPECT_EQ(sendRes, commSuccess) << "clean send iter " << iter;
        EXPECT_EQ(recvRes, commSuccess) << "clean recv iter " << iter;
      }
    } // transports + EventBase threads destroyed here

    const auto t1 = std::chrono::steady_clock::now();
    if (iter == 0 || (iter + 1) % 10 == 0 || iter + 1 == kIters || fault >= 3) {
      std::cout << "[LifecycleFaults] rank " << globalRank << " iter "
                << (iter + 1) << "/" << kIters << " fault " << fault
                << " send=" << static_cast<int>(sendRes)
                << " recv=" << static_cast<int>(recvRes) << " "
                << std::chrono::duration<double, std::milli>(t1 - t0).count()
                << " ms" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
  EXPECT_EQ(cudaFree(dstGpu), cudaSuccess);
  MPI_Barrier(MPI_COMM_WORLD);
}

// --- main ---

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
