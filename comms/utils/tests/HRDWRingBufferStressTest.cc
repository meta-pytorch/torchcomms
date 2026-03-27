// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// GPU stress tests for HRDWRingBuffer. These exercise real GPU-CPU concurrency
// to validate __threadfence_system() ordering, torn-read detection, and the
// snapshot-before-callback design.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/HRDWRingBuffer.h"
#include "comms/utils/HRDWRingBufferReader.h"

using meta::comms::colltrace::HRDWRingBuffer;
using meta::comms::colltrace::HRDWRingBufferReader;
using meta::comms::colltrace::launchRingBufferEndWrite;
using meta::comms::colltrace::launchRingBufferStartWrite;

namespace {

// RAII wrapper for per-collective pinned memory (slot storage).
struct PerCollState {
  uint64_t* slotStorage{nullptr};

  PerCollState() {
    void* ptr = nullptr;
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaHostAlloc(&ptr, sizeof(uint64_t), cudaHostAllocDefault);
    slotStorage = static_cast<uint64_t*>(ptr);
    *slotStorage = 0;
  }

  ~PerCollState() {
    if (slotStorage) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaFreeHost(slotStorage);
    }
  }

  PerCollState(const PerCollState&) = delete;
  PerCollState& operator=(const PerCollState&) = delete;
};

} // namespace

class HRDWRingBufferStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Skip if no GPU available.
    int deviceCount = 0;
    auto err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaSetDevice(0);
  }
};

// ---------------------------------------------------------------------------
// Stress config: shared by both eager and graph tests.
//   ringSize   — ring buffer size (power of 2)
//   numStreams  — number of concurrent streams (eager) or colls per graph
//   numWrites  — writes per stream (eager) or replays (graph)
// ---------------------------------------------------------------------------

struct StressConfig {
  uint32_t ringSize;
  int numStreams;
  int numWrites;

  int totalWrites() const {
    return numStreams * numWrites;
  }
};

static const StressConfig kStressConfigs[] = {
    {64, 4, 500}, // Small ring, forces lapping.
    {256, 8, 200}, // Medium ring, balanced.
    {4096, 1, 2000}, // Single stream, no lapping, monotonic ordering.
    {4096, 10, 200}, // Large ring, many streams/colls.
    {16, 1, 1000}, // Tiny ring, maximum lapping pressure.
};

// Helper: launch totalWrites start+end pairs across numStreams streams.
static void launchEagerWrites(
    const std::vector<cudaStream_t>& streams,
    HRDWRingBuffer<void*>& buf,
    std::vector<std::unique_ptr<PerCollState>>& states) {
  auto totalWrites = static_cast<int>(states.size());
  auto numStreams = static_cast<int>(streams.size());
  for (int i = 0; i < totalWrites; ++i) {
    auto streamIdx = i % numStreams;
    launchRingBufferStartWrite(
        streams[streamIdx],
        buf.ring(),
        buf.writeIndex(),
        buf.mask(),
        nullptr,
        states[i]->slotStorage);
    launchRingBufferEndWrite(
        streams[streamIdx], buf.ring(), states[i]->slotStorage, buf.mask());
  }
}

// Define an eager stress test that iterates over all configs. Provides:
//   cfg       — StressConfig
//   buf       — HRDWRingBuffer<void*>
//   streams   — vector of cudaStream_t (cfg.numStreams)
//   states    — vector of PerCollState (cfg.totalWrites())
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define EAGER_STRESS_TEST(name)                                 \
  static void eagerStressBody_##name(                           \
      const StressConfig&,                                      \
      HRDWRingBuffer<void*>&,                                   \
      std::vector<cudaStream_t>&,                               \
      std::vector<std::unique_ptr<PerCollState>>&);             \
  TEST_F(HRDWRingBufferStressTest, name) {                      \
    for (const auto& cfg : kStressConfigs) {                    \
      SCOPED_TRACE(                                             \
          "ring=" + std::to_string(cfg.ringSize) +              \
          " streams=" + std::to_string(cfg.numStreams) +        \
          " writes=" + std::to_string(cfg.numWrites));          \
      HRDWRingBuffer<void*> buf(cfg.ringSize);                  \
      ASSERT_TRUE(buf.valid());                                 \
      std::vector<cudaStream_t> streams(cfg.numStreams);        \
      for (auto& s : streams) {                                 \
        ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);           \
      }                                                         \
      std::vector<std::unique_ptr<PerCollState>> states;        \
      states.reserve(cfg.totalWrites());                        \
      for (int i = 0; i < cfg.totalWrites(); ++i) {             \
        states.push_back(std::make_unique<PerCollState>());     \
      }                                                         \
      eagerStressBody_##name(cfg, buf, streams, states);        \
      for (auto& s : streams) {                                 \
        /* NOLINTNEXTLINE(facebook-cuda-safe-api-call-check) */ \
        cudaStreamDestroy(s);                                   \
      }                                                         \
    }                                                           \
  }                                                             \
  static void eagerStressBody_##name(                           \
      const StressConfig& cfg,                                  \
      HRDWRingBuffer<void*>& buf,                               \
      std::vector<cudaStream_t>& streams,                       \
      std::vector<std::unique_ptr<PerCollState>>& states)

// Launch start+end pairs across multiple streams, sync, then poll.
// Every delivered entry must have valid timestamps.
EAGER_STRESS_TEST(MultiStreamTimestampOrdering) {
  HRDWRingBufferReader<void*> reader(buf);
  launchEagerWrites(streams, buf, states);

  for (auto& s : streams) {
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
  }

  uint64_t badEntries = 0;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    if (e.start_ns == 0 || e.end_ns == 0 || e.end_ns < e.start_ns) {
      ++badEntries;
    }
  });

  EXPECT_EQ(badEntries, 0u)
      << "Entries with invalid timestamps (threadfence_system ordering failure)";
  EXPECT_EQ(
      result.entriesRead + result.entriesLost,
      static_cast<uint64_t>(cfg.totalWrites()));
}

// Launch writes while a background CPU thread polls continuously.
EAGER_STRESS_TEST(ConcurrentWriteAndPoll) {
  std::atomic<uint64_t> totalRead{0};
  std::atomic<uint64_t> totalLost{0};
  std::atomic<uint64_t> totalBad{0};
  std::atomic<bool> writersDone{false};

  std::thread readerThread([&]() {
    HRDWRingBufferReader<void*> reader(buf);
    while (!writersDone.load(std::memory_order_acquire)) {
      auto result = reader.poll([&](const auto& e, uint64_t) {
        if (e.start_ns == 0 || e.end_ns == 0 || e.end_ns < e.start_ns) {
          totalBad.fetch_add(1, std::memory_order_relaxed);
        }
      });
      totalRead.fetch_add(result.entriesRead, std::memory_order_relaxed);
      totalLost.fetch_add(result.entriesLost, std::memory_order_relaxed);
    }
    // Final drain.
    auto result = reader.poll([&](const auto& e, uint64_t) {
      if (e.start_ns == 0 || e.end_ns == 0 || e.end_ns < e.start_ns) {
        totalBad.fetch_add(1, std::memory_order_relaxed);
      }
    });
    totalRead.fetch_add(result.entriesRead, std::memory_order_relaxed);
    totalLost.fetch_add(result.entriesLost, std::memory_order_relaxed);
  });

  launchEagerWrites(streams, buf, states);

  for (auto& s : streams) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamSynchronize(s);
  }
  writersDone.store(true, std::memory_order_release);
  readerThread.join();

  EXPECT_EQ(totalBad.load(), 0u)
      << "Corrupted entries during concurrent polling";

  auto accounted = totalRead.load() + totalLost.load();
  EXPECT_EQ(accounted, static_cast<uint64_t>(cfg.totalWrites()));
}

// Single-stream configs only: verify monotonic timestamp ordering.
// On a single stream, start_ns of entry N+1 >= end_ns of entry N.
EAGER_STRESS_TEST(SingleStreamMonotonicOrdering) {
  if (cfg.numStreams != 1) {
    return; // Only meaningful for single-stream configs.
  }

  HRDWRingBufferReader<void*> reader(buf);
  launchEagerWrites(streams, buf, states);
  ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);

  uint64_t badEntries = 0;
  uint64_t prevEndNs = 0;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    if (e.start_ns == 0 || e.end_ns == 0 || e.end_ns < e.start_ns) {
      ++badEntries;
    }
    if (prevEndNs > 0 && e.start_ns < prevEndNs) {
      ++badEntries;
    }
    prevEndNs = e.end_ns;
  });

  EXPECT_EQ(badEntries, 0u);
  EXPECT_EQ(
      result.entriesRead + result.entriesLost,
      static_cast<uint64_t>(cfg.totalWrites()));
  EXPECT_FALSE(result.tornReadDetected);
}

// NOTE: graph tests reuse kStressConfigs: numStreams = numColls, numWrites =
// replays.

// Helper: capture N serial start+end pairs into a graph, return the
// graph and instance. Caller owns cleanup.
struct CapturedGraph {
  cudaGraph_t graph{nullptr};
  cudaGraphExec_t instance{nullptr};
  std::vector<std::unique_ptr<PerCollState>> states;
};

static CapturedGraph
captureGraph(cudaStream_t stream, HRDWRingBuffer<void*>& buf, int numColls) {
  CapturedGraph cg;
  cg.states.reserve(numColls);
  for (int i = 0; i < numColls; ++i) {
    cg.states.push_back(std::make_unique<PerCollState>());
  }

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < numColls; ++i) {
    launchRingBufferStartWrite(
        stream,
        buf.ring(),
        buf.writeIndex(),
        buf.mask(),
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(static_cast<uintptr_t>(i)),
        cg.states[i]->slotStorage);
    launchRingBufferEndWrite(
        stream, buf.ring(), cg.states[i]->slotStorage, buf.mask());
  }
  cudaStreamEndCapture(stream, &cg.graph);
  if (cg.graph) {
    cudaGraphInstantiate(&cg.instance, cg.graph, nullptr, nullptr, 0);
  }
  return cg;
}

static void destroyGraph(CapturedGraph& cg) {
  if (cg.instance) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGraphExecDestroy(cg.instance);
  }
  if (cg.graph) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGraphDestroy(cg.graph);
  }
}

// Define a graph stress test that iterates over all configs. Provides:
//   cfg.ringSize, cfg.numStreams, cfg.numWrites
//   buf     — HRDWRingBuffer<void*> sized to cfg.ringSize
//   stream  — cudaStream_t
//   cg      — CapturedGraph with cfg.numStreams start+end pairs
// The test body goes after the macro invocation as a brace-enclosed block.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GRAPH_STRESS_TEST(name)                               \
  static void graphStressBody_##name(                         \
      const StressConfig&,                                    \
      HRDWRingBuffer<void*>&,                                 \
      cudaStream_t,                                           \
      CapturedGraph&);                                        \
  TEST_F(HRDWRingBufferStressTest, name) {                    \
    for (const auto& cfg : kStressConfigs) {                  \
      SCOPED_TRACE(                                           \
          "ring=" + std::to_string(cfg.ringSize) +            \
          " colls=" + std::to_string(cfg.numStreams) +        \
          " replays=" + std::to_string(cfg.numWrites));       \
      HRDWRingBuffer<void*> buf(cfg.ringSize);                \
      ASSERT_TRUE(buf.valid());                               \
      cudaStream_t stream;                                    \
      ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);      \
      auto cg = captureGraph(stream, buf, cfg.numStreams);    \
      ASSERT_NE(cg.graph, nullptr);                           \
      ASSERT_NE(cg.instance, nullptr);                        \
      graphStressBody_##name(cfg, buf, stream, cg);           \
      destroyGraph(cg);                                       \
      /* NOLINTNEXTLINE(facebook-cuda-safe-api-call-check) */ \
      cudaStreamDestroy(stream);                              \
    }                                                         \
  }                                                           \
  static void graphStressBody_##name(                         \
      const StressConfig& cfg,                                \
      [[maybe_unused]] HRDWRingBuffer<void*>& buf,            \
      cudaStream_t stream,                                    \
      CapturedGraph& cg)

// Replay graph, then poll. Every delivered entry must have valid timestamps.
GRAPH_STRESS_TEST(GraphReplayTimestampOrdering) {
  HRDWRingBufferReader<void*> reader(buf);

  for (int r = 0; r < cfg.numWrites; ++r) {
    ASSERT_EQ(cudaGraphLaunch(cg.instance, stream), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  uint64_t badEntries = 0;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    if (e.start_ns == 0 || e.end_ns == 0 || e.end_ns < e.start_ns) {
      ++badEntries;
    }
  });

  EXPECT_EQ(badEntries, 0u)
      << "Graph replay produced entries with invalid timestamps";

  uint64_t totalWrites = static_cast<uint64_t>(cfg.numStreams) * cfg.numWrites;
  EXPECT_EQ(result.entriesRead + result.entriesLost, totalWrites);
}

// Replay graph while a background CPU thread polls continuously.
GRAPH_STRESS_TEST(GraphReplayConcurrentPoll) {
  std::atomic<uint64_t> totalRead{0};
  std::atomic<uint64_t> totalLost{0};
  std::atomic<uint64_t> totalBad{0};
  std::atomic<bool> replaysDone{false};

  std::thread readerThread([&]() {
    HRDWRingBufferReader<void*> reader(buf);
    while (!replaysDone.load(std::memory_order_acquire)) {
      auto result = reader.poll([&](const auto& e, uint64_t) {
        if (e.start_ns == 0 || e.end_ns == 0 || e.end_ns < e.start_ns) {
          totalBad.fetch_add(1, std::memory_order_relaxed);
        }
      });
      totalRead.fetch_add(result.entriesRead, std::memory_order_relaxed);
      totalLost.fetch_add(result.entriesLost, std::memory_order_relaxed);
    }
    // Final drain.
    auto result = reader.poll([&](const auto& e, uint64_t) {
      if (e.start_ns == 0 || e.end_ns == 0 || e.end_ns < e.start_ns) {
        totalBad.fetch_add(1, std::memory_order_relaxed);
      }
    });
    totalRead.fetch_add(result.entriesRead, std::memory_order_relaxed);
    totalLost.fetch_add(result.entriesLost, std::memory_order_relaxed);
  });

  for (int r = 0; r < cfg.numWrites; ++r) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGraphLaunch(cg.instance, stream);
  }
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamSynchronize(stream);
  replaysDone.store(true, std::memory_order_release);
  readerThread.join();

  EXPECT_EQ(totalBad.load(), 0u)
      << "Graph replay produced corrupted entries during concurrent polling";

  auto accounted = totalRead.load() + totalLost.load();
  EXPECT_EQ(accounted, static_cast<uint64_t>(cfg.totalWrites()));
}

// Replay with scanIncomplete=true to exercise per-collective in-flight
// detection. Verify write-pending entries have valid collId tags.
GRAPH_STRESS_TEST(GraphReplayIncompleteDetection) {
  std::atomic<uint64_t> totalCompleted{0};
  std::atomic<uint64_t> totalLost{0};
  std::atomic<uint64_t> totalPending{0};
  std::atomic<uint64_t> totalBad{0};
  std::atomic<bool> replaysDone{false};

  std::thread readerThread([&]() {
    HRDWRingBufferReader<void*> reader(buf);
    while (!replaysDone.load(std::memory_order_acquire)) {
      auto result = reader.poll(
          [&](const auto& e, uint64_t) {
            auto tag =
                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(e.data));
            if (tag >= static_cast<uint32_t>(cfg.numStreams)) {
              totalBad.fetch_add(1, std::memory_order_relaxed);
            }
            if (e.sequence == HRDW_RINGBUFFER_WRITE_PENDING) {
              totalPending.fetch_add(1, std::memory_order_relaxed);
            } else {
              if (e.start_ns == 0 || e.end_ns == 0 || e.end_ns < e.start_ns) {
                totalBad.fetch_add(1, std::memory_order_relaxed);
              }
              totalCompleted.fetch_add(1, std::memory_order_relaxed);
            }
          },
          /*scanIncomplete=*/true);
      totalLost.fetch_add(result.entriesLost, std::memory_order_relaxed);
    }
    // Final drain.
    auto result = reader.poll(
        [&](const auto& e, uint64_t) {
          if (e.sequence != HRDW_RINGBUFFER_WRITE_PENDING) {
            totalCompleted.fetch_add(1, std::memory_order_relaxed);
          }
        },
        /*scanIncomplete=*/true);
    totalLost.fetch_add(result.entriesLost, std::memory_order_relaxed);
  });

  for (int r = 0; r < cfg.numWrites; ++r) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGraphLaunch(cg.instance, stream);
  }
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamSynchronize(stream);
  replaysDone.store(true, std::memory_order_release);
  readerThread.join();

  EXPECT_EQ(totalBad.load(), 0u)
      << "Write-pending or completed entries had invalid data";

  auto accounted = totalCompleted.load() + totalLost.load();
  EXPECT_EQ(accounted, static_cast<uint64_t>(cfg.totalWrites()));
}
