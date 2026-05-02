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
#include "comms/utils/tests/HRDWRingBufferTestTypes.h"

using meta::comms::colltrace::HRDWRingBuffer;
using meta::comms::colltrace::HRDWRingBufferReader;

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

// Helper: launch totalWrites write pairs (2 writes each) across streams.
static void launchEagerWrites(
    const std::vector<cudaStream_t>& streams,
    HRDWRingBuffer<TestEvent>& buf,
    int totalPairs) {
  auto numStreams = static_cast<int>(streams.size());
  for (int i = 0; i < totalPairs; ++i) {
    auto streamIdx = i % numStreams;
    buf.write(streams[streamIdx], TestEvent{0});
    buf.write(streams[streamIdx], TestEvent{0});
  }
}

// Define an eager stress test that iterates over all configs. Provides:
//   cfg       — StressConfig
//   buf       — HRDWRingBuffer<TestEvent>
//   streams   — vector of cudaStream_t (cfg.numStreams)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define EAGER_STRESS_TEST(name)                                 \
  static void eagerStressBody_##name(                           \
      const StressConfig&,                                      \
      HRDWRingBuffer<TestEvent>&,                               \
      std::vector<cudaStream_t>&);                              \
  TEST_F(HRDWRingBufferStressTest, name) {                      \
    for (const auto& cfg : kStressConfigs) {                    \
      SCOPED_TRACE(                                             \
          "ring=" + std::to_string(cfg.ringSize) +              \
          " streams=" + std::to_string(cfg.numStreams) +        \
          " writes=" + std::to_string(cfg.numWrites));          \
      HRDWRingBuffer<TestEvent> buf(cfg.ringSize);              \
      ASSERT_TRUE(buf.valid());                                 \
      std::vector<cudaStream_t> streams(cfg.numStreams);        \
      for (auto& s : streams) {                                 \
        ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);           \
      }                                                         \
      eagerStressBody_##name(cfg, buf, streams);                \
      for (auto& s : streams) {                                 \
        /* NOLINTNEXTLINE(facebook-cuda-safe-api-call-check) */ \
        cudaStreamDestroy(s);                                   \
      }                                                         \
    }                                                           \
  }                                                             \
  static void eagerStressBody_##name(                           \
      const StressConfig& cfg,                                  \
      HRDWRingBuffer<TestEvent>& buf,                           \
      std::vector<cudaStream_t>& streams)

// Launch write pairs across multiple streams, sync, then poll.
// Every delivered entry must have valid timestamps.
EAGER_STRESS_TEST(MultiStreamTimestampOrdering) {
  HRDWRingBufferReader<TestEvent> reader(buf);
  launchEagerWrites(streams, buf, cfg.totalWrites());

  for (auto& s : streams) {
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
  }

  uint64_t badEntries = 0;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    if (e.timestamp == 0) {
      ++badEntries;
    }
  });

  EXPECT_EQ(badEntries, 0u)
      << "Entries with invalid timestamps (threadfence_system ordering failure)";
  // Each pair produces 2 ring entries.
  EXPECT_EQ(
      result.entriesRead + result.entriesLost,
      static_cast<uint64_t>(2 * cfg.totalWrites()));
}

// Launch writes while a background CPU thread polls continuously.
EAGER_STRESS_TEST(ConcurrentWriteAndPoll) {
  std::atomic<uint64_t> totalRead{0};
  std::atomic<uint64_t> totalLost{0};
  std::atomic<uint64_t> totalBad{0};
  std::atomic<bool> writersDone{false};

  std::thread readerThread([&]() {
    HRDWRingBufferReader<TestEvent> reader(buf);
    while (!writersDone.load(std::memory_order_acquire)) {
      auto result = reader.poll([&](const auto& e, uint64_t) {
        if (e.timestamp == 0) {
          totalBad.fetch_add(1, std::memory_order_relaxed);
        }
      });
      totalRead.fetch_add(result.entriesRead, std::memory_order_relaxed);
      totalLost.fetch_add(result.entriesLost, std::memory_order_relaxed);
    }
    // Final drain.
    auto result = reader.poll([&](const auto& e, uint64_t) {
      if (e.timestamp == 0) {
        totalBad.fetch_add(1, std::memory_order_relaxed);
      }
    });
    totalRead.fetch_add(result.entriesRead, std::memory_order_relaxed);
    totalLost.fetch_add(result.entriesLost, std::memory_order_relaxed);
  });

  launchEagerWrites(streams, buf, cfg.totalWrites());

  for (auto& s : streams) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamSynchronize(s);
  }
  writersDone.store(true, std::memory_order_release);
  readerThread.join();

  EXPECT_EQ(totalBad.load(), 0u)
      << "Corrupted entries during concurrent polling";

  auto accounted = totalRead.load() + totalLost.load();
  EXPECT_EQ(accounted, static_cast<uint64_t>(2 * cfg.totalWrites()));
}

// Single-stream configs only: verify monotonic timestamp ordering.
EAGER_STRESS_TEST(SingleStreamMonotonicOrdering) {
  if (cfg.numStreams != 1) {
    return; // Only meaningful for single-stream configs.
  }

  HRDWRingBufferReader<TestEvent> reader(buf);
  launchEagerWrites(streams, buf, cfg.totalWrites());
  ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);

  uint64_t badEntries = 0;
  uint32_t prevTimestamp = 0;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    if (e.timestamp == 0) {
      ++badEntries;
    }
    if (prevTimestamp > 0 && e.timestamp < prevTimestamp) {
      ++badEntries;
    }
    prevTimestamp = e.timestamp;
  });

  EXPECT_EQ(badEntries, 0u);
  EXPECT_EQ(
      result.entriesRead + result.entriesLost,
      static_cast<uint64_t>(2 * cfg.totalWrites()));
}

// NOTE: graph tests reuse kStressConfigs: numStreams = numColls, numWrites =
// replays.

// Helper: capture N serial write pairs into a graph, return the
// graph and instance. Caller owns cleanup.
struct CapturedGraph {
  cudaGraph_t graph{nullptr};
  cudaGraphExec_t instance{nullptr};
};

static CapturedGraph captureGraph(
    cudaStream_t stream,
    HRDWRingBuffer<TestEvent>& buf,
    int numColls) {
  CapturedGraph cg;

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < numColls; ++i) {
    TestEvent event{static_cast<uint32_t>(i)};
    buf.write(stream, event);
    buf.write(stream, event);
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
//   buf     — HRDWRingBuffer<TestEvent> sized to cfg.ringSize
//   stream  — cudaStream_t
//   cg      — CapturedGraph with cfg.numStreams write pairs
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GRAPH_STRESS_TEST(name)                               \
  static void graphStressBody_##name(                         \
      const StressConfig&,                                    \
      HRDWRingBuffer<TestEvent>&,                             \
      cudaStream_t,                                           \
      CapturedGraph&);                                        \
  TEST_F(HRDWRingBufferStressTest, name) {                    \
    for (const auto& cfg : kStressConfigs) {                  \
      SCOPED_TRACE(                                           \
          "ring=" + std::to_string(cfg.ringSize) +            \
          " colls=" + std::to_string(cfg.numStreams) +        \
          " replays=" + std::to_string(cfg.numWrites));       \
      HRDWRingBuffer<TestEvent> buf(cfg.ringSize);            \
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
      [[maybe_unused]] HRDWRingBuffer<TestEvent>& buf,        \
      cudaStream_t stream,                                    \
      CapturedGraph& cg)

// Replay graph, then poll. Every delivered entry must have valid timestamps.
GRAPH_STRESS_TEST(GraphReplayTimestampOrdering) {
  HRDWRingBufferReader<TestEvent> reader(buf);

  for (int r = 0; r < cfg.numWrites; ++r) {
    ASSERT_EQ(cudaGraphLaunch(cg.instance, stream), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  uint64_t badEntries = 0;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    if (e.timestamp == 0) {
      ++badEntries;
    }
  });

  EXPECT_EQ(badEntries, 0u)
      << "Graph replay produced entries with invalid timestamps";

  uint64_t totalSlots =
      static_cast<uint64_t>(cfg.numStreams) * cfg.numWrites * 2;
  EXPECT_EQ(result.entriesRead + result.entriesLost, totalSlots);
}

// Replay graph while a background CPU thread polls continuously.
GRAPH_STRESS_TEST(GraphReplayConcurrentPoll) {
  std::atomic<uint64_t> totalRead{0};
  std::atomic<uint64_t> totalLost{0};
  std::atomic<uint64_t> totalBad{0};
  std::atomic<bool> replaysDone{false};

  std::thread readerThread([&]() {
    HRDWRingBufferReader<TestEvent> reader(buf);
    while (!replaysDone.load(std::memory_order_acquire)) {
      auto result = reader.poll([&](const auto& e, uint64_t) {
        if (e.timestamp == 0) {
          totalBad.fetch_add(1, std::memory_order_relaxed);
        }
      });
      totalRead.fetch_add(result.entriesRead, std::memory_order_relaxed);
      totalLost.fetch_add(result.entriesLost, std::memory_order_relaxed);
    }
    // Final drain.
    auto result = reader.poll([&](const auto& e, uint64_t) {
      if (e.timestamp == 0) {
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
  EXPECT_EQ(accounted, static_cast<uint64_t>(cfg.totalWrites()) * 2);
}

// Replay and verify entries have valid data tags and timestamps.
GRAPH_STRESS_TEST(GraphReplayEventValidation) {
  std::atomic<uint64_t> totalCompleted{0};
  std::atomic<uint64_t> totalLost{0};
  std::atomic<uint64_t> totalBad{0};
  std::atomic<bool> replaysDone{false};

  std::thread readerThread([&]() {
    HRDWRingBufferReader<TestEvent> reader(buf);
    while (!replaysDone.load(std::memory_order_acquire)) {
      auto result = reader.poll([&](const auto& e, uint64_t) {
        auto tag = e.data.tag;
        if (tag >= static_cast<uint32_t>(cfg.numStreams)) {
          totalBad.fetch_add(1, std::memory_order_relaxed);
        }
        if (e.timestamp == 0) {
          totalBad.fetch_add(1, std::memory_order_relaxed);
        }
        totalCompleted.fetch_add(1, std::memory_order_relaxed);
      });
      totalLost.fetch_add(result.entriesLost, std::memory_order_relaxed);
    }
    // Final drain.
    auto result = reader.poll([&](const auto& e, uint64_t) {
      (void)e;
      totalCompleted.fetch_add(1, std::memory_order_relaxed);
    });
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
      << "Entries had invalid data tags or timestamps";

  auto accounted = totalCompleted.load() + totalLost.load();
  EXPECT_EQ(accounted, static_cast<uint64_t>(cfg.totalWrites()) * 2);
}

// Each write claims its own slot — no contention. Fill the ring, then
// write again to verify wrap-around succeeds immediately.
TEST_F(HRDWRingBufferStressTest, WrapAroundImmediateNoContention) {
  constexpr uint32_t kSize = 8;
  HRDWRingBuffer<TestEvent> buf(kSize);
  ASSERT_TRUE(buf.valid());

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  // Fill the ring with kSize writes, then one more to wrap around.
  for (uint32_t i = 0; i <= kSize; ++i) {
    buf.write(stream, TestEvent{0});
  }
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  // Poll and verify all entries are readable (including the wrap-around).
  HRDWRingBufferReader<TestEvent> reader(buf);
  uint64_t readCount = 0;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    EXPECT_NE(e.timestamp, 0u);
    ++readCount;
  });
  // kSize + 1 writes total. Some may be lost due to lapping on tiny ring.
  EXPECT_EQ(result.entriesRead + result.entriesLost, kSize + 1);

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamDestroy(stream);
}

// Multi-writer stress test: 4 streams, all writing concurrently.
// No contention — all streams complete immediately.
TEST_F(HRDWRingBufferStressTest, MultiWriterNeverBlocks) {
  constexpr uint32_t kSize = 16;
  constexpr int kNumStreams = 4;
  constexpr int kWritesPerStream = 32;
  HRDWRingBuffer<TestEvent> buf(kSize);
  ASSERT_TRUE(buf.valid());

  cudaStream_t streams[kNumStreams];
  for (auto& s : streams) {
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
  }

  for (int w = 0; w < kWritesPerStream; ++w) {
    for (int si = 0; si < kNumStreams; ++si) {
      buf.write(streams[si], TestEvent{0});
    }
  }

  for (auto& s : streams) {
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
  }

  HRDWRingBufferReader<TestEvent> reader(buf);
  uint64_t badEntries = 0;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    if (e.timestamp == 0) {
      ++badEntries;
    }
  });
  EXPECT_EQ(badEntries, 0u);
  uint64_t totalSlots = static_cast<uint64_t>(kNumStreams) * kWritesPerStream;
  EXPECT_EQ(result.entriesRead + result.entriesLost, totalSlots);

  for (auto& s : streams) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamDestroy(s);
  }
}
