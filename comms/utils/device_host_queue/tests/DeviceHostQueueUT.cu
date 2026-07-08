// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/utils/device_host_queue/DeviceHostQueue.cuh"

using meta::comms::ConsumerPolicy;
using meta::comms::DeviceHostQueue;
using meta::comms::Direction;
using meta::comms::QueueOpStatus;

namespace {

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = (cmd);                                                \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// Torn-publish detector payload. `guard[]` is filled with `value` on every
// publish and spans the rest of a 64B cache line, so a consistently-published
// command has guard[j] == value for all j. If the consumer ever observes the
// readySeq marker without all payload writes being visible (a publish-ordering
// bug — possible on weak memory, e.g. aarch64/Grace, if the release/acquire
// pairing were wrong), some guard words would still hold a prior occupant's
// value and the check below would fire.
struct Cmd {
  uint32_t producer;
  uint32_t value;
  uint32_t guard[14]; // == value on a consistent publish; sizeof(Cmd) == 64
};
static_assert(sizeof(Cmd) == 64);

using IntQueue = DeviceHostQueue<uint64_t>;
using CmdQueue = DeviceHostQueue<Cmd>;
using MultiCmdQueue =
    DeviceHostQueue<Cmd, Direction::D2H, ConsumerPolicy::Multi>;

// Each thread is a producer that publishes `perThread` Cmd commands, tagged
// with its global thread id and an increasing per-thread value (guard words
// mirror value). Templated on the producer handle so it serves both the Single
// and Multi consumer-policy queues — the producer protocol is identical.
template <class ProducerHandle>
__global__ void producerKernel(ProducerHandle q, uint32_t perThread) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = 0; i < perThread; ++i) {
    Cmd c;
    c.producer = tid;
    c.value = i;
    for (int j = 0; j < 14; ++j) {
      c.guard[j] = i; // all guard words mirror value
    }
    q.blockingWrite(c);
  }
}

// Single-thread producer: blocking-writes values base..base+n-1 in order.
__global__ void
intBlockingWriteKernel(IntQueue::Producer q, uint32_t n, uint64_t base) {
  for (uint32_t i = 0; i < n; ++i) {
    q.blockingWrite(base + i);
  }
}

// Single-thread producer using the non-blocking write(); records each call's
// status into st[] so the host can assert Ok/Full behaviour.
__global__ void intWriteStatusKernel(
    IntQueue::Producer q,
    uint32_t n,
    uint64_t base,
    QueueOpStatus* st) {
  for (uint32_t i = 0; i < n; ++i) {
    st[i] = q.write(base + i);
  }
}

// N producer threads each attempt exactly one non-blocking write(); st[tid]
// records Ok/Full so the host can verify the credit guard under real
// contention.
__global__ void concurrentWriteKernel(IntQueue::Producer q, QueueOpStatus* st) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  st[tid] = q.write(tid);
}

} // namespace

TEST(DeviceHostQueueTest, CapacityMustBePowerOfTwo) {
  EXPECT_THROW(IntQueue(0), std::runtime_error);
  EXPECT_THROW(IntQueue(3), std::runtime_error);
  EXPECT_NO_THROW(IntQueue(1));
  EXPECT_NO_THROW(IntQueue(8));
}

TEST(DeviceHostQueueTest, EmptyReadReturnsEmpty) {
  IntQueue q(8);
  IntQueue::ReadResult r;
  EXPECT_EQ(q.read(r), QueueOpStatus::Empty);
}

TEST(DeviceHostQueueTest, DeviceWriteHostReadRoundTrip) {
  IntQueue q(8);
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  // One device producer writes a single value 42.
  intBlockingWriteKernel<<<1, 1, 0, stream>>>(q.producerHandle(), 1, 42);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaStreamSynchronize(stream));

  IntQueue::ReadResult r;
  ASSERT_EQ(q.read(r), QueueOpStatus::Ok);
  EXPECT_EQ(r.value, 42u);
  EXPECT_EQ(r.seq, 0u);
  EXPECT_EQ(q.read(r), QueueOpStatus::Empty);
  CUDACHECK(cudaStreamDestroy(stream));
}

TEST(DeviceHostQueueTest, FullReturnsFullUntilCreditAdvances) {
  IntQueue q(4);
  QueueOpStatus* st = nullptr;
  CUDACHECK(cudaMallocManaged(&st, sizeof(QueueOpStatus) * 5));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Five non-blocking writes into a capacity-4 ring with nothing consumed yet:
  // the first four claim credit (Ok), the fifth finds the ring full (Full).
  intWriteStatusKernel<<<1, 1, 0, stream>>>(q.producerHandle(), 5, 0, st);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(st[i], QueueOpStatus::Ok) << "i=" << i;
  }
  EXPECT_EQ(st[4], QueueOpStatus::Full);

  // Consume one (frees its slot); one more write now fits.
  IntQueue::ReadResult r;
  ASSERT_EQ(q.read(r), QueueOpStatus::Ok);
  intWriteStatusKernel<<<1, 1, 0, stream>>>(q.producerHandle(), 1, 99, st);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaStreamSynchronize(stream));
  EXPECT_EQ(st[0], QueueOpStatus::Ok);

  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(st));
}

TEST(DeviceHostQueueTest, ConcurrentWriteRespectsCapacity) {
  const uint32_t cap = 8;
  const uint32_t nThreads = 256; // >> cap: heavy contention on the credit guard
  IntQueue q(cap);
  QueueOpStatus* st = nullptr;
  CUDACHECK(cudaMallocManaged(&st, sizeof(QueueOpStatus) * nThreads));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // nThreads producers race on write() into a cap-8 ring with nothing consumed.
  concurrentWriteKernel<<<nThreads / 32, 32, 0, stream>>>(
      q.producerHandle(), st);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaStreamSynchronize(stream));

  uint32_t ok = 0, full = 0;
  for (uint32_t i = 0; i < nThreads; ++i) {
    if (st[i] == QueueOpStatus::Ok) {
      ++ok;
    } else if (st[i] == QueueOpStatus::Full) {
      ++full;
    }
  }
  EXPECT_EQ(ok, cap); // exactly `cap` slots claimed: the guard admits no more
  EXPECT_EQ(ok + full, nThreads); // every attempt resolved to Ok or Full

  // Every Full claimed nothing (no hole), so exactly the `cap` Ok writes are
  // published and drain back.
  IntQueue::ReadResult r;
  uint32_t drained = 0;
  while (q.read(r) == QueueOpStatus::Ok) {
    ++drained;
  }
  EXPECT_EQ(drained, cap);

  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(st));
}

TEST(DeviceHostQueueTest, WraparoundPreservesFifo) {
  IntQueue q(4);
  const uint64_t kIters = 1000; // many capacity cycles through a small ring
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // One device producer streams values 1..kIters; the host drains concurrently
  // so the capacity-4 ring wraps ~250 times. blockingWrite spins for credit,
  // so the producer cannot outrun the consumer.
  intBlockingWriteKernel<<<1, 1, 0, stream>>>(
      q.producerHandle(), static_cast<uint32_t>(kIters), /*base=*/1);
  CUDACHECK(cudaGetLastError());

  IntQueue::ReadResult r;
  for (uint64_t i = 0; i < kIters; ++i) {
    while (q.read(r) == QueueOpStatus::Empty) {
      // spin until the producer publishes the next item
    }
    EXPECT_EQ(r.value, i + 1); // values are 1..kIters in order
    EXPECT_EQ(r.seq, i);
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaStreamDestroy(stream));
}

TEST(DeviceHostQueueTest, ReadMultiDrainsInOrder) {
  IntQueue q(4);
  const uint64_t kIters = 1000; // wraps the capacity-4 ring many times
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // One device producer streams 1..kIters; the host drains in batches via
  // readMulti (max 4 per call). batch == capacity exercises full and partial
  // batches and the single-ci-store reuse path under wraparound.
  intBlockingWriteKernel<<<1, 1, 0, stream>>>(
      q.producerHandle(), static_cast<uint32_t>(kIters), /*base=*/1);
  CUDACHECK(cudaGetLastError());

  IntQueue::ReadResult batch[4];
  uint64_t got = 0;
  while (got < kIters) {
    size_t n = q.readMulti(batch, 4);
    for (size_t i = 0; i < n; ++i) {
      EXPECT_EQ(batch[i].seq, got); // strict sequence order, no loss/dup
      EXPECT_EQ(batch[i].value, got + 1); // values are 1..kIters
      ++got;
    }
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaStreamDestroy(stream));
}

TEST(DeviceHostQueueTest, SizeApprox) {
  IntQueue q(8);
  EXPECT_EQ(q.sizeApprox(), 0u);
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  // Three published, none consumed -> backlog of 3.
  intBlockingWriteKernel<<<1, 1, 0, stream>>>(q.producerHandle(), 3, 1);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaStreamSynchronize(stream));
  EXPECT_EQ(q.sizeApprox(), 3u);
  CUDACHECK(cudaStreamDestroy(stream));
}

// Many GPU producers -> single host consumer. Verifies: no loss/duplication
// (each producer's commands all arrive), global FIFO by sequence, each
// producer's values arrive in its publish order, and — via the guard words —
// that no publish is ever observed torn (payload inconsistent with the
// readySeq marker). The last check is what actually exercises the publish
// release / consume acquire ordering on weak memory (aarch64/Grace); a large
// command count widens the window for any ordering bug to surface.
TEST(DeviceHostQueueTest, DeviceProducersHostConsumer) {
  const uint32_t blocks = 8;
  const uint32_t threads = 32;
  const uint32_t perThread = 2000;
  const uint32_t producers = blocks * threads;
  const uint64_t total = uint64_t(producers) * perThread;

  CmdQueue q(1024);
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  producerKernel<<<blocks, threads, 0, stream>>>(q.producerHandle(), perThread);
  CUDACHECK(cudaGetLastError());

  std::vector<uint32_t> counts(producers, 0);
  std::vector<uint32_t> nextValue(producers, 0);
  uint64_t consumed = 0;
  uint64_t expectedSeq = 0;
  uint64_t tornPublishes = 0;
  CmdQueue::ReadResult r;
  while (consumed < total) {
    if (q.read(r) == QueueOpStatus::Ok) {
      ASSERT_LT(r.value.producer, producers);
      EXPECT_EQ(r.seq, expectedSeq); // host reads in strict sequence order
      EXPECT_EQ(
          r.value.value, nextValue[r.value.producer]); // per-producer FIFO
      // Torn-publish check: every guard word must equal value.
      for (int j = 0; j < 14; ++j) {
        if (r.value.guard[j] != r.value.value) {
          ++tornPublishes;
          break; // count once per command
        }
      }
      ++nextValue[r.value.producer];
      ++counts[r.value.producer];
      ++consumed;
      ++expectedSeq;
    }
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  EXPECT_EQ(tornPublishes, 0u)
      << tornPublishes << " of " << total
      << " commands were observed torn (payload visible before/without the "
         "readySeq marker it was published under)";
  for (uint32_t p = 0; p < producers; ++p) {
    EXPECT_EQ(counts[p], perThread) << "producer " << p;
  }
  CUDACHECK(cudaStreamDestroy(stream));
}

// =============================================================================
// ConsumerPolicy::Multi (MPMC) tests
// =============================================================================

TEST(DeviceHostQueueTest, MultiConsumerEmptyReadReturnsEmpty) {
  MultiCmdQueue q(8);
  MultiCmdQueue::ReadResult r;
  EXPECT_EQ(q.read(r), QueueOpStatus::Empty);
}

// Many GPU producers -> several concurrent CPU consumers (MPMC). After the
// queue is fully drained, verifies: every sequence is consumed exactly once (no
// loss, no duplication), every producer's perThread values each arrive exactly
// once, and no command is observed torn. Processing order across consumers is
// NOT globally ordered by design; only exactly-once delivery is guaranteed.
TEST(DeviceHostQueueTest, MultiConsumerNoLossNoDup) {
  const uint32_t blocks = 8;
  const uint32_t threads = 32;
  const uint32_t perThread = 1000;
  const uint32_t producers = blocks * threads;
  const uint64_t total = uint64_t(producers) * perThread;

  MultiCmdQueue q(1024);
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  producerKernel<<<blocks, threads, 0, stream>>>(q.producerHandle(), perThread);
  CUDACHECK(cudaGetLastError());

  struct Item {
    uint64_t seq;
    uint32_t producer;
    uint32_t value;
    bool torn;
  };
  const int nConsumers = 8;
  std::vector<std::vector<Item>> got(nConsumers);
  std::atomic<uint64_t> consumed{0};

  // Each consumer drains until all `total` commands have been taken. read() is
  // safe to call concurrently under ConsumerPolicy::Multi; every Ok hands the
  // caller a command no other consumer will see.
  auto worker = [&](int id) {
    MultiCmdQueue::ReadResult r;
    std::vector<Item>& mine = got[id];
    while (consumed.load(std::memory_order_relaxed) < total) {
      if (q.read(r) == QueueOpStatus::Ok) {
        bool torn = false;
        for (int j = 0; j < 14; ++j) {
          if (r.value.guard[j] != r.value.value) {
            torn = true;
            break;
          }
        }
        mine.push_back({r.seq, r.value.producer, r.value.value, torn});
        consumed.fetch_add(1, std::memory_order_relaxed);
      } else {
        std::this_thread::yield();
      }
    }
  };

  std::vector<std::thread> consumers;
  consumers.reserve(nConsumers);
  for (int i = 0; i < nConsumers; ++i) {
    consumers.emplace_back(worker, i);
  }
  for (auto& t : consumers) {
    t.join();
  }
  CUDACHECK(cudaStreamSynchronize(stream));

  // Aggregate single-threaded and verify exactly-once delivery.
  std::vector<uint8_t> seqSeen(total, 0);
  std::vector<uint8_t> pairSeen(
      total, 0); // index = producer * perThread + value
  uint64_t torn = 0;
  uint64_t totalConsumed = 0;
  for (int i = 0; i < nConsumers; ++i) {
    for (const Item& it : got[i]) {
      ++totalConsumed;
      if (it.torn) {
        ++torn;
      }
      ASSERT_LT(it.seq, total);
      ++seqSeen[it.seq];
      ASSERT_LT(it.producer, producers);
      ASSERT_LT(it.value, perThread);
      ++pairSeen[it.producer * perThread + it.value];
    }
  }
  EXPECT_EQ(totalConsumed, total);
  EXPECT_EQ(torn, 0u) << torn << " commands observed torn";

  uint64_t seqDup = 0, seqMissing = 0, pairBad = 0;
  for (uint64_t s = 0; s < total; ++s) {
    if (seqSeen[s] > 1) {
      ++seqDup;
    } else if (seqSeen[s] == 0) {
      ++seqMissing;
    }
    if (pairSeen[s] != 1) {
      ++pairBad;
    }
  }
  EXPECT_EQ(seqDup, 0u) << seqDup << " sequences consumed more than once";
  EXPECT_EQ(seqMissing, 0u) << seqMissing << " sequences never consumed";
  EXPECT_EQ(pairBad, 0u)
      << pairBad << " (producer,value) pairs not delivered exactly once";

  CUDACHECK(cudaStreamDestroy(stream));
}

// Same exactly-once contract as MultiConsumerNoLossNoDup, but consumers drain
// via the batched readMulti (one head CAS per run of up to kBatch commands)
// instead of per-item read(). Exercises the multi-consumer batched-claim path.
TEST(DeviceHostQueueTest, MultiConsumerReadMultiNoLossNoDup) {
  const uint32_t blocks = 8;
  const uint32_t threads = 32;
  const uint32_t perThread = 1000;
  const uint32_t producers = blocks * threads;
  const uint64_t total = uint64_t(producers) * perThread;

  MultiCmdQueue q(1024);
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  producerKernel<<<blocks, threads, 0, stream>>>(q.producerHandle(), perThread);
  CUDACHECK(cudaGetLastError());

  struct Item {
    uint64_t seq;
    uint32_t producer;
    uint32_t value;
    bool torn;
  };
  const int nConsumers = 8;
  const size_t kBatch = 16;
  std::vector<std::vector<Item>> got(nConsumers);
  std::atomic<uint64_t> consumed{0};

  // readMulti is safe to call concurrently under ConsumerPolicy::Multi; each
  // call claims a contiguous run via one head CAS and returns it to this caller
  // only.
  auto worker = [&](int id) {
    MultiCmdQueue::ReadResult batch[kBatch];
    std::vector<Item>& mine = got[id];
    while (consumed.load(std::memory_order_relaxed) < total) {
      size_t n = q.readMulti(batch, kBatch);
      if (n == 0) {
        std::this_thread::yield();
        continue;
      }
      for (size_t k = 0; k < n; ++k) {
        const MultiCmdQueue::ReadResult& r = batch[k];
        bool torn = false;
        for (int j = 0; j < 14; ++j) {
          if (r.value.guard[j] != r.value.value) {
            torn = true;
            break;
          }
        }
        mine.push_back({r.seq, r.value.producer, r.value.value, torn});
      }
      consumed.fetch_add(n, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> consumers;
  consumers.reserve(nConsumers);
  for (int i = 0; i < nConsumers; ++i) {
    consumers.emplace_back(worker, i);
  }
  for (auto& t : consumers) {
    t.join();
  }
  CUDACHECK(cudaStreamSynchronize(stream));

  // Aggregate single-threaded and verify exactly-once delivery.
  std::vector<uint8_t> seqSeen(total, 0);
  std::vector<uint8_t> pairSeen(total, 0);
  uint64_t torn = 0;
  uint64_t totalConsumed = 0;
  for (int i = 0; i < nConsumers; ++i) {
    for (const Item& it : got[i]) {
      ++totalConsumed;
      if (it.torn) {
        ++torn;
      }
      ASSERT_LT(it.seq, total);
      ++seqSeen[it.seq];
      ASSERT_LT(it.producer, producers);
      ASSERT_LT(it.value, perThread);
      ++pairSeen[it.producer * perThread + it.value];
    }
  }
  EXPECT_EQ(totalConsumed, total);
  EXPECT_EQ(torn, 0u) << torn << " commands observed torn";

  uint64_t seqDup = 0, seqMissing = 0, pairBad = 0;
  for (uint64_t s = 0; s < total; ++s) {
    if (seqSeen[s] > 1) {
      ++seqDup;
    } else if (seqSeen[s] == 0) {
      ++seqMissing;
    }
    if (pairSeen[s] != 1) {
      ++pairBad;
    }
  }
  EXPECT_EQ(seqDup, 0u) << seqDup << " sequences consumed more than once";
  EXPECT_EQ(seqMissing, 0u) << seqMissing << " sequences never consumed";
  EXPECT_EQ(pairBad, 0u)
      << pairBad << " (producer,value) pairs not delivered exactly once";

  CUDACHECK(cudaStreamDestroy(stream));
}
