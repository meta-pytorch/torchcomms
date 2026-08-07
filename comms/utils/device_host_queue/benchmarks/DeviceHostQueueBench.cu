// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Benchmark for DeviceHostQueue: GPU producers -> CPU consumer(s).
// Reports steady-state throughput (Mops/s and effective GB/s) across payload
// sizes and consumer policies, plus round-trip request->publish->read latency.
//
// Standalone binary (not a CI test): run it on a GPU host with
//   buck run @fbcode//mode/opt -c hpc_comms.use_ncclx=stable \
//     fbcode//comms/utils/device_host_queue/benchmarks:device_host_proxy_cmd_queue_bench

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "comms/common/AtomicUtils.cuh"
#include "comms/utils/device_host_queue/DeviceHostQueue.cuh"

using meta::comms::ConsumerPolicy;
using meta::comms::DeviceHostQueue;
using meta::comms::Direction;
using meta::comms::QueueOpStatus;

#define CUDA_CHECK(cmd)                \
  do {                                 \
    cudaError_t e = (cmd);             \
    if (e != cudaSuccess) {            \
      fprintf(                         \
          stderr,                      \
          "CUDA error at %s:%d: %s\n", \
          __FILE__,                    \
          __LINE__,                    \
          cudaGetErrorString(e));      \
      std::abort();                    \
    }                                  \
  } while (0)

namespace {

using Clock = std::chrono::steady_clock;
double secondsSince(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Fixed-size payload of N 32-bit words (sizeof == 4*N bytes).
template <int N>
struct Payload {
  uint32_t w[N];
};

// Each GPU thread blocking-writes `perThread` commands.
template <class T, class ProducerHandle>
__global__ void benchProducerKernel(ProducerHandle q, uint32_t perThread) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  T c;
  for (uint32_t i = 0; i < perThread; ++i) {
    c.w[0] = tid + i;
    q.blockingWrite(c);
  }
}

// Latency producer: for each iteration i, wait until the host requests it
// (go > i), then publish one command. One thread.
template <class T, class ProducerHandle>
__global__ void
latProducerKernel(ProducerHandle q, uint64_t* go, uint32_t iters) {
  T c;
  for (uint32_t i = 0; i < iters; ++i) {
    while (::comms::device::ld_acquire_sys_global(go) <= i) {
      // spin until the host requests iteration i
    }
    c.w[0] = i;
    q.blockingWrite(c);
  }
}

// Tracer tag for the contended-latency test: the host distinguishes the one
// measured command from filler by w[0] (filler writes 0).
constexpr uint32_t kTracerTag = 0xFFFFFFFFu;

// Contended-latency producer kernel. The tracer (block 0, thread 0) and the
// fillers (block 1, threads < nFiller) run in SEPARATE blocks -- hence separate
// warps/wavefronts. This matters on AMD/CDNA: a gated thread (the tracer's
// `while (go <= i)` spin) sharing a wavefront with a forever-spinning filler is
// starved by it (no per-thread forward-progress guarantee within a wavefront),
// which hangs the host's drain-until-tracer loop. Separate blocks are
// independently scheduled, so both make progress (NVIDIA Volta+ is fine either
// way via independent thread scheduling). Launched as <<<2, max(1,nFiller)>>>;
// surplus threads in block 0 just return. Fillers use blockingWrite (fetch_add,
// bounded progress); the host drains throughout (incl. teardown) so the
// no-escape blockingWrite always completes.
template <class T, class ProducerHandle>
__global__ void latContendedKernel(
    ProducerHandle q,
    const uint64_t* go,
    const uint64_t* stop,
    uint32_t iters,
    uint32_t nFiller) {
  if (blockIdx.x == 0) {
    if (threadIdx.x != 0) {
      return; // only thread 0 of block 0 is the tracer
    }
    T c;
    c.w[0] = kTracerTag;
    for (uint32_t i = 0; i < iters; ++i) {
      while (::comms::device::ld_acquire_sys_global(go) <= i) {
        // await request i
      }
      q.blockingWrite(c);
    }
  } else {
    if (threadIdx.x >= nFiller) {
      return;
    }
    T c;
    c.w[0] = 0;
    while (::comms::device::ld_relaxed_sys_global(stop) == 0) {
      q.blockingWrite(c);
    }
  }
}

// Drive `total` commands from `blocks*threads` GPU producers through `q` and
// drain them on the host; returns Mops/s and effective GB/s end to end.
template <class T, ConsumerPolicy Policy>
void runThroughput(
    int blocks,
    int threads,
    uint64_t total,
    int nConsumers,
    uint32_t cap,
    double& mops,
    double& gbps) {
  using Q = DeviceHostQueue<T, Direction::D2H, Policy>;
  Q q(cap);
  const uint32_t perThread =
      static_cast<uint32_t>(total / (uint64_t(blocks) * threads));
  const uint64_t real = uint64_t(blocks) * threads * perThread;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto t0 = Clock::now();
  benchProducerKernel<T>
      <<<blocks, threads, 0, stream>>>(q.producerHandle(), perThread);
  CUDA_CHECK(cudaGetLastError());

  if constexpr (Policy == ConsumerPolicy::Single) {
    typename Q::ReadResult r;
    uint64_t got = 0;
    while (got < real) {
      if (q.read(r) == QueueOpStatus::Ok) {
        ++got;
      }
    }
  } else {
    std::atomic<uint64_t> consumed{0};
    auto worker = [&]() {
      typename Q::ReadResult r;
      while (consumed.load(std::memory_order_relaxed) < real) {
        if (q.read(r) == QueueOpStatus::Ok) {
          consumed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    };
    std::vector<std::thread> ts;
    ts.reserve(nConsumers);
    for (int i = 0; i < nConsumers; ++i) {
      ts.emplace_back(worker);
    }
    for (auto& t : ts) {
      t.join();
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  double secs = secondsSince(t0);
  mops = double(real) / secs / 1e6;
  gbps = double(real) * sizeof(T) / secs / 1e9;
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// MPSC drain via readMulti: same producer load, but the host dequeues in
// batches of up to `batchMax` -- one ci release store per batch instead of per
// command. Single consumer.
template <class T>
void runThroughputReadMulti(
    int blocks,
    int threads,
    uint64_t total,
    uint32_t cap,
    size_t batchMax,
    double& mops,
    double& gbps) {
  using Q = DeviceHostQueue<T, Direction::D2H, ConsumerPolicy::Single>;
  Q q(cap);
  const uint32_t perThread =
      static_cast<uint32_t>(total / (uint64_t(blocks) * threads));
  const uint64_t real = uint64_t(blocks) * threads * perThread;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto t0 = Clock::now();
  benchProducerKernel<T>
      <<<blocks, threads, 0, stream>>>(q.producerHandle(), perThread);
  CUDA_CHECK(cudaGetLastError());

  std::vector<typename Q::ReadResult> batch(batchMax);
  uint64_t got = 0;
  while (got < real) {
    got += q.readMulti(batch.data(), batchMax);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  double secs = secondsSince(t0);
  mops = double(real) / secs / 1e6;
  gbps = double(real) * sizeof(T) / secs / 1e9;
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// Flags for the host->GPU signaling words (go/stop). On AMD the coherent flag
// is MANDATORY: a non-coherent mapped flag is not reliably visible to the
// device mid-kernel under load, so the gated producer can miss the host's
// update and the host then spins forever (observed as a hang on MI300 at >=32
// producers).
#if defined(__HIP_PLATFORM_AMD__)
constexpr unsigned kFlagAllocFlags =
    hipHostMallocMapped | hipHostMallocCoherent;
#else
constexpr unsigned kFlagAllocFlags = cudaHostAllocMapped;
#endif

template <class T, ConsumerPolicy Policy = ConsumerPolicy::Single>
void runLatency(
    uint32_t iters,
    uint32_t cap,
    double& avgNs,
    double& p50Ns,
    double& p99Ns,
    double& minNs) {
  // One consumer thread drives the chosen read path; Multi exercises the MPMC
  // claim (head CAS + done marker + reuse-credit advance) on the hot path.
  using Q = DeviceHostQueue<T, Direction::D2H, Policy>;
  Q q(cap);

  void* goHostV = nullptr;
  CUDA_CHECK(cudaHostAlloc(&goHostV, sizeof(uint64_t), kFlagAllocFlags));
  auto* goHost = static_cast<uint64_t*>(goHostV);
  *goHost = 0;
  void* goDevV = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&goDevV, goHostV, 0));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  latProducerKernel<T><<<1, 1, 0, stream>>>(
      q.producerHandle(), static_cast<uint64_t*>(goDevV), iters);
  CUDA_CHECK(cudaGetLastError());

  std::vector<double> samples;
  samples.reserve(iters);
  typename Q::ReadResult r;
  std::atomic_ref<uint64_t> go(*goHost);
  for (uint32_t i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    go.store(i + 1, std::memory_order_release); // request iteration i
    q.blockingRead(r);
    samples.push_back(
        std::chrono::duration<double, std::nano>(Clock::now() - t0).count());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFreeHost(goHostV));

  std::sort(samples.begin(), samples.end());
  double sum = 0;
  for (double s : samples) {
    sum += s;
  }
  avgNs = sum / samples.size();
  p50Ns = samples[samples.size() / 2];
  p99Ns = samples[(samples.size() * 99) / 100];
  minNs = samples.front();
}

// Round-trip latency of a single tagged "tracer" command while `nProducers` GPU
// threads contend (thread 0 = tracer, the rest hammer write()). One host
// consumer (MPSC); timing is host-only (request -> dequeue of the tracer). At
// high nProducers the queue stays backlogged, so this is the tracer round-trip
// INCLUDING queue residency under load -- the operational tail, not the
// empty-queue publish path.
template <class T>
void runLatencyContended(
    int nProducers,
    uint32_t iters,
    uint32_t cap,
    double& avgNs,
    double& p50Ns,
    double& p99Ns,
    double& minNs) {
  using Q = DeviceHostQueue<T, Direction::D2H, ConsumerPolicy::Single>;
  Q q(cap);

  void* goHostV = nullptr;
  CUDA_CHECK(cudaHostAlloc(&goHostV, sizeof(uint64_t), kFlagAllocFlags));
  auto* goHost = static_cast<uint64_t*>(goHostV);
  *goHost = 0;
  void* goDevV = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&goDevV, goHostV, 0));

  void* stopHostV = nullptr;
  CUDA_CHECK(cudaHostAlloc(&stopHostV, sizeof(uint64_t), kFlagAllocFlags));
  auto* stopHost = static_cast<uint64_t*>(stopHostV);
  *stopHost = 0;
  void* stopDevV = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&stopDevV, stopHostV, 0));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  // Block 0 = tracer, block 1 = fillers (separate wavefronts; see kernel doc).
  const uint32_t nFiller = static_cast<uint32_t>(nProducers - 1);
  const int fillerThreads = nFiller > 0 ? static_cast<int>(nFiller) : 1;
  latContendedKernel<T><<<2, fillerThreads, 0, stream>>>(
      q.producerHandle(),
      static_cast<uint64_t*>(goDevV),
      static_cast<uint64_t*>(stopDevV),
      iters,
      nFiller);
  CUDA_CHECK(cudaGetLastError());

  std::vector<double> samples;
  samples.reserve(iters);
  typename Q::ReadResult r;
  std::atomic_ref<uint64_t> go(*goHost);
  for (uint32_t i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    go.store(i + 1, std::memory_order_release); // request tracer i
    for (;;) { // drain filler in FIFO until the tracer surfaces
      if (q.read(r) == QueueOpStatus::Ok && r.value.w[0] == kTracerTag) {
        break;
      }
    }
    samples.push_back(
        std::chrono::duration<double, std::nano>(Clock::now() - t0).count());
  }

  // Teardown: filler uses blockingWrite (no escape), so a filler blocked for
  // credit only observes stop after it gets credit and completes its write --
  // keep draining until the kernel finishes, else cudaStreamSynchronize hangs.
  std::atomic_ref<uint64_t>(*stopHost).store(1, std::memory_order_release);
  while (cudaStreamQuery(stream) == cudaErrorNotReady) {
    q.read(r); // free credit so a blocked filler can finish and observe stop
  }
  while (q.read(r) == QueueOpStatus::Ok) {
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFreeHost(goHostV));
  CUDA_CHECK(cudaFreeHost(stopHostV));

  std::sort(samples.begin(), samples.end());
  double sum = 0;
  for (double s : samples) {
    sum += s;
  }
  avgNs = sum / samples.size();
  p50Ns = samples[samples.size() / 2];
  p99Ns = samples[(samples.size() * 99) / 100];
  minNs = samples.front();
}

// Each throughput point is run kReps times; we report the median and sample
// stddev so a later diff can tell a real change from run-to-run noise (the
// GB200 fast path in particular is noisy).
constexpr int kReps = 7;

struct MopsStat {
  double median;
  double stddev;
};

template <class Fn>
MopsStat repeatMops(Fn&& fn) {
  std::vector<double> v;
  v.reserve(kReps);
  for (int i = 0; i < kReps; ++i) {
    v.push_back(fn());
  }
  std::sort(v.begin(), v.end());
  double median = v[v.size() / 2];
  double mean = 0;
  for (double x : v) {
    mean += x;
  }
  mean /= v.size();
  double var = 0;
  for (double x : v) {
    var += (x - mean) * (x - mean);
  }
  return {median, std::sqrt(var / v.size())};
}

// One throughput-table row: kReps runs of runThroughput, median Mops/s +
// stddev, GB/s derived from the median (GB/s = Mops/s * sizeof(T) / 1e3).
template <class T, ConsumerPolicy Policy>
void throughputRow(
    int payloadB,
    int producers,
    int consumers,
    int blocks,
    int threads,
    uint64_t total,
    int nConsumers,
    uint32_t cap) {
  MopsStat s = repeatMops([&] {
    double m = 0, g = 0;
    runThroughput<T, Policy>(blocks, threads, total, nConsumers, cap, m, g);
    return m;
  });
  printf(
      "%-10d %-10d %-10d %-10.2f %-10.2f %-10.3f\n",
      payloadB,
      producers,
      consumers,
      s.median,
      s.stddev,
      s.median * sizeof(T) / 1e3);
}

// One readMulti (MPSC batched) throughput row, same median+stddev treatment.
template <class T>
void readMultiRow(
    int payloadB,
    int blocks,
    int threads,
    uint64_t total,
    uint32_t cap,
    size_t batchMax) {
  MopsStat s = repeatMops([&] {
    double m = 0, g = 0;
    runThroughputReadMulti<T>(blocks, threads, total, cap, batchMax, m, g);
    return m;
  });
  printf(
      "%-10d %-10d %-10d %-10.2f %-10.2f %-10.3f\n",
      payloadB,
      256,
      1,
      s.median,
      s.stddev,
      s.median * sizeof(T) / 1e3);
}

} // namespace

int main() {
  const uint32_t cap = 1024;
  const uint64_t total = 2'000'000;
  double mops = 0, gbps = 0;

  // Warm up CUDA context / JIT before timing.
  runThroughput<Payload<16>, ConsumerPolicy::Single>(
      8, 32, 200'000, 1, cap, mops, gbps);

  printf("\n=== Throughput: GPU producers -> 1 host consumer (MPSC) ===\n");
  printf(
      "%-10s %-10s %-10s %-10s %-10s %-10s\n",
      "payloadB",
      "producers",
      "consumers",
      "Mops/s",
      "stddev",
      "GB/s");
  fflush(stdout);
  throughputRow<Payload<4>, ConsumerPolicy::Single>(
      16, 256, 1, 8, 32, total, 1, cap);
  throughputRow<Payload<16>, ConsumerPolicy::Single>(
      64, 256, 1, 8, 32, total, 1, cap);
  throughputRow<Payload<32>, ConsumerPolicy::Single>(
      128, 256, 1, 8, 32, total, 1, cap);
  throughputRow<Payload<64>, ConsumerPolicy::Single>(
      256, 256, 1, 8, 32, total, 1, cap);
  fflush(stdout);

  printf(
      "\n=== Throughput: GPU producers -> 1 host consumer (MPSC, readMulti) ===\n");
  printf(
      "%-10s %-10s %-10s %-10s %-10s %-10s\n",
      "payloadB",
      "producers",
      "consumers",
      "Mops/s",
      "stddev",
      "GB/s");
  readMultiRow<Payload<4>>(16, 8, 32, total, cap, 256);
  readMultiRow<Payload<16>>(64, 8, 32, total, cap, 256);
  readMultiRow<Payload<32>>(128, 8, 32, total, cap, 256);
  readMultiRow<Payload<64>>(256, 8, 32, total, cap, 256);
  fflush(stdout);

  printf("\n=== Throughput: GPU producers -> 4 host consumers (MPMC) ===\n");
  printf(
      "%-10s %-10s %-10s %-10s %-10s %-10s\n",
      "payloadB",
      "producers",
      "consumers",
      "Mops/s",
      "stddev",
      "GB/s");
  throughputRow<Payload<4>, ConsumerPolicy::Multi>(
      16, 256, 4, 8, 32, total, 4, cap);
  throughputRow<Payload<16>, ConsumerPolicy::Multi>(
      64, 256, 4, 8, 32, total, 4, cap);
  throughputRow<Payload<32>, ConsumerPolicy::Multi>(
      128, 256, 4, 8, 32, total, 4, cap);
  throughputRow<Payload<64>, ConsumerPolicy::Multi>(
      256, 256, 4, 8, 32, total, 4, cap);
  fflush(stdout);

  printf(
      "\n=== Scaling: 256 GPU producers -> N host consumers, 64B payload ===\n");
  printf("%-12s %-10s %-10s %-10s\n", "consumers", "Mops/s", "stddev", "GB/s");
  {
    MopsStat s = repeatMops([&] {
      double m = 0, g = 0;
      runThroughput<Payload<16>, ConsumerPolicy::Single>(
          8, 32, total, 1, cap, m, g);
      return m;
    });
    printf(
        "%-12s %-10.2f %-10.2f %-10.3f\n",
        "1 (MPSC)",
        s.median,
        s.stddev,
        s.median * sizeof(Payload<16>) / 1e3);
  }
  for (int nc : {2, 4, 8}) {
    MopsStat s = repeatMops([&] {
      double m = 0, g = 0;
      runThroughput<Payload<16>, ConsumerPolicy::Multi>(
          8, 32, total, nc, cap, m, g);
      return m;
    });
    printf(
        "%-12d %-10.2f %-10.2f %-10.3f\n",
        nc,
        s.median,
        s.stddev,
        s.median * sizeof(Payload<16>) / 1e3);
  }
  fflush(stdout);

  printf(
      "\n=== Capacity sweep: 256 GPU producers -> 1 host consumer, 64B (MPSC) ===\n");
  printf("%-10s %-10s %-10s %-10s\n", "cap", "Mops/s", "stddev", "GB/s");
  for (uint32_t capN : {64u, 256u, 1024u, 4096u}) {
    MopsStat s = repeatMops([&] {
      double m = 0, g = 0;
      runThroughput<Payload<16>, ConsumerPolicy::Single>(
          8, 32, total, 1, capN, m, g);
      return m;
    });
    printf(
        "%-10u %-10.2f %-10.2f %-10.3f\n",
        capN,
        s.median,
        s.stddev,
        s.median * sizeof(Payload<16>) / 1e3);
  }
  fflush(stdout);

  printf(
      "\n=== Round-trip latency (MPSC): request -> GPU publish -> host read ===\n");
  printf(
      "%-10s %-9s %-9s %-9s %-9s\n",
      "payloadB",
      "min(ns)",
      "p50(ns)",
      "avg(ns)",
      "p99(ns)");
  double a, p50, p99, mn;
  runLatency<Payload<4>>(20'000, cap, a, p50, p99, mn);
  printf("%-10d %-9.0f %-9.0f %-9.0f %-9.0f\n", 16, mn, p50, a, p99);
  runLatency<Payload<16>>(20'000, cap, a, p50, p99, mn);
  printf("%-10d %-9.0f %-9.0f %-9.0f %-9.0f\n", 64, mn, p50, a, p99);
  runLatency<Payload<32>>(20'000, cap, a, p50, p99, mn);
  printf("%-10d %-9.0f %-9.0f %-9.0f %-9.0f\n", 128, mn, p50, a, p99);
  fflush(stdout);

  printf(
      "\n=== Round-trip latency (MPMC read path): request -> publish -> read ===\n");
  printf(
      "%-10s %-9s %-9s %-9s %-9s\n",
      "payloadB",
      "min(ns)",
      "p50(ns)",
      "avg(ns)",
      "p99(ns)");
  runLatency<Payload<4>, ConsumerPolicy::Multi>(20'000, cap, a, p50, p99, mn);
  printf("%-10d %-9.0f %-9.0f %-9.0f %-9.0f\n", 16, mn, p50, a, p99);
  runLatency<Payload<16>, ConsumerPolicy::Multi>(20'000, cap, a, p50, p99, mn);
  printf("%-10d %-9.0f %-9.0f %-9.0f %-9.0f\n", 64, mn, p50, a, p99);
  runLatency<Payload<32>, ConsumerPolicy::Multi>(20'000, cap, a, p50, p99, mn);
  printf("%-10d %-9.0f %-9.0f %-9.0f %-9.0f\n", 128, mn, p50, a, p99);
  fflush(stdout);

  printf(
      "\n=== Latency under producer contention (MPSC, 64B; tracer round-trip "
      "incl. queue residency) ===\n");
  printf(
      "%-12s %-9s %-9s %-9s %-9s\n",
      "producers",
      "min(ns)",
      "p50(ns)",
      "avg(ns)",
      "p99(ns)");
  // Fewer iters than the empty-queue latency above: under contention each
  // tracer drains ~capacity backlog in FIFO, so cost is iters * capacity --
  // 2k samples is plenty for a p99 and keeps the 256-producer case tractable
  // (and not multi-minute on MI300's relaxed memory).
  for (int np : {1, 32, 256}) {
    runLatencyContended<Payload<16>>(np, 2'000, cap, a, p50, p99, mn);
    printf("%-12d %-9.0f %-9.0f %-9.0f %-9.0f\n", np, mn, p50, a, p99);
  }
  fflush(stdout);

  return 0;
}
