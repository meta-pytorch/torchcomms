// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <thread>

#include <common/init/Init.h>
#include <folly/Benchmark.h>
#include <glog/logging.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/common/fault_tolerance/benchmarks/AbortBench.cuh"

namespace comms::fault_tolerance::benchmark {
namespace {

// Both signal budgets are global to a benchmark invocation, not per exchange,
// so a degraded run fails once rather than once per iteration. They therefore
// have to scale with the iteration count folly picks, or a healthy run with a
// large enough `iters` trips the guard on wall clock alone.
//
// The device budget must stay strictly shorter than the host budget: on a real
// co-residency failure we want the kernel to record its own sentinel and the
// host to report that, rather than the host CHECK firing first and tearing down
// the context before the diagnostic can be read.
constexpr auto kSignalWaitBase = std::chrono::milliseconds{1000};
constexpr auto kSignalWaitPerExchange = std::chrono::microseconds{50};
constexpr int kHostSignalWaitSlack = 5;
constexpr int kDeviceLoopIterations = 100000;
constexpr int kManyBlockPollingBlocks = 128;
constexpr int kManyBlockPollingThreads = 32;
constexpr int kManyBlockPollIterations = 1024;

using HostAtomicInt = std::atomic_ref<int>;

struct CudaFreeDeleter {
  template <typename T>
  void operator()(T* ptr) const {
    if (ptr != nullptr) {
      CHECK_EQ(cudaFree(ptr), cudaSuccess);
    }
  }
};

template <typename T>
using DeviceValue = std::unique_ptr<T, CudaFreeDeleter>;

template <typename T>
DeviceValue<T> makeDeviceValue(T value = 0) {
  T* ptr = nullptr;
  CHECK_EQ(cudaMalloc(&ptr, sizeof(T)), cudaSuccess);
  CHECK_EQ(
      cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
  return DeviceValue<T>{ptr};
}

template <typename T>
DeviceValue<T> makeDeviceBuffer(std::size_t count) {
  T* ptr = nullptr;
  CHECK_EQ(cudaMalloc(&ptr, sizeof(T) * count), cudaSuccess);
  CHECK_EQ(cudaMemset(ptr, 0, sizeof(T) * count), cudaSuccess);
  return DeviceValue<T>{ptr};
}

template <typename T>
T readDeviceValue(const DeviceValue<T>& ptr) {
  T value = 0;
  CHECK_EQ(
      cudaMemcpy(&value, ptr.get(), sizeof(T), cudaMemcpyDeviceToHost),
      cudaSuccess);
  return value;
}

// Reads one element out of a multi-element device buffer. Separate from
// readDeviceValue() so a single-element read of a buffer is obviously
// deliberate: the benchmarks only need one element to defeat dead-code
// elimination, not the whole allocation.
template <typename T>
T readDeviceElement(const DeviceValue<T>& buffer, std::size_t index = 0) {
  T value = 0;
  CHECK_EQ(
      cudaMemcpy(
          &value, buffer.get() + index, sizeof(T), cudaMemcpyDeviceToHost),
      cudaSuccess);
  return value;
}

class LegacyAbortModel {
 public:
  explicit LegacyAbortModel(bool enabled = true) : enabled_(enabled) {}

  bool Test() const {
    if (!enabled_) {
      return false;
    }
    if (abort_.load(std::memory_order_acquire)) {
      return true;
    }
    if (!hasTimeout_.load(std::memory_order_acquire)) {
      return false;
    }
    return TimedOut();
  }

  void SetDefaultTimeoutDuration(std::chrono::milliseconds duration) {
    if (!enabled_) {
      return;
    }
    timeoutMs_.store(duration.count(), std::memory_order_release);
  }

  std::optional<std::chrono::milliseconds> GetDefaultTimeoutDuration() const {
    if (!enabled_) {
      return std::nullopt;
    }
    const auto timeoutMs = timeoutMs_.load(std::memory_order_acquire);
    if (timeoutMs < 0) {
      return std::nullopt;
    }
    return std::chrono::milliseconds{timeoutMs};
  }

 private:
  bool TimedOut() const {
    return false;
  }

  const bool enabled_;
  std::atomic<int> abort_{0};
  std::atomic<bool> hasTimeout_{false};
  std::atomic<int64_t> timeoutMs_{-1};
};

FOLLY_NOINLINE int loadStdAtomicAcquire(const std::atomic<int>* flag) {
  return flag->load(std::memory_order_acquire);
}

FOLLY_NOINLINE int loadHostAtomicAcquire(const HostAtomicInt* flag) {
  return flag->load(std::memory_order_acquire);
}

FOLLY_NOINLINE bool testLegacyAbort(const LegacyAbortModel* abort) {
  return abort->Test();
}

FOLLY_NOINLINE bool testAbort(Abort* abort) {
  return abort->isAborted();
}

FOLLY_NOINLINE void setLegacyDefaultTimeout(
    LegacyAbortModel* abort,
    std::chrono::milliseconds timeout) {
  abort->SetDefaultTimeoutDuration(timeout);
}

FOLLY_NOINLINE int64_t
getLegacyDefaultTimeoutMs(const LegacyAbortModel* abort) {
  return abort->GetDefaultTimeoutDuration()
      .value_or(std::chrono::milliseconds{-1})
      .count();
}

FOLLY_NOINLINE void setAbortDefaultTimeout(
    Abort* abort,
    std::chrono::milliseconds timeout) {
  abort->setDefaultTimeout(timeout);
}

FOLLY_NOINLINE int64_t getAbortDefaultTimeoutMs(const Abort* abort) {
  return abort->getDefaultTimeout()
      .value_or(std::chrono::milliseconds{-1})
      .count();
}

FOLLY_NOINLINE void setAbortTimeout(
    Abort* abort,
    std::chrono::milliseconds timeout) {
  abort->startTimeout(timeout);
}

FOLLY_NOINLINE std::chrono::milliseconds timeRemaining(Abort* abort) {
  return abort->getTimeRemaining();
}

FOLLY_NOINLINE void cancelAbortTimeout(Abort* abort) {
  abort->cancelTimeout();
}

class MappedFlag {
 public:
  MappedFlag() {
    void* host = nullptr;
    CHECK_EQ(
        cudaHostAlloc(&host, sizeof(int), cudaHostAllocMapped), cudaSuccess);
    host_ = static_cast<int*>(host);

    void* device = nullptr;
    CHECK_EQ(cudaHostGetDevicePointer(&device, host_, 0), cudaSuccess);
    device_ = static_cast<int*>(device);
    reset();
  }

  ~MappedFlag() {
    CHECK_EQ(cudaFreeHost(host_), cudaSuccess);
  }

  MappedFlag(const MappedFlag&) = delete;
  MappedFlag& operator=(const MappedFlag&) = delete;
  MappedFlag(MappedFlag&&) = delete;
  MappedFlag& operator=(MappedFlag&&) = delete;

  int* device() const {
    return device_;
  }

  HostAtomicInt hostAtomic() {
    return HostAtomicInt{*host_};
  }

  void reset() {
    hostAtomic().store(0, std::memory_order_release);
  }

 private:
  int* host_{nullptr};
  int* device_{nullptr};
};

void recordOps(folly::UserCounters& counters, uint64_t ops) {
  counters["ops"] = folly::UserMetric(ops, folly::UserMetric::Type::METRIC);
}

int checkedIterationCount(uint32_t iters) {
  CHECK_LE(iters, static_cast<uint32_t>(std::numeric_limits<int>::max()));
  return static_cast<int>(iters);
}

std::chrono::milliseconds deviceSignalWaitBudget(int iterations) {
  return kSignalWaitBase +
      std::chrono::duration_cast<std::chrono::milliseconds>(
             kSignalWaitPerExchange * iterations);
}

std::chrono::milliseconds hostSignalWaitBudget(int iterations) {
  return deviceSignalWaitBudget(iterations) * kHostSignalWaitSlack;
}

uint64_t deviceWaitCycles(std::chrono::milliseconds budget) {
  int device = 0;
  CHECK_EQ(cudaGetDevice(&device), cudaSuccess);
  return static_cast<uint64_t>(budget.count()) *
      detail::hostDeviceCyclesPerMs(device);
}

std::chrono::steady_clock::time_point hostSignalDeadline(int iterations) {
  const auto hostBudget = hostSignalWaitBudget(iterations);
  CHECK_GT(hostBudget.count(), deviceSignalWaitBudget(iterations).count())
      << "host guard must outlast the device budget so the kernel sentinel wins";
  return std::chrono::steady_clock::now() + hostBudget;
}

void waitForCounterAtLeast(
    const HostAtomicInt* counter,
    int expected,
    std::chrono::steady_clock::time_point deadline,
    const char* label) {
  while (loadHostAtomicAcquire(counter) < expected) {
    CHECK(std::chrono::steady_clock::now() < deadline) << label;
    std::this_thread::yield();
  }
}

} // namespace

BENCHMARK_COUNTERS(StdAtomicHostLoad, counters, iters) {
  folly::BenchmarkSuspender suspender;
  std::atomic<int> flag{1};
  auto* flagPtr = &flag;
  folly::doNotOptimizeAway(flagPtr);
  suspender.dismiss();

  int sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    auto value = loadStdAtomicAcquire(flagPtr);
    folly::doNotOptimizeAway(value);
    sink += value;
  }
  folly::doNotOptimizeAway(sink);
  recordOps(counters, iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(MappedPinnedHostLoad, counters, iters) {
  folly::BenchmarkSuspender suspender;
  MappedFlag flag;
  auto atomic = flag.hostAtomic();
  atomic.store(1, std::memory_order_release);
  auto* atomicPtr = &atomic;
  folly::doNotOptimizeAway(atomicPtr);
  suspender.dismiss();

  int sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    auto value = loadHostAtomicAcquire(atomicPtr);
    folly::doNotOptimizeAway(value);
    sink += value;
  }
  folly::doNotOptimizeAway(sink);
  recordOps(counters, iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(MappedPinnedAbortFlagLoad, counters, iters) {
  folly::BenchmarkSuspender suspender;
  MappedFlag flag;
  auto abortFlag = flag.hostAtomic();
  abortFlag.store(
      static_cast<int>(AbortReason::ABORTED), std::memory_order_release);
  auto* abortFlagPtr = &abortFlag;
  folly::doNotOptimizeAway(abortFlagPtr);
  suspender.dismiss();

  int sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    auto value = loadHostAtomicAcquire(abortFlagPtr);
    folly::doNotOptimizeAway(value);
    sink += value;
  }
  folly::doNotOptimizeAway(sink);
  recordOps(counters, iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(LegacyAbortTestNoTimeout, counters, iters) {
  folly::BenchmarkSuspender suspender;
  LegacyAbortModel abort;
  auto* abortPtr = &abort;
  folly::doNotOptimizeAway(abortPtr);
  suspender.dismiss();

  int sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    auto value = testLegacyAbort(abortPtr) ? 1 : 0;
    folly::doNotOptimizeAway(value);
    sink += value;
  }
  folly::doNotOptimizeAway(sink);
  recordOps(counters, iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(AbortTestNoTimeout, counters, iters) {
  folly::BenchmarkSuspender suspender;
  Abort abort{/*enabled=*/true};
  auto* abortPtr = &abort;
  folly::doNotOptimizeAway(abortPtr);
  suspender.dismiss();

  int sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    sink += testAbort(abortPtr) ? 1 : 0;
  }
  folly::doNotOptimizeAway(sink);
  recordOps(counters, iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(AbortTestWithFutureTimeout, counters, iters) {
  folly::BenchmarkSuspender suspender;
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::hours(1));
  auto* abortPtr = &abort;
  folly::doNotOptimizeAway(abortPtr);
  suspender.dismiss();

  int sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    sink += testAbort(abortPtr) ? 1 : 0;
  }
  folly::doNotOptimizeAway(sink);
  recordOps(counters, iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(LegacyDefaultTimeoutSetGet, counters, iters) {
  folly::BenchmarkSuspender suspender;
  LegacyAbortModel abort;
  auto* abortPtr = &abort;
  folly::doNotOptimizeAway(abortPtr);
  suspender.dismiss();

  int64_t sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    setLegacyDefaultTimeout(abortPtr, std::chrono::milliseconds{1000});
    auto timeout = getLegacyDefaultTimeoutMs(abortPtr);
    folly::doNotOptimizeAway(timeout);
    sink += timeout;
  }
  folly::doNotOptimizeAway(sink);
  recordOps(counters, 2ULL * iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(AbortDefaultTimeoutSetGet, counters, iters) {
  folly::BenchmarkSuspender suspender;
  Abort abort{/*enabled=*/true};
  auto* abortPtr = &abort;
  setAbortDefaultTimeout(abortPtr, std::chrono::milliseconds{1000});
  CHECK(abort.getDefaultTimeout().has_value());
  folly::doNotOptimizeAway(abortPtr);
  suspender.dismiss();

  int64_t sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    setAbortDefaultTimeout(abortPtr, std::chrono::milliseconds{1000});
    sink += getAbortDefaultTimeoutMs(abortPtr);
  }
  folly::doNotOptimizeAway(sink);
  recordOps(counters, 2ULL * iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(AbortTimeRemaining, counters, iters) {
  folly::BenchmarkSuspender suspender;
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::hours(1));
  auto* abortPtr = &abort;
  folly::doNotOptimizeAway(abortPtr);
  suspender.dismiss();

  int64_t sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    sink += timeRemaining(abortPtr).count();
  }
  folly::doNotOptimizeAway(sink);
  recordOps(counters, iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(AbortSetTimeoutCancel, counters, iters) {
  folly::BenchmarkSuspender suspender;
  Abort abort{/*enabled=*/true};
  auto* abortPtr = &abort;
  folly::doNotOptimizeAway(abortPtr);
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    setAbortTimeout(abortPtr, std::chrono::hours(1));
    cancelAbortTimeout(abortPtr);
  }
  recordOps(counters, 2ULL * iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(CudaAtomicDeviceLoadLoop, counters, iters) {
  folly::BenchmarkSuspender suspender;
  MappedFlag flag;
  flag.hostAtomic().store(1, std::memory_order_release);
  auto sink = makeDeviceValue<int>();
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    CHECK_EQ(
        launchDeviceLoadLoop(
            flag.device(),
            sink.get(),
            kDeviceLoopIterations,
            /*stream=*/nullptr),
        cudaSuccess);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  folly::doNotOptimizeAway(readDeviceValue(sink));
  counters["deviceAtomicLoads"] = folly::UserMetric(
      static_cast<double>(iters) * kDeviceLoopIterations,
      folly::UserMetric::Type::METRIC);
  suspender.rehire();
}

BENCHMARK_COUNTERS(CudaAtomicManyBlockDeviceLoadLoop, counters, iters) {
  folly::BenchmarkSuspender suspender;
  MappedFlag flag;
  flag.hostAtomic().store(1, std::memory_order_release);
  auto sink =
      makeDeviceBuffer<int>(kManyBlockPollingBlocks * kManyBlockPollingThreads);
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    CHECK_EQ(
        launchManyBlockDeviceLoadLoop(
            flag.device(),
            sink.get(),
            kManyBlockPollingBlocks,
            kManyBlockPollingThreads,
            kManyBlockPollIterations,
            /*stream=*/nullptr),
        cudaSuccess);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  folly::doNotOptimizeAway(readDeviceElement(sink));
  counters["deviceAtomicLoads"] = folly::UserMetric(
      static_cast<double>(iters) * kManyBlockPollingBlocks *
          kManyBlockPollingThreads * kManyBlockPollIterations,
      folly::UserMetric::Type::METRIC);
  counters["pollingBlocks"] = folly::UserMetric(
      kManyBlockPollingBlocks, folly::UserMetric::Type::METRIC);
  counters["pollingThreadsPerBlock"] = folly::UserMetric(
      kManyBlockPollingThreads, folly::UserMetric::Type::METRIC);
  suspender.rehire();
}

BENCHMARK_COUNTERS(CudaAtomicDeviceStoreLoop, counters, iters) {
  folly::BenchmarkSuspender suspender;
  MappedFlag flag;
  flag.reset();
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    CHECK_EQ(
        launchDeviceStoreLoop(
            flag.device(), kDeviceLoopIterations, /*stream=*/nullptr),
        cudaSuccess);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  counters["deviceAtomicStores"] = folly::UserMetric(
      static_cast<double>(iters) * kDeviceLoopIterations,
      folly::UserMetric::Type::METRIC);
  suspender.rehire();
}

BENCHMARK_COUNTERS(AbortDeviceDefaultTimeoutLoadLoop, counters, iters) {
  folly::BenchmarkSuspender suspender;
  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(std::chrono::milliseconds{1000});
  auto sink = makeDeviceValue<int64_t>();
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    CHECK_EQ(
        launchAbortDeviceDefaultTimeoutLoadLoop(
            abort.getDeviceHandle(),
            sink.get(),
            kDeviceLoopIterations,
            /*stream=*/nullptr),
        cudaSuccess);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  CHECK_GT(readDeviceValue(sink), 0);
  counters["deviceTimeoutLoads"] = folly::UserMetric(
      static_cast<double>(iters) * kDeviceLoopIterations,
      folly::UserMetric::Type::METRIC);
  suspender.rehire();
}

BENCHMARK_COUNTERS(AbortDeviceIsAbortedLoadLoop, counters, iters) {
  folly::BenchmarkSuspender suspender;
  Abort abort{/*enabled=*/true};
  auto sink = makeDeviceValue<int>();
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    CHECK_EQ(
        launchAbortDeviceIsAbortedLoadLoop(
            abort.getDeviceHandle(),
            sink.get(),
            kDeviceLoopIterations,
            /*startTimeout=*/false,
            /*stream=*/nullptr),
        cudaSuccess);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  CHECK_EQ(readDeviceValue(sink), 0);
  counters["deviceIsAbortedPolls"] = folly::UserMetric(
      static_cast<double>(iters) * kDeviceLoopIterations,
      folly::UserMetric::Type::METRIC);
  suspender.rehire();
}

BENCHMARK_COUNTERS(AbortDeviceIsAbortedWithDeadlineLoadLoop, counters, iters) {
  folly::BenchmarkSuspender suspender;
  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(std::chrono::hours(1));
  auto sink = makeDeviceValue<int>();
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    CHECK_EQ(
        launchAbortDeviceIsAbortedLoadLoop(
            abort.getDeviceHandle(),
            sink.get(),
            kDeviceLoopIterations,
            /*startTimeout=*/true,
            /*stream=*/nullptr),
        cudaSuccess);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  CHECK_EQ(readDeviceValue(sink), 0);
  counters["deviceIsAbortedPolls"] = folly::UserMetric(
      static_cast<double>(iters) * kDeviceLoopIterations,
      folly::UserMetric::Type::METRIC);
  suspender.rehire();
}

BENCHMARK_COUNTERS(MappedPinnedHostStoreLoad, counters, iters) {
  folly::BenchmarkSuspender suspender;
  MappedFlag flag;
  auto atomic = flag.hostAtomic();
  auto* atomicPtr = &atomic;
  folly::doNotOptimizeAway(atomicPtr);
  flag.reset();
  suspender.dismiss();

  int sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    atomic.store(1, std::memory_order_release);
    sink += loadHostAtomicAcquire(atomicPtr);
  }
  folly::doNotOptimizeAway(sink);
  recordOps(counters, 2ULL * iters);
  suspender.rehire();
}

BENCHMARK_COUNTERS(AbortSignalHostDeviceRoundTrip, counters, iters) {
  folly::BenchmarkSuspender suspender;
  const auto iterations = checkedIterationCount(iters);
  MappedFlag request;
  MappedFlag response;
  MappedFlag ready;
  auto requestAtomic = request.hostAtomic();
  auto responseAtomic = response.hostAtomic();
  auto readyAtomic = ready.hostAtomic();
  auto* responseAtomicPtr = &responseAtomic;
  auto* readyAtomicPtr = &readyAtomic;
  auto observed = makeDeviceValue<int>();
  const auto deviceTimeoutCycles =
      deviceWaitCycles(deviceSignalWaitBudget(iterations));
  folly::doNotOptimizeAway(responseAtomicPtr);
  CHECK_EQ(
      launchDeviceToHostRoundTrip(
          request.device(),
          response.device(),
          ready.device(),
          observed.get(),
          iterations,
          deviceTimeoutCycles,
          /*stream=*/nullptr),
      cudaSuccess);
  const auto deadline = hostSignalDeadline(iterations);
  waitForCounterAtLeast(
      readyAtomicPtr,
      1,
      deadline,
      "Host/device benchmark kernel did not start");
  suspender.dismiss();

  for (int expected = 1; expected <= iterations; ++expected) {
    requestAtomic.store(expected, std::memory_order_release);
    waitForCounterAtLeast(
        responseAtomicPtr,
        expected,
        deadline,
        "Host/device benchmark response timed out");
  }
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  suspender.rehire();

  const auto observedValue = readDeviceValue(observed);
  CHECK_EQ(observedValue, iterations)
      << "Host/device benchmark missed a signal";
  folly::doNotOptimizeAway(observedValue);
  folly::doNotOptimizeAway(loadHostAtomicAcquire(responseAtomicPtr));
  recordOps(counters, 2ULL * iters);
  counters["mixedRoundTrips"] =
      folly::UserMetric(iters, folly::UserMetric::Type::METRIC);
}

// Attributes the per-launch cost of arming a device deadline, which is the
// leading suspect for FT's fixed healthy-path overhead.
//
// `startTimeout()` resolves the communicator deadline through
// `getTimeoutMs()`, an uncached read of mapped pinned host memory. Every block
// does it at kernel entry, and they all hit the same cacheline. This launches
// a kernel that does nothing else, so the enabled-minus-disabled difference is
// the arm cost with launch overhead subtracted out.
void runArmOnlyBenchmark(
    folly::UserCounters& counters,
    uint32_t iters,
    bool enabled,
    int blocks,
    int threads) {
  folly::BenchmarkSuspender suspender;
  Abort abort{enabled};
  if (enabled) {
    abort.setDefaultTimeout(std::chrono::milliseconds{60000});
  }
  auto handle = abort.getDeviceHandle();
  auto sink = makeDeviceValue<uint64_t>();
  CHECK_NE(sink, nullptr);
  // Warm the context so the first launch's lazy init is not attributed here.
  CHECK_EQ(
      launchAbortDeviceArmOnly(
          handle, sink.get(), blocks, threads, /*stream=*/nullptr),
      cudaSuccess);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  suspender.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    CHECK_EQ(
        launchAbortDeviceArmOnly(
            handle, sink.get(), blocks, threads, /*stream=*/nullptr),
        cudaSuccess);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  suspender.rehire();
  counters["warps"] = folly::UserMetric(
      static_cast<int64_t>(blocks) * ((threads + 31) / 32),
      folly::UserMetric::Type::METRIC);
}

// 1x640 is the real AllReduce tree/ring launch shape (`kBlockSize = 640`).
// The rest bracket it so the scaling is visible rather than asserted.
BENCHMARK_COUNTERS(ArmOnlyDisabled1x1, counters, iters) {
  runArmOnlyBenchmark(counters, iters, /*enabled=*/false, 1, 1);
}

BENCHMARK_COUNTERS(ArmOnlyEnabled1x1, counters, iters) {
  runArmOnlyBenchmark(counters, iters, /*enabled=*/true, 1, 1);
}

BENCHMARK_COUNTERS(ArmOnlyDisabled1x640, counters, iters) {
  runArmOnlyBenchmark(counters, iters, /*enabled=*/false, 1, 640);
}

BENCHMARK_COUNTERS(ArmOnlyEnabled1x640, counters, iters) {
  runArmOnlyBenchmark(counters, iters, /*enabled=*/true, 1, 640);
}

BENCHMARK_COUNTERS(ArmOnlyDisabled8x640, counters, iters) {
  runArmOnlyBenchmark(counters, iters, /*enabled=*/false, 8, 640);
}

BENCHMARK_COUNTERS(ArmOnlyEnabled8x640, counters, iters) {
  runArmOnlyBenchmark(counters, iters, /*enabled=*/true, 8, 640);
}

BENCHMARK_COUNTERS(AbortSignalDeviceDevicePingPong, counters, iters) {
  folly::BenchmarkSuspender suspender;
  const auto iterations = checkedIterationCount(iters);
  MappedFlag request;
  MappedFlag response;
  MappedFlag ready;
  MappedFlag start;
  auto readyAtomic = ready.hostAtomic();
  auto startAtomic = start.hostAtomic();
  auto* readyAtomicPtr = &readyAtomic;
  auto observed = makeDeviceBuffer<int>(kPingPongBlocks);
  const auto deviceTimeoutCycles =
      deviceWaitCycles(deviceSignalWaitBudget(iterations));
  CHECK_EQ(
      launchDeviceToDevicePingPong(
          request.device(),
          response.device(),
          ready.device(),
          start.device(),
          observed.get(),
          iterations,
          deviceTimeoutCycles,
          /*stream=*/nullptr),
      cudaSuccess);
  const auto deadline = hostSignalDeadline(iterations);
  waitForCounterAtLeast(
      readyAtomicPtr,
      kPingPongBlocks,
      deadline,
      "Device/device benchmark blocks did not start");
  suspender.dismiss();

  startAtomic.store(1, std::memory_order_release);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  suspender.rehire();

  // Both blocks report independently, so a timeout in either is visible even
  // when the other completed.
  for (int block = 0; block < kPingPongBlocks; ++block) {
    const auto observedValue = readDeviceElement(observed, block);
    CHECK_EQ(observedValue, iterations)
        << "Device-to-device benchmark missed a signal in block " << block;
    folly::doNotOptimizeAway(observedValue);
  }
  recordOps(counters, 2ULL * iters);
  counters["pingPongs"] =
      folly::UserMetric(iters, folly::UserMetric::Type::METRIC);
}

} // namespace comms::fault_tolerance::benchmark

int main(int argc, char** argv) {
  facebook::initFacebook(&argc, &argv, true);
  folly::runBenchmarks();
  return 0;
}
