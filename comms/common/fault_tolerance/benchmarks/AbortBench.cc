// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/common/fault_tolerance/benchmarks/AbortBench.cuh"

namespace comms::fault_tolerance::benchmark {
namespace {

constexpr int kDevicePollIterations = 10000000;
constexpr int kDeviceLoopIterations = 1000000;
constexpr int kManyBlockPollingBlocks = 128;
constexpr int kManyBlockPollingThreads = 32;
constexpr int kManyBlockPollIterations = 64;

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

class LegacyAbortModel {
 public:
  bool Test() const {
    return abort_.load(std::memory_order_acquire) != 0;
  }

  void SetDefaultTimeoutDuration(std::chrono::milliseconds duration) {
    timeoutMs_.store(duration.count(), std::memory_order_release);
  }

  std::chrono::milliseconds GetDefaultTimeoutDuration() const {
    return std::chrono::milliseconds{
        timeoutMs_.load(std::memory_order_acquire)};
  }

 private:
  std::atomic<int> abort_{0};
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
  return abort->GetDefaultTimeoutDuration().count();
}

FOLLY_NOINLINE void setAbortDefaultTimeout(
    Abort* abort,
    std::chrono::milliseconds timeout) {
  abort->setDefaultTimeout(timeout);
}

FOLLY_NOINLINE int64_t getAbortDefaultTimeoutMs(const Abort* abort) {
  auto timeout = abort->getDefaultTimeout();
  CHECK(timeout.has_value());
  return timeout->count();
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

void waitForCounterAtLeast(
    const HostAtomicInt* counter,
    int expected,
    const char* label) {
  int polls = 0;
  while (loadHostAtomicAcquire(counter) < expected) {
    CHECK_LT(++polls, kDevicePollIterations) << label;
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
  folly::doNotOptimizeAway(readDeviceValue(sink));
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

BENCHMARK_COUNTERS(AbortSignalHostDeviceHalfRoundTrip, counters, iters) {
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
  folly::doNotOptimizeAway(responseAtomicPtr);
  CHECK_EQ(
      launchDeviceToHostRoundTrip(
          request.device(),
          response.device(),
          ready.device(),
          iterations,
          /*stream=*/nullptr),
      cudaSuccess);
  waitForCounterAtLeast(
      readyAtomicPtr, 1, "Host/device benchmark kernel did not start");
  suspender.dismiss();

  for (int expected = 1; expected <= iterations; ++expected) {
    requestAtomic.store(expected, std::memory_order_release);
    waitForCounterAtLeast(
        responseAtomicPtr,
        expected,
        "Host/device benchmark response timed out");
  }
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  folly::doNotOptimizeAway(loadHostAtomicAcquire(responseAtomicPtr));
  recordOps(counters, 2ULL * iters);
  counters["mixedRoundTrips"] =
      folly::UserMetric(iters, folly::UserMetric::Type::METRIC);
  suspender.rehire();
}

BENCHMARK_COUNTERS(AbortSignalDeviceDeviceHalfRoundTrip, counters, iters) {
  folly::BenchmarkSuspender suspender;
  const auto iterations = checkedIterationCount(iters);
  MappedFlag request;
  MappedFlag response;
  MappedFlag ready;
  MappedFlag start;
  auto readyAtomic = ready.hostAtomic();
  auto startAtomic = start.hostAtomic();
  auto* readyAtomicPtr = &readyAtomic;
  auto observed = makeDeviceValue<int>();
  CHECK_EQ(
      launchDeviceToDevicePingPong(
          request.device(),
          response.device(),
          ready.device(),
          start.device(),
          observed.get(),
          iterations,
          kDevicePollIterations,
          /*stream=*/nullptr),
      cudaSuccess);
  waitForCounterAtLeast(
      readyAtomicPtr, 2, "Device/device benchmark blocks did not start");
  suspender.dismiss();

  startAtomic.store(1, std::memory_order_release);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  suspender.rehire();

  const auto observedValue = readDeviceValue(observed);
  CHECK_EQ(observedValue, iterations)
      << "Device-to-device benchmark missed a signal";
  folly::doNotOptimizeAway(observedValue);
  recordOps(counters, 2ULL * iters);
  counters["pingPongs"] =
      folly::UserMetric(iters, folly::UserMetric::Type::METRIC);
}

} // namespace comms::fault_tolerance::benchmark

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
