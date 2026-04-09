// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>

#ifdef ENABLE_FOLLY
#include <folly/io/async/ScopedEventBaseThread.h>
#endif // ENABLE_FOLLY

#include "comms/uniflow/executor/LockFreeEventBase.h"
#include "comms/uniflow/executor/MutexEventBase.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"

// ===== EventBaseAdapter =====
//
// Maps a uniform benchmark API to concrete EventBase types, similar to the
// adapter in StressFloodTest.cpp but extended with ScopedThread creation
// and dispatchInline.

template <typename EventBaseT>
struct EventBaseAdapter {
  using ScopedThread = uniflow::TScopedEventBaseThread<EventBaseT>;
  using EvbPtr = uniflow::EventBase*;

  static EvbPtr getEventBase(ScopedThread& t) {
    return t.getEventBase();
  }

  template <typename F>
  static void dispatch(EvbPtr evb, F&& func) {
    evb->dispatch(std::forward<F>(func));
  }

  template <typename F>
  static void dispatchAndWait(EvbPtr evb, F&& func) {
    evb->dispatchAndWait(std::forward<F>(func));
  }

  template <typename F>
  static void dispatchInline(EvbPtr evb, F&& func) {
    evb->dispatchInline(std::forward<F>(func));
  }
};

#ifdef ENABLE_FOLLY

template <>
struct EventBaseAdapter<folly::EventBase> {
  using ScopedThread = folly::ScopedEventBaseThread;
  using EvbPtr = folly::EventBase*;

  static EvbPtr getEventBase(ScopedThread& t) {
    return t.getEventBase();
  }

  template <typename F>
  static void dispatch(EvbPtr evb, F&& func) {
    evb->runInEventBaseThread(std::forward<F>(func));
  }

  template <typename F>
  static void dispatchAndWait(EvbPtr evb, F&& func) {
    evb->runInEventBaseThreadAndWait(std::forward<F>(func));
  }

  template <typename F>
  static void dispatchInline(EvbPtr evb, F&& func) {
    evb->runImmediatelyOrRunInEventBaseThread(std::forward<F>(func));
  }
};

#endif // ENABLE_FOLLY

// ===== Type aliases for readable benchmark names =====

using MutexEvb = uniflow::MutexEventBase;
using LockFreeEvb = uniflow::LockFreeEventBase;
#ifdef ENABLE_FOLLY
using FollyEvb = folly::EventBase;
#endif // ENABLE_FOLLY

// ===== Templated benchmarks =====

template <typename EventBaseT>
static void BM_Dispatch(benchmark::State& state) {
  using Adapter = EventBaseAdapter<EventBaseT>;
  typename Adapter::ScopedThread evbThread("bench");
  auto* evb = Adapter::getEventBase(evbThread);
  size_t count{0};
  for (auto _ : state) {
    Adapter::dispatch(evb, [&count]() noexcept { ++count; });
  }

  benchmark::DoNotOptimize(count);

  state.counters["ops/s"] =
      benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

template <typename EventBaseT>
static void BM_DispatchAndWait(benchmark::State& state) {
  using Adapter = EventBaseAdapter<EventBaseT>;
  typename Adapter::ScopedThread evbThread("bench");
  auto* evb = Adapter::getEventBase(evbThread);

  size_t count{0};
  for (auto _ : state) {
    Adapter::dispatchAndWait(evb, [&count]() noexcept { ++count; });
  }

  benchmark::DoNotOptimize(count);

  Adapter::dispatchAndWait(evb, []() noexcept {});

  state.counters["ops/s"] =
      benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

template <typename EventBaseT>
static void BM_DispatchInlineFromLoopThread(benchmark::State& state) {
  using Adapter = EventBaseAdapter<EventBaseT>;
  typename Adapter::ScopedThread evbThread("bench");
  auto* evb = Adapter::getEventBase(evbThread);

  size_t count{0};

  Adapter::dispatchAndWait(evb, [evb, &state, &count]() noexcept {
    for (auto _ : state) {
      Adapter::dispatchInline(evb, [&count]() noexcept { ++count; });
    }
  });
  benchmark::DoNotOptimize(count);

  state.counters["ops/s"] =
      benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

// Concurrent dispatch benchmark: N threads dispatch to the Evb.
// Bounding the number of in-flight tasks prevents allocator thrashing
// and allows measuring true steady-state throughput.
template <typename EventBaseT>
static void BM_ConcurrentDispatchBounded(benchmark::State& state) {
  using Adapter = EventBaseAdapter<EventBaseT>;
  typename Adapter::ScopedThread evbThread("bench");
  auto* evb = Adapter::getEventBase(evbThread);

  const int numThreads = state.range(0);
  std::atomic<bool> go{false};
  std::atomic<bool> done{false};
  std::atomic<int64_t> totalOps{0};

  constexpr int kMaxInFlightPerThread = 1024;

  std::vector<std::thread> producers;
  producers.reserve(numThreads);
  for (int i = 0; i < numThreads; ++i) {
    producers.emplace_back([evb, &go, &done, &totalOps] {
      while (!go.load(std::memory_order_acquire)) {
      }
      int64_t ops = 0;
      std::atomic<int> inFlight{0};
      while (!done.load(std::memory_order_relaxed)) {
        if (inFlight.load(std::memory_order_relaxed) < kMaxInFlightPerThread) {
          inFlight.fetch_add(1, std::memory_order_relaxed);
          Adapter::dispatch(evb, [&inFlight]() noexcept {
            inFlight.fetch_sub(1, std::memory_order_relaxed);
          });
          ++ops;
        } else {
          std::this_thread::yield();
        }
      }
      // Wait for all in-flight tasks from this producer to finish
      while (inFlight.load(std::memory_order_relaxed) > 0) {
        std::this_thread::yield();
      }
      totalOps.fetch_add(ops, std::memory_order_relaxed);
    });
  }

  go.store(true, std::memory_order_release);

  std::atomic<int> mainInFlight{0};
  for (auto _ : state) {
    while (mainInFlight.load(std::memory_order_relaxed) >=
           kMaxInFlightPerThread) {
      std::this_thread::yield();
    }
    mainInFlight.fetch_add(1, std::memory_order_relaxed);
    Adapter::dispatch(evb, [&mainInFlight]() noexcept {
      mainInFlight.fetch_sub(1, std::memory_order_relaxed);
    });
  }

  done.store(true, std::memory_order_release);
  for (auto& p : producers) {
    p.join();
  }

  while (mainInFlight.load(std::memory_order_relaxed) > 0) {
    std::this_thread::yield();
  }

  Adapter::dispatchAndWait(evb, []() noexcept {});

  int64_t allOps =
      totalOps.load(std::memory_order_relaxed) + state.iterations();
  state.counters["total_ops"] = allOps;
  state.counters["ops/s"] =
      benchmark::Counter(allOps, benchmark::Counter::kIsRate);
}

template <typename EventBaseT>
static void BM_DispatchBatch(benchmark::State& state) {
  using Adapter = EventBaseAdapter<EventBaseT>;
  typename Adapter::ScopedThread evbThread("bench");
  auto* evb = Adapter::getEventBase(evbThread);

  const int batchSize = state.range(0);

  for (auto _ : state) {
    for (int i = 0; i < batchSize; ++i) {
      Adapter::dispatch(evb, []() noexcept {});
    }
  }

  Adapter::dispatchAndWait(evb, []() noexcept {});

  state.counters["ops/s"] = benchmark::Counter(
      state.iterations() * batchSize, benchmark::Counter::kIsRate);
}

template <typename EventBaseT>
static void BM_PingPong(benchmark::State& state) {
  using Adapter = EventBaseAdapter<EventBaseT>;
  typename Adapter::ScopedThread evbThreadA("benchA");
  typename Adapter::ScopedThread evbThreadB("benchB");
  auto* evbA = Adapter::getEventBase(evbThreadA);
  auto* evbB = Adapter::getEventBase(evbThreadB);

  std::mutex mutex;
  std::condition_variable cv;
  bool done = false;

  for (auto _ : state) {
    done = false;
    Adapter::dispatch(evbA, [evbB, &mutex, &cv, &done]() noexcept {
      Adapter::dispatch(evbB, [&mutex, &cv, &done]() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
        cv.notify_one();
      });
    });

    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&done] { return done; });
  }

  state.counters["ops/s"] =
      benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

// ===== Benchmark registration =====

// NOLINTBEGIN(facebook-avoid-non-const-global-variables)
constexpr double kWarmUpTime = 0.5; // seconds
constexpr double kMinTime = 2.0; // seconds

#define REGISTER_BENCHMARK(name, ...)   \
  BENCHMARK_TEMPLATE(name, __VA_ARGS__) \
      ->MinWarmUpTime(kWarmUpTime)      \
      ->MinTime(kMinTime)

// --- Dispatch ---
REGISTER_BENCHMARK(BM_Dispatch, MutexEvb);
REGISTER_BENCHMARK(BM_Dispatch, LockFreeEvb);
#ifdef ENABLE_FOLLY
REGISTER_BENCHMARK(BM_Dispatch, FollyEvb);
#endif // ENABLE_FOLLY

// --- DispatchAndWait ---
REGISTER_BENCHMARK(BM_DispatchAndWait, MutexEvb);
REGISTER_BENCHMARK(BM_DispatchAndWait, LockFreeEvb);
#ifdef ENABLE_FOLLY
REGISTER_BENCHMARK(BM_DispatchAndWait, FollyEvb);
#endif // ENABLE_FOLLY

// --- DispatchInlineFromLoopThread ---
REGISTER_BENCHMARK(BM_DispatchInlineFromLoopThread, MutexEvb);
REGISTER_BENCHMARK(BM_DispatchInlineFromLoopThread, LockFreeEvb);
#ifdef ENABLE_FOLLY
REGISTER_BENCHMARK(BM_DispatchInlineFromLoopThread, FollyEvb);
#endif // ENABLE_FOLLY

// --- ConcurrentDispatchBounded ---
REGISTER_BENCHMARK(BM_ConcurrentDispatchBounded, MutexEvb)
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32);
REGISTER_BENCHMARK(BM_ConcurrentDispatchBounded, LockFreeEvb)
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32);
#ifdef ENABLE_FOLLY
REGISTER_BENCHMARK(BM_ConcurrentDispatchBounded, FollyEvb)
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32);
#endif // ENABLE_FOLLY

// --- DispatchBatch ---
REGISTER_BENCHMARK(BM_DispatchBatch, MutexEvb)->Arg(100)->Arg(1000);
REGISTER_BENCHMARK(BM_DispatchBatch, LockFreeEvb)->Arg(100)->Arg(1000);
#ifdef ENABLE_FOLLY
REGISTER_BENCHMARK(BM_DispatchBatch, FollyEvb)->Arg(100)->Arg(1000);
#endif // ENABLE_FOLLY

// --- PingPong ---
REGISTER_BENCHMARK(BM_PingPong, MutexEvb);
REGISTER_BENCHMARK(BM_PingPong, LockFreeEvb);
#ifdef ENABLE_FOLLY
REGISTER_BENCHMARK(BM_PingPong, FollyEvb);
#endif // ENABLE_FOLLY
// NOLINTEND(facebook-avoid-non-const-global-variables)

BENCHMARK_MAIN();
