// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <benchmark/benchmark.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>

#include "comms/ctran/backends/ib/CtranIbSingleton.h"
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/utils/cvars/nccl_cvars.h"

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

namespace {

struct CqPoolBenchFixture {
  std::shared_ptr<CtranIbSingleton> singleton;
  int devIdx = 0;
  int maxCqe = 0;
  bool initialized = false;

  void ensureInit() {
    if (initialized) {
      return;
    }
    initialized = true;
    singleton = CtranIbSingleton::getInstance();
    if (!singleton || singleton->ibvDevices.empty()) {
      return;
    }
    auto devAttr = singleton->ibvDevices[devIdx].queryDevice();
    if (devAttr) {
      maxCqe = devAttr->max_cqe;
    }
    if (maxCqe <= 0) {
      singleton = nullptr;
    }
  }
};

CqPoolBenchFixture& getFixture() {
  static CqPoolBenchFixture fixture;
  fixture.ensureInit();
  return fixture;
}

void BM_CqCreate_ColdPath(benchmark::State& state) {
  auto& fix = getFixture();
  if (!fix.singleton || fix.singleton->ibvDevices.empty()) {
    state.SkipWithError("No IB hardware available");
    return;
  }

  auto origPoolEnable = NCCL_CTRAN_IB_CQ_POOL_ENABLE;
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = false;

  for (auto _ : state) {
    auto cq = fix.singleton->checkoutCq(fix.devIdx, fix.maxCqe);
    if (!cq) {
      state.SkipWithError("checkoutCq failed");
      break;
    }
    // IbvCq destructor calls ibv_destroy_cq at end of iteration
  }

  NCCL_CTRAN_IB_CQ_POOL_ENABLE = origPoolEnable;
}

void BM_CqCheckout_WarmPath(benchmark::State& state) {
  auto& fix = getFixture();
  if (!fix.singleton || fix.singleton->ibvDevices.empty()) {
    state.SkipWithError("No IB hardware available");
    return;
  }

  auto origPoolEnable = NCCL_CTRAN_IB_CQ_POOL_ENABLE;
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = true;

  // Seed pool with 1 CQ
  auto seedCq = fix.singleton->ibvDevices[fix.devIdx].createCq(
      fix.maxCqe, nullptr, nullptr, 0);
  if (!seedCq) {
    state.SkipWithError("Failed to create seed CQ");
    NCCL_CTRAN_IB_CQ_POOL_ENABLE = origPoolEnable;
    return;
  }
  fix.singleton->checkinCq(fix.devIdx, std::move(*seedCq));

  for (auto _ : state) {
    auto cq = fix.singleton->checkoutCq(fix.devIdx, fix.maxCqe);
    if (!cq) {
      state.SkipWithError("checkoutCq failed");
      break;
    }
    fix.singleton->checkinCq(fix.devIdx, std::move(*cq));
  }

  NCCL_CTRAN_IB_CQ_POOL_ENABLE = origPoolEnable;
}

} // namespace

static auto* registered_cold_path =
    benchmark::RegisterBenchmark("BM_CqCreate_ColdPath", BM_CqCreate_ColdPath)
        ->UseRealTime()
        ->Unit(benchmark::kMillisecond)
        ->Iterations(10);

static auto* registered_warm_path = benchmark::RegisterBenchmark(
                                        "BM_CqCheckout_WarmPath",
                                        BM_CqCheckout_WarmPath)
                                        ->UseRealTime()
                                        ->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  folly::init(&argc, &argv);
  ncclCvarInit();
  ::benchmark::RunSpecifiedBenchmarks();
  getFixture().singleton.reset();
  return 0;
}
