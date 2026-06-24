// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

#include "comms/ctran/backends/ib/CtranIbSingleton.h"
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual

class CqPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ncclCvarInit();
    ctran::logging::initCtranLogging(true);
    singleton_ = CtranIbSingleton::getInstance();
    ASSERT_NE(singleton_, nullptr);
    ASSERT_FALSE(singleton_->ibvDevices.empty());
    devIdx_ = 0;
    auto devAttr = singleton_->ibvDevices[devIdx_].queryDevice();
    ASSERT_TRUE(devAttr) << devAttr.error().errStr;
    maxCqe_ = devAttr->max_cqe;
    origPoolEnable_ = NCCL_CTRAN_IB_CQ_POOL_ENABLE;
  }

  // TearDown only restores the CVAR. Pool cleanup is intentionally omitted:
  // LIFO ordering guarantees each test controls its own pool top, so
  // residual CQs from prior tests do not affect subsequent tests.
  void TearDown() override {
    NCCL_CTRAN_IB_CQ_POOL_ENABLE = origPoolEnable_;
  }

  // Helper: create a fresh CQ directly (bypassing pool).
  ibverbx::IbvCq createCq(int devIdx) {
    auto cq =
        singleton_->ibvDevices[devIdx].createCq(maxCqe_, nullptr, nullptr, 0);
    EXPECT_TRUE(cq) << cq.error().errStr;
    if (!cq) {
      return ibverbx::IbvCq{};
    }
    return std::move(*cq);
  }

  ibverbx::IbvCq createCq() {
    return createCq(devIdx_);
  }

  std::shared_ptr<CtranIbSingleton> singleton_;
  int devIdx_;
  int maxCqe_;
  bool origPoolEnable_;
};

// checkoutCq always returns a valid, non-null CQ regardless of pool state.
TEST_F(CqPoolTest, Checkout_ReturnsValidCq) {
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = true;
  auto cq = singleton_->checkoutCq(devIdx_, maxCqe_);
  ASSERT_TRUE(cq) << cq.error().errStr;
  EXPECT_NE(cq->cq(), nullptr);
}

// Pool hit: checkin a CQ, checkout returns same ibv_cq* pointer.
TEST_F(CqPoolTest, CheckinThenCheckout_ReturnsSameCq) {
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = true;

  auto cq = createCq();
  ASSERT_NE(cq.cq(), nullptr);
  auto* rawPtr = cq.cq();

  singleton_->checkinCq(devIdx_, std::move(cq));

  auto checkedOut = singleton_->checkoutCq(devIdx_, maxCqe_);
  ASSERT_TRUE(checkedOut) << checkedOut.error().errStr;
  EXPECT_EQ(checkedOut->cq(), rawPtr);
}

// LIFO ordering: 2 CQs checked in, checked out in reverse order.
TEST_F(CqPoolTest, MultipleCheckinCheckout_LIFOOrder) {
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = true;

  auto cq1 = createCq();
  auto cq2 = createCq();
  ASSERT_NE(cq1.cq(), nullptr);
  ASSERT_NE(cq2.cq(), nullptr);

  auto* ptr1 = cq1.cq();
  auto* ptr2 = cq2.cq();

  singleton_->checkinCq(devIdx_, std::move(cq1));
  singleton_->checkinCq(devIdx_, std::move(cq2));

  // LIFO: cq2 was pushed last, should come out first
  auto out1 = singleton_->checkoutCq(devIdx_, maxCqe_);
  ASSERT_TRUE(out1) << out1.error().errStr;
  EXPECT_EQ(out1->cq(), ptr2);

  auto out2 = singleton_->checkoutCq(devIdx_, maxCqe_);
  ASSERT_TRUE(out2) << out2.error().errStr;
  EXPECT_EQ(out2->cq(), ptr1);
}

// CQs keyed by singletonDevIdx don't leak across device indices.
TEST_F(CqPoolTest, MultiDeviceIsolation) {
  if (singleton_->ibvDevices.size() < 2) {
    GTEST_SKIP() << "Need at least 2 IB devices for multi-device test";
  }
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = true;

  // Create and checkin a CQ for device 0
  auto cq0 = createCq(0);
  ASSERT_NE(cq0.cq(), nullptr);
  auto* ptr0 = cq0.cq();
  singleton_->checkinCq(0, std::move(cq0));

  // Checkout from device 1 — must NOT get the device-0 CQ
  auto devAttr1 = singleton_->ibvDevices[1].queryDevice();
  ASSERT_TRUE(devAttr1) << devAttr1.error().errStr;

  auto out1 = singleton_->checkoutCq(1, devAttr1->max_cqe);
  ASSERT_TRUE(out1) << out1.error().errStr;
  EXPECT_NE(out1->cq(), ptr0);
}

// CQ survives pool round-trip and remains functional.
TEST_F(CqPoolTest, CqUsableAfterRoundTrip) {
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = true;

  auto cq = createCq();
  ASSERT_NE(cq.cq(), nullptr);

  singleton_->checkinCq(devIdx_, std::move(cq));

  auto checkedOut = singleton_->checkoutCq(devIdx_, maxCqe_);
  ASSERT_TRUE(checkedOut) << checkedOut.error().errStr;
  ASSERT_NE(checkedOut->cq(), nullptr);

  // pollCq on an idle CQ should succeed with an empty result
  auto result = checkedOut->pollCq(1);
  ASSERT_TRUE(result) << result.error().errStr;
  EXPECT_TRUE(result->empty());
}

// Pool-disabled creation path still returns a valid CQ.
TEST_F(CqPoolTest, PoolDisabled_CheckoutCreatesValidCq) {
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = false;

  auto cq = singleton_->checkoutCq(devIdx_, maxCqe_);
  ASSERT_TRUE(cq) << cq.error().errStr;
  EXPECT_NE(cq->cq(), nullptr);
}

// With pool disabled, checked-in CQs are destroyed, not pooled.
TEST_F(CqPoolTest, PoolDisabled_CheckinDoesNotPool) {
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = true;

  // Checkin a marker CQ with pool enabled — it sits at LIFO top
  auto marker = createCq();
  ASSERT_NE(marker.cq(), nullptr);
  auto* markerPtr = marker.cq();
  singleton_->checkinCq(devIdx_, std::move(marker));

  // Disable pool, create and checkin another CQ — should be destroyed
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = false;
  auto discarded = createCq();
  ASSERT_NE(discarded.cq(), nullptr);
  singleton_->checkinCq(devIdx_, std::move(discarded));

  // Re-enable pool, checkout — should get the marker, proving the
  // disabled-path CQ was not added to the pool
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = true;
  auto out = singleton_->checkoutCq(devIdx_, maxCqe_);
  ASSERT_TRUE(out) << out.error().errStr;
  EXPECT_EQ(out->cq(), markerPtr);
}

// Thread safety under contention: concurrent checkout/checkin cycles.
TEST_F(CqPoolTest, ConcurrentCheckoutCheckin_ThreadSafe) {
  NCCL_CTRAN_IB_CQ_POOL_ENABLE = true;

  constexpr int kNumThreads = 4;
  constexpr int kIterations = 10;

  // Pre-seed pool so concurrent checkouts get pool hits, avoiding
  // ~183ms ibv_create_cq per miss
  for (int i = 0; i < kNumThreads; i++) {
    singleton_->checkinCq(devIdx_, createCq());
  }

  std::atomic<int> errors{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; t++) {
    threads.emplace_back([&]() {
      for (int i = 0; i < kIterations; i++) {
        auto cq = singleton_->checkoutCq(devIdx_, maxCqe_);
        if (!cq || cq->cq() == nullptr) {
          errors.fetch_add(1);
          continue;
        }
        singleton_->checkinCq(devIdx_, std::move(*cq));
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(errors.load(), 0);
}
