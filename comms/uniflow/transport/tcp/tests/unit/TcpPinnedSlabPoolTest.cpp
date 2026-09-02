// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpPinnedSlabPool.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"

namespace uniflow {

// The pool's job is to make the reader's staging copy non-blocking (pinned
// memory) without letting put() take every slab. The two properties that
// matter, and that nothing else in the transport can enforce, are: a reserved
// slab is unreachable from the bulk path, and a bulk acquire is all-or-nothing
// so two of them cannot each hold a partial set and wait for the other.
class TcpPinnedSlabPoolTest : public ::testing::Test {
 protected:
  static constexpr size_t kSlabSize = 128;
  static constexpr size_t kSlabCount = 4;
  static constexpr size_t kReserved = 1;

  void SetUp() override {
    cudaApi_ = std::make_shared<::testing::NiceMock<MockCudaApi>>();
    // Real memory, so a test can prove the slabs are distinct, non-overlapping
    // windows onto one region rather than trusting the arithmetic.
    region_.assign(kSlabSize * kSlabCount, uint8_t{0});
    ON_CALL(*cudaApi_, hostAlloc(::testing::_, ::testing::_))
        .WillByDefault(
            ::testing::Return(
                Result<void*>(static_cast<void*>(region_.data()))));
    ON_CALL(*cudaApi_, hostFree(::testing::_))
        .WillByDefault(::testing::Return(Ok()));
  }

  std::shared_ptr<TcpPinnedSlabPool> makePool(
      size_t slabCount = kSlabCount,
      size_t reserved = kReserved) {
    auto pool =
        TcpPinnedSlabPool::create(cudaApi_, kSlabSize, slabCount, reserved);
    EXPECT_TRUE(pool.hasValue()) << "pool creation should succeed";
    return pool.hasValue() ? pool.value() : nullptr;
  }

  std::shared_ptr<::testing::NiceMock<MockCudaApi>> cudaApi_;
  std::vector<uint8_t> region_;
};

TEST_F(TcpPinnedSlabPoolTest, SlabsAreDistinctWindowsOntoTheRegion) {
  auto pool = makePool();
  std::set<uint8_t*> seen;
  std::vector<TcpPinnedSlab> held;
  for (size_t i = 0; i < kSlabCount; ++i) {
    auto slab = pool->tryAcquire(/*allowReserved=*/true);
    ASSERT_TRUE(static_cast<bool>(slab)) << "slab " << i << " should be free";
    EXPECT_EQ(slab.capacity(), kSlabSize);
    EXPECT_GE(slab.data(), region_.data());
    EXPECT_LE(slab.data() + slab.capacity(), region_.data() + region_.size());
    EXPECT_TRUE(seen.insert(slab.data()).second)
        << "two live leases handed out the same slab";
    held.push_back(std::move(slab));
  }
  EXPECT_FALSE(static_cast<bool>(pool->tryAcquire(/*allowReserved=*/true)))
      << "the pool is fixed-size; it must not invent a slab past the region";
}

TEST_F(TcpPinnedSlabPoolTest, ADestroyedLeaseReturnsItsSlab) {
  auto pool = makePool(/*slabCount=*/1, /*reserved=*/0);
  uint8_t* first = nullptr;
  {
    auto slab = pool->tryAcquire(/*allowReserved=*/false);
    ASSERT_TRUE(static_cast<bool>(slab));
    first = slab.data();
    EXPECT_FALSE(static_cast<bool>(pool->tryAcquire(/*allowReserved=*/true)));
  }
  auto again = pool->tryAcquire(/*allowReserved=*/false);
  ASSERT_TRUE(static_cast<bool>(again));
  EXPECT_EQ(again.data(), first) << "the released slab should be reusable";
}

TEST_F(TcpPinnedSlabPoolTest, TheReserveIsUnreachableFromTheBulkPath) {
  auto pool = makePool();
  // Take everything the unreserved path can reach.
  auto bulk = pool->acquire(kSlabCount - kReserved);
  ASSERT_TRUE(bulk.hasValue());
  EXPECT_EQ(bulk.value().size(), kSlabCount - kReserved);

  EXPECT_FALSE(static_cast<bool>(pool->tryAcquire(/*allowReserved=*/false)))
      << "put() must see the pool as exhausted while only the reserve is left";
  auto readerSlab = pool->tryAcquire(/*allowReserved=*/true);
  EXPECT_TRUE(static_cast<bool>(readerSlab))
      << "the responder must still get a slab with every put slab occupied; "
         "that is what the reserve is for";
}

TEST_F(TcpPinnedSlabPoolTest, ABulkAcquireLargerThanTheUnreservedSetIsRefused) {
  auto pool = makePool();
  auto tooMany = pool->acquire(kSlabCount);
  EXPECT_TRUE(tooMany.hasError())
      << "a request that can never be satisfied must fail rather than park the "
         "caller forever";
  EXPECT_EQ(tooMany.error().code(), ErrCode::InvalidArgument);
}

TEST_F(TcpPinnedSlabPoolTest, ABulkAcquireIsAllOrNothing) {
  auto pool = makePool();
  const size_t bulkCount = kSlabCount - kReserved;
  auto held = pool->acquire(bulkCount);
  ASSERT_TRUE(held.hasValue());

  std::atomic<bool> completed{false};
  std::thread waiter([&]() {
    auto second = pool->acquire(bulkCount);
    EXPECT_TRUE(second.hasValue());
    EXPECT_EQ(second.value().size(), bulkCount);
    completed.store(true, std::memory_order_release);
  });

  // Hand back one short of the full set. A waiter that took what it could get
  // would have made partial progress here, and two such waiters would deadlock;
  // this one must still be holding nothing.
  held.value().pop_back();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(completed.load(std::memory_order_acquire))
      << "a bulk acquire must not proceed on a partial set";

  held.value().clear();
  waiter.join();
  EXPECT_TRUE(completed.load(std::memory_order_acquire));
}

TEST_F(TcpPinnedSlabPoolTest, CloseWakesWaitersAndRefusesNewLeases) {
  auto pool = makePool();
  auto held = pool->acquire(kSlabCount - kReserved);
  ASSERT_TRUE(held.hasValue());

  std::atomic<bool> refused{false};
  std::thread waiter([&]() {
    auto blocked = pool->acquire(kSlabCount - kReserved);
    EXPECT_TRUE(blocked.hasError());
    refused.store(true, std::memory_order_release);
  });

  // Nothing is released, so only close() can end that wait -- which is what
  // stops a caller parked here from outliving the transport at shutdown.
  pool->close();
  waiter.join();
  EXPECT_TRUE(refused.load(std::memory_order_acquire));
  EXPECT_FALSE(static_cast<bool>(pool->tryAcquire(/*allowReserved=*/true)));
}

TEST_F(TcpPinnedSlabPoolTest, CreateRejectsAnUnusableConfiguration) {
  EXPECT_TRUE(
      TcpPinnedSlabPool::create(cudaApi_, 0, kSlabCount, kReserved).hasError());
  EXPECT_TRUE(TcpPinnedSlabPool::create(cudaApi_, kSlabSize, 0, 0).hasError());
  EXPECT_TRUE(
      TcpPinnedSlabPool::create(cudaApi_, kSlabSize, kSlabCount, kSlabCount)
          .hasError())
      << "reserving every slab would leave put() unable to make progress";
  EXPECT_TRUE(
      TcpPinnedSlabPool::create(nullptr, kSlabSize, kSlabCount, kReserved)
          .hasError());
}

TEST_F(TcpPinnedSlabPoolTest, CreatePropagatesAnAllocationFailure) {
  ON_CALL(*cudaApi_, hostAlloc(::testing::_, ::testing::_))
      .WillByDefault(
          ::testing::Return(
              Result<void*>(
                  Err(ErrCode::DriverError,
                      "test: out of pinned host "
                      "memory"))));
  auto pool =
      TcpPinnedSlabPool::create(cudaApi_, kSlabSize, kSlabCount, kReserved);
  ASSERT_TRUE(pool.hasError())
      << "a failed pinned allocation must fail one transfer, not throw out of "
         "the transport";
  EXPECT_EQ(pool.error().code(), ErrCode::DriverError);
}

TEST_F(TcpPinnedSlabPoolTest, TheRegionIsFreedExactlyOnce) {
  EXPECT_CALL(*cudaApi_, hostFree(static_cast<void*>(region_.data()))).Times(1);
  {
    auto pool = makePool();
    auto slab = pool->tryAcquire(/*allowReserved=*/true);
    // The lease outliving this scope's shared_ptr keeps the pool alive, so the
    // region is not freed under a copy that may still be running.
    pool.reset();
  }
}

} // namespace uniflow
