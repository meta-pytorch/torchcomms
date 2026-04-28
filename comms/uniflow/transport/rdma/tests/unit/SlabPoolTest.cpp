// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/SlabPool.h"

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/drivers/ibverbs/mock/MockIbvApi.h"

#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace uniflow;
using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

class SlabPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mockCudaApi_ = std::make_shared<testing::NiceMock<MockCudaApi>>();
    mockIbvApi_ = std::make_shared<testing::NiceMock<MockIbvApi>>();

    ON_CALL(*mockCudaApi_, hostAlloc(_, _))
        .WillByDefault(Invoke([](size_t size, unsigned int) -> Result<void*> {
          void* ptr = std::calloc(1, size);
          if (!ptr) {
            return Err(ErrCode::DriverError, "calloc failed");
          }
          return ptr;
        }));
    ON_CALL(*mockCudaApi_, hostFree(_))
        .WillByDefault(Invoke([](void* ptr) -> Status {
          std::free(ptr);
          return Ok();
        }));

    ON_CALL(*mockIbvApi_, regMr(_, _, _, _))
        .WillByDefault(Return(Result<ibv_mr*>(&fakeMr0_)));
    ON_CALL(*mockIbvApi_, deregMr(_)).WillByDefault(Return(Ok()));
  }

  std::shared_ptr<testing::NiceMock<MockCudaApi>> mockCudaApi_;
  std::shared_ptr<testing::NiceMock<MockIbvApi>> mockIbvApi_;

  ibv_pd fakePd0_{};
  ibv_pd fakePd1_{};
  ibv_mr fakeMr0_{};
  ibv_mr fakeMr1_{};

  std::vector<NicResources> makeNics(size_t count = 1) {
    std::vector<NicResources> nics;
    nics.push_back({.pd = &fakePd0_});
    if (count > 1) {
      nics.push_back({.pd = &fakePd1_});
    }
    return nics;
  }
};

TEST_F(SlabPoolTest, Construction_RegistersMrPerNic) {
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd0_, _, _, _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr0_)));
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd1_, _, _, _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr1_)));

  SlabPoolConfig config{.totalSize = 2 * 1024 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics(2));

  EXPECT_EQ(pool.numSlabs(), 4);
  EXPECT_EQ(pool.slabSize(), 512 * 1024);
}

TEST_F(SlabPoolTest, Construction_ZeroSlabSize_Throws) {
  SlabPoolConfig config{.totalSize = 1024, .slabSize = 0};
  EXPECT_THROW(
      SlabPool(config, mockCudaApi_, mockIbvApi_, makeNics()),
      std::invalid_argument);
}

TEST_F(SlabPoolTest, Construction_TotalSizeLessThanSlabSize_Throws) {
  SlabPoolConfig config{.totalSize = 256, .slabSize = 512};
  EXPECT_THROW(
      SlabPool(config, mockCudaApi_, mockIbvApi_, makeNics()),
      std::invalid_argument);
}

TEST_F(SlabPoolTest, Construction_RegMrFails_Throws) {
  EXPECT_CALL(*mockIbvApi_, regMr(_, _, _, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "ibv_reg_mr failed")));

  SlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  EXPECT_THROW(
      SlabPool(config, mockCudaApi_, mockIbvApi_, makeNics()),
      std::runtime_error);
}

TEST_F(SlabPoolTest, AcquireRelease_BasicCycle) {
  SlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());

  auto r1 = pool.acquire();
  ASSERT_TRUE(r1.hasValue());
  EXPECT_NE(r0.value(), r1.value());

  // Pool has 2 slabs, both acquired — next should fail
  auto r2 = pool.acquire();
  EXPECT_TRUE(r2.hasError());

  pool.release(r0.value());
  pool.release(r1.value());

  // After release, acquire should succeed again
  auto r3 = pool.acquire();
  ASSERT_TRUE(r3.hasValue());
}

TEST_F(SlabPoolTest, Acquire_PoolExhausted_ReturnsError) {
  SlabPoolConfig config{.totalSize = 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
  EXPECT_EQ(pool.numSlabs(), 1);

  auto r0 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());

  auto r1 = pool.acquire();
  ASSERT_TRUE(r1.hasError());
  EXPECT_EQ(r1.error().code(), ErrCode::ResourceExhausted);
}

TEST_F(SlabPoolTest, Acquire_ReleaseAndReacquire) {
  SlabPoolConfig config{.totalSize = 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());
  uint16_t idx = r0.value();

  pool.release(idx);

  auto r1 = pool.acquire();
  ASSERT_TRUE(r1.hasValue());
  EXPECT_EQ(r1.value(), idx);
}

TEST_F(SlabPoolTest, SlabPtr_ReturnsDistinctAddresses) {
  SlabPoolConfig config{.totalSize = 2 * 1024 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool.acquire();
  auto r1 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());
  ASSERT_TRUE(r1.hasValue());

  void* p0 = pool.slabPtr(r0.value());
  void* p1 = pool.slabPtr(r1.value());

  EXPECT_NE(p0, nullptr);
  EXPECT_NE(p1, nullptr);
  EXPECT_NE(p0, p1);

  auto diff = std::abs(static_cast<char*>(p1) - static_cast<char*>(p0));
  EXPECT_GE(static_cast<size_t>(diff), pool.slabSize());
}

TEST_F(SlabPoolTest, SlabAddr_MatchesSlabPtr) {
  SlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());
  EXPECT_EQ(
      pool.slabAddr(r0.value()),
      reinterpret_cast<uint64_t>(pool.slabPtr(r0.value())));
}

TEST_F(SlabPoolTest, LkeyRkey_ReturnsMrKeys) {
  fakeMr0_.lkey = 0xAAAA;
  fakeMr0_.rkey = 0xBBBB;
  fakeMr1_.lkey = 0xCCCC;
  fakeMr1_.rkey = 0xDDDD;

  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd0_, _, _, _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr0_)));
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd1_, _, _, _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr1_)));

  SlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics(2));

  EXPECT_EQ(pool.lkey(0), 0xAAAA);
  EXPECT_EQ(pool.rkey(0), 0xBBBB);
  EXPECT_EQ(pool.lkey(1), 0xCCCC);
  EXPECT_EQ(pool.rkey(1), 0xDDDD);
}

TEST_F(SlabPoolTest, SlabMemory_IsReadableWritable) {
  SlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());

  void* ptr = pool.slabPtr(r0.value());
  std::memset(ptr, 0xAB, pool.slabSize());

  auto* bytes = static_cast<uint8_t*>(ptr);
  EXPECT_EQ(bytes[0], 0xAB);
  EXPECT_EQ(bytes[pool.slabSize() - 1], 0xAB);
}

TEST_F(SlabPoolTest, Destruction_DeregistersMrs) {
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd0_, _, _, _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr0_)));
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd1_, _, _, _))
      .WillOnce(Return(Result<ibv_mr*>(&fakeMr1_)));
  EXPECT_CALL(*mockIbvApi_, deregMr(&fakeMr0_)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mockIbvApi_, deregMr(&fakeMr1_)).WillOnce(Return(Ok()));

  SlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  {
    SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics(2));
  }
}

// --- Atomic path tests (numSlabs <= 128) ---

TEST_F(SlabPoolTest, Atomic_AcquireAll64Slabs) {
  SlabPoolConfig config{.totalSize = 64 * 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
  EXPECT_EQ(pool.numSlabs(), 64);

  std::vector<uint16_t> acquired;
  for (int i = 0; i < 64; ++i) {
    auto r = pool.acquire();
    ASSERT_TRUE(r.hasValue()) << "Failed to acquire slab " << i;
    acquired.push_back(r.value());
  }

  // All 64 slabs acquired — next should fail
  EXPECT_TRUE(pool.acquire().hasError());

  // All indices should be unique and in [0, 64)
  std::sort(acquired.begin(), acquired.end());
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(acquired[i], i);
  }

  for (auto idx : acquired) {
    pool.release(idx);
  }

  // All freed — should be able to acquire again
  EXPECT_TRUE(pool.acquire().hasValue());
}

TEST_F(SlabPoolTest, Atomic_AcquireAll128Slabs_CrossesBitmapBoundary) {
  SlabPoolConfig config{.totalSize = 128 * 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
  EXPECT_EQ(pool.numSlabs(), 128);

  std::vector<uint16_t> acquired;
  for (int i = 0; i < 128; ++i) {
    auto r = pool.acquire();
    ASSERT_TRUE(r.hasValue()) << "Failed to acquire slab " << i;
    acquired.push_back(r.value());
  }

  EXPECT_TRUE(pool.acquire().hasError());

  // Verify indices span both low (0-63) and high (64-127)
  std::sort(acquired.begin(), acquired.end());
  for (int i = 0; i < 128; ++i) {
    EXPECT_EQ(acquired[i], i);
  }

  for (auto idx : acquired) {
    pool.release(idx);
  }
  EXPECT_TRUE(pool.acquire().hasValue());
}

TEST_F(SlabPoolTest, Atomic_ReleaseLowAndHighIndependently) {
  SlabPoolConfig config{.totalSize = 128 * 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  // Acquire all 128
  std::vector<uint16_t> acquired;
  for (int i = 0; i < 128; ++i) {
    auto r = pool.acquire();
    ASSERT_TRUE(r.hasValue());
    acquired.push_back(r.value());
  }
  EXPECT_TRUE(pool.acquire().hasError());

  // Release only slabs in the high range (64-127)
  for (auto idx : acquired) {
    if (idx >= 64) {
      pool.release(idx);
    }
  }

  // Should be able to acquire from high range
  auto r = pool.acquire();
  ASSERT_TRUE(r.hasValue());
  EXPECT_GE(r.value(), 64);

  // Release all remaining
  pool.release(r.value());
  for (auto idx : acquired) {
    if (idx < 64) {
      pool.release(idx);
    }
  }
}

TEST_F(SlabPoolTest, Atomic_LowBitmapFullThenHighUsed) {
  SlabPoolConfig config{.totalSize = 65 * 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
  EXPECT_EQ(pool.numSlabs(), 65);

  // Acquire 65 slabs — 64 from low bitmap, 1 from high
  std::vector<uint16_t> acquired;
  for (int i = 0; i < 65; ++i) {
    auto r = pool.acquire();
    ASSERT_TRUE(r.hasValue());
    acquired.push_back(r.value());
  }
  EXPECT_TRUE(pool.acquire().hasError());

  // Verify at least one index >= 64 (from high bitmap)
  bool hasHigh = false;
  for (auto idx : acquired) {
    if (idx >= 64) {
      hasHigh = true;
    }
  }
  EXPECT_TRUE(hasHigh);

  for (auto idx : acquired) {
    pool.release(idx);
  }
}

TEST_F(SlabPoolTest, Atomic_ConcurrentHighContention) {
  SlabPoolConfig config{.totalSize = 4 * 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
  EXPECT_EQ(pool.numSlabs(), 4);

  constexpr int kThreads = 8;
  constexpr int kOpsPerThread = 1000;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto r = pool.acquire();
        if (r.hasValue()) {
          pool.release(r.value());
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  // All 4 slabs should be back
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(pool.acquire().hasValue());
  }
  EXPECT_TRUE(pool.acquire().hasError());
}

// --- Mutex path tests (numSlabs > 128) ---

TEST_F(SlabPoolTest, Mutex_AcquireRelease_BasicCycle) {
  // 129 slabs forces mutex path
  SlabPoolConfig config{.totalSize = 129 * 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
  EXPECT_EQ(pool.numSlabs(), 129);

  auto r0 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());

  auto r1 = pool.acquire();
  ASSERT_TRUE(r1.hasValue());
  EXPECT_NE(r0.value(), r1.value());

  pool.release(r0.value());
  pool.release(r1.value());

  auto r2 = pool.acquire();
  ASSERT_TRUE(r2.hasValue());
}

TEST_F(SlabPoolTest, Mutex_PoolExhausted_ReturnsError) {
  SlabPoolConfig config{.totalSize = 129 * 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  std::vector<uint16_t> acquired;
  for (size_t i = 0; i < 129; ++i) {
    auto r = pool.acquire();
    ASSERT_TRUE(r.hasValue());
    acquired.push_back(r.value());
  }

  auto r = pool.acquire();
  ASSERT_TRUE(r.hasError());
  EXPECT_EQ(r.error().code(), ErrCode::ResourceExhausted);

  for (auto idx : acquired) {
    pool.release(idx);
  }
}

TEST_F(SlabPoolTest, Mutex_ConcurrentAcquireRelease) {
  SlabPoolConfig config{.totalSize = 129 * 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  constexpr int kThreads = 4;
  constexpr int kOpsPerThread = 200;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto r = pool.acquire();
        if (r.hasValue()) {
          pool.release(r.value());
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  // All 129 slabs should be back
  for (size_t i = 0; i < 129; ++i) {
    EXPECT_TRUE(pool.acquire().hasValue());
  }
  EXPECT_TRUE(pool.acquire().hasError());
}

// --- Original concurrent test (exercises whichever path numSlabs selects) ---

TEST_F(SlabPoolTest, ConcurrentAcquireRelease_NoRace) {
  SlabPoolConfig config{.totalSize = 8 * 512 * 1024, .slabSize = 512 * 1024};
  SlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
  EXPECT_EQ(pool.numSlabs(), 8);

  constexpr int kThreads = 4;
  constexpr int kOpsPerThread = 100;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto r = pool.acquire();
        if (r.hasValue()) {
          pool.release(r.value());
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  // All slabs should be back — acquire all 8 to verify
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(pool.acquire().hasValue());
  }
  EXPECT_TRUE(pool.acquire().hasError());
}
