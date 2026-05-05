// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/drivers/ibverbs/mock/MockIbvApi.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace uniflow;
using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

class RdmaSlabPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mockCudaApi_ = std::make_shared<testing::NiceMock<MockCudaApi>>();
    mockIbvApi_ = std::make_shared<testing::NiceMock<MockIbvApi>>();

    ON_CALL(*mockCudaApi_, hostAlloc(_, _))
        .WillByDefault([](size_t size, unsigned int) -> Result<void*> {
          void* ptr = std::calloc(1, size);
          if (!ptr) {
            return Err(ErrCode::DriverError, "calloc failed");
          }
          return ptr;
        });
    ON_CALL(*mockCudaApi_, hostFree(_)).WillByDefault([](void* ptr) -> Status {
      std::free(ptr);
      return Ok();
    });

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

  std::shared_ptr<std::vector<NicResources>> makeNics(size_t count = 1) {
    auto nics = std::make_shared<std::vector<NicResources>>();

    auto setupNic = [this](ibv_device* dev, ibv_context* ctx, ibv_pd* pd) {
      ON_CALL(*mockIbvApi_, openDevice(dev))
          .WillByDefault(Return(Result<ibv_context*>(ctx)));
      ON_CALL(*mockIbvApi_, allocPd(ctx))
          .WillByDefault(Return(Result<ibv_pd*>(pd)));
      ON_CALL(*mockIbvApi_, isDmaBufSupported(pd))
          .WillByDefault(Return(Result<bool>(false)));
      ON_CALL(*mockIbvApi_, queryPort(ctx, 1, _))
          .WillByDefault([](ibv_context*, uint8_t, ibv_port_attr* attr) {
            attr->state = IBV_PORT_ACTIVE;
            attr->lid = 0;
            attr->active_mtu = IBV_MTU_4096;
            attr->link_layer = IBV_LINK_LAYER_ETHERNET;
            return Ok();
          });
      ON_CALL(*mockIbvApi_, queryGid(ctx, 1, _, _)).WillByDefault(Return(Ok()));
      ON_CALL(*mockIbvApi_, queryDevice(ctx, _))
          .WillByDefault([](ibv_context*, ibv_device_attr* attr) {
            attr->phys_port_cnt = 1;
            return Ok();
          });
      ON_CALL(*mockIbvApi_, deallocPd(pd)).WillByDefault(Return(Ok()));
      ON_CALL(*mockIbvApi_, closeDevice(ctx)).WillByDefault(Return(Ok()));
    };

    nics->reserve(count);
    setupNic(&fakeDev0_, &fakeCtx0_, &fakePd0_);
    nics->emplace_back(&fakeDev0_, mockIbvApi_, 3, uint8_t{1});
    if (count > 1) {
      setupNic(&fakeDev1_, &fakeCtx1_, &fakePd1_);
      nics->emplace_back(&fakeDev1_, mockIbvApi_, 3, uint8_t{1});
    }
    return nics;
  }

  ibv_device fakeDev0_{};
  ibv_device fakeDev1_{};
  ibv_context fakeCtx0_{.device = &fakeDev0_};
  ibv_context fakeCtx1_{.device = &fakeDev1_};
};

TEST_F(RdmaSlabPoolTest, Construction_RegistersMrPerNic) {
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd0_, _, _, _))
      .WillRepeatedly(Return(Result<ibv_mr*>(&fakeMr0_)));
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd1_, _, _, _))
      .WillRepeatedly(Return(Result<ibv_mr*>(&fakeMr1_)));

  RdmaSlabPoolConfig config{
      .totalSize = 2 * 1024 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics(2));

  EXPECT_EQ(pool.numSlabs(), 4);
  EXPECT_EQ(pool.slabSize(), 512 * 1024);
}

TEST_F(RdmaSlabPoolTest, Construction_ZeroSlabSize_Throws) {
  RdmaSlabPoolConfig config{.totalSize = 1024, .slabSize = 0};
  EXPECT_THROW(
      RdmaSlabPool(config, mockCudaApi_, mockIbvApi_, makeNics()),
      std::invalid_argument);
}

TEST_F(RdmaSlabPoolTest, Construction_TotalSizeLessThanSlabSize_Throws) {
  RdmaSlabPoolConfig config{.totalSize = 256, .slabSize = 512};
  EXPECT_THROW(
      RdmaSlabPool(config, mockCudaApi_, mockIbvApi_, makeNics()),
      std::invalid_argument);
}

TEST_F(RdmaSlabPoolTest, Construction_RegMrFails_Throws) {
  EXPECT_CALL(*mockIbvApi_, regMr(_, _, _, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "ibv_reg_mr failed")));

  RdmaSlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  EXPECT_THROW(
      RdmaSlabPool(config, mockCudaApi_, mockIbvApi_, makeNics()),
      std::runtime_error);
}

TEST_F(RdmaSlabPoolTest, AcquireRelease_BasicCycle) {
  RdmaSlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

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

TEST_F(RdmaSlabPoolTest, Acquire_PoolExhausted_ReturnsError) {
  RdmaSlabPoolConfig config{.totalSize = 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
  EXPECT_EQ(pool.numSlabs(), 1);

  auto r0 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());

  auto r1 = pool.acquire();
  ASSERT_TRUE(r1.hasError());
  EXPECT_EQ(r1.error().code(), ErrCode::ResourceExhausted);
}

TEST_F(RdmaSlabPoolTest, Acquire_ReleaseAndReacquire) {
  RdmaSlabPoolConfig config{.totalSize = 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());
  uint16_t idx = r0.value();

  pool.release(idx);

  auto r1 = pool.acquire();
  ASSERT_TRUE(r1.hasValue());
  EXPECT_EQ(r1.value(), idx);
}

TEST_F(RdmaSlabPoolTest, SlabPtr_ReturnsDistinctAddresses) {
  RdmaSlabPoolConfig config{
      .totalSize = 2 * 1024 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

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

TEST_F(RdmaSlabPoolTest, SlabAddr_MatchesSlabPtr) {
  RdmaSlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());
  EXPECT_EQ(
      pool.slabAddr(r0.value()),
      reinterpret_cast<uint64_t>(pool.slabPtr(r0.value())));
}

TEST_F(RdmaSlabPoolTest, LkeyRkey_ReturnsMrKeys) {
  fakeMr0_.lkey = 0xAAAA;
  fakeMr0_.rkey = 0xBBBB;
  fakeMr1_.lkey = 0xCCCC;
  fakeMr1_.rkey = 0xDDDD;

  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd0_, _, _, _))
      .WillRepeatedly(Return(Result<ibv_mr*>(&fakeMr0_)));
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd1_, _, _, _))
      .WillRepeatedly(Return(Result<ibv_mr*>(&fakeMr1_)));

  RdmaSlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics(2));

  EXPECT_EQ(pool.slabLkey(0), 0xAAAA);
  EXPECT_EQ(pool.slabRkey(0), 0xBBBB);
  EXPECT_EQ(pool.slabLkey(1), 0xCCCC);
  EXPECT_EQ(pool.slabRkey(1), 0xDDDD);
}

TEST_F(RdmaSlabPoolTest, SlabMemory_IsReadableWritable) {
  RdmaSlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool.acquire();
  ASSERT_TRUE(r0.hasValue());

  void* ptr = pool.slabPtr(r0.value());
  std::memset(ptr, 0xAB, pool.slabSize());

  auto* bytes = static_cast<uint8_t*>(ptr);
  EXPECT_EQ(bytes[0], 0xAB);
  EXPECT_EQ(bytes[pool.slabSize() - 1], 0xAB);
}

TEST_F(RdmaSlabPoolTest, Destruction_DeregistersMrs) {
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd0_, _, _, _))
      .WillRepeatedly(Return(Result<ibv_mr*>(&fakeMr0_)));
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd1_, _, _, _))
      .WillRepeatedly(Return(Result<ibv_mr*>(&fakeMr1_)));
  EXPECT_CALL(*mockIbvApi_, deregMr(&fakeMr0_)).WillRepeatedly(Return(Ok()));
  EXPECT_CALL(*mockIbvApi_, deregMr(&fakeMr1_)).WillRepeatedly(Return(Ok()));

  RdmaSlabPoolConfig config{.totalSize = 1024 * 1024, .slabSize = 512 * 1024};
  {
    RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics(2));
  }
}

// --- Atomic path tests (numSlabs <= 128) ---

TEST_F(RdmaSlabPoolTest, Atomic_AcquireAll64Slabs) {
  RdmaSlabPoolConfig config{
      .totalSize = 64 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
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

TEST_F(RdmaSlabPoolTest, Atomic_AcquireAll128Slabs_CrossesBitmapBoundary) {
  RdmaSlabPoolConfig config{
      .totalSize = 128 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
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

TEST_F(RdmaSlabPoolTest, Atomic_ReleaseLowAndHighIndependently) {
  RdmaSlabPoolConfig config{
      .totalSize = 128 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

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

TEST_F(RdmaSlabPoolTest, Atomic_LowBitmapFullThenHighUsed) {
  RdmaSlabPoolConfig config{
      .totalSize = 65 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
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

TEST_F(RdmaSlabPoolTest, Atomic_ConcurrentHighContention) {
  RdmaSlabPoolConfig config{
      .totalSize = 4 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
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

TEST_F(RdmaSlabPoolTest, Mutex_AcquireRelease_BasicCycle) {
  // 129 slabs forces mutex path
  RdmaSlabPoolConfig config{
      .totalSize = 129 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
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

TEST_F(RdmaSlabPoolTest, Mutex_PoolExhausted_ReturnsError) {
  RdmaSlabPoolConfig config{
      .totalSize = 129 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

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

TEST_F(RdmaSlabPoolTest, Mutex_ConcurrentAcquireRelease) {
  RdmaSlabPoolConfig config{
      .totalSize = 129 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

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

// --- Batch acquire/release tests (atomic path) ---

TEST_F(RdmaSlabPoolTest, Atomic_BatchAcquire_Success) {
  RdmaSlabPoolConfig config{
      .totalSize = 8 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool.acquire(4);
  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(r.value().size(), 4);

  // All indices should be unique
  auto& slabs = r.value();
  std::set<uint16_t> unique(slabs.begin(), slabs.end());
  EXPECT_EQ(unique.size(), 4);

  // Should still be able to acquire 4 more
  auto r2 = pool.acquire(4);
  ASSERT_TRUE(r2.hasValue());
  EXPECT_EQ(r2.value().size(), 4);

  // Pool exhausted
  EXPECT_TRUE(pool.acquire(1).hasError());

  pool.release(slabs);
  pool.release(r2.value());
}

TEST_F(RdmaSlabPoolTest, Atomic_BatchAcquire_InsufficientSlabs) {
  RdmaSlabPoolConfig config{
      .totalSize = 2 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool.acquire(4);
  ASSERT_TRUE(r.hasError());
  EXPECT_EQ(r.error().code(), ErrCode::ResourceExhausted);

  // All slabs should still be free after rollback
  auto r2 = pool.acquire(2);
  ASSERT_TRUE(r2.hasValue());
  EXPECT_EQ(r2.value().size(), 2);
  pool.release(r2.value());
}

TEST_F(RdmaSlabPoolTest, Atomic_BatchRelease) {
  RdmaSlabPoolConfig config{
      .totalSize = 4 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool.acquire(4);
  ASSERT_TRUE(r.hasValue());
  EXPECT_TRUE(pool.acquire().hasError());

  pool.release(r.value());

  // All 4 should be available again
  auto r2 = pool.acquire(4);
  ASSERT_TRUE(r2.hasValue());
  pool.release(r2.value());
}

TEST_F(RdmaSlabPoolTest, Atomic_BatchAcquire_CrossesBitmapBoundary) {
  RdmaSlabPoolConfig config{
      .totalSize = 128 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  // Acquire all 128 via batch
  auto r = pool.acquire(128);
  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(r.value().size(), 128);

  std::set<uint16_t> unique(r.value().begin(), r.value().end());
  EXPECT_EQ(unique.size(), 128);

  EXPECT_TRUE(pool.acquire(1).hasError());

  // Batch release all
  pool.release(r.value());
  EXPECT_TRUE(pool.acquire(1).hasValue());
}

TEST_F(RdmaSlabPoolTest, Atomic_BatchRelease_MixedLowHigh) {
  RdmaSlabPoolConfig config{
      .totalSize = 128 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool.acquire(128);
  ASSERT_TRUE(r.hasValue());

  // Release a mix of low and high indices in one call
  std::vector<uint16_t> mixed;
  for (auto idx : r.value()) {
    if (idx == 0 || idx == 63 || idx == 64 || idx == 127) {
      mixed.push_back(idx);
    }
  }
  pool.release(mixed);

  // Should be able to acquire exactly those back
  auto r2 = pool.acquire(static_cast<uint32_t>(mixed.size()));
  ASSERT_TRUE(r2.hasValue());
  EXPECT_EQ(r2.value().size(), mixed.size());

  pool.release(r2.value());
  // Release remaining
  for (auto idx : r.value()) {
    bool wasMixed = false;
    for (auto m : mixed) {
      if (idx == m) {
        wasMixed = true;
        break;
      }
    }
    if (!wasMixed) {
      pool.release(idx);
    }
  }
}

TEST_F(RdmaSlabPoolTest, Atomic_BatchAcquireRelease_Concurrent) {
  RdmaSlabPoolConfig config{
      .totalSize = 16 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  constexpr int kThreads = 4;
  constexpr int kOpsPerThread = 200;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto r = pool.acquire(2);
        if (r.hasValue()) {
          pool.release(r.value());
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  // All 16 slabs should be back
  auto r = pool.acquire(16);
  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(r.value().size(), 16);
  EXPECT_TRUE(pool.acquire(1).hasError());
  pool.release(r.value());
}

// --- Batch acquire/release tests (mutex path) ---

TEST_F(RdmaSlabPoolTest, Mutex_BatchAcquire_Success) {
  RdmaSlabPoolConfig config{
      .totalSize = 129 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool.acquire(4);
  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(r.value().size(), 4);

  std::set<uint16_t> unique(r.value().begin(), r.value().end());
  EXPECT_EQ(unique.size(), 4);

  pool.release(r.value());
}

TEST_F(RdmaSlabPoolTest, Mutex_BatchAcquire_InsufficientSlabs) {
  RdmaSlabPoolConfig config{
      .totalSize = 129 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  // Acquire all but 2
  auto r1 = pool.acquire(127);
  ASSERT_TRUE(r1.hasValue());

  // Try to acquire 4 more — only 2 available
  auto r2 = pool.acquire(4);
  ASSERT_TRUE(r2.hasError());
  EXPECT_EQ(r2.error().code(), ErrCode::ResourceExhausted);

  // The 2 remaining should still be free (no partial allocation)
  auto r3 = pool.acquire(2);
  ASSERT_TRUE(r3.hasValue());
  EXPECT_EQ(r3.value().size(), 2);

  pool.release(r1.value());
  pool.release(r3.value());
}

TEST_F(RdmaSlabPoolTest, Mutex_BatchRelease) {
  RdmaSlabPoolConfig config{
      .totalSize = 129 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool.acquire(10);
  ASSERT_TRUE(r.hasValue());

  pool.release(r.value());

  // All 10 should be available again
  auto r2 = pool.acquire(10);
  ASSERT_TRUE(r2.hasValue());
  EXPECT_EQ(r2.value().size(), 10);
  pool.release(r2.value());
}

TEST_F(RdmaSlabPoolTest, Mutex_BatchAcquireRelease_Concurrent) {
  RdmaSlabPoolConfig config{
      .totalSize = 129 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());

  constexpr int kThreads = 4;
  constexpr int kOpsPerThread = 100;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto r = pool.acquire(3);
        if (r.hasValue()) {
          pool.release(r.value());
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  auto r = pool.acquire(129);
  ASSERT_TRUE(r.hasValue());
  EXPECT_TRUE(pool.acquire(1).hasError());
  pool.release(r.value());
}

// --- Original concurrent test (exercises whichever path numSlabs selects) ---

TEST_F(RdmaSlabPoolTest, ConcurrentAcquireRelease_NoRace) {
  RdmaSlabPoolConfig config{
      .totalSize = 8 * 512 * 1024, .slabSize = 512 * 1024};
  RdmaSlabPool pool(config, mockCudaApi_, mockIbvApi_, makeNics());
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
