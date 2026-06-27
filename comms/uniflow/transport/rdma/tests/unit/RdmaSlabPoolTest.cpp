// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/drivers/ibverbs/mock/MockIbvApi.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace uniflow;
using ::testing::_;
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
    ON_CALL(*mockCudaApi_, hostGetDevicePointer(_))
        .WillByDefault([](void* hostPtr) -> Result<void*> { return hostPtr; });

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

  static constexpr uint8_t kTestGidIndex = 3;
  static constexpr uint8_t kMockGidTblLen = kTestGidIndex + 1;

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
            attr->gid_tbl_len = kMockGidTblLen;
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
    nics->emplace_back(&fakeDev0_, mockIbvApi_, -1, kTestGidIndex, uint8_t{1});
    if (count > 1) {
      setupNic(&fakeDev1_, &fakeCtx1_, &fakePd1_);
      nics->emplace_back(
          &fakeDev1_, mockIbvApi_, -1, kTestGidIndex, uint8_t{1});
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

  RdmaSlabPoolConfig config{.slabNum = 4, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics(2));

  EXPECT_EQ(pool->numSlabs(), 4);
  EXPECT_EQ(pool->slabSize(), 512 * 1024);
}

TEST_F(RdmaSlabPoolTest, Construction_ZeroSlabSize_Throws) {
  RdmaSlabPoolConfig config{.slabNum = 1, .slabSize = 0};
  EXPECT_THROW(
      std::make_shared<RdmaSlabPool>(
          config, mockCudaApi_, mockIbvApi_, makeNics()),
      std::invalid_argument);
}

TEST_F(RdmaSlabPoolTest, Construction_ZeroSlabNum_Throws) {
  RdmaSlabPoolConfig config{.slabNum = 0, .slabSize = 512};
  EXPECT_THROW(
      std::make_shared<RdmaSlabPool>(
          config, mockCudaApi_, mockIbvApi_, makeNics()),
      std::invalid_argument);
}

TEST_F(RdmaSlabPoolTest, Construction_RegMrFails_Throws) {
  EXPECT_CALL(*mockIbvApi_, regMr(_, _, _, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "ibv_reg_mr failed")));

  RdmaSlabPoolConfig config{.slabNum = 2, .slabSize = 512 * 1024};
  EXPECT_THROW(
      std::make_shared<RdmaSlabPool>(
          config, mockCudaApi_, mockIbvApi_, makeNics()),
      std::runtime_error);
}

TEST_F(RdmaSlabPoolTest, AcquireRelease_BasicCycle) {
  RdmaSlabPoolConfig config{.slabNum = 2, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  {
    auto r0 = pool->acquire();
    ASSERT_TRUE(r0.hasValue());

    auto r1 = pool->acquire();
    ASSERT_TRUE(r1.hasValue());
    EXPECT_NE(r0.value().index(), r1.value().index());

    EXPECT_TRUE(pool->acquire().hasError());
  }

  EXPECT_TRUE(pool->acquire().hasValue());
}

TEST_F(RdmaSlabPoolTest, Acquire_PoolExhausted_ReturnsError) {
  RdmaSlabPoolConfig config{.slabNum = 1, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());
  EXPECT_EQ(pool->numSlabs(), 1);

  auto r0 = pool->acquire();
  ASSERT_TRUE(r0.hasValue());

  auto r1 = pool->acquire();
  ASSERT_TRUE(r1.hasError());
  EXPECT_EQ(r1.error().code(), ErrCode::ResourceExhausted);
}

TEST_F(RdmaSlabPoolTest, Acquire_ReleaseAndReacquire) {
  RdmaSlabPoolConfig config{.slabNum = 1, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  uint16_t idx;
  {
    auto r0 = pool->acquire();
    ASSERT_TRUE(r0.hasValue());
    idx = r0.value().index();
  }

  auto r1 = pool->acquire();
  ASSERT_TRUE(r1.hasValue());
  EXPECT_EQ(r1.value().index(), idx);
}

TEST_F(RdmaSlabPoolTest, RdmaSlab_AutoReleasesOnDestruction) {
  RdmaSlabPoolConfig config{.slabNum = 1, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  {
    auto r = pool->acquire();
    ASSERT_TRUE(r.hasValue());
    EXPECT_TRUE(pool->acquire().hasError());
  }
  EXPECT_TRUE(pool->acquire().hasValue());
}

TEST_F(RdmaSlabPoolTest, RdmaSlab_MoveTransfersOwnership) {
  RdmaSlabPoolConfig config{.slabNum = 1, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool->acquire();
  ASSERT_TRUE(r.hasValue());
  uint16_t idx = r.value().index();

  {
    RdmaSlab moved = std::move(r.value());
    EXPECT_EQ(moved.index(), idx);
    EXPECT_TRUE(pool->acquire().hasError());
  }
  EXPECT_TRUE(pool->acquire().hasValue());
}

TEST_F(RdmaSlabPoolTest, RdmaSlab_KeepsPoolAlive) {
  RdmaSlabPoolConfig config{.slabNum = 1, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool->acquire();
  ASSERT_TRUE(r.hasValue());

  auto slab = std::move(r.value());
  pool.reset();
}

TEST_F(RdmaSlabPoolTest, SlabPtr_ReturnsDistinctAddresses) {
  RdmaSlabPoolConfig config{.slabNum = 4, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool->acquire();
  auto r1 = pool->acquire();
  ASSERT_TRUE(r0.hasValue());
  ASSERT_TRUE(r1.hasValue());

  void* p0 = pool->slabPtr(r0.value().index());
  void* p1 = pool->slabPtr(r1.value().index());

  EXPECT_NE(p0, nullptr);
  EXPECT_NE(p1, nullptr);
  EXPECT_NE(p0, p1);

  auto diff = std::abs(static_cast<char*>(p1) - static_cast<char*>(p0));
  EXPECT_GE(static_cast<size_t>(diff), pool->slabSize());
}

TEST_F(RdmaSlabPoolTest, SlabAddr_MatchesSlabPtr) {
  RdmaSlabPoolConfig config{.slabNum = 2, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool->acquire();
  ASSERT_TRUE(r0.hasValue());
  uint16_t idx = r0.value().index();
  EXPECT_EQ(
      pool->slabAddr(idx), reinterpret_cast<uint64_t>(pool->slabPtr(idx)));
}

TEST_F(RdmaSlabPoolTest, StatePtr_ReturnsDistinctFromSlabPtr) {
  RdmaSlabPoolConfig config{.slabNum = 2, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool->acquire();
  ASSERT_TRUE(r0.hasValue());
  uint16_t idx = r0.value().index();

  EXPECT_NE(pool->statePtr(idx), nullptr);
  EXPECT_NE(pool->statePtr(idx), pool->slabPtr(idx));
  EXPECT_EQ(
      pool->stateAddr(idx), reinterpret_cast<uint64_t>(pool->statePtr(idx)));
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

  RdmaSlabPoolConfig config{.slabNum = 2, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics(2));

  EXPECT_EQ(pool->slabLkey(0), 0xAAAA);
  EXPECT_EQ(pool->slabRkey(0), 0xBBBB);
  EXPECT_EQ(pool->slabLkey(1), 0xCCCC);
  EXPECT_EQ(pool->slabRkey(1), 0xDDDD);
}

TEST_F(RdmaSlabPoolTest, SlabMemory_IsReadableWritable) {
  RdmaSlabPoolConfig config{.slabNum = 2, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r0 = pool->acquire();
  ASSERT_TRUE(r0.hasValue());

  void* ptr = pool->slabPtr(r0.value().index());
  std::memset(ptr, 0xAB, pool->slabSize());

  auto* bytes = static_cast<uint8_t*>(ptr);
  EXPECT_EQ(bytes[0], 0xAB);
  EXPECT_EQ(bytes[pool->slabSize() - 1], 0xAB);
}

TEST_F(RdmaSlabPoolTest, Destruction_DeregistersMrs) {
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd0_, _, _, _))
      .WillRepeatedly(Return(Result<ibv_mr*>(&fakeMr0_)));
  EXPECT_CALL(*mockIbvApi_, regMr(&fakePd1_, _, _, _))
      .WillRepeatedly(Return(Result<ibv_mr*>(&fakeMr1_)));
  EXPECT_CALL(*mockIbvApi_, deregMr(&fakeMr0_)).WillRepeatedly(Return(Ok()));
  EXPECT_CALL(*mockIbvApi_, deregMr(&fakeMr1_)).WillRepeatedly(Return(Ok()));

  RdmaSlabPoolConfig config{.slabNum = 2, .slabSize = 512 * 1024};
  {
    auto pool = std::make_shared<RdmaSlabPool>(
        config, mockCudaApi_, mockIbvApi_, makeNics(2));
  }
}

// --- Atomic path tests (numSlabs <= 128) ---

TEST_F(RdmaSlabPoolTest, Atomic_AcquireAll64Slabs) {
  RdmaSlabPoolConfig config{.slabNum = 64, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  std::vector<RdmaSlab> slabs;
  for (int i = 0; i < 64; ++i) {
    auto r = pool->acquire();
    ASSERT_TRUE(r.hasValue()) << "Failed to acquire slab " << i;
    slabs.push_back(std::move(r.value()));
  }

  EXPECT_TRUE(pool->acquire().hasError());

  std::vector<uint16_t> indices;
  indices.reserve(slabs.size());
  for (const auto& s : slabs) {
    indices.push_back(s.index());
  }
  std::sort(indices.begin(), indices.end());
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(indices[i], i);
  }

  slabs.clear();
  EXPECT_TRUE(pool->acquire().hasValue());
}

TEST_F(RdmaSlabPoolTest, Atomic_AcquireAll128Slabs_CrossesBitmapBoundary) {
  RdmaSlabPoolConfig config{.slabNum = 128, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  std::vector<RdmaSlab> slabs;
  for (int i = 0; i < 128; ++i) {
    auto r = pool->acquire();
    ASSERT_TRUE(r.hasValue()) << "Failed to acquire slab " << i;
    slabs.push_back(std::move(r.value()));
  }

  EXPECT_TRUE(pool->acquire().hasError());

  std::vector<uint16_t> indices;
  indices.reserve(slabs.size());
  for (const auto& s : slabs) {
    indices.push_back(s.index());
  }
  std::sort(indices.begin(), indices.end());
  for (int i = 0; i < 128; ++i) {
    EXPECT_EQ(indices[i], i);
  }

  slabs.clear();
  EXPECT_TRUE(pool->acquire().hasValue());
}

TEST_F(RdmaSlabPoolTest, Atomic_ReleaseLowAndHighIndependently) {
  RdmaSlabPoolConfig config{.slabNum = 128, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  std::vector<RdmaSlab> slabs;
  for (int i = 0; i < 128; ++i) {
    auto r = pool->acquire();
    ASSERT_TRUE(r.hasValue());
    slabs.push_back(std::move(r.value()));
  }
  EXPECT_TRUE(pool->acquire().hasError());

  std::erase_if(slabs, [](const RdmaSlab& s) { return s.index() >= 64; });

  auto r = pool->acquire();
  ASSERT_TRUE(r.hasValue());
  EXPECT_GE(r.value().index(), 64);

  slabs.clear();
}

TEST_F(RdmaSlabPoolTest, Atomic_LowBitmapFullThenHighUsed) {
  RdmaSlabPoolConfig config{.slabNum = 65, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  std::vector<RdmaSlab> slabs;
  for (int i = 0; i < 65; ++i) {
    auto r = pool->acquire();
    ASSERT_TRUE(r.hasValue());
    slabs.push_back(std::move(r.value()));
  }
  EXPECT_TRUE(pool->acquire().hasError());

  bool hasHigh = std::any_of(slabs.begin(), slabs.end(), [](const RdmaSlab& s) {
    return s.index() >= 64;
  });
  EXPECT_TRUE(hasHigh);
}

TEST_F(RdmaSlabPoolTest, Atomic_ConcurrentHighContention) {
  RdmaSlabPoolConfig config{.slabNum = 4, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  constexpr int kThreads = 8;
  constexpr int kOpsPerThread = 1000;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto r = pool->acquire();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  std::vector<RdmaSlab> slabs;
  for (int i = 0; i < 4; ++i) {
    auto r = pool->acquire();
    ASSERT_TRUE(r.hasValue());
    slabs.push_back(std::move(r.value()));
  }
  EXPECT_TRUE(pool->acquire().hasError());
}

// --- Mutex path tests (numSlabs > 128) ---

TEST_F(RdmaSlabPoolTest, Mutex_AcquireRelease_BasicCycle) {
  RdmaSlabPoolConfig config{.slabNum = 129, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  {
    auto r0 = pool->acquire();
    ASSERT_TRUE(r0.hasValue());

    auto r1 = pool->acquire();
    ASSERT_TRUE(r1.hasValue());
    EXPECT_NE(r0.value().index(), r1.value().index());
  }

  EXPECT_TRUE(pool->acquire().hasValue());
}

TEST_F(RdmaSlabPoolTest, Mutex_PoolExhausted_ReturnsError) {
  RdmaSlabPoolConfig config{.slabNum = 129, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  std::vector<RdmaSlab> slabs;
  for (size_t i = 0; i < 129; ++i) {
    auto r = pool->acquire();
    ASSERT_TRUE(r.hasValue());
    slabs.push_back(std::move(r.value()));
  }

  auto r = pool->acquire();
  ASSERT_TRUE(r.hasError());
  EXPECT_EQ(r.error().code(), ErrCode::ResourceExhausted);
}

TEST_F(RdmaSlabPoolTest, Mutex_ConcurrentAcquireRelease) {
  RdmaSlabPoolConfig config{.slabNum = 129, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  constexpr int kThreads = 4;
  constexpr int kOpsPerThread = 200;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto r = pool->acquire();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  std::vector<RdmaSlab> slabs;
  for (size_t i = 0; i < 129; ++i) {
    auto r = pool->acquire();
    ASSERT_TRUE(r.hasValue());
    slabs.push_back(std::move(r.value()));
  }
  EXPECT_TRUE(pool->acquire().hasError());
}

// --- Batch acquire tests (atomic path) ---

TEST_F(RdmaSlabPoolTest, Atomic_BatchAcquire_Success) {
  RdmaSlabPoolConfig config{.slabNum = 8, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool->acquire(4);
  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(r.value().size(), 4);

  std::set<uint16_t> unique;
  for (const auto& s : r.value()) {
    unique.insert(s.index());
  }
  EXPECT_EQ(unique.size(), 4);

  auto r2 = pool->acquire(4);
  ASSERT_TRUE(r2.hasValue());
  EXPECT_EQ(r2.value().size(), 4);

  EXPECT_TRUE(pool->acquire(1).hasError());
}

TEST_F(RdmaSlabPoolTest, Atomic_BatchAcquire_InsufficientSlabs) {
  RdmaSlabPoolConfig config{.slabNum = 2, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool->acquire(4);
  ASSERT_TRUE(r.hasError());
  EXPECT_EQ(r.error().code(), ErrCode::ResourceExhausted);

  auto r2 = pool->acquire(2);
  ASSERT_TRUE(r2.hasValue());
  EXPECT_EQ(r2.value().size(), 2);
}

TEST_F(RdmaSlabPoolTest, Atomic_BatchAcquire_ReleaseAndReacquire) {
  RdmaSlabPoolConfig config{.slabNum = 4, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  {
    auto r = pool->acquire(4);
    ASSERT_TRUE(r.hasValue());
    EXPECT_TRUE(pool->acquire().hasError());
  }

  auto r2 = pool->acquire(4);
  ASSERT_TRUE(r2.hasValue());
}

TEST_F(RdmaSlabPoolTest, Atomic_BatchAcquire_CrossesBitmapBoundary) {
  RdmaSlabPoolConfig config{.slabNum = 128, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool->acquire(128);
  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(r.value().size(), 128);

  std::set<uint16_t> unique;
  for (const auto& s : r.value()) {
    unique.insert(s.index());
  }
  EXPECT_EQ(unique.size(), 128);

  EXPECT_TRUE(pool->acquire(1).hasError());

  r.value().clear();
  EXPECT_TRUE(pool->acquire(1).hasValue());
}

TEST_F(RdmaSlabPoolTest, Atomic_BatchRelease_MixedLowHigh) {
  RdmaSlabPoolConfig config{.slabNum = 128, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool->acquire(128);
  ASSERT_TRUE(r.hasValue());

  std::erase_if(r.value(), [](const RdmaSlab& s) {
    uint16_t idx = s.index();
    return idx == 0 || idx == 63 || idx == 64 || idx == 127;
  });

  size_t released = 128 - r.value().size();
  auto r2 = pool->acquire(static_cast<uint32_t>(released));
  ASSERT_TRUE(r2.hasValue());
  EXPECT_EQ(r2.value().size(), released);
}

TEST_F(RdmaSlabPoolTest, Atomic_BatchAcquireRelease_Concurrent) {
  RdmaSlabPoolConfig config{.slabNum = 16, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  constexpr int kThreads = 4;
  constexpr int kOpsPerThread = 200;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto r = pool->acquire(2);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  auto r = pool->acquire(16);
  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(r.value().size(), 16);
  EXPECT_TRUE(pool->acquire(1).hasError());
}

// --- Batch acquire tests (mutex path) ---

TEST_F(RdmaSlabPoolTest, Mutex_BatchAcquire_Success) {
  RdmaSlabPoolConfig config{.slabNum = 129, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r = pool->acquire(4);
  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(r.value().size(), 4);

  std::set<uint16_t> unique;
  for (const auto& s : r.value()) {
    unique.insert(s.index());
  }
  EXPECT_EQ(unique.size(), 4);
}

TEST_F(RdmaSlabPoolTest, Mutex_BatchAcquire_InsufficientSlabs) {
  RdmaSlabPoolConfig config{.slabNum = 129, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  auto r1 = pool->acquire(127);
  ASSERT_TRUE(r1.hasValue());

  auto r2 = pool->acquire(4);
  ASSERT_TRUE(r2.hasError());
  EXPECT_EQ(r2.error().code(), ErrCode::ResourceExhausted);

  auto r3 = pool->acquire(2);
  ASSERT_TRUE(r3.hasValue());
  EXPECT_EQ(r3.value().size(), 2);
}

TEST_F(RdmaSlabPoolTest, Mutex_BatchAcquire_ReleaseAndReacquire) {
  RdmaSlabPoolConfig config{.slabNum = 129, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  {
    auto r = pool->acquire(10);
    ASSERT_TRUE(r.hasValue());
  }

  auto r2 = pool->acquire(10);
  ASSERT_TRUE(r2.hasValue());
  EXPECT_EQ(r2.value().size(), 10);
}

TEST_F(RdmaSlabPoolTest, Mutex_BatchAcquireRelease_Concurrent) {
  RdmaSlabPoolConfig config{.slabNum = 129, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  constexpr int kThreads = 4;
  constexpr int kOpsPerThread = 100;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto r = pool->acquire(3);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  auto r = pool->acquire(129);
  ASSERT_TRUE(r.hasValue());
  EXPECT_TRUE(pool->acquire(1).hasError());
}

// --- General concurrent test ---

TEST_F(RdmaSlabPoolTest, ConcurrentAcquireRelease_NoRace) {
  RdmaSlabPoolConfig config{.slabNum = 8, .slabSize = 512 * 1024};
  auto pool = std::make_shared<RdmaSlabPool>(
      config, mockCudaApi_, mockIbvApi_, makeNics());

  constexpr int kThreads = 4;
  constexpr int kOpsPerThread = 100;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto r = pool->acquire();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  std::vector<RdmaSlab> slabs;
  for (int i = 0; i < 8; ++i) {
    auto r = pool->acquire();
    ASSERT_TRUE(r.hasValue());
    slabs.push_back(std::move(r.value()));
  }
  EXPECT_TRUE(pool->acquire().hasError());
}
