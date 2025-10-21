// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "nccl.h"

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/algos/common/BufManager.h"
#include "comms/testinfra/TestUtils.h"

#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestsDistUtils.h"

class BufManagerTest : public NcclxBaseTest {
 public:
  BufManagerTest() = default;
  void SetUp() override {
    // This test requires CTRAN to be enabled
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    // TODO: remove this when memCache does not rely on colltrace
    setenv("NCCL_COLLTRACE", "trace", 1);
    NcclxBaseTest::SetUp();
    comm_ = createNcclComm(globalRank, numRanks, localRank);
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(comm_));
    NcclxBaseTest::TearDown();
  }

 protected:
  ncclComm_t comm_{nullptr};
};

using ctran::algos::BufManager;
using ctran::algos::bufmanager::BasicBuf;
using ctran::algos::bufmanager::MemType;
using ctran::algos::bufmanager::RegBuf;
using ctran::algos::bufmanager::RemRegBuf;

class BufManagerTestParamFixture
    : public BufManagerTest,
      // numProducers, count
      public ::testing::WithParamInterface<MemType> {};

TEST_P(BufManagerTestParamFixture, Alloc) {
  auto memType = GetParam();

  if (!ctranInitialized(comm_->ctranComm_.get())) {
    GTEST_SKIP() << "Skip test because ctranInitialized returns false";
  }

  constexpr int numIter = 10;
  enum class TestModuleBufNames { kBuf1, kBuf2, kBuf3, kNumBufs };

  for (int x = 0; x < numIter; x++) {
    const auto statex = comm_->ctranComm_->statex_.get();
    // as we destroy bufMngr in each iteration, the same key can be reused;
    // otherwise mempool may complain duplicated keys
    auto memKey = folly::sformat("test-module-{:#x}", statex->commHash());
    auto bufMngr = std::make_unique<
        BufManager<TestModuleBufNames, TestModuleBufNames::kNumBufs>>(
        statex,
        comm_->ctranComm_->ctran_->mapper.get(),
        &comm_->logMetaData,
        memKey);

    ASSERT_NE(bufMngr, nullptr);

    bufMngr->insert(memType, TestModuleBufNames::kBuf1, 1024);
    bufMngr->insert(memType, TestModuleBufNames::kBuf2, 1024);
    ASSERT_EQ(bufMngr->commit(), commSuccess);

    ASSERT_TRUE(bufMngr->isCommitted());
    ASSERT_FALSE(bufMngr->isExchanged());

    ASSERT_TRUE(bufMngr->contains(memType, TestModuleBufNames::kBuf1));
    ASSERT_TRUE(bufMngr->contains(memType, TestModuleBufNames::kBuf2));
    ASSERT_FALSE(bufMngr->contains(memType, TestModuleBufNames::kBuf3));

    BasicBuf buf1, buf2;
    ASSERT_TRUE(bufMngr->assignBuf(memType, TestModuleBufNames::kBuf1, buf1));
    ASSERT_NE(buf1.ptr, nullptr);
    ASSERT_GE(buf1.size, 1024); // bufMngr may align the size

    ASSERT_TRUE(bufMngr->assignBuf(memType, TestModuleBufNames::kBuf2, buf2));
    ASSERT_NE(buf2.ptr, nullptr);
    ASSERT_GE(buf2.size, 1024); // bufMngr may align the size
    std::cout << "TEST asigned buf1 " << buf1.toString() << std::endl;
    std::cout << "TEST asigned buf2 " << buf2.toString() << std::endl;

    // buf1 and buf2 should be assigned in different memory regions sequentially
    ASSERT_GE((uintptr_t)buf2.ptr, (uintptr_t)buf1.ptr + buf1.size);
  }
}

TEST_P(BufManagerTestParamFixture, Exchange) {
  auto memType = GetParam();
  if (!ctranInitialized(comm_->ctranComm_.get())) {
    GTEST_SKIP() << "Skip test because ctranInitialized returns false";
  }

  constexpr int numIter = 10;
  enum class TestModuleBufNames { kBuf1, kBuf2, kBuf3, kNumBufs };
  const auto nRanks = comm_->ctranComm_->statex_->nRanks();

  for (int x = 0; x < numIter; x++) {
    const auto statex = comm_->ctranComm_->statex_.get();
    auto memKey = folly::sformat("test-module-{:#x}", statex->commHash());
    auto bufMngr = std::make_unique<
        BufManager<TestModuleBufNames, TestModuleBufNames::kNumBufs>>(
        statex,
        comm_->ctranComm_->ctran_->mapper.get(),
        &comm_->logMetaData,
        memKey);

    bufMngr->insert(memType, TestModuleBufNames::kBuf1, 8199);
    bufMngr->insert(memType, TestModuleBufNames::kBuf2, 1024);
    ASSERT_EQ(bufMngr->commit(), commSuccess);

    std::vector<int> peerRanks;
    for (int i = 0; i < nRanks; i++) {
      peerRanks.push_back(i);
    }
    ASSERT_EQ(bufMngr->exchange(peerRanks, nRanks), commSuccess);
    ASSERT_TRUE(bufMngr->isExchanged());

    RegBuf buf1, buf2;
    ASSERT_TRUE(
        bufMngr->assignRegBuf(memType, TestModuleBufNames::kBuf1, buf1));
    ASSERT_NE(buf1.ptr, nullptr);
    ASSERT_GE(buf1.size, 1024); // bufMngr may align the size
    ASSERT_NE(buf1.regHdl, nullptr);

    ASSERT_TRUE(
        bufMngr->assignRegBuf(memType, TestModuleBufNames::kBuf2, buf2));
    ASSERT_NE(buf2.ptr, nullptr);
    ASSERT_GE(buf2.size, 1024); // bufMngr may align the size
    ASSERT_NE(buf2.regHdl, nullptr);

    std::vector<int> peerRanks1;
    for (int i = 0; i < std::max(2, nRanks); i++) {
      peerRanks1.push_back(i);
    }

    std::vector<RemRegBuf> remBufs1;
    ASSERT_TRUE(bufMngr->assignRemRegBuf(
        memType, TestModuleBufNames::kBuf1, peerRanks1, remBufs1));
    ASSERT_EQ(remBufs1.size(), peerRanks1.size());
    for (int i = 0; i < peerRanks1.size(); i++) {
      ASSERT_NE(remBufs1[i].ptr, nullptr);
      ASSERT_EQ(remBufs1[i].peerRank, peerRanks1[i]);

      if (statex->rank() == peerRanks1[i]) {
        ASSERT_EQ(remBufs1[i].rkey.backend, CtranMapperBackend::UNSET);
      } else if (
          // For device memory buffer exchanged with intra-node peers, we expect
          // it is exporded via NVL backend
          comm_->ctranComm_->statex_->isSameNode(
              statex->rank(), peerRanks1[i]) &&
          memType == MemType::kDevice) {
        ASSERT_EQ(remBufs1[i].rkey.backend, CtranMapperBackend::NVL);
      } else {
        // For inter-node peers or host memory, we expect it is exported via IB
        ASSERT_EQ(remBufs1[i].rkey.backend, CtranMapperBackend::IB);
      }
    }
  }
}

TEST_F(BufManagerTest, InvalidAssign) {
  const auto memType = MemType::kDevice;
  if (!ctranInitialized(comm_->ctranComm_.get())) {
    GTEST_SKIP() << "Skip test because ctranInitialized returns false";
  }

  enum class TestModuleBufNames { kBuf1, kBuf2, kBuf3, kNumBufs };
  const auto nRanks = comm_->ctranComm_->statex_->nRanks();

  const auto statex = comm_->ctranComm_->statex_.get();
  auto memKey = folly::sformat("test-module-{:#x}", statex->commHash());
  auto bufMngr = std::make_unique<
      BufManager<TestModuleBufNames, TestModuleBufNames::kNumBufs>>(
      statex,
      comm_->ctranComm_->ctran_->mapper.get(),
      &comm_->logMetaData,
      memKey);

  bufMngr->insert(memType, TestModuleBufNames::kBuf1, 8199);

  // Invalid basic buff assignment before commit
  BasicBuf buf;
  ASSERT_FALSE(bufMngr->contains(memType, TestModuleBufNames::kBuf2));
  ASSERT_FALSE(bufMngr->assignBuf(memType, TestModuleBufNames::kBuf2, buf));

  ASSERT_EQ(bufMngr->commit(), commSuccess);

  // Invalid reg buff assignment before exchange
  RegBuf buf1;
  ASSERT_FALSE(bufMngr->assignRegBuf(memType, TestModuleBufNames::kBuf1, buf1));

  std::vector<int> peerRanks1;
  for (int i = 0; i < std::max(2, nRanks); i++) {
    peerRanks1.push_back(i);
  }

  // Invalid rem reg buff assignment before exchange
  std::vector<RemRegBuf> remBufs1;
  ASSERT_FALSE(bufMngr->assignRemRegBuf(
      memType, TestModuleBufNames::kBuf1, peerRanks1, remBufs1));
}

TEST_P(BufManagerTestParamFixture, DifferentMemKeysDifferentAddresses) {
  auto memType = GetParam();

  if (!ctranInitialized(comm_->ctranComm_.get())) {
    GTEST_SKIP() << "Skip test because ctranInitialized returns false";
  }

  enum class TestModuleBufNames { kBuf1, kBuf2, kNumBufs };
  const auto statex = comm_->ctranComm_->statex_.get();

  // Create first BufManager with memKey1
  auto memKey1 = folly::sformat("test-module-key1-{:#x}", statex->commHash());
  auto bufMngr1 = std::make_unique<
      BufManager<TestModuleBufNames, TestModuleBufNames::kNumBufs>>(
      statex,
      comm_->ctranComm_->ctran_->mapper.get(),
      &comm_->logMetaData,
      memKey1);

  ASSERT_NE(bufMngr1, nullptr);

  // Create second BufManager with different memKey2
  auto memKey2 = folly::sformat("test-module-key2-{:#x}", statex->commHash());
  auto bufMngr2 = std::make_unique<
      BufManager<TestModuleBufNames, TestModuleBufNames::kNumBufs>>(
      statex,
      comm_->ctranComm_->ctran_->mapper.get(),
      &comm_->logMetaData,
      memKey2);

  ASSERT_NE(bufMngr2, nullptr);

  // Insert and commit buffers for first BufManager
  bufMngr1->insert(memType, TestModuleBufNames::kBuf1, 1024);
  ASSERT_EQ(bufMngr1->commit(), commSuccess);

  // Insert and commit buffers for second BufManager
  bufMngr2->insert(memType, TestModuleBufNames::kBuf1, 1024);
  ASSERT_EQ(bufMngr2->commit(), commSuccess);

  // Assign buffers from both managers
  BasicBuf buf1, buf2;
  ASSERT_TRUE(bufMngr1->assignBuf(memType, TestModuleBufNames::kBuf1, buf1));
  ASSERT_NE(buf1.ptr, nullptr);
  ASSERT_GE(buf1.size, 1024);

  ASSERT_TRUE(bufMngr2->assignBuf(memType, TestModuleBufNames::kBuf1, buf2));
  ASSERT_NE(buf2.ptr, nullptr);
  ASSERT_GE(buf2.size, 1024);

  std::cout << "TEST BufManager1 assigned buf1 " << buf1.toString()
            << std::endl;
  std::cout << "TEST BufManager2 assigned buf2 " << buf2.toString()
            << std::endl;

  // Verify that memory addresses for the two buffers are different
  // Since they use different memory keys, they should be allocated in different
  // memory regions
  ASSERT_NE(buf1.ptr, buf2.ptr)
      << "Expected different memory addresses for buffers with different memKeys";

  bufMngr1->release();
  bufMngr2->release();
}

TEST_P(BufManagerTestParamFixture, DifferentCommsDifferentAddresses) {
  auto memType = GetParam();

  if (!ctranInitialized(comm_->ctranComm_.get())) {
    GTEST_SKIP() << "Skip test because ctranInitialized returns false";
  }

  enum class TestModuleBufNames { kBuf1, kBuf2, kNumBufs };

  // Create second communicator (comm2) identical to the first one
  auto comm2_ = createNcclComm(globalRank, numRanks, localRank);
  ASSERT_NE(comm2_, nullptr);

  // Get statex from both communicators
  const auto statex1 = comm_->ctranComm_->statex_.get();
  const auto statex2 = comm2_->ctranComm_->statex_.get();

  // Create BufManager for first communicator
  auto memKey1 = folly::sformat("test-module-comm1-{:#x}", statex1->commHash());
  auto bufMngr1 = std::make_unique<
      BufManager<TestModuleBufNames, TestModuleBufNames::kNumBufs>>(
      statex1,
      comm_->ctranComm_->ctran_->mapper.get(),
      &comm_->logMetaData,
      memKey1);

  ASSERT_NE(bufMngr1, nullptr);

  // Create BufManager for second communicator
  auto memKey2 = folly::sformat("test-module-comm2-{:#x}", statex2->commHash());
  auto bufMngr2 = std::make_unique<
      BufManager<TestModuleBufNames, TestModuleBufNames::kNumBufs>>(
      statex2,
      comm2_->ctranComm_->ctran_->mapper.get(),
      &comm2_->logMetaData,
      memKey2);

  ASSERT_NE(bufMngr2, nullptr);

  // Insert and commit buffers for first BufManager
  bufMngr1->insert(memType, TestModuleBufNames::kBuf1, 1024);
  ASSERT_EQ(bufMngr1->commit(), commSuccess);

  // Insert and commit buffers for second BufManager
  bufMngr2->insert(memType, TestModuleBufNames::kBuf1, 1024);
  ASSERT_EQ(bufMngr2->commit(), commSuccess);

  // Assign buffers from both managers
  BasicBuf buf1, buf2;
  ASSERT_TRUE(bufMngr1->assignBuf(memType, TestModuleBufNames::kBuf1, buf1));
  ASSERT_NE(buf1.ptr, nullptr);
  ASSERT_GE(buf1.size, 1024);

  ASSERT_TRUE(bufMngr2->assignBuf(memType, TestModuleBufNames::kBuf1, buf2));
  ASSERT_NE(buf2.ptr, nullptr);
  ASSERT_GE(buf2.size, 1024);

  std::cout << "TEST Comm1 BufManager assigned buf1 " << buf1.toString()
            << std::endl;
  std::cout << "TEST Comm2 BufManager assigned buf2 " << buf2.toString()
            << std::endl;

  // Verify that memory addresses for the two buffers are different
  // Since they use different communicators, they should be allocated in
  // different memory regions
  ASSERT_NE(buf1.ptr, buf2.ptr)
      << "Expected different memory addresses for buffers from different communicators";

  // Clean up second communicator
  NCCLCHECK_TEST(ncclCommDestroy(comm2_));
}

TEST_P(BufManagerTestParamFixture, DifferentSizes) {
  auto memType = GetParam();

  if (!ctranInitialized(comm_->ctranComm_.get())) {
    GTEST_SKIP() << "Skip test because ctranInitialized returns false";
  }

  enum class TestModuleBufNames { kBuf1, kBuf2, kNumBufs };
  const auto statex = comm_->ctranComm_->statex_.get();

  // Create first BufManager with memKey
  auto memKey = folly::sformat("test-module-key-{:#x}", statex->commHash());
  auto bufMngr1 = std::make_unique<
      BufManager<TestModuleBufNames, TestModuleBufNames::kNumBufs>>(
      statex,
      comm_->ctranComm_->ctran_->mapper.get(),
      &comm_->logMetaData,
      memKey);

  ASSERT_NE(bufMngr1, nullptr);

  // Create second BufManager with the same key
  auto bufMngr2 = std::make_unique<
      BufManager<TestModuleBufNames, TestModuleBufNames::kNumBufs>>(
      statex,
      comm_->ctranComm_->ctran_->mapper.get(),
      &comm_->logMetaData,
      memKey);

  ASSERT_NE(bufMngr2, nullptr);

  auto size1 = 1 << 10; // 1KB
  auto size2 = 1 << 20; // 2MB

  // Insert and commit buffers for first BufManager with 1KB
  bufMngr1->insert(memType, TestModuleBufNames::kBuf1, size1);
  ASSERT_EQ(bufMngr1->commit(), commSuccess);

  // Insert and commit buffers for second BufManager with 2MB
  bufMngr2->insert(memType, TestModuleBufNames::kBuf1, size2);
  ASSERT_EQ(bufMngr2->commit(), commSuccess);

  // Assign buffers from both managers
  BasicBuf buf1, buf2;
  ASSERT_TRUE(bufMngr1->assignBuf(memType, TestModuleBufNames::kBuf1, buf1));
  ASSERT_NE(buf1.ptr, nullptr);
  ASSERT_GE(buf1.size, size1);

  ASSERT_TRUE(bufMngr2->assignBuf(memType, TestModuleBufNames::kBuf1, buf2));
  ASSERT_NE(buf2.ptr, nullptr);
  ASSERT_GE(buf2.size, size2);

  std::cout << "TEST BufManager1 assigned buf1 " << buf1.toString()
            << std::endl;
  std::cout << "TEST BufManager2 assigned buf2 " << buf2.toString()
            << std::endl;

  // Verify that memory addresses for the two buffers are different
  // Since they use different memory keys, they should be allocated in different
  // memory regions
  ASSERT_NE(buf1.ptr, buf2.ptr)
      << "Expected different memory addresses for buffers with different sizes";
}

INSTANTIATE_TEST_SUITE_P(
    BufManagerTest,
    BufManagerTestParamFixture,
    ::testing::Values(MemType::kDevice, MemType::kHostPinned),
    [&](const testing::TestParamInfo<BufManagerTestParamFixture::ParamType>&
            info) {
      return ctran::algos::bufmanager::memTypeToStr(info.param);
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
