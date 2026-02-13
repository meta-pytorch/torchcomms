// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/tests/IbverbxTestFixture.h"

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

namespace ibverbx {

TEST_F(IbverbxTestFixture, IbvVirtualQp) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto virtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
  ASSERT_TRUE(virtualQp);
  ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
  ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

  // Store the first QP's raw pointer, and notifyQp's raw pointer for comparison
  // after move
  const auto firstQpPtr = virtualQp->getQpsRef()[0].qp();
  const auto notifyQpPtr = virtualQp->getNotifyQpRef().qp();

  // move constructor
  auto qpg1 = std::move(*virtualQp);
  ASSERT_TRUE(
      virtualQp->getQpsRef().empty()); // After move, vector should be empty
  ASSERT_EQ(qpg1.getQpsRef().size(), totalQps); // Size should match original
  ASSERT_EQ(qpg1.getQpsRef()[0].qp(), firstQpPtr); // First element should match
  ASSERT_EQ(qpg1.getNotifyQpRef().qp(), notifyQpPtr); // Notify QP should match

  IbvVirtualQp qpg2(std::move(qpg1));
  ASSERT_TRUE(qpg1.getQpsRef().empty()); // After move, vector should be empty
  ASSERT_EQ(qpg2.getQpsRef().size(), totalQps); // Size should match original
  ASSERT_EQ(qpg2.getQpsRef()[0].qp(), firstQpPtr); // First element should match
  ASSERT_EQ(qpg2.getNotifyQpRef().qp(), notifyQpPtr); // Notify QP should match
}

TEST_F(IbverbxTestFixture, IbvVirtualQpMultiThreadUniqueQpNum) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  constexpr int kNumThreads = 4;
  constexpr int kVirtualQpsPerThread = 10;
  constexpr int kTotalVirtualQps = kNumThreads * kVirtualQpsPerThread;

  std::set<uint32_t> virtualQpNums;
  std::set<uint32_t> virtualCqNums;
  std::mutex numsMutex;

  auto createVirtualQps = [&]() {
    std::vector<uint32_t> localQpNums;
    std::vector<uint32_t> localCqNums;
    localQpNums.reserve(kVirtualQpsPerThread);
    localCqNums.reserve(kVirtualQpsPerThread);

    for (int i = 0; i < kVirtualQpsPerThread; i++) {
      int cqe = 100;
      auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
      ASSERT_TRUE(maybeVirtualCq);
      auto virtualCq = std::move(*maybeVirtualCq);

      auto initAttr =
          makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
      auto pd = device.allocPd();
      ASSERT_TRUE(pd);

      int totalQps = 4;
      auto virtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
      ASSERT_TRUE(virtualQp);

      localQpNums.push_back(virtualQp->getVirtualQpNum());
      localCqNums.push_back(virtualCq.getVirtualCqNum());
    }

    std::lock_guard<std::mutex> lock(numsMutex);
    virtualQpNums.insert(localQpNums.begin(), localQpNums.end());
    virtualCqNums.insert(localCqNums.begin(), localCqNums.end());
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; i++) {
    threads.emplace_back(createVirtualQps);
  }

  for (auto& t : threads) {
    t.join();
  }

  ASSERT_EQ(virtualQpNums.size(), kTotalVirtualQps)
      << "All virtual QP numbers should be distinct";
  ASSERT_EQ(virtualCqNums.size(), kTotalVirtualQps)
      << "All virtual CQ numbers should be distinct";
}

TEST_F(IbverbxTestFixture, IbvVirtualQpFindAvailableSendQp) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 4;
  int maxMsgCntPerQp = 100;

  // Test setting 1: default maxMsgCntPerQp (100)
  {
    auto virtualQp =
        pd->createVirtualQp(totalQps, &initAttr, &virtualCq, maxMsgCntPerQp);
    ASSERT_TRUE(virtualQp);
    ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
    ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

    // Mock send maxMsgCntPerQp-1 messages on all QPs
    for (int i = 0; i < totalQps; i++) {
      for (int j = 0; j < maxMsgCntPerQp - 1; j++) {
        virtualQp->getQpsRef().at(i).enquePhysicalSendWrStatus(0, 0);
      }
    }

    // Find available QP should return 0. Then, after mock send 1 more
    // message on QP 0, this QP should no longer be available
    int curQp = 0;
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // Find available QP should return 1. Then, after mock send 1 more
    // message on QP 1, this QP should no longer be available
    curQp = 1;
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // Find available QP should return 2. Then, after mock send 1 more
    // message on QP 2, this QP should no longer be available
    curQp = 2;
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // Find available QP should return 3. Then, after mock send 1 more
    // message on QP 3, this QP should no longer be available
    curQp = 3;
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // Now, all QPs are full, find available QP should return -1
    ASSERT_EQ(virtualQp->findAvailableSendQp(), -1);

    // After clear up on QP 2, QP 2 should be available again. Then, mock send 1
    // more message on this QP again and it should no longer be available
    curQp = 2;
    virtualQp->getQpsRef().at(curQp).dequePhysicalSendWrStatus();
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // After clear up on QP 3, QP 3 should be available again. Then, mock send 1
    // more message on this QP again and it should no longer be available
    curQp = 3;
    virtualQp->getQpsRef().at(curQp).dequePhysicalSendWrStatus();
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // After clear up on QP 0, QP 0 should be available again. Then, mock send 1
    // more message on this QP again and it should no longer be available
    curQp = 0;
    virtualQp->getQpsRef().at(curQp).dequePhysicalSendWrStatus();
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // After clear up on QP 1, QP 1 should be available again. Then, mock send 1
    // more message on this QP again and it should no longer be available
    curQp = 1;
    virtualQp->getQpsRef().at(curQp).dequePhysicalSendWrStatus();
    ASSERT_EQ(virtualQp->findAvailableSendQp(), curQp);
    virtualQp->getQpsRef().at(curQp).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(
        virtualQp->getQpsRef().at(curQp).isSendQueueAvailable(maxMsgCntPerQp));

    // Now, all QPs are full, find available QP should return -1
    ASSERT_EQ(virtualQp->findAvailableSendQp(), -1);
  }

  // Test setting 2: No limit on maxMsgCntPerQp and maxMsgSize
  {
    auto virtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq, -1);
    ASSERT_TRUE(virtualQp);
    ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
    ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

    // Mock send default maxMsgCntPerQp messages on all QPs
    for (int i = 0; i < totalQps; i++) {
      for (int j = 0; j < maxMsgCntPerQp; j++) {
        virtualQp->getQpsRef().at(i).enquePhysicalSendWrStatus(0, 0);
      }
    }

    // findAvailableSendQp should return 0, 1, 2, 3, in order
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 0);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 1);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 2);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 3);

    // After mock send one more message on all QPs, findAvailableSendQp should
    // return 0, 1, 2, 3, in order again
    for (int i = 0; i < totalQps; i++) {
      virtualQp->getQpsRef().at(i).enquePhysicalSendWrStatus(0, 0);
    }
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 0);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 1);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 2);
    ASSERT_EQ(virtualQp->findAvailableSendQp(), 3);
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpBusinessCard) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto virtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
  ASSERT_TRUE(virtualQp);
  ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
  ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

  auto virtualQpBusinessCard = virtualQp->getVirtualQpBusinessCard();
  ASSERT_EQ(virtualQpBusinessCard.qpNums_.size(), totalQps);
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCard.qpNums_.at(i),
        virtualQp->getQpsRef().at(i).qp()->qp_num);
  }
  ASSERT_EQ(
      virtualQpBusinessCard.notifyQpNum_,
      virtualQp->getNotifyQpRef().qp()->qp_num);

  // move constructor
  auto virtualQpBusinessCard1 = std::move(virtualQpBusinessCard);
  ASSERT_TRUE(virtualQpBusinessCard.qpNums_.empty());
  ASSERT_EQ(virtualQpBusinessCard1.qpNums_.size(), totalQps);
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCard1.qpNums_.at(i),
        virtualQp->getQpsRef().at(i).qp()->qp_num);
  }
  ASSERT_EQ(
      virtualQpBusinessCard1.notifyQpNum_,
      virtualQp->getNotifyQpRef().qp()->qp_num);

  auto virtualQpBusinessCard2(std::move(virtualQpBusinessCard1));
  ASSERT_TRUE(virtualQpBusinessCard1.qpNums_.empty());
  ASSERT_EQ(virtualQpBusinessCard2.qpNums_.size(), totalQps);
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCard2.qpNums_.at(i),
        virtualQp->getQpsRef().at(i).qp()->qp_num);
  }
  ASSERT_EQ(
      virtualQpBusinessCard2.notifyQpNum_,
      virtualQp->getNotifyQpRef().qp()->qp_num);

  // Copy constructor
  IbvVirtualQpBusinessCard virtualQpBusinessCardCopy1(virtualQpBusinessCard2);
  ASSERT_EQ(
      virtualQpBusinessCardCopy1.qpNums_.size(),
      virtualQpBusinessCard2.qpNums_.size());
  ASSERT_EQ(
      virtualQpBusinessCardCopy1.qpNums_.size(),
      virtualQpBusinessCard2.qpNums_.size());
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCardCopy1.qpNums_.at(i),
        virtualQpBusinessCard2.qpNums_.at(i));
  }
  ASSERT_EQ(
      virtualQpBusinessCardCopy1.notifyQpNum_,
      virtualQpBusinessCard2.notifyQpNum_);

  IbvVirtualQpBusinessCard virtualQpBusinessCardCopy2;
  virtualQpBusinessCardCopy2 = virtualQpBusinessCard2;
  ASSERT_EQ(
      virtualQpBusinessCardCopy2.qpNums_.size(),
      virtualQpBusinessCard2.qpNums_.size());
  ASSERT_EQ(
      virtualQpBusinessCardCopy2.qpNums_.size(),
      virtualQpBusinessCard2.qpNums_.size());
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCardCopy2.qpNums_.at(i),
        virtualQpBusinessCard2.qpNums_.at(i));
  }
  ASSERT_EQ(
      virtualQpBusinessCardCopy2.notifyQpNum_,
      virtualQpBusinessCard2.notifyQpNum_);
}

TEST_F(IbverbxTestFixture, IbvVirtualQpBusinessCardSerializeAndDeserialize) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 16;
  auto virtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
  ASSERT_TRUE(virtualQp);
  ASSERT_EQ(virtualQp->getQpsRef().size(), totalQps);
  ASSERT_EQ(virtualQp->getTotalQps(), totalQps);

  auto virtualQpBusinessCard = virtualQp->getVirtualQpBusinessCard();
  ASSERT_EQ(virtualQpBusinessCard.qpNums_.size(), totalQps);
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        virtualQpBusinessCard.qpNums_.at(i),
        virtualQp->getQpsRef().at(i).qp()->qp_num);
  }
  ASSERT_EQ(
      virtualQpBusinessCard.notifyQpNum_,
      virtualQp->getNotifyQpRef().qp()->qp_num);

  // Serialize and deserialize
  auto serializedVirtualQpBusinessCard = virtualQpBusinessCard.serialize();
  auto maybeDeserializedVirtualQpBusinessCard =
      IbvVirtualQpBusinessCard::deserialize(serializedVirtualQpBusinessCard);
  ASSERT_TRUE(maybeDeserializedVirtualQpBusinessCard);
  auto& deserializedVirtualQpBusinessCard =
      *maybeDeserializedVirtualQpBusinessCard;
  ASSERT_EQ(
      deserializedVirtualQpBusinessCard.qpNums_.size(),
      virtualQpBusinessCard.qpNums_.size());
  for (auto i = 0; i < totalQps; i++) {
    ASSERT_EQ(
        deserializedVirtualQpBusinessCard.qpNums_.at(i),
        virtualQp->getQpsRef().at(i).qp()->qp_num);
  }
  ASSERT_EQ(
      deserializedVirtualQpBusinessCard.notifyQpNum_,
      virtualQp->getNotifyQpRef().qp()->qp_num);

  // Deserialize fail case
  std::string emptyStr;
  std::string& jsonStr = emptyStr;
  auto maybeDeserializedVirtualQpBusinessCardError =
      IbvVirtualQpBusinessCard::deserialize(jsonStr);
  ASSERT_FALSE(maybeDeserializedVirtualQpBusinessCardError);
}

TEST_F(IbverbxTestFixture, IbvVirtualQpIsMultiQp) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;

  // totalQps > 1 → isMultiQp() returns true
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    auto virtualQp = pd->createVirtualQp(4, &initAttr, &virtualCq);
    ASSERT_TRUE(virtualQp);
    ASSERT_TRUE(virtualQp->isMultiQp());
  }

  // totalQps == 1 → isMultiQp() returns false
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    auto virtualQp = pd->createVirtualQp(1, &initAttr, &virtualCq);
    ASSERT_TRUE(virtualQp);
    ASSERT_FALSE(virtualQp->isMultiQp());
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpRegisterUnregisterWithVirtualCq) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;

  // Test 1: Constructor auto-registers all physical QPs + notifyQp
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    auto maybeVirtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    // Should have totalQps + 1 (notify QP) registrations
    ASSERT_EQ(virtualCq.registeredQps_.size(), totalQps + 1);

    // Verify virtualCq_ back-pointer
    ASSERT_EQ(virtualQp.virtualCq_, &virtualCq);

    // Verify each physical QP is registered correctly
    for (size_t i = 0; i < virtualQp.getQpsRef().size(); i++) {
      auto& pqp = virtualQp.getQpsRef().at(i);
      auto* info =
          virtualCq.findRegisteredQpInfo(pqp.qp()->qp_num, pqp.getDeviceId());
      ASSERT_NE(info, nullptr);
      ASSERT_EQ(info->vqp, &virtualQp);
      ASSERT_TRUE(info->isMultiQp);
      ASSERT_EQ(info->virtualQpNum, virtualQp.getVirtualQpNum());
    }

    // Verify notify QP is registered
    auto& notifyQp = virtualQp.getNotifyQpRef();
    auto* notifyInfo = virtualCq.findRegisteredQpInfo(
        notifyQp.qp()->qp_num, notifyQp.getDeviceId());
    ASSERT_NE(notifyInfo, nullptr);
    ASSERT_EQ(notifyInfo->vqp, &virtualQp);
  }

  // Test 2: Destructor auto-unregisters all physical QPs
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    std::vector<std::pair<uint32_t, int32_t>> qpIds;

    {
      auto maybeVirtualQp =
          pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
      ASSERT_TRUE(maybeVirtualQp);
      auto virtualQp = std::move(*maybeVirtualQp);

      // Record QP identifiers before destruction
      for (size_t i = 0; i < virtualQp.getQpsRef().size(); i++) {
        auto& pqp = virtualQp.getQpsRef().at(i);
        qpIds.emplace_back(pqp.qp()->qp_num, pqp.getDeviceId());
      }
      auto& notifyQp = virtualQp.getNotifyQpRef();
      qpIds.emplace_back(notifyQp.qp()->qp_num, notifyQp.getDeviceId());

      ASSERT_EQ(virtualCq.registeredQps_.size(), totalQps + 1);
    } // virtualQp destroyed here

    // All registrations should be gone
    ASSERT_TRUE(virtualCq.registeredQps_.empty());
    for (const auto& [qpNum, deviceId] : qpIds) {
      ASSERT_EQ(virtualCq.findRegisteredQpInfo(qpNum, deviceId), nullptr);
    }
  }

  // Test 3: Move constructor re-registers with new pointer
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    auto maybeVirtualQp = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp1 = std::move(*maybeVirtualQp);

    // Move-construct
    IbvVirtualQp virtualQp2(std::move(virtualQp1));

    // Old virtualCq_ should be nullptr
    ASSERT_EQ(virtualQp1.virtualCq_, nullptr);

    // Registration count unchanged
    ASSERT_EQ(virtualCq.registeredQps_.size(), totalQps + 1);

    // All entries point to new VirtualQp
    for (size_t i = 0; i < virtualQp2.getQpsRef().size(); i++) {
      auto& pqp = virtualQp2.getQpsRef().at(i);
      auto* info =
          virtualCq.findRegisteredQpInfo(pqp.qp()->qp_num, pqp.getDeviceId());
      ASSERT_NE(info, nullptr);
      ASSERT_EQ(info->vqp, &virtualQp2);
    }
  }

  // Test 4: Move assignment re-registers with new pointer
  {
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    ASSERT_TRUE(maybeVirtualCq);
    auto virtualCq = std::move(*maybeVirtualCq);

    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto pd = device.allocPd();
    ASSERT_TRUE(pd);

    int totalQps = 4;
    auto maybeVirtualQp1 = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
    ASSERT_TRUE(maybeVirtualQp1);
    auto virtualQp1 = std::move(*maybeVirtualQp1);

    auto maybeVirtualQp2 = pd->createVirtualQp(totalQps, &initAttr, &virtualCq);
    ASSERT_TRUE(maybeVirtualQp2);
    auto virtualQp2 = std::move(*maybeVirtualQp2);

    // Record QP1's physical QP ids before move
    std::vector<std::pair<uint32_t, int32_t>> qp1Ids;
    for (size_t i = 0; i < virtualQp1.getQpsRef().size(); i++) {
      auto& pqp = virtualQp1.getQpsRef().at(i);
      qp1Ids.emplace_back(pqp.qp()->qp_num, pqp.getDeviceId());
    }

    // Move-assign: virtualQp2's old QPs should be unregistered,
    // virtualQp1's QPs re-registered under virtualQp2's pointer
    virtualQp2 = std::move(virtualQp1);

    // Source's virtualCq_ should be nullptr
    ASSERT_EQ(virtualQp1.virtualCq_, nullptr);

    // virtualQp1's physical QPs should now be registered under virtualQp2
    for (const auto& [qpNum, deviceId] : qp1Ids) {
      auto* info = virtualCq.findRegisteredQpInfo(qpNum, deviceId);
      ASSERT_NE(info, nullptr);
      ASSERT_EQ(info->vqp, &virtualQp2);
    }
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpUpdateWrState) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  auto maybeVirtualQp = pd->createVirtualQp(4, &initAttr, &virtualCq);
  ASSERT_TRUE(maybeVirtualQp);
  auto virtualQp = std::move(*maybeVirtualQp);

  // Test 1: Basic decrement and opcode update
  {
    ActiveVirtualWr wr;
    wr.userWrId = 100;
    wr.remainingMsgCnt = 3;
    uint64_t id = virtualQp.sendTracker_.add(std::move(wr));

    auto result = virtualQp.updateWrState(
        virtualQp.sendTracker_, id, IBV_WC_SUCCESS, IBV_WC_RDMA_WRITE);
    ASSERT_TRUE(result.hasValue());

    auto* found = virtualQp.sendTracker_.find(id);
    ASSERT_NE(found, nullptr);
    ASSERT_EQ(found->remainingMsgCnt, 2);
    ASSERT_EQ(found->wcOpcode, IBV_WC_RDMA_WRITE);
    ASSERT_EQ(found->aggregatedStatus, IBV_WC_SUCCESS);

    virtualQp.sendTracker_.remove(id);
  }

  // Test 2: First-error-wins aggregation
  {
    ActiveVirtualWr wr;
    wr.userWrId = 200;
    wr.remainingMsgCnt = 3;
    uint64_t id = virtualQp.sendTracker_.add(std::move(wr));

    // First call: success
    auto r1 = virtualQp.updateWrState(
        virtualQp.sendTracker_, id, IBV_WC_SUCCESS, IBV_WC_RDMA_WRITE);
    ASSERT_TRUE(r1.hasValue());
    ASSERT_EQ(
        virtualQp.sendTracker_.find(id)->aggregatedStatus, IBV_WC_SUCCESS);

    // Second call: error
    auto r2 = virtualQp.updateWrState(
        virtualQp.sendTracker_, id, IBV_WC_REM_ACCESS_ERR, IBV_WC_RDMA_WRITE);
    ASSERT_TRUE(r2.hasValue());
    ASSERT_EQ(
        virtualQp.sendTracker_.find(id)->aggregatedStatus,
        IBV_WC_REM_ACCESS_ERR);

    // Third call: different error — first error should stick
    auto r3 = virtualQp.updateWrState(
        virtualQp.sendTracker_, id, IBV_WC_RETRY_EXC_ERR, IBV_WC_RDMA_WRITE);
    ASSERT_TRUE(r3.hasValue());
    ASSERT_EQ(
        virtualQp.sendTracker_.find(id)->aggregatedStatus,
        IBV_WC_REM_ACCESS_ERR);

    ASSERT_EQ(virtualQp.sendTracker_.find(id)->remainingMsgCnt, 0);

    virtualQp.sendTracker_.remove(id);
  }

  // Test 3: Not-found WR triggers CHECK failure
  {
    ASSERT_DEATH(
        virtualQp.updateWrState(
            virtualQp.sendTracker_, 99999, IBV_WC_SUCCESS, IBV_WC_SEND),
        "not found in tracker");
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpBuildVirtualWc) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  auto maybeVirtualQp = pd->createVirtualQp(4, &initAttr, &virtualCq);
  ASSERT_TRUE(maybeVirtualQp);
  auto virtualQp = std::move(*maybeVirtualQp);

  ActiveVirtualWr wr;
  wr.userWrId = 42;
  wr.aggregatedStatus = IBV_WC_REM_ACCESS_ERR;
  wr.length = 8192;
  wr.immData = 0xDEAD;
  wr.wcOpcode = IBV_WC_RDMA_WRITE;

  IbvVirtualWc wc = virtualQp.buildVirtualWc(wr);
  ASSERT_EQ(wc.wrId, 42);
  ASSERT_EQ(wc.status, IBV_WC_REM_ACCESS_ERR);
  ASSERT_EQ(wc.byteLen, 8192);
  ASSERT_EQ(wc.immData, 0xDEAD);
  ASSERT_EQ(wc.opcode, IBV_WC_RDMA_WRITE);
  ASSERT_EQ(wc.qpNum, virtualQp.getVirtualQpNum());
}

TEST_F(IbverbxTestFixture, IbvVirtualQpIsSendOpcode) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  auto maybeVirtualQp = pd->createVirtualQp(4, &initAttr, &virtualCq);
  ASSERT_TRUE(maybeVirtualQp);
  auto virtualQp = std::move(*maybeVirtualQp);

  // Send opcodes → true
  ASSERT_TRUE(virtualQp.isSendOpcode(IBV_WC_SEND));
  ASSERT_TRUE(virtualQp.isSendOpcode(IBV_WC_RDMA_WRITE));
  ASSERT_TRUE(virtualQp.isSendOpcode(IBV_WC_RDMA_READ));
  ASSERT_TRUE(virtualQp.isSendOpcode(IBV_WC_FETCH_ADD));
  ASSERT_TRUE(virtualQp.isSendOpcode(IBV_WC_COMP_SWAP));

  // Recv opcodes → false
  ASSERT_FALSE(virtualQp.isSendOpcode(IBV_WC_RECV));
  ASSERT_FALSE(virtualQp.isSendOpcode(IBV_WC_RECV_RDMA_WITH_IMM));
}

TEST_F(IbverbxTestFixture, IbvVirtualQpHasQpCapacity) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int totalQps = 4;
  int maxMsgCntPerQp = 2;

  // With limit: capacity depends on outstanding count
  {
    auto maybeVirtualQp =
        pd->createVirtualQp(totalQps, &initAttr, &virtualCq, maxMsgCntPerQp);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    ASSERT_TRUE(virtualQp.hasQpCapacity(0));

    // Fill to limit
    virtualQp.getQpsRef().at(0).enquePhysicalSendWrStatus(0, 0);
    ASSERT_TRUE(virtualQp.hasQpCapacity(0));
    virtualQp.getQpsRef().at(0).enquePhysicalSendWrStatus(0, 0);
    ASSERT_FALSE(virtualQp.hasQpCapacity(0));

    // Free one, capacity returns
    virtualQp.getQpsRef().at(0).dequePhysicalSendWrStatus();
    ASSERT_TRUE(virtualQp.hasQpCapacity(0));
  }

  // No limit (maxMsgCntPerQp == -1): always has capacity
  {
    auto maybeVirtualQp =
        pd->createVirtualQp(totalQps, &initAttr, &virtualCq, -1);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    virtualQp.getQpsRef().at(0).enquePhysicalSendWrStatus(0, 0);
    virtualQp.getQpsRef().at(0).enquePhysicalSendWrStatus(0, 0);
    virtualQp.getQpsRef().at(0).enquePhysicalSendWrStatus(0, 0);
    ASSERT_TRUE(virtualQp.hasQpCapacity(0));
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualQpBuildPhysicalSendWr) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  auto virtualCq = std::move(*maybeVirtualCq);

  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  int32_t deviceId = 0;

  // Test 1: IBV_WR_RDMA_WRITE (default SPRAY mode)
  {
    auto maybeVirtualQp =
        pd->createVirtualQp(4, &initAttr, &virtualCq, 100, 1024);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    ActiveVirtualWr pending;
    pending.localAddr = reinterpret_cast<void*>(0x1000);
    pending.remoteAddr = 0x2000;
    pending.length = 2048;
    pending.offset = 512;
    pending.opcode = IBV_WR_RDMA_WRITE;
    pending.deviceKeys[deviceId] = MemoryRegionKeys{111, 222};

    uint32_t fragLen = 1024;
    auto [sendWr, sendSge] =
        virtualQp.buildPhysicalSendWr(pending, deviceId, fragLen);
    sendWr.sg_list = &sendSge;

    ASSERT_EQ(sendSge.addr, 0x1000 + 512);
    ASSERT_EQ(sendSge.length, fragLen);
    ASSERT_EQ(sendSge.lkey, 111);
    ASSERT_EQ(sendWr.sg_list, &sendSge);
    ASSERT_EQ(sendWr.num_sge, 1);
    ASSERT_EQ(sendWr.send_flags, IBV_SEND_SIGNALED);
    ASSERT_EQ(sendWr.wr.rdma.remote_addr, 0x2000 + 512);
    ASSERT_EQ(sendWr.wr.rdma.rkey, 222);
    ASSERT_EQ(sendWr.opcode, IBV_WR_RDMA_WRITE);
  }

  // Test 2: IBV_WR_RDMA_WRITE_WITH_IMM in SPRAY mode → opcode downgraded
  //         to IBV_WR_RDMA_WRITE (notify sent separately)
  {
    auto maybeVirtualQp =
        pd->createVirtualQp(4, &initAttr, &virtualCq, 100, 1024);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    ActiveVirtualWr pending;
    pending.localAddr = reinterpret_cast<void*>(0x1000);
    pending.remoteAddr = 0x2000;
    pending.length = 2048;
    pending.offset = 0;
    pending.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    pending.deviceKeys[deviceId] = MemoryRegionKeys{111, 222};

    auto [sendWr, sendSge] =
        virtualQp.buildPhysicalSendWr(pending, deviceId, 1024);
    sendWr.sg_list = &sendSge;

    ASSERT_EQ(sendWr.opcode, IBV_WR_RDMA_WRITE);
  }

  // Test 3: IBV_WR_RDMA_WRITE_WITH_IMM in DQPLB mode → opcode preserved,
  //         imm_data populated
  {
    auto maybeVirtualQp = pd->createVirtualQp(
        4, &initAttr, &virtualCq, 100, 1024, LoadBalancingScheme::DQPLB);
    ASSERT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    ActiveVirtualWr pending;
    pending.localAddr = reinterpret_cast<void*>(0x1000);
    pending.remoteAddr = 0x2000;
    pending.length = 1024;
    pending.offset = 0;
    pending.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    pending.deviceKeys[deviceId] = MemoryRegionKeys{111, 222};

    // Last fragment (offset + fragLen >= length) → notify bit set
    auto [sendWr, sendSge] =
        virtualQp.buildPhysicalSendWr(pending, deviceId, 1024);
    sendWr.sg_list = &sendSge;

    ASSERT_EQ(sendWr.opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
    // imm_data should have notify bit set (last fragment)
    ASSERT_NE(sendWr.imm_data & (1U << kNotifyBit), 0);
  }
}

} // namespace ibverbx

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
