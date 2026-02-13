// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/tests/IbverbxTestFixture.h"

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

namespace ibverbx {

TEST_F(IbverbxTestFixture, ActiveVirtualWr) {
  // isComplete
  {
    ActiveVirtualWr wr;
    wr.remainingMsgCnt = 2;
    ASSERT_FALSE(wr.isComplete());
    wr.remainingMsgCnt = 1;
    ASSERT_FALSE(wr.isComplete());
    wr.remainingMsgCnt = 0;
    ASSERT_TRUE(wr.isComplete());
  }

  // isSendOp
  {
    ActiveVirtualWr wr;
    for (auto opcode :
         {IBV_WR_SEND,
          IBV_WR_RDMA_WRITE,
          IBV_WR_RDMA_WRITE_WITH_IMM,
          IBV_WR_RDMA_READ,
          IBV_WR_ATOMIC_FETCH_AND_ADD,
          IBV_WR_ATOMIC_CMP_AND_SWP}) {
      wr.opcode = opcode;
      ASSERT_TRUE(wr.isSendOp()) << "opcode " << opcode << " should be send op";
    }
    wr.opcode = IBV_WR_SEND_WITH_IMM;
    ASSERT_FALSE(wr.isSendOp());
  }
}

TEST_F(IbverbxTestFixture, WrTracker) {
  // add, find, remove
  {
    WrTracker<ActiveVirtualWr> tracker;

    ActiveVirtualWr wr0;
    wr0.userWrId = 100;
    ActiveVirtualWr wr1;
    wr1.userWrId = 200;

    ASSERT_EQ(tracker.add(std::move(wr0)), 0);
    ASSERT_EQ(tracker.add(std::move(wr1)), 1);
    ASSERT_EQ(tracker.activeCount(), 2);

    ASSERT_NE(tracker.find(0), nullptr);
    ASSERT_EQ(tracker.find(0)->userWrId, 100);
    ASSERT_NE(tracker.find(1), nullptr);
    ASSERT_EQ(tracker.find(1)->userWrId, 200);

    // const find
    const auto& constTracker = tracker;
    ASSERT_NE(constTracker.find(0), nullptr);

    // find non-existent
    ASSERT_EQ(tracker.find(999), nullptr);

    // remove
    tracker.remove(0);
    ASSERT_EQ(tracker.activeCount(), 1);
    ASSERT_EQ(tracker.find(0), nullptr);
    ASSERT_NE(tracker.find(1), nullptr);

    // remove non-existent is a no-op
    tracker.remove(999);
    ASSERT_EQ(tracker.activeCount(), 1);
  }

  // queues and counts
  {
    WrTracker<ActiveVirtualWr> tracker;
    ASSERT_FALSE(tracker.hasPendingPost());
    ASSERT_FALSE(tracker.hasPendingCompletion());

    for (int i = 0; i < 3; i++) {
      ActiveVirtualWr wr;
      wr.userWrId = i;
      tracker.add(std::move(wr));
    }
    ASSERT_EQ(tracker.activeCount(), 3);
    ASSERT_EQ(tracker.pendingPostCount(), 3);
    ASSERT_EQ(tracker.pendingCompletionCount(), 3);

    // FIFO order for pending queue
    ASSERT_EQ(tracker.frontPendingPost(), 0);
    tracker.popPendingPost();
    ASSERT_EQ(tracker.frontPendingPost(), 1);
    tracker.popPendingPost();
    ASSERT_EQ(tracker.frontPendingPost(), 2);
    tracker.popPendingPost();
    ASSERT_FALSE(tracker.hasPendingPost());

    // Outstanding queue is independent
    ASSERT_EQ(tracker.pendingCompletionCount(), 3);
    ASSERT_EQ(tracker.frontPendingCompletion(), 0);
    tracker.popPendingCompletion();
    ASSERT_EQ(tracker.frontPendingCompletion(), 1);
    tracker.popPendingCompletion();
    tracker.popPendingCompletion();
    ASSERT_FALSE(tracker.hasPendingCompletion());

    // Popping queues does not affect active map
    ASSERT_EQ(tracker.activeCount(), 3);
  }

  // move semantics
  {
    WrTracker<ActiveVirtualWr> tracker;

    ActiveVirtualWr wr;
    wr.userWrId = 42;
    wr.deviceKeys[0] = MemoryRegionKeys{111, 222};
    wr.deviceKeys[1] = MemoryRegionKeys{333, 444};

    uint64_t id = tracker.add(std::move(wr));
    ASSERT_TRUE(wr.deviceKeys.empty()); // NOLINT(bugprone-use-after-move)

    auto* found = tracker.find(id);
    ASSERT_NE(found, nullptr);
    ASSERT_EQ(found->deviceKeys.size(), 2);
    ASSERT_EQ(found->deviceKeys[0].lkey, 111);
    ASSERT_EQ(found->deviceKeys[1].rkey, 444);
  }
}

TEST_F(IbverbxTestFixture, IbvVirtualWcDefaults) {
  IbvVirtualWc wc;
  ASSERT_EQ(wc.wrId, 0);
  ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
  ASSERT_EQ(wc.opcode, IBV_WC_SEND);
  ASSERT_EQ(wc.qpNum, 0);
  ASSERT_EQ(wc.immData, 0);
  ASSERT_EQ(wc.byteLen, 0);
}

TEST_F(IbverbxTestFixture, DqplbSeqTrackerGetSendImm) {
  DqplbSeqTracker tracker;

  // Test 1: Basic sequence number generation
  {
    uint32_t imm1 = tracker.getSendImm(2); // remainingMsgCnt = 2, not last msg
    ASSERT_EQ(imm1 & kSeqNumMask, 0); // First sequence number should be 0
    ASSERT_EQ(imm1 & (1U << kNotifyBit), 0); // Notify bit should not be set

    uint32_t imm2 = tracker.getSendImm(2); // remainingMsgCnt = 2, not last msg
    ASSERT_EQ(imm2 & kSeqNumMask, 1); // Second sequence number should be 1
    ASSERT_EQ(imm2 & (1U << kNotifyBit), 0); // Notify bit should not be set

    uint32_t imm3 = tracker.getSendImm(2); // remainingMsgCnt = 2, not last msg
    ASSERT_EQ(imm3 & kSeqNumMask, 2); // Third sequence number should be 2
    ASSERT_EQ(imm3 & (1U << kNotifyBit), 0); // Notify bit should not be set
  }

  // Test 2: Notify bit is set when remainingMsgCnt == 1
  {
    uint32_t imm = tracker.getSendImm(1); // remainingMsgCnt = 1, last msg
    ASSERT_EQ(imm & kSeqNumMask, 3); // Fourth sequence number should be 3
    ASSERT_EQ(
        imm & (1U << kNotifyBit),
        (1U << kNotifyBit)); // Notify bit should be set
  }

  // Test 3: Sequence number wraps around after reaching kSeqNumMask
  {
    // Continue incrementing to test wrap-around
    // Current sequence is 4 (from previous tests: 0, 1, 2, 3)
    // kSeqNumMask = 0xFFFFFF (24 bits), so we need to reach 0xFFFFFF and wrap
    // to 0

    // Fast forward to near the end of sequence space
    for (uint32_t i = 4; i < kSeqNumMask - 1; i++) {
      uint32_t imm = tracker.getSendImm(2);
      ASSERT_EQ(imm & kSeqNumMask, i);
      ASSERT_EQ(imm & (1U << kNotifyBit), 0);
    }

    // Test wrap-around: sequence should go from kSeqNumMask - 1 to 0
    uint32_t immBeforeWrap = tracker.getSendImm(2);
    ASSERT_EQ(immBeforeWrap & kSeqNumMask, kSeqNumMask - 1);
    ASSERT_EQ(immBeforeWrap & (1U << kNotifyBit), 0);

    uint32_t immAfterWrap = tracker.getSendImm(2);
    ASSERT_EQ(immAfterWrap & kSeqNumMask, 0); // Should wrap to 0
    ASSERT_EQ(immAfterWrap & (1U << kNotifyBit), 0);

    // Verify it continues from 0
    uint32_t immAfterWrap2 = tracker.getSendImm(1);
    ASSERT_EQ(immAfterWrap2 & kSeqNumMask, 1);
    ASSERT_EQ(
        immAfterWrap2 & (1U << kNotifyBit),
        (1U << kNotifyBit)); // Last message has notify bit
  }
}

TEST_F(IbverbxTestFixture, DqplbSeqTrackerProcessReceivedImm) {
  // Test 1: Process messages in order without notify bit
  {
    DqplbSeqTracker tracker;

    uint32_t imm0 = 0 % kSeqNumMask; // Seq 0, no notify
    uint32_t imm1 = 1 % kSeqNumMask; // Seq 1, no notify
    uint32_t imm2 = 2 % kSeqNumMask; // Seq 2, no notify

    int notify0 = tracker.processReceivedImm(imm0);
    ASSERT_EQ(notify0, 0); // No notify bit, so notify count is 0

    int notify1 = tracker.processReceivedImm(imm1);
    ASSERT_EQ(notify1, 0); // No notify bit, so notify count is 0

    int notify2 = tracker.processReceivedImm(imm2);
    ASSERT_EQ(notify2, 0); // No notify bit, so notify count is 0
  }

  // Test 2: Process messages in order with notify bit
  {
    DqplbSeqTracker tracker2;

    uint32_t imm0 = (0 % kSeqNumMask) | (1U << kNotifyBit); // Seq 0 with notify
    uint32_t imm1 = (1 % kSeqNumMask); // Seq 1, no notify
    uint32_t imm2 = (2 % kSeqNumMask) | (1U << kNotifyBit); // Seq 2 with notify

    int notify0 = tracker2.processReceivedImm(imm0);
    ASSERT_EQ(notify0, 1); // Notify bit set, expect notify count 1

    int notify1 = tracker2.processReceivedImm(imm1);
    ASSERT_EQ(notify1, 0); // No notify bit, so notify count is 0

    int notify2 = tracker2.processReceivedImm(imm2);
    ASSERT_EQ(notify2, 1); // Notify bit set, expect notify count 1
  }

  // Test 3: Process messages out of order - later messages arrive first
  {
    DqplbSeqTracker tracker3;

    // Receive messages out of order: 2, 1, 0
    uint32_t imm2 = (2 % kSeqNumMask) | (1U << kNotifyBit); // Seq 2 with notify
    uint32_t imm1 = (1 % kSeqNumMask); // Seq 1, no notify
    uint32_t imm0 = (0 % kSeqNumMask); // Seq 0, no notify

    // Receive seq 2 first (out of order)
    int notify2 = tracker3.processReceivedImm(imm2);
    ASSERT_EQ(
        notify2,
        0); // Even though notify bit is set, can't process until seq 0, 1
            // arrive

    // Receive seq 1 (still out of order)
    int notify1 = tracker3.processReceivedImm(imm1);
    ASSERT_EQ(notify1, 0); // Still waiting for seq 0

    // Receive seq 0 (now in order)
    int notify0 = tracker3.processReceivedImm(imm0);
    ASSERT_EQ(
        notify0,
        1); // Now all messages are in order, process all: 0 (no notify), 1 (no
            // notify), 2 (notify) = 1 notify total
  }

  // Test 4: Process messages out of order with multiple notify bits
  {
    DqplbSeqTracker tracker4;

    // Receive messages: 3, 1, 0, 2
    uint32_t imm0 = (0 % kSeqNumMask) | (1U << kNotifyBit); // Seq 0 with notify
    uint32_t imm1 = (1 % kSeqNumMask); // Seq 1, no notify
    uint32_t imm2 = (2 % kSeqNumMask) | (1U << kNotifyBit); // Seq 2 with notify
    uint32_t imm3 = (3 % kSeqNumMask) | (1U << kNotifyBit); // Seq 3 with notify

    // Receive seq 3 first (far out of order)
    int notify3 = tracker4.processReceivedImm(imm3);
    ASSERT_EQ(notify3, 0); // Waiting for seq 0

    // Receive seq 1 (out of order)
    int notify1 = tracker4.processReceivedImm(imm1);
    ASSERT_EQ(notify1, 0); // Still waiting for seq 0

    // Receive seq 0 (can process 0 and 1 now)
    int notify0 = tracker4.processReceivedImm(imm0);
    ASSERT_EQ(
        notify0, 1); // Process seq 0 (notify) and seq 1 (no notify) = 1 notify

    // Receive seq 2 (can process 2 and 3 now)
    int notify2 = tracker4.processReceivedImm(imm2);
    ASSERT_EQ(
        notify2, 2); // Process seq 2 (notify) and seq 3 (notify) = 2 notifies
  }

  // Test 5: Process with sequence wrap-around
  {
    DqplbSeqTracker tracker5;

    // Fast forward receiveNext_ to near the end of sequence space
    for (uint32_t i = 0; i < kSeqNumMask - 2; i++) {
      uint32_t imm = (i % kSeqNumMask);
      tracker5.processReceivedImm(imm);
    }

    // Now receiveNext_ should be at kSeqNumMask - 2
    // Test wrap-around: receive messages kSeqNumMask - 2, kSeqNumMask - 1, 0, 1
    uint32_t immBeforeWrap1 =
        ((kSeqNumMask - 2) % kSeqNumMask) | (1U << kNotifyBit);
    uint32_t immBeforeWrap2 = ((kSeqNumMask - 1) % kSeqNumMask);
    uint32_t immAfterWrap1 = (kSeqNumMask % kSeqNumMask);
    uint32_t immAfterWrap2 =
        ((kSeqNumMask + 1) % kSeqNumMask) | (1U << kNotifyBit);

    int notify1 = tracker5.processReceivedImm(immBeforeWrap1);
    ASSERT_EQ(notify1, 1); // Seq kSeqNumMask - 2 with notify = 1

    int notify2 = tracker5.processReceivedImm(immBeforeWrap2);
    ASSERT_EQ(notify2, 0); // Seq kSeqNumMask - 1, no notify = 0

    int notify3 = tracker5.processReceivedImm(immAfterWrap1);
    ASSERT_EQ(notify3, 0); // Seq 0, no notify = 0 (wrap around works)

    int notify4 = tracker5.processReceivedImm(immAfterWrap2);
    ASSERT_EQ(notify4, 1); // Seq 1 with notify = 1
  }

  // Test 6: Process with sequence wrap-around and out-of-order
  {
    DqplbSeqTracker tracker6;

    // Fast forward receiveNext_ to near the end of sequence space
    for (uint32_t i = 0; i < kSeqNumMask - 2; i++) {
      uint32_t imm = (i & kSeqNumMask);
      tracker6.processReceivedImm(imm);
    }

    // Now receiveNext_ should be at kSeqNumMask - 2
    // Test wrap-around: receive messages kSeqNumMask - 2, kSeqNumMask - 1, 0, 1
    uint32_t immBeforeWrap1 =
        ((kSeqNumMask - 2) % kSeqNumMask) | (1U << kNotifyBit);
    uint32_t immBeforeWrap2 = ((kSeqNumMask - 1) % kSeqNumMask);
    uint32_t immAfterWrap1 = (kSeqNumMask % kSeqNumMask);
    uint32_t immAfterWrap2 =
        ((kSeqNumMask + 1) % kSeqNumMask) | (1U << kNotifyBit);

    int notify1 = tracker6.processReceivedImm(immBeforeWrap2);
    ASSERT_EQ(
        notify1,
        0); // Receiver receives immBeforeWrap2, no notification so notify1 is 0

    int notify2 = tracker6.processReceivedImm(immAfterWrap2);
    ASSERT_EQ(
        notify2, 0); // Receiver receives immAfterWrap2. Though notification bit
                     // is set, but received out of order so notify2 is 0

    int notify3 = tracker6.processReceivedImm(immBeforeWrap1);
    ASSERT_EQ(notify3, 1); // Receiver receives immBeforeWrap1, notification bit
                           // is set, and received in order, so notify3 is 1

    int notify4 = tracker6.processReceivedImm(immAfterWrap1);
    ASSERT_EQ(notify4, 1); // Receiver receives immAfterWrap1, notification bit
                           // is not set. We received immAfterWrap2 before with
                           // notification bit set. So notify4 is 1
  }
}

} // namespace ibverbx

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
