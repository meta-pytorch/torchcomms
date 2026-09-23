// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibrc/P2pIbrcHostWriter.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>

namespace comms::prims {
namespace {

class P2pIbrcHostWriterTest : public ::testing::Test {
 protected:
  P2pIbrcHostWriter makeWriter(
      uint32_t nic = 1,
      std::shared_ptr<void> lease = nullptr) {
    return P2pIbrcHostWriter(
        descs_.data(),
        &pi_,
        &ci_,
        &status_,
        static_cast<uint32_t>(descs_.size()),
        nic,
        std::move(lease));
  }

  IbgdaLocalBuffer localBuffer(int keyCount = 2) {
    NetworkLKeys keys(keyCount);
    for (int i = 0; i < keyCount; ++i) {
      keys[i] = NetworkLKey{static_cast<uint32_t>(0x1000 + i)};
    }
    return IbgdaLocalBuffer(reinterpret_cast<void*>(0x100000), keys);
  }

  IbgdaRemoteBuffer remoteBuffer(int keyCount = 2) {
    NetworkRKeys keys(keyCount);
    for (int i = 0; i < keyCount; ++i) {
      keys[i] = NetworkRKey{static_cast<uint32_t>(0x2000 + i)};
    }
    return IbgdaRemoteBuffer(reinterpret_cast<void*>(0x200000), keys);
  }

  // The reason a wait gave up, which distinguishes the per-wait bound from a
  // shared operation budget.
  static std::string waitFailure(P2pIbrcHostWriter& writer) {
    uint64_t counter = 0;
    try {
      writer.poll_counter(&counter, 1);
    } catch (const std::runtime_error& e) {
      return e.what();
    }
    return "";
  }

  std::array<IbrcDesc, 4> descs_{};
  uint64_t pi_{0};
  uint64_t ci_{0};
  IbrcNicStatus status_{};
};

TEST_F(P2pIbrcHostWriterTest, RejectsInvalidConstruction) {
  EXPECT_THROW(
      P2pIbrcHostWriter(nullptr, &pi_, &ci_, &status_, descs_.size(), 0),
      std::runtime_error);
  EXPECT_THROW(
      P2pIbrcHostWriter(
          descs_.data(), nullptr, &ci_, &status_, descs_.size(), 0),
      std::runtime_error);
  EXPECT_THROW(
      P2pIbrcHostWriter(descs_.data(), &pi_, &ci_, &status_, 3, 0),
      std::runtime_error);
}

TEST_F(P2pIbrcHostWriterTest, PublishesPutDescriptor) {
  auto writer = makeWriter();
  const auto local = localBuffer();
  const auto remote = remoteBuffer();
  const auto signal = remoteBuffer();
  uint64_t counter = 0;

  const uint64_t seq = writer.put(
      local,
      remote,
      /*nbytes=*/64,
      &signal,
      /*signalVal=*/7,
      &counter,
      /*counterVal=*/11);

  EXPECT_EQ(seq, 0);
  EXPECT_EQ(pi_, 1);
  const IbrcDesc& desc = descs_[0];
  EXPECT_EQ(desc.ready_seq, 0);
  EXPECT_EQ(desc.op, static_cast<uint16_t>(IbrcOp::PUT));
  EXPECT_EQ(desc.local_addr, reinterpret_cast<uint64_t>(local.ptr));
  EXPECT_EQ(desc.remote_addr, reinterpret_cast<uint64_t>(remote.ptr));
  EXPECT_EQ(desc.bytes, 64);
  EXPECT_EQ(desc.signal_addr, reinterpret_cast<uint64_t>(signal.ptr));
  EXPECT_EQ(desc.signal_value, 7);
  EXPECT_EQ(desc.counter_addr, reinterpret_cast<uint64_t>(&counter));
  EXPECT_EQ(desc.counter_value, 11);
  EXPECT_EQ(desc.lkey_device_order, local.lkey_per_device[1].value);
  EXPECT_EQ(desc.rkey_device_order, remote.rkey_per_device[1].value);
  EXPECT_EQ(desc.signal_rkey_device_order, signal.rkey_per_device[1].value);
  EXPECT_EQ(desc.flags, IBRC_HAS_SIGNAL | IBRC_SIGNAL_ADD | IBRC_HAS_COUNTER);
}

TEST_F(P2pIbrcHostWriterTest, RejectsUnalignedSignalTarget) {
  auto writer = makeWriter();
  NetworkRKeys keys(2);
  keys[0] = NetworkRKey{0x2000};
  keys[1] = NetworkRKey{0x2001};
  const IbgdaRemoteBuffer misaligned(reinterpret_cast<void*>(0x200004), keys);

  EXPECT_THROW(
      writer.put(localBuffer(), remoteBuffer(), 64, &misaligned),
      std::runtime_error);
  EXPECT_THROW(writer.signal(misaligned), std::runtime_error);
}

TEST_F(P2pIbrcHostWriterTest, RejectsOversizedTransfer) {
  auto writer = makeWriter();
  EXPECT_THROW(
      writer.put(localBuffer(), remoteBuffer(), std::size_t{1} << 32),
      std::runtime_error);
}

TEST_F(P2pIbrcHostWriterTest, RejectsInvalidArguments) {
  auto writer = makeWriter();
  const auto local = localBuffer();
  const auto remote = remoteBuffer();
  const IbgdaRemoteBuffer nullSignal;

  EXPECT_THROW(writer.put(local, remote, 0), std::runtime_error);
  EXPECT_THROW(writer.put(local, remote, 1, &nullSignal), std::runtime_error);
  EXPECT_THROW(writer.put(localBuffer(1), remote, 1), std::runtime_error);
  EXPECT_THROW(writer.put(local, remoteBuffer(1), 1), std::runtime_error);
  EXPECT_THROW(writer.signal(remoteBuffer(1)), std::runtime_error);
  EXPECT_THROW(writer.poll_counter(nullptr, 1), std::runtime_error);
}

TEST_F(P2pIbrcHostWriterTest, WaitPolicyBoundsPollAndBackpressure) {
  auto writer = makeWriter();
  writer.set_timeout(std::chrono::milliseconds(1));

  uint64_t counter = 0;
  EXPECT_THROW(writer.poll_counter(&counter, 1), std::runtime_error);

  pi_ = descs_.size();
  ci_ = 0;
  EXPECT_THROW(writer.signal(remoteBuffer()), std::runtime_error);
}

TEST_F(P2pIbrcHostWriterTest, AbortPredicateBreaksWait) {
  auto writer = makeWriter();
  writer.set_wait_policy(std::chrono::seconds(10), [] { return true; });

  uint64_t counter = 0;
  EXPECT_THROW(writer.poll_counter(&counter, 1), std::runtime_error);
}

// set_timeout() must not disturb the predicate the transport armed at
// getHostWriter() time; a caller adjusting only its deadline would otherwise
// silently drop back to waiting out the full timeout on an aborted job.
TEST_F(P2pIbrcHostWriterTest, TimeoutChangeKeepsAbortPredicate) {
  auto writer = makeWriter();
  writer.set_abort_predicate([] { return true; });
  writer.set_timeout(std::chrono::seconds(10));

  uint64_t counter = 0;
  const auto start = std::chrono::steady_clock::now();
  EXPECT_THROW(writer.poll_counter(&counter, 1), std::runtime_error);
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
}

// An empty ring never reaches spin_until()'s predicate, so the abort has to be
// caught before the reserve. Otherwise the producer index still advances and a
// descriptor is published, which the proxy would post to the wire.
TEST_F(P2pIbrcHostWriterTest, AbortBeforeReserveLeavesRingUntouched) {
  auto writer = makeWriter();
  writer.set_abort_predicate([] { return true; });
  descs_[0].ready_seq = kIbrcInvalidReadySeq;

  EXPECT_THROW(
      writer.put(localBuffer(), remoteBuffer(), /*nbytes=*/64),
      std::runtime_error);
  EXPECT_THROW(writer.signal(remoteBuffer()), std::runtime_error);

  EXPECT_EQ(pi_, 0);
  EXPECT_EQ(descs_[0].ready_seq, kIbrcInvalidReadySeq);
}

// A caller's operation timeout has to cover the whole operation. Each wait
// restarts the per-wait bound on its own, so an operation built from several
// waits would otherwise run to N * timeout; under a budget they share one
// absolute deadline.
TEST_F(P2pIbrcHostWriterTest, OperationBudgetIsSharedAcrossWaits) {
  auto writer = makeWriter();
  writer.set_timeout(std::chrono::seconds(30));

  const auto start = std::chrono::steady_clock::now();
  {
    const auto budget = writer.begin_op(std::chrono::milliseconds(200));
    EXPECT_NE(waitFailure(writer), "");
    EXPECT_NE(waitFailure(writer), "");
  }
  // Two waits on the 30s per-wait bound would take a minute.
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
}

TEST_F(P2pIbrcHostWriterTest, NestedOperationBudgetNarrowsOnly) {
  auto writer = makeWriter();
  writer.set_timeout(std::chrono::seconds(30));

  const auto start = std::chrono::steady_clock::now();
  {
    const auto outer = writer.begin_op(std::chrono::milliseconds(200));
    const auto inner = writer.begin_op(std::chrono::seconds(30));
    EXPECT_NE(waitFailure(writer), "");
  }
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
}

TEST_F(P2pIbrcHostWriterTest, OperationBudgetRestoresPerWaitTimeout) {
  auto writer = makeWriter();
  writer.set_timeout(std::chrono::milliseconds(1));

  EXPECT_NE(waitFailure(writer).find("timed out after 1ms"), std::string::npos);
  {
    const auto budget = writer.begin_op(std::chrono::milliseconds(1));
    EXPECT_NE(
        waitFailure(writer).find("operation budget expired"),
        std::string::npos);
  }
  EXPECT_NE(waitFailure(writer).find("timed out after 1ms"), std::string::npos);
}

// A moved-from guard must not restore; otherwise the destination's budget is
// torn down when the source dies, which is what a container of guards does.
TEST_F(P2pIbrcHostWriterTest, MovedOperationBudgetRestoresOnce) {
  auto writer = makeWriter();
  writer.set_timeout(std::chrono::milliseconds(1));

  {
    auto budget = writer.begin_op(std::chrono::milliseconds(1));
    auto moved = std::move(budget);
    // Still under a budget: a per-wait message here would mean the moved-from
    // guard restored.
    EXPECT_NE(
        waitFailure(writer).find("operation budget expired"),
        std::string::npos);
  }
  EXPECT_NE(waitFailure(writer).find("timed out after 1ms"), std::string::npos);
}

TEST_F(P2pIbrcHostWriterTest, MovedWriterRemainsUsable) {
  auto writer = makeWriter();
  auto moved = std::move(writer);

  EXPECT_NO_THROW(moved.fence());
}

/*
 * The producer claim a writer carries is the transport's, and it has to behave
 * like the one P2pIbrcHostLanes holds: alive for the holder's lifetime,
 * released when it dies. Without that a bare writer records nothing, and a
 * second writer -- or a lanes object -- silently becomes a second producer on
 * rings whose backpressure check is only sound with one.
 *
 * Stands in for the transport the same way P2pIbrcHostLanesTest does: the
 * token is the whole mechanism, so exercising it here needs no fabric.
 */
TEST_F(P2pIbrcHostWriterTest, ProducerLeaseIsHeldAndReleased) {
  std::weak_ptr<void> issued;
  {
    auto lease = std::make_shared<char>();
    issued = lease;
    auto writer = makeWriter(1, std::move(lease));
    EXPECT_FALSE(issued.expired())
        << "claim not held while the writer is alive";
  }
  EXPECT_TRUE(issued.expired()) << "claim outlived the writer";
}

// Moving a writer must carry the claim with it, or the peer looks free while a
// live writer is still driving its rings.
TEST_F(P2pIbrcHostWriterTest, ProducerLeaseSurvivesMove) {
  std::weak_ptr<void> issued;
  {
    auto lease = std::make_shared<char>();
    issued = lease;
    auto writer = makeWriter(1, std::move(lease));

    auto moved = std::move(writer);
    EXPECT_FALSE(issued.expired()) << "claim dropped by the move";
  }
  EXPECT_TRUE(issued.expired()) << "claim outlived the moved-to writer";
}

/*
 * Two positions on one peer -- the ring whose down and up peers coincide at
 * two ranks -- share the one writer rather than acquiring twice. Sharing keeps
 * a single claim, which is what makes it legal; acquiring twice is what the
 * transport refuses.
 */
TEST_F(P2pIbrcHostWriterTest, SharedWriterKeepsOneClaimForBothPositions) {
  std::weak_ptr<void> issued;
  {
    auto lease = std::make_shared<char>();
    issued = lease;
    auto writer =
        std::make_shared<P2pIbrcHostWriter>(makeWriter(1, std::move(lease)));

    auto down = writer;
    auto up = writer; // same peer at two ranks
    EXPECT_EQ(down.get(), up.get()) << "both positions must share one producer";

    writer.reset();
    down.reset();
    EXPECT_FALSE(issued.expired()) << "claim released while a holder remains";
  }
  EXPECT_TRUE(issued.expired()) << "claim outlived the last holder";
}

} // namespace
} // namespace comms::prims
