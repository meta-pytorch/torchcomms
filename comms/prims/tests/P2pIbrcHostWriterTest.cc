// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibrc/P2pIbrcHostWriter.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

namespace comms::prims {
namespace {

class P2pIbrcHostWriterTest : public ::testing::Test {
 protected:
  P2pIbrcHostWriter makeWriter(uint32_t nic = 1) {
    return P2pIbrcHostWriter(
        descs_.data(),
        &pi_,
        &ci_,
        &status_,
        static_cast<uint32_t>(descs_.size()),
        nic);
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
  const auto counter = localBuffer();

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
  EXPECT_EQ(desc.counter_addr, reinterpret_cast<uint64_t>(counter.ptr));
  EXPECT_EQ(desc.counter_value, 11);
  EXPECT_EQ(desc.lkey_device_order, local.lkey_per_device[1].value);
  EXPECT_EQ(desc.rkey_device_order, remote.rkey_per_device[1].value);
  EXPECT_EQ(desc.signal_rkey_device_order, signal.rkey_per_device[1].value);
  EXPECT_EQ(desc.flags, IBRC_HAS_SIGNAL | IBRC_SIGNAL_ADD | IBRC_HAS_COUNTER);
}

TEST_F(P2pIbrcHostWriterTest, RejectsInvalidArguments) {
  auto writer = makeWriter();
  const auto local = localBuffer();
  const auto remote = remoteBuffer();
  const IbgdaRemoteBuffer nullSignal;
  const IbgdaLocalBuffer nullCounter;

  EXPECT_THROW(writer.put(local, remote, 0), std::runtime_error);
  EXPECT_THROW(writer.put(local, remote, 1, &nullSignal), std::runtime_error);
  EXPECT_THROW(
      writer.put(local, remote, 1, nullptr, 1, &nullCounter),
      std::runtime_error);
  EXPECT_THROW(writer.put(localBuffer(1), remote, 1), std::runtime_error);
  EXPECT_THROW(writer.put(local, remoteBuffer(1), 1), std::runtime_error);
  EXPECT_THROW(writer.signal(remoteBuffer(1)), std::runtime_error);
  EXPECT_THROW(writer.poll_counter(nullptr, 1), std::runtime_error);
}

TEST_F(P2pIbrcHostWriterTest, WaitPolicyBoundsPollAndBackpressure) {
  auto writer = makeWriter();
  writer.set_wait_policy(std::chrono::milliseconds(1));

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

TEST_F(P2pIbrcHostWriterTest, MovedWriterRemainsUsable) {
  auto writer = makeWriter();
  auto moved = std::move(writer);

  EXPECT_NO_THROW(moved.fence());
}

} // namespace
} // namespace comms::prims
