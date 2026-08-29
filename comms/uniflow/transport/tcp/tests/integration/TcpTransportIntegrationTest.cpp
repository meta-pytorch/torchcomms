// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/MultiTransport.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/tcp/TcpRegistrationHandle.h"
#include "comms/uniflow/transport/tcp/TcpTransport.h"
#include "comms/uniflow/transport/tcp/TcpWireProtocol.h"

namespace uniflow {

// Friend wrapper (name must be exactly "SegmentTest") to build a
// RegisteredSegment with a handle for the direct-transport send/recv test.
class SegmentTest {
 public:
  static RegisteredSegment makeRegistered(
      Segment& segment,
      std::unique_ptr<RegistrationHandle> handle) {
    RegisteredSegment reg(segment);
    reg.handles_.push_back(std::move(handle));
    return reg;
  }
};

namespace {

// TCP-only factory options: a non-matching netdev prefix excludes RDMA NIC
// discovery and deviceId=-1 excludes NVLink, so only the TCP transport is
// registered — making this test host-independent (no RDMA/GPU required).
MultiTransportFactoryOptions tcpOnlyOptions() {
  MultiTransportFactoryOptions opt;
  opt.netdevPrefix = "uniflow_tcp_test_nonic";
  opt.preferredTransport = TransportType::TCP;
  opt.enableTcp = true;
  opt.tcpBindHost = "127.0.0.1";
  return opt;
}

// Same-host loopback: two in-process agents exchange bind info, connect over
// TCP, and one puts a DRAM buffer into the other's registered segment.
TEST(TcpTransportIntegration, DramPutRoundTrip) {
  constexpr size_t kLen = 8192;
  std::vector<uint8_t> src(kLen);
  std::vector<uint8_t> dst(kLen, 0);
  for (size_t i = 0; i < kLen; ++i) {
    src[i] = static_cast<uint8_t>((i * 131 + 7) & 0xFF);
  }

  MultiTransportFactory factoryA(/*deviceId=*/-1, tcpOnlyOptions());
  MultiTransportFactory factoryB(/*deviceId=*/-1, tcpOnlyOptions());

  Segment srcSeg(src.data(), kLen, MemoryType::DRAM);
  Segment dstSeg(dst.data(), kLen, MemoryType::DRAM);

  auto regA = factoryA.registerSegment(srcSeg);
  ASSERT_FALSE(regA.hasError()) << regA.error().message();
  auto regB = factoryB.registerSegment(dstSeg);
  ASSERT_FALSE(regB.hasError()) << regB.error().message();

  auto exportB = regB.value().exportId();
  ASSERT_FALSE(exportB.hasError()) << exportB.error().message();
  auto remoteOnA = factoryA.importSegment(exportB.value());
  ASSERT_FALSE(remoteOnA.hasError()) << remoteOnA.error().message();

  auto mtAResult = factoryA.createTransport(factoryB.getTopology());
  ASSERT_FALSE(mtAResult.hasError()) << mtAResult.error().message();
  auto mtBResult = factoryB.createTransport(factoryA.getTopology());
  ASSERT_FALSE(mtBResult.hasError()) << mtBResult.error().message();
  auto& mtA = mtAResult.value();
  auto& mtB = mtBResult.value();

  auto bindA = mtA->bind();
  ASSERT_FALSE(bindA.hasError()) << bindA.error().message();
  auto bindB = mtB->bind();
  ASSERT_FALSE(bindB.hasError()) << bindB.error().message();

  // connect() blocks on the listener side, so run both peers concurrently.
  Status ca = Ok();
  Status cb = Ok();
  std::thread tA([&]() { ca = mtA->connect(bindB.value()); });
  std::thread tB([&]() { cb = mtB->connect(bindA.value()); });
  tA.join();
  tB.join();
  ASSERT_FALSE(ca.hasError()) << ca.error().message();
  ASSERT_FALSE(cb.hasError()) << cb.error().message();

  std::vector<TransferRequest> requests;
  requests.push_back(
      TransferRequest{
          regA.value().span(size_t{0}, kLen),
          remoteOnA.value().span(size_t{0}, kLen)});

  auto putStatus = mtA->put(requests).get();
  ASSERT_FALSE(putStatus.hasError()) << putStatus.error().message();

  EXPECT_EQ(dst, src);

  mtA->shutdown();
  mtB->shutdown();
}

// Same-host loopback: A pulls (get) a DRAM buffer from B's registered segment.
TEST(TcpTransportIntegration, DramGetRoundTrip) {
  constexpr size_t kLen = 8192;
  std::vector<uint8_t> remoteSrc(kLen);
  std::vector<uint8_t> localDst(kLen, 0);
  for (size_t i = 0; i < kLen; ++i) {
    remoteSrc[i] = static_cast<uint8_t>((i * 17 + 3) & 0xFF);
  }

  MultiTransportFactory factoryA(/*deviceId=*/-1, tcpOnlyOptions());
  MultiTransportFactory factoryB(/*deviceId=*/-1, tcpOnlyOptions());

  Segment srcSeg(remoteSrc.data(), kLen, MemoryType::DRAM);
  Segment dstSeg(localDst.data(), kLen, MemoryType::DRAM);

  auto regB = factoryB.registerSegment(srcSeg);
  ASSERT_FALSE(regB.hasError()) << regB.error().message();
  auto regA = factoryA.registerSegment(dstSeg);
  ASSERT_FALSE(regA.hasError()) << regA.error().message();

  auto exportB = regB.value().exportId();
  ASSERT_FALSE(exportB.hasError()) << exportB.error().message();
  auto remoteOnA = factoryA.importSegment(exportB.value());
  ASSERT_FALSE(remoteOnA.hasError()) << remoteOnA.error().message();

  auto mtAResult = factoryA.createTransport(factoryB.getTopology());
  ASSERT_FALSE(mtAResult.hasError()) << mtAResult.error().message();
  auto mtBResult = factoryB.createTransport(factoryA.getTopology());
  ASSERT_FALSE(mtBResult.hasError()) << mtBResult.error().message();
  auto& mtA = mtAResult.value();
  auto& mtB = mtBResult.value();

  auto bindA = mtA->bind();
  ASSERT_FALSE(bindA.hasError()) << bindA.error().message();
  auto bindB = mtB->bind();
  ASSERT_FALSE(bindB.hasError()) << bindB.error().message();

  Status ca = Ok();
  Status cb = Ok();
  std::thread tA([&]() { ca = mtA->connect(bindB.value()); });
  std::thread tB([&]() { cb = mtB->connect(bindA.value()); });
  tA.join();
  tB.join();
  ASSERT_FALSE(ca.hasError()) << ca.error().message();
  ASSERT_FALSE(cb.hasError()) << cb.error().message();

  std::vector<TransferRequest> requests;
  requests.push_back(
      TransferRequest{
          regA.value().span(size_t{0}, kLen),
          remoteOnA.value().span(size_t{0}, kLen)});

  auto getStatus = mtA->get(requests).get();
  ASSERT_FALSE(getStatus.hasError()) << getStatus.error().message();

  EXPECT_EQ(localDst, remoteSrc);

  mtA->shutdown();
  mtB->shutdown();
}

// Both peers issue a concurrent get() to each other with large payloads. This
// exercises the mutual-READ path that would deadlock if reply sends blocked the
// reader; the dedicated sender thread must keep both readers draining.
TEST(TcpTransportIntegration, MutualGetNoDeadlock) {
  constexpr size_t kLen = 262144; // 256 KiB to stress socket buffers
  std::vector<uint8_t> aSrc(kLen), aDst(kLen, 0);
  std::vector<uint8_t> bSrc(kLen), bDst(kLen, 0);
  for (size_t i = 0; i < kLen; ++i) {
    aSrc[i] = static_cast<uint8_t>((i * 5 + 1) & 0xFF);
    bSrc[i] = static_cast<uint8_t>((i * 9 + 2) & 0xFF);
  }

  MultiTransportFactory factoryA(/*deviceId=*/-1, tcpOnlyOptions());
  MultiTransportFactory factoryB(/*deviceId=*/-1, tcpOnlyOptions());

  Segment aSrcSeg(aSrc.data(), kLen, MemoryType::DRAM);
  Segment aDstSeg(aDst.data(), kLen, MemoryType::DRAM);
  Segment bSrcSeg(bSrc.data(), kLen, MemoryType::DRAM);
  Segment bDstSeg(bDst.data(), kLen, MemoryType::DRAM);

  auto regASrc = factoryA.registerSegment(aSrcSeg);
  auto regADst = factoryA.registerSegment(aDstSeg);
  auto regBSrc = factoryB.registerSegment(bSrcSeg);
  auto regBDst = factoryB.registerSegment(bDstSeg);
  ASSERT_FALSE(regASrc.hasError());
  ASSERT_FALSE(regADst.hasError());
  ASSERT_FALSE(regBSrc.hasError());
  ASSERT_FALSE(regBDst.hasError());

  auto exportASrc = regASrc.value().exportId();
  auto exportBSrc = regBSrc.value().exportId();
  ASSERT_FALSE(exportASrc.hasError());
  ASSERT_FALSE(exportBSrc.hasError());
  auto remoteBOnA = factoryA.importSegment(exportBSrc.value());
  auto remoteAOnB = factoryB.importSegment(exportASrc.value());
  ASSERT_FALSE(remoteBOnA.hasError());
  ASSERT_FALSE(remoteAOnB.hasError());

  auto mtAResult = factoryA.createTransport(factoryB.getTopology());
  auto mtBResult = factoryB.createTransport(factoryA.getTopology());
  ASSERT_FALSE(mtAResult.hasError());
  ASSERT_FALSE(mtBResult.hasError());
  auto& mtA = mtAResult.value();
  auto& mtB = mtBResult.value();

  auto bindA = mtA->bind();
  auto bindB = mtB->bind();
  ASSERT_FALSE(bindA.hasError());
  ASSERT_FALSE(bindB.hasError());

  Status ca = Ok();
  Status cb = Ok();
  std::thread tA([&]() { ca = mtA->connect(bindB.value()); });
  std::thread tB([&]() { cb = mtB->connect(bindA.value()); });
  tA.join();
  tB.join();
  ASSERT_FALSE(ca.hasError()) << ca.error().message();
  ASSERT_FALSE(cb.hasError()) << cb.error().message();

  std::vector<TransferRequest> reqA;
  reqA.push_back(
      TransferRequest{
          regADst.value().span(size_t{0}, kLen),
          remoteBOnA.value().span(size_t{0}, kLen)});
  std::vector<TransferRequest> reqB;
  reqB.push_back(
      TransferRequest{
          regBDst.value().span(size_t{0}, kLen),
          remoteAOnB.value().span(size_t{0}, kLen)});

  auto futA = mtA->get(reqA);
  auto futB = mtB->get(reqB);
  auto statusA = futA.get();
  auto statusB = futB.get();
  ASSERT_FALSE(statusA.hasError()) << statusA.error().message();
  ASSERT_FALSE(statusB.hasError()) << statusB.error().message();

  EXPECT_EQ(aDst, bSrc);
  EXPECT_EQ(bDst, aSrc);

  mtA->shutdown();
  mtB->shutdown();
}

// Same-host loopback: two-sided send/recv between two direct TcpTransports.
TEST(TcpTransportIntegration, SendRecvRoundTrip) {
  constexpr size_t kLen = 4096;
  std::vector<uint8_t> src(kLen);
  std::vector<uint8_t> dst(kLen, 0);
  for (size_t i = 0; i < kLen; ++i) {
    src[i] = static_cast<uint8_t>((i * 37 + 5) & 0xFF);
  }

  ScopedEventBaseThread evbtA;
  ScopedEventBaseThread evbtB;
  TcpTransportFactory factoryA(/*deviceId=*/-1, evbtA.getEventBase());
  TcpTransportFactory factoryB(/*deviceId=*/-1, evbtB.getEventBase());

  Segment srcSeg(src.data(), kLen, MemoryType::DRAM);
  Segment dstSeg(dst.data(), kLen, MemoryType::DRAM);
  auto rA = factoryA.registerSegment(srcSeg);
  auto rB = factoryB.registerSegment(dstSeg);
  ASSERT_FALSE(rA.hasError()) << rA.error().message();
  ASSERT_FALSE(rB.hasError()) << rB.error().message();
  auto regA = SegmentTest::makeRegistered(srcSeg, std::move(rA.value()));
  auto regB = SegmentTest::makeRegistered(dstSeg, std::move(rB.value()));

  auto taResult = factoryA.createTransport(factoryB.getTopology());
  auto tbResult = factoryB.createTransport(factoryA.getTopology());
  ASSERT_FALSE(taResult.hasError()) << taResult.error().message();
  ASSERT_FALSE(tbResult.hasError()) << tbResult.error().message();
  auto& ta = taResult.value();
  auto& tb = tbResult.value();

  auto ia = ta->bind();
  auto ib = tb->bind();
  std::thread t1([&]() { (void)ta->connect(ib); });
  (void)tb->connect(ia);
  t1.join();

  auto fRecv = tb->recv(regB.span(size_t{0}, kLen));
  auto fSend = ta->send(regA.span(size_t{0}, kLen));
  ASSERT_FALSE(fSend.get().hasError());
  ASSERT_FALSE(fRecv.get().hasError());
  EXPECT_EQ(dst, src);

  ta->shutdown();
  tb->shutdown();
}

// A send() payload exceeding the 64 MiB wire-frame cap is rejected with a clear
// InvalidArgument. send() is single-frame (no chunking); large transfers should
// use put/get, which chunk.
TEST(TcpTransportIntegration, DramOversizeSendRejected) {
  constexpr size_t kLen = (64u << 20) + 1; // just over the 64 MiB frame cap
  std::vector<uint8_t> src(kLen);

  ScopedEventBaseThread evbtA;
  ScopedEventBaseThread evbtB;
  TcpTransportFactory factoryA(/*deviceId=*/-1, evbtA.getEventBase());
  TcpTransportFactory factoryB(/*deviceId=*/-1, evbtB.getEventBase());

  Segment srcSeg(src.data(), kLen, MemoryType::DRAM);
  auto rA = factoryA.registerSegment(srcSeg);
  ASSERT_FALSE(rA.hasError()) << rA.error().message();
  auto regA = SegmentTest::makeRegistered(srcSeg, std::move(rA.value()));

  auto taResult = factoryA.createTransport(factoryB.getTopology());
  auto tbResult = factoryB.createTransport(factoryA.getTopology());
  ASSERT_FALSE(taResult.hasError()) << taResult.error().message();
  ASSERT_FALSE(tbResult.hasError()) << tbResult.error().message();
  auto& ta = taResult.value();
  auto& tb = tbResult.value();

  auto ia = ta->bind();
  auto ib = tb->bind();
  std::thread t1([&]() { (void)ta->connect(ib); });
  (void)tb->connect(ia);
  t1.join();

  auto st = ta->send(regA.span(size_t{0}, kLen)).get();
  EXPECT_TRUE(st.hasError());
  EXPECT_EQ(st.error().code(), ErrCode::InvalidArgument);

  ta->shutdown();
  tb->shutdown();
}

// Exercises payload chunking: a >4 MiB transfer spans multiple wire frames.
// Verifies both put (write) and get (pull) reassemble correctly.
TEST(TcpTransportIntegration, DramLargeChunkedPutGet) {
  constexpr size_t kLen = 10 * 1024 * 1024; // 10 MiB > 4 MiB chunk
  std::vector<uint8_t> src(kLen);
  std::vector<uint8_t> dst(kLen, 0);
  std::vector<uint8_t> back(kLen, 0);
  for (size_t i = 0; i < kLen; ++i) {
    src[i] = static_cast<uint8_t>((i * 2654435761u) >> 24);
  }

  MultiTransportFactory factoryA(/*deviceId=*/-1, tcpOnlyOptions());
  MultiTransportFactory factoryB(/*deviceId=*/-1, tcpOnlyOptions());

  Segment srcSeg(src.data(), kLen, MemoryType::DRAM);
  Segment backSeg(back.data(), kLen, MemoryType::DRAM);
  Segment dstSeg(dst.data(), kLen, MemoryType::DRAM);

  auto regASrc = factoryA.registerSegment(srcSeg);
  auto regABack = factoryA.registerSegment(backSeg);
  auto regBDst = factoryB.registerSegment(dstSeg);
  ASSERT_FALSE(regASrc.hasError());
  ASSERT_FALSE(regABack.hasError());
  ASSERT_FALSE(regBDst.hasError());

  auto exportB = regBDst.value().exportId();
  ASSERT_FALSE(exportB.hasError());
  auto remoteBOnA = factoryA.importSegment(exportB.value());
  ASSERT_FALSE(remoteBOnA.hasError());

  auto mtAResult = factoryA.createTransport(factoryB.getTopology());
  auto mtBResult = factoryB.createTransport(factoryA.getTopology());
  ASSERT_FALSE(mtAResult.hasError());
  ASSERT_FALSE(mtBResult.hasError());
  auto& mtA = mtAResult.value();
  auto& mtB = mtBResult.value();

  auto bindA = mtA->bind();
  auto bindB = mtB->bind();
  ASSERT_FALSE(bindA.hasError());
  ASSERT_FALSE(bindB.hasError());
  Status ca = Ok();
  Status cb = Ok();
  std::thread tA([&]() { ca = mtA->connect(bindB.value()); });
  std::thread tB([&]() { cb = mtB->connect(bindA.value()); });
  tA.join();
  tB.join();
  ASSERT_FALSE(ca.hasError()) << ca.error().message();
  ASSERT_FALSE(cb.hasError()) << cb.error().message();

  std::vector<TransferRequest> putReqs;
  putReqs.push_back(
      TransferRequest{
          regASrc.value().span(size_t{0}, kLen),
          remoteBOnA.value().span(size_t{0}, kLen)});
  ASSERT_FALSE(mtA->put(putReqs).get().hasError());
  EXPECT_EQ(dst, src);

  std::vector<TransferRequest> getReqs;
  getReqs.push_back(
      TransferRequest{
          regABack.value().span(size_t{0}, kLen),
          remoteBOnA.value().span(size_t{0}, kLen)});
  ASSERT_FALSE(mtA->get(getReqs).get().hasError());
  EXPECT_EQ(back, src);

  mtA->shutdown();
  mtB->shutdown();
}

// Byte value at an absolute offset, from a hash with a period far longer than
// any transfer here. This matters: a pattern that repeats every 256 bytes (say
// `offset & 0xFF`) is invisible to a whole-chunk misplacement, because chunk
// boundaries are multiples of 256. With this, a chunk landing at the wrong
// offset -- or duplicated, dropped, or reordered -- changes the bytes.
uint8_t patternAt(uint64_t offset, uint64_t round) {
  uint64_t x = offset + 0x9E3779B97F4A7C15ULL * (round + 1);
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDULL;
  x ^= x >> 33;
  x *= 0xC4CEB9FE1A85EC53ULL;
  x ^= x >> 33;
  return static_cast<uint8_t>(x);
}

// Transfer sizes for the correctness sweep. Random sizes alone are weak here:
// every failure this transport can have lives on a boundary, and a uniform draw
// almost never lands on one. So the boundaries are enumerated explicitly and
// the random sizes only add coverage between them.
std::vector<size_t> sweepSizes(size_t maxLen) {
  constexpr size_t kChunk = 4UL * 1024 * 1024; // kMaxChunkSize in TcpTransport
  std::vector<size_t> sizes{
      0, // the (len == 0) ? 1 : ... chunk-count special case
      1,
      3,
      4095, // unaligned tails
      4096,
      4097,
      kChunk - 1, // chunk boundary: +1 leaves a 1-byte final chunk
      kChunk,
      kChunk + 1,
      2 * kChunk, // exact multiple: no remainder chunk at all
      2 * kChunk + 1,
      kMaxFrameSize - 1, // wire-frame cap / ReadRequest rejection edge
      kMaxFrameSize,
      kMaxFrameSize + 1,
      kMaxFrameSize + kChunk, // just over the outbound queue cap
      2 * kMaxFrameSize, // 128 MiB: the size that used to kill the connection
  };

  // Log-uniform, not uniform: a uniform draw over [1, maxLen] puts ~99% of
  // samples above 2 MiB and would essentially never exercise small transfers.
  // Fixed seed so a failure reproduces exactly.
  constexpr uint64_t kSeed = 0x5EED1234;
  std::mt19937_64 rng(kSeed);
  std::uniform_real_distribution<double> logDist(
      0.0, std::log(static_cast<double>(maxLen)));
  for (int i = 0; i < 12; ++i) {
    sizes.push_back(static_cast<size_t>(std::exp(logDist(rng))));
  }

  std::erase_if(sizes, [maxLen](size_t s) { return s > maxLen; });
  return sizes;
}

// Boundary + randomized size/offset sweep with content verification, over one
// connection. Two gaps this closes: nothing exercised a transfer larger than
// kMaxOutQueueBytes (which is how a 128 MiB get killing the connection reached
// a published diff), and nothing used a non-zero remote offset, so the
// `offset <= entry->len - len` bounds check and the `baseOffset + off` chunk
// arithmetic were only ever tested at offset 0.
TEST(TcpTransportIntegration, DramSizeAndOffsetSweepPutGet) {
  // Segment must exceed the largest transfer so a non-zero offset still fits.
  constexpr size_t kMaxLen = 2 * kMaxFrameSize; // 128 MiB
  constexpr size_t kSegLen = kMaxLen + 4UL * 1024 * 1024;
  std::vector<uint8_t> src(kSegLen, 0);
  std::vector<uint8_t> dst(kSegLen, 0);
  std::vector<uint8_t> back(kSegLen, 0);

  MultiTransportFactory factoryA(/*deviceId=*/-1, tcpOnlyOptions());
  MultiTransportFactory factoryB(/*deviceId=*/-1, tcpOnlyOptions());

  Segment srcSeg(src.data(), kSegLen, MemoryType::DRAM);
  Segment backSeg(back.data(), kSegLen, MemoryType::DRAM);
  Segment dstSeg(dst.data(), kSegLen, MemoryType::DRAM);

  auto regASrc = factoryA.registerSegment(srcSeg);
  auto regABack = factoryA.registerSegment(backSeg);
  auto regBDst = factoryB.registerSegment(dstSeg);
  ASSERT_FALSE(regASrc.hasError()) << regASrc.error().message();
  ASSERT_FALSE(regABack.hasError()) << regABack.error().message();
  ASSERT_FALSE(regBDst.hasError()) << regBDst.error().message();

  auto exportB = regBDst.value().exportId();
  ASSERT_FALSE(exportB.hasError()) << exportB.error().message();
  auto remoteBOnA = factoryA.importSegment(exportB.value());
  ASSERT_FALSE(remoteBOnA.hasError()) << remoteBOnA.error().message();

  auto mtAResult = factoryA.createTransport(factoryB.getTopology());
  auto mtBResult = factoryB.createTransport(factoryA.getTopology());
  ASSERT_FALSE(mtAResult.hasError()) << mtAResult.error().message();
  ASSERT_FALSE(mtBResult.hasError()) << mtBResult.error().message();
  auto& mtA = mtAResult.value();
  auto& mtB = mtBResult.value();

  auto bindA = mtA->bind();
  auto bindB = mtB->bind();
  ASSERT_FALSE(bindA.hasError()) << bindA.error().message();
  ASSERT_FALSE(bindB.hasError()) << bindB.error().message();
  Status ca = Ok();
  Status cb = Ok();
  std::thread tA([&]() { ca = mtA->connect(bindB.value()); });
  std::thread tB([&]() { cb = mtB->connect(bindA.value()); });
  tA.join();
  tB.join();
  ASSERT_FALSE(ca.hasError()) << ca.error().message();
  ASSERT_FALSE(cb.hasError()) << cb.error().message();

  const std::vector<size_t> sizes = sweepSizes(kMaxLen);
  std::mt19937_64 offsetRng(0xA5A5A5A5);
  uint64_t round = 0;

  for (size_t len : sizes) {
    // Offset 0 always, plus one unaligned offset that still fits the segment.
    std::vector<size_t> offsets{0};
    if (kSegLen > len) {
      offsets.push_back(
          std::uniform_int_distribution<size_t>(1, kSegLen - len)(offsetRng));
    }

    for (size_t offset : offsets) {
      SCOPED_TRACE(
          "len=" + std::to_string(len) + " offset=" + std::to_string(offset));
      // Re-seed every round. A transfer that silently moved nothing therefore
      // fails, because the destination still holds the previous round's
      // pattern rather than this round's.
      ++round;
      for (size_t i = 0; i < len; ++i) {
        src[offset + i] = patternAt(offset + i, round);
        dst[offset + i] = 0;
        back[offset + i] = 0;
      }

      std::vector<TransferRequest> putReqs{TransferRequest{
          regASrc.value().span(offset, len),
          remoteBOnA.value().span(offset, len)}};
      auto putStatus = mtA->put(putReqs).get();
      ASSERT_FALSE(putStatus.hasError()) << putStatus.error().message();
      for (size_t i = 0; i < len; ++i) {
        ASSERT_EQ(dst[offset + i], patternAt(offset + i, round))
            << "put mismatch at byte " << i;
      }

      std::vector<TransferRequest> getReqs{TransferRequest{
          regABack.value().span(offset, len),
          remoteBOnA.value().span(offset, len)}};
      auto getStatus = mtA->get(getReqs).get();
      ASSERT_FALSE(getStatus.hasError()) << getStatus.error().message();
      for (size_t i = 0; i < len; ++i) {
        ASSERT_EQ(back[offset + i], patternAt(offset + i, round))
            << "get mismatch at byte " << i;
      }
    }
  }

  mtA->shutdown();
  mtB->shutdown();
}

} // namespace
} // namespace uniflow
