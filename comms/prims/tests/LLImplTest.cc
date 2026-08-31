// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/prims/core/LlxPacket.cuh"
#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/tests/LLImplTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::prims {

// Memcpy opts into the LL protocol: it provides packet-aware sendLL/recvLL over
// LlxPacket geometry, so the detection traits used by the IBGDA transport's LL
// dispatch report it as LL-capable.
static_assert(has_sendLL_v<Memcpy, LlxPacketGeometry>);
static_assert(has_recvLL_v<Memcpy, LlxPacketGeometry>);
static_assert(has_forwardLL_v<Memcpy, LlxPacketGeometry>);

namespace {
// has_forwardLL_v probes a FIXED 8-argument list, so it silently reports false
// for a hook whose signature drifts -- and the transport turns that false into
// a static_assert in whichever .cu instantiated the LL forward, far from the
// edit that caused it. Pin the negative verdicts here, next to the trait.
//
// Declarations only: these are never called, and the trait is an unevaluated
// decltype.
struct NoLlHooks {};

// forwardLL present, but one argument short -- the shape a hook would have if
// it relayed packets verbatim, with no downstream generation to re-stamp with.
struct MissingFwdFlag {
  template <typename P>
  static void forwardLL(
      ThreadGroup&,
      char*,
      char*,
      const char*,
      std::size_t,
      std::size_t,
      typename P::FlagType);
};

static_assert(!has_forwardLL_v<NoLlHooks, LlxPacketGeometry>);
static_assert(!has_forwardLL_v<MissingFwdFlag, LlxPacketGeometry>);
} // namespace

namespace {

using LaunchFn = void (*)(const char*, char*, char*, std::size_t, uint32_t*);

// Drive one pack->unpack round-trip on device for geometry P and verify the
// payload comes back byte-identical.
template <typename P>
void roundTrip(LaunchFn launch, std::size_t nbytes) {
  const std::size_t wire = P::wire_bytes(nbytes);

  std::vector<char> h_src(nbytes);
  for (std::size_t i = 0; i < nbytes; ++i) {
    h_src[i] = static_cast<char>(i * 131u + 7u);
  }

  DeviceBuffer src(nbytes);
  DeviceBuffer staging(wire);
  DeviceBuffer dst(nbytes);
  DeviceBuffer errBuf(sizeof(uint32_t));
  auto* err_d = static_cast<uint32_t*>(errBuf.get());

  CUDACHECK_TEST(
      cudaMemcpy(src.get(), h_src.data(), nbytes, cudaMemcpyHostToDevice));
  // Poison staging so a missed flag/decode surfaces as a mismatch.
  CUDACHECK_TEST(cudaMemset(staging.get(), 0xEE, wire));
  CUDACHECK_TEST(cudaMemset(dst.get(), 0, nbytes));
  CUDACHECK_TEST(cudaMemset(err_d, 0, sizeof(uint32_t)));

  launch(
      static_cast<const char*>(src.get()),
      static_cast<char*>(staging.get()),
      static_cast<char*>(dst.get()),
      nbytes,
      err_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t err_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&err_h, err_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(err_h, 0u) << "payload round-trip mismatch at nbytes=" << nbytes;

  std::vector<char> h_dst(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(h_dst.data(), dst.get(), nbytes, cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_dst, h_src) << "decoded payload differs at nbytes=" << nbytes;
}

} // namespace

class LLImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }
  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

TEST_F(LLImplTest, LlRoundTrip) {
  // Exact multiples and partial final packet (kData = 4).
  for (std::size_t n :
       {std::size_t(4),
        std::size_t(7),
        std::size_t(64),
        std::size_t(1000),
        std::size_t(4096)}) {
    roundTrip<LlxPacketGeometry>(test::test_ll_pack_unpack, n);
  }
}

namespace {

// Seed accum, pack src, reduce src into accum on device, return accum.
// `expected` is built independently on the host from the same inputs.
template <typename T, typename Launch>
std::vector<T> runUnpackReduce(
    const std::vector<T>& src,
    const std::vector<T>& seed,
    Launch launch) {
  const std::size_t nelems = src.size();
  const std::size_t nbytes = nelems * sizeof(T);

  DeviceBuffer srcBuf(nbytes);
  DeviceBuffer accumBuf(nbytes);
  DeviceBuffer staging(LlxPacketGeometry::wire_bytes(nbytes));

  CUDACHECK_TEST(
      cudaMemcpy(srcBuf.get(), src.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(
      cudaMemcpy(accumBuf.get(), seed.data(), nbytes, cudaMemcpyHostToDevice));
  // Poison staging: unpack_reduce must wait for pack()'s flags, so a missing
  // readiness poll shows up as reduced garbage rather than a pass.
  CUDACHECK_TEST(
      cudaMemset(staging.get(), 0xEE, LlxPacketGeometry::wire_bytes(nbytes)));

  launch(srcBuf.get(), static_cast<char*>(staging.get()), accumBuf.get());
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<T> out(nelems);
  CUDACHECK_TEST(
      cudaMemcpy(out.data(), accumBuf.get(), nbytes, cudaMemcpyDeviceToHost));
  return out;
}

} // namespace

// sizeof(T) == kData: one element per packet, all three reduce ops.
// Values are small integers so float arithmetic is exact and the expected
// vector can be compared with ==.
TEST_F(LLImplTest, UnpackReduceOneElemPerPacket) {
  constexpr std::size_t kElems = 133; // not a multiple of the block size
  std::vector<float> src(kElems), seed(kElems);
  for (std::size_t i = 0; i < kElems; ++i) {
    src[i] = static_cast<float>(i % 17);
    seed[i] = static_cast<float>((i % 5) + 1);
  }

  const std::array<test::ReduceKind, 3> kinds{
      test::ReduceKind::Sum, test::ReduceKind::Max, test::ReduceKind::Min};
  for (const auto kind : kinds) {
    std::vector<float> expected(kElems);
    for (std::size_t i = 0; i < kElems; ++i) {
      switch (kind) {
        case test::ReduceKind::Sum:
          expected[i] = seed[i] + src[i];
          break;
        case test::ReduceKind::Max:
          expected[i] = std::max(seed[i], src[i]);
          break;
        case test::ReduceKind::Min:
          expected[i] = std::min(seed[i], src[i]);
          break;
      }
    }

    const auto actual =
        runUnpackReduce<float>(src, seed, [&](void* s, char* st, void* a) {
          test::test_ll_unpack_reduce_f32(
              static_cast<const float*>(s),
              st,
              static_cast<float*>(a),
              kElems,
              kind);
        });
    EXPECT_EQ(actual, expected)
        << "float reduce mismatch for kind " << static_cast<int>(kind);
  }
}

// sizeof(T) < kData: two elements per packet. kElems is odd so the final
// packet carries one valid element and one that must be left alone -- the
// valid_payload mask inside unpack_reduce.
TEST_F(LLImplTest, UnpackReduceTwoElemsPerPacketPartialTail) {
  constexpr std::size_t kElems = 101;
  std::vector<__half> src(kElems), seed(kElems);
  std::vector<float> expected(kElems);
  for (std::size_t i = 0; i < kElems; ++i) {
    const float s = static_cast<float>(i % 7);
    const float a = static_cast<float>(i % 3);
    src[i] = __float2half(s);
    seed[i] = __float2half(a);
    expected[i] = s + a; // <= 8, exactly representable in fp16
  }

  const auto actual =
      runUnpackReduce<__half>(src, seed, [&](void* s, char* st, void* a) {
        test::test_ll_unpack_reduce_f16(s, st, a, kElems);
      });

  std::vector<float> actualF(kElems);
  for (std::size_t i = 0; i < kElems; ++i) {
    actualF[i] = __half2float(actual[i]);
  }
  EXPECT_EQ(actualF, expected) << "fp16 two-elements-per-packet reduce wrong";
}

// sizeof(T) > kData: one element spans two packets and is reassembled
// byte-wise. Integers make any byte-order or packet-ordering slip obvious
// rather than a small numeric drift.
TEST_F(LLImplTest, UnpackReduceElementSpansPackets) {
  constexpr std::size_t kElems = 97;
  std::vector<int64_t> src(kElems), seed(kElems), expected(kElems);
  for (std::size_t i = 0; i < kElems; ++i) {
    // Distinct high and low words so a swapped packet is not self-masking.
    src[i] = static_cast<int64_t>((i + 1) * 0x0001'0000'0007ULL);
    seed[i] = static_cast<int64_t>(i);
    expected[i] = seed[i] + src[i];
  }

  const auto actual =
      runUnpackReduce<int64_t>(src, seed, [&](void* s, char* st, void* a) {
        test::test_ll_unpack_reduce_i64(
            static_cast<const int64_t*>(s),
            st,
            static_cast<int64_t*>(a),
            kElems);
      });
  EXPECT_EQ(actual, expected) << "int64 spanning-packet reduce wrong";
}

namespace {

// pack `nbytes`, optionally break packet `corruptPacket`, return the verdict.
uint32_t runAllFlagsSet(std::size_t nbytes, int corruptPacket) {
  using P = LlxPacketGeometry;
  std::vector<char> h_src(nbytes);
  for (std::size_t i = 0; i < nbytes; ++i) {
    h_src[i] = static_cast<char>(i * 131u + 7u);
  }

  DeviceBuffer src(nbytes);
  DeviceBuffer staging(P::wire_bytes(nbytes));
  DeviceBuffer readyBuf(sizeof(uint32_t));
  auto* ready_d = static_cast<uint32_t*>(readyBuf.get());

  CUDACHECK_TEST(
      cudaMemcpy(src.get(), h_src.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(staging.get(), 0xEE, P::wire_bytes(nbytes)));
  CUDACHECK_TEST(cudaMemset(ready_d, 0xFF, sizeof(uint32_t)));

  test::test_ll_all_flags_set(
      static_cast<const char*>(src.get()),
      static_cast<char*>(staging.get()),
      nbytes,
      corruptPacket,
      ready_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t ready = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&ready, ready_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  return ready;
}

} // namespace

// A fully packed chunk reports ready. kData = 4, so the sweep covers every
// `nbytes % kData` remainder as well as sizes spanning many packets.
TEST_F(LLImplTest, AllFlagsSetTrueOnCompleteChunk) {
  for (std::size_t n :
       {std::size_t(1),
        std::size_t(3),
        std::size_t(4),
        std::size_t(7),
        std::size_t(64),
        std::size_t(533),
        std::size_t(4096)}) {
    EXPECT_EQ(runAllFlagsSet(n, /*corruptPacket=*/-1), 1u)
        << "nbytes=" << n << ": complete chunk reported not-ready";
  }
}

// One packet carrying a different generation must sink the whole verdict --
// and must come back as `false`, not as a hang. The group AND-reduce is the
// part being pinned: a thread whose own packets are all fine must still agree
// with the thread that found the bad one.
TEST_F(LLImplTest, AllFlagsSetFalseOnAnyMissingPacket) {
  constexpr std::size_t kBytes = 533; // 134 packets, tail is partial
  const int nPackets =
      static_cast<int>(LlxPacketGeometry::packet_count(kBytes));

  // First, middle and last: the first is found by thread 0 immediately, the
  // last only by whichever thread owns the tail, and the middle by neither
  // edge case -- so all three exercise different threads reporting.
  for (int p : {0, nPackets / 2, nPackets - 1}) {
    EXPECT_EQ(runAllFlagsSet(kBytes, p), 0u)
        << "packet " << p << " of " << nPackets
        << " had the wrong generation but the chunk reported ready";
  }
}

namespace {

struct RepackResult {
  uint32_t flagErrors;
  std::vector<char> dst; // decoded by the relay itself (empty when useDst)
  std::vector<char> packetOut; // decoded back out of the forwarded packets
};

// pack -> repack -> inspect. See test_ll_repack() for the two-pass split.
RepackResult runRepack(const std::vector<char>& src, bool useDst) {
  using P = LlxPacketGeometry;
  const std::size_t nbytes = src.size();
  const std::size_t wire = P::wire_bytes(nbytes);

  DeviceBuffer srcBuf(nbytes);
  DeviceBuffer recvStaging(wire);
  DeviceBuffer fwdStaging(wire);
  DeviceBuffer dstBuf(nbytes);
  DeviceBuffer packetOut(nbytes);
  DeviceBuffer errBuf(sizeof(uint32_t));

  CUDACHECK_TEST(
      cudaMemcpy(srcBuf.get(), src.data(), nbytes, cudaMemcpyHostToDevice));
  // Poison both outputs: a relay that writes nothing must not look like a pass.
  CUDACHECK_TEST(cudaMemset(dstBuf.get(), 0xA5, nbytes));
  CUDACHECK_TEST(cudaMemset(packetOut.get(), 0xA5, nbytes));
  // Poison the forward staging with the UPSTREAM generation, so a relay that
  // copies packets verbatim cannot accidentally leave the right flag behind.
  CUDACHECK_TEST(cudaMemset(fwdStaging.get(), 0x07, wire));
  CUDACHECK_TEST(cudaMemset(errBuf.get(), 0, sizeof(uint32_t)));

  test::test_ll_repack(
      static_cast<const char*>(srcBuf.get()),
      static_cast<char*>(recvStaging.get()),
      static_cast<char*>(fwdStaging.get()),
      static_cast<char*>(dstBuf.get()),
      static_cast<char*>(packetOut.get()),
      nbytes,
      useDst,
      static_cast<uint32_t*>(errBuf.get()));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  RepackResult out;
  CUDACHECK_TEST(cudaMemcpy(
      &out.flagErrors, errBuf.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
  out.packetOut.resize(nbytes);
  CUDACHECK_TEST(cudaMemcpy(
      out.packetOut.data(), packetOut.get(), nbytes, cudaMemcpyDeviceToHost));
  if (useDst) {
    out.dst.resize(nbytes);
    CUDACHECK_TEST(cudaMemcpy(
        out.dst.data(), dstBuf.get(), nbytes, cudaMemcpyDeviceToHost));
  }
  return out;
}

std::vector<char> pattern(std::size_t nbytes) {
  // Position-dependent, so a relay that shifts or duplicates a packet shows up.
  std::vector<char> v(nbytes);
  for (std::size_t i = 0; i < nbytes; ++i) {
    v[i] = static_cast<char>(i * 131u + 7u);
  }
  return v;
}

} // namespace

// kData = 4, so `nbytes % 4` decides how much of the final packet is the
// packer's zero padding. Sweep every remainder, plus sizes on either side of a
// single packet and a few large enough to give every thread several packets.
TEST_F(LLImplTest, RepackReStampsAndPreservesPayload) {
  for (std::size_t n :
       {std::size_t(1),
        std::size_t(2),
        std::size_t(3),
        std::size_t(4),
        std::size_t(5),
        std::size_t(7),
        std::size_t(64),
        std::size_t(533),
        std::size_t(1000),
        std::size_t(4096)}) {
    const auto src = pattern(n);
    const auto got = runRepack(src, /*useDst=*/true);

    EXPECT_EQ(got.flagErrors, 0u)
        << "nbytes=" << n << ": " << got.flagErrors
        << " forwarded packet(s) still carry the upstream generation";
    EXPECT_EQ(got.dst, src)
        << "nbytes=" << n << ": relay decode into dst wrong";
    EXPECT_EQ(got.packetOut, src)
        << "nbytes=" << n << ": forwarded packet payload wrong";
  }
}

// Forward-only relay: the chain's intermediate ranks pass dst == nullptr, and
// the packets must still be re-stamped and carry the payload.
TEST_F(LLImplTest, RepackForwardOnlyNullDst) {
  constexpr std::size_t kBytes = 1000;
  const auto src = pattern(kBytes);
  const auto got = runRepack(src, /*useDst=*/false);

  EXPECT_EQ(got.flagErrors, 0u)
      << got.flagErrors << " forwarded packet(s) still carry the upstream "
      << "generation with dst == nullptr";
  EXPECT_EQ(got.packetOut, src) << "forward-only relay payload wrong";
}

TEST_F(LLImplTest, FlagRoundTrip) {
  DeviceBuffer p8(LlxPacketGeometry::kPacketBytes);
  CUDACHECK_TEST(cudaMemset(p8.get(), 0, LlxPacketGeometry::kPacketBytes));

  DeviceBuffer errBuf(sizeof(uint32_t));
  auto* err_d = static_cast<uint32_t*>(errBuf.get());
  CUDACHECK_TEST(cudaMemset(err_d, 0, sizeof(uint32_t)));

  test::test_ll_flag_roundtrip(p8.get(), err_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t err_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&err_h, err_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(err_h, 0u) << "store_flag/load_flag/is_flag_set round-trip wrong";
}

} // namespace comms::prims
