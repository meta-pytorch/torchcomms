// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Forced-path tests for AnsCompress's adaptive plain/compressed encoding
// (`AnsCompress::SendArgs`). A scripted feedback source stands in for the
// transport, so each sub-chunk's decision is deterministic.

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "comms/prims/collectives/tests/AnsAdaptiveEncodingTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::prims::test {
namespace {

// 1.5 ANS pieces per sub-chunk, and a partial last sub-chunk.
constexpr std::size_t kChunkBytes = 384 * 1024;
constexpr std::size_t kNbytes = 6 * kChunkBytes + 4096 + 16;
constexpr std::size_t kNumChunks = (kNbytes + kChunkBytes - 1) / kChunkBytes;
// Comfortably above the ~1.31x compressed worst case and the plain layout.
constexpr std::size_t kStagingStride = 2 * kChunkBytes;
constexpr std::size_t kPieceBytes = 256 * 1024;
constexpr unsigned long long kPlainFlag = 1ULL << 63;
constexpr std::size_t kThreshold = 1 << 20;
constexpr std::size_t kAlwaysPlain = std::numeric_limits<std::size_t>::max();

std::size_t align16(std::size_t n) {
  return (n + 15) & ~std::size_t{15};
}

std::size_t chunkBytes(std::size_t k) {
  return std::min(kChunkBytes, kNbytes - k * kChunkBytes);
}

// Header table + raw payload, each padded to 16 bytes.
std::size_t plainStagedBytes(std::size_t k) {
  const std::size_t pieces = (chunkBytes(k) + kPieceBytes - 1) / kPieceBytes;
  return align16(pieces * sizeof(uint64_t)) + align16(chunkBytes(k));
}

class AnsAdaptiveEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
    // Low-entropy bytes, so the compressed path really shrinks them.
    src_.resize(kNbytes);
    uint32_t state = 12345;
    for (auto& b : src_) {
      state = state * 1103515245 + 12345;
      b = static_cast<uint8_t>((state >> 16) % 5);
    }
    CUDACHECK_TEST(
        cudaMemcpy(src_d_.get(), src_.data(), kNbytes, cudaMemcpyHostToDevice));
  }

  struct Encoded {
    std::vector<uint64_t> sizes;
    std::vector<bool> plain;
  };

  // Encodes src_ into staging_d_; sub-chunk k's backlog query sees pending[k].
  Encoded encode(
      const std::vector<uint64_t>& pending,
      bool feedback,
      std::size_t minPendingBytes) {
    CUDACHECK_TEST(cudaMemcpy(
        pending_d_.get(),
        pending.data(),
        pending.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice));
    ans_adaptive_send(
        static_cast<const char*>(src_d_.get()),
        static_cast<char*>(staging_d_.get()),
        kNbytes,
        kChunkBytes,
        kStagingStride,
        static_cast<const uint64_t*>(pending_d_.get()),
        feedback,
        minPendingBytes,
        static_cast<uint64_t*>(sizes_d_.get()));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    Encoded out{std::vector<uint64_t>(kNumChunks), headerFlags(staging_d_)};
    CUDACHECK_TEST(cudaMemcpy(
        out.sizes.data(),
        sizes_d_.get(),
        kNumChunks * sizeof(uint64_t),
        cudaMemcpyDeviceToHost));
    return out;
  }

  std::vector<bool> headerFlags(const DeviceBuffer& staging) {
    std::vector<bool> plain(kNumChunks);
    for (std::size_t k = 0; k < kNumChunks; ++k) {
      unsigned long long header = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &header,
          static_cast<const char*>(staging.get()) + k * kStagingStride,
          sizeof(header),
          cudaMemcpyDeviceToHost));
      plain[k] = (header & kPlainFlag) != 0;
    }
    return plain;
  }

  std::vector<uint8_t> toHost(const DeviceBuffer& buf) {
    std::vector<uint8_t> out(kNbytes);
    CUDACHECK_TEST(
        cudaMemcpy(out.data(), buf.get(), kNbytes, cudaMemcpyDeviceToHost));
    return out;
  }

  std::vector<uint8_t> decode(const DeviceBuffer& staging) {
    CUDACHECK_TEST(cudaMemset(dst_d_.get(), 0, kNbytes));
    ans_adaptive_recv(
        static_cast<const char*>(staging.get()),
        static_cast<char*>(dst_d_.get()),
        kNbytes,
        kChunkBytes,
        kStagingStride);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    return toHost(dst_d_);
  }

  std::vector<uint8_t> src_;
  DeviceBuffer src_d_{kNbytes};
  DeviceBuffer dst_d_{kNbytes};
  DeviceBuffer staging_d_{kNumChunks * kStagingStride};
  DeviceBuffer fwdStaging_d_{kNumChunks * kStagingStride};
  DeviceBuffer pending_d_{kNumChunks * sizeof(uint64_t)};
  DeviceBuffer sizes_d_{kNumChunks * sizeof(uint64_t)};
};

TEST_F(AnsAdaptiveEncodingTest, DisabledThresholdAlwaysCompresses) {
  const auto enc = encode(std::vector<uint64_t>(kNumChunks, 0), true, 0);
  EXPECT_EQ(enc.plain, std::vector<bool>(kNumChunks, false));
  for (std::size_t k = 0; k < kNumChunks; ++k) {
    EXPECT_LT(enc.sizes[k], chunkBytes(k)) << "sub-chunk " << k;
  }
  EXPECT_EQ(decode(staging_d_), src_);
}

TEST_F(AnsAdaptiveEncodingTest, BacklogBelowThresholdShipsPlain) {
  const auto enc =
      encode(std::vector<uint64_t>(kNumChunks, 0), true, kAlwaysPlain);
  EXPECT_EQ(enc.plain, std::vector<bool>(kNumChunks, true));
  for (std::size_t k = 0; k < kNumChunks; ++k) {
    EXPECT_EQ(enc.sizes[k], plainStagedBytes(k)) << "sub-chunk " << k;
  }
  EXPECT_EQ(decode(staging_d_), src_);
}

TEST_F(AnsAdaptiveEncodingTest, DecisionIsPerSubChunk) {
  std::vector<uint64_t> pending(kNumChunks);
  std::vector<bool> expectedPlain(kNumChunks);
  for (std::size_t k = 0; k < kNumChunks; ++k) {
    expectedPlain[k] = (k % 2 == 0);
    pending[k] = expectedPlain[k] ? 0 : 2 * kThreshold;
  }
  const auto enc = encode(pending, true, kThreshold);
  EXPECT_EQ(enc.plain, expectedPlain);
  EXPECT_EQ(decode(staging_d_), src_);
}

TEST_F(AnsAdaptiveEncodingTest, NoFeedbackAlwaysCompresses) {
  const auto enc =
      encode(std::vector<uint64_t>(kNumChunks, 0), false, kAlwaysPlain);
  EXPECT_EQ(enc.plain, std::vector<bool>(kNumChunks, false));
  EXPECT_EQ(decode(staging_d_), src_);
}

// recv_forward() must decode both encodings and forward each sub-chunk with
// its encoding intact, so the next hop decodes it too.
TEST_F(AnsAdaptiveEncodingTest, ForwardPreservesEncoding) {
  std::vector<uint64_t> pending(kNumChunks);
  for (std::size_t k = 0; k < kNumChunks; ++k) {
    pending[k] = (k % 2 == 0) ? 0 : 2 * kThreshold;
  }
  const auto enc = encode(pending, true, kThreshold);

  CUDACHECK_TEST(cudaMemset(dst_d_.get(), 0, kNbytes));
  ans_adaptive_forward(
      static_cast<const char*>(staging_d_.get()),
      static_cast<char*>(fwdStaging_d_.get()),
      static_cast<char*>(dst_d_.get()),
      kNbytes,
      kChunkBytes,
      kStagingStride);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  EXPECT_EQ(toHost(dst_d_), src_);
  EXPECT_EQ(headerFlags(fwdStaging_d_), enc.plain);
  EXPECT_EQ(decode(fwdStaging_d_), src_);
}

} // namespace
} // namespace comms::prims::test
