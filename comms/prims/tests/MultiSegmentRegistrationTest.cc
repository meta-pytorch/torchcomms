// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include <gmock/gmock.h>

#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/ibrc/MultipeerIbrcTransport.h"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims::tests {

struct DisjointBuffer {
  CUdeviceptr va{0};
  std::size_t total_size{0};
  std::size_t segment_size{0};
  int num_segments{0};
  std::vector<CUmemGenericAllocationHandle> handles;

  static DisjointBuffer allocate(std::size_t size, int segments, int device) {
    DisjointBuffer buf;
    buf.total_size = size;
    buf.num_segments = segments;
    buf.segment_size = size / segments;

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    std::size_t granularity = 0;
    auto res = cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (res != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemGetAllocationGranularity failed");
    }
    if (buf.segment_size % granularity != 0) {
      throw std::runtime_error(
          "segment_size must be a multiple of allocation granularity");
    }

    res = cuMemAddressReserve(&buf.va, size, granularity, 0, 0);
    if (res != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemAddressReserve failed");
    }

    buf.handles.resize(segments);
    for (int i = 0; i < segments; ++i) {
      res = cuMemCreate(&buf.handles[i], buf.segment_size, &prop, 0);
      if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuMemCreate failed for segment");
      }
      res = cuMemMap(
          buf.va + i * buf.segment_size,
          buf.segment_size,
          0,
          buf.handles[i],
          0);
      if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuMemMap failed for segment");
      }
    }

    CUmemAccessDesc accessDesc{};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    res = cuMemSetAccess(buf.va, size, &accessDesc, 1);
    if (res != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemSetAccess failed");
    }

    return buf;
  }

  void free() {
    if (va == 0) {
      return;
    }
    for (int i = 0; i < num_segments; ++i) {
      cuMemUnmap(va + i * segment_size, segment_size);
      cuMemRelease(handles[i]);
    }
    cuMemAddressFree(va, total_size);
    va = 0;
  }

  void* ptr() const {
    // CUdeviceptr is an integer handle from the CUDA driver; converting to
    // void* is required to pass it into the transport API.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(va);
  }
};

enum class IbTestBackend {
  Ibgda,
  Ibrc,
};

const char* backendName(IbTestBackend backend) {
  switch (backend) {
    case IbTestBackend::Ibgda:
      return "IBGDA";
    case IbTestBackend::Ibrc:
      return "IBRC";
  }
  return "unknown";
}

std::string backendParamName(
    const ::testing::TestParamInfo<IbTestBackend>& info) {
  return backendName(info.param);
}

MultipeerIbTransportConfig makeConfig() {
  return MultipeerIbTransportConfig{
      .cudaDevice = 0,
      .numSignalSlots = 1,
      .numCounterSlots = 1,
      .maxGroups = 64,
  };
}

// Create a transport without MPI — uses a mock bootstrap and skips exchange().
// registerBuffer/deregisterBuffer are purely local (only need PDs from the
// constructor), so no inter-rank communication is required.
struct TransportHandle {
  std::unique_ptr<MultipeerIbgdaTransport> ibgda;
  std::unique_ptr<MultipeerIbrcTransport> ibrc;

  IbgdaLocalBuffer
  registerBuffer(void* ptr, std::size_t size, bool relaxedOrdering = false) {
    return ibgda ? ibgda->registerBuffer(ptr, size, relaxedOrdering)
                 : ibrc->registerBuffer(ptr, size, relaxedOrdering);
  }

  void deregisterBuffer(void* ptr) {
    if (ibgda) {
      ibgda->deregisterBuffer(ptr);
    } else {
      ibrc->deregisterBuffer(ptr);
    }
  }

  IbBufferRegistration registerIbBufferRange(void* ptr, std::size_t size) {
    return ibgda ? ibgda->registerIbBufferRange(ptr, size)
                 : ibrc->registerIbBufferRange(ptr, size);
  }

  void deregisterIbBufferRange(IbBufferRegistration& registration) {
    if (ibgda) {
      ibgda->deregisterIbBufferRange(registration);
    } else {
      ibrc->deregisterIbBufferRange(registration);
    }
  }
};

TransportHandle createTransport(
    IbTestBackend backend,
    const MultipeerIbTransportConfig& config) {
  auto bootstrap = std::make_shared<
      testing::NiceMock<meta::comms::testing::MockBootstrap>>();
  TransportHandle handle;
  if (backend == IbTestBackend::Ibgda) {
    handle.ibgda = std::make_unique<MultipeerIbgdaTransport>(
        0, 2, std::move(bootstrap), config);
  } else {
    handle.ibrc = std::make_unique<MultipeerIbrcTransport>(
        0, 2, std::move(bootstrap), config);
  }
  return handle;
}

TransportHandle createTransport(IbTestBackend backend) {
  return createTransport(backend, makeConfig());
}

class MultiSegmentRegistrationTest
    : public ::testing::TestWithParam<IbTestBackend> {};

INSTANTIATE_TEST_SUITE_P(
    IbBackends,
    MultiSegmentRegistrationTest,
    ::testing::Values(IbTestBackend::Ibgda, IbTestBackend::Ibrc),
    backendParamName);

TEST_P(MultiSegmentRegistrationTest, DisjointBufferRegistration) {
  CUDACHECK_TEST(cudaSetDevice(0));

  TransportHandle transport;
  try {
    transport = createTransport(GetParam());
  } catch (const std::exception& e) {
    GTEST_SKIP() << backendName(GetParam())
                 << " transport not available: " << e.what();
  }

  constexpr std::size_t kTotalSize = 8 * 1024 * 1024; // 8 MB
  constexpr int kNumSegments = 4; // 4 x 2 MB segments

  DisjointBuffer disjointBuf;
  try {
    disjointBuf = DisjointBuffer::allocate(kTotalSize, kNumSegments, 0);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "cuMem VMM allocation failed: " << e.what();
  }

  // Register the full disjoint buffer — exercises the widening logic in
  // registerBuffer when cuMemGetAddressRange returns only the first segment.
  auto reg = transport.registerBuffer(disjointBuf.ptr(), kTotalSize);
  EXPECT_NE(reg.ptr, nullptr);

  // Register a sub-buffer from the third segment — should hit the containment
  // cache (no new MR, refcount incremented).
  std::size_t segmentOffset = 2 * disjointBuf.segment_size;
  void* subPtr = static_cast<char*>(disjointBuf.ptr()) + segmentOffset;
  std::size_t subSize = disjointBuf.segment_size;

  auto subReg = transport.registerBuffer(subPtr, subSize);
  EXPECT_NE(subReg.ptr, nullptr);
  EXPECT_EQ(subReg.ptr, subPtr);
  // Sub-registration should hit the containment cache and reuse the same
  // per-NIC lkeys as the parent registration.
  ASSERT_EQ(subReg.lkey_per_device.size, reg.lkey_per_device.size);
  for (int i = 0; i < subReg.lkey_per_device.size; ++i) {
    EXPECT_EQ(subReg.lkey_per_device[i], reg.lkey_per_device[i]);
  }

  // Deregister sub-buffer first (decrements refcount).
  transport.deregisterBuffer(subPtr);

  // Deregister main buffer (drops refcount to zero, frees MR).
  transport.deregisterBuffer(disjointBuf.ptr());

  disjointBuf.free();
}

TEST_P(MultiSegmentRegistrationTest, ContiguousBufferRegistration) {
  CUDACHECK_TEST(cudaSetDevice(0));

  TransportHandle transport;
  try {
    transport = createTransport(GetParam());
  } catch (const std::exception& e) {
    GTEST_SKIP() << backendName(GetParam())
                 << " transport not available: " << e.what();
  }

  constexpr std::size_t kSize = 4 * 1024 * 1024; // 4 MB
  void* devPtr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&devPtr, kSize));

  auto reg = transport.registerBuffer(devPtr, kSize);
  EXPECT_NE(reg.ptr, nullptr);
  EXPECT_EQ(reg.ptr, devPtr);

  transport.deregisterBuffer(devPtr);
  CUDACHECK_TEST(cudaFree(devPtr));
}

TEST_P(MultiSegmentRegistrationTest, ExactRangeSpansDisjointSegments) {
  CUDACHECK_TEST(cudaSetDevice(0));

  TransportHandle transport;
  try {
    transport = createTransport(GetParam());
  } catch (const std::exception& e) {
    GTEST_SKIP() << backendName(GetParam())
                 << " transport not available: " << e.what();
  }

  constexpr std::size_t kTotalSize = 8 * 1024 * 1024;
  constexpr int kNumSegments = 4;
  DisjointBuffer disjointBuffer;
  try {
    disjointBuffer = DisjointBuffer::allocate(kTotalSize, kNumSegments, 0);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "cuMem VMM allocation failed: " << e.what();
  }

  auto registration =
      transport.registerIbBufferRange(disjointBuffer.ptr(), kTotalSize);
  EXPECT_TRUE(registration.valid());
  EXPECT_EQ(registration.localBuffer.ptr, disjointBuffer.ptr());
  EXPECT_EQ(registration.size, kTotalSize);

  transport.deregisterIbBufferRange(registration);
  disjointBuffer.free();
}

TEST_P(MultiSegmentRegistrationTest, ExactRangeRegistrationUsesRequestedVa) {
  CUDACHECK_TEST(cudaSetDevice(0));

  TransportHandle transport;
  try {
    transport = createTransport(GetParam());
  } catch (const std::exception& e) {
    GTEST_SKIP() << backendName(GetParam())
                 << " transport not available: " << e.what();
  }

  constexpr std::size_t kAllocationSize = 4 * 1024 * 1024;
  constexpr std::size_t kRangeOffset = 512 * 1024;
  constexpr std::size_t kRangeSize = 2 * 1024 * 1024;
  void* allocation = nullptr;
  CUDACHECK_TEST(cudaMalloc(&allocation, kAllocationSize));
  auto* const rangePtr = static_cast<char*>(allocation) + kRangeOffset;

  EXPECT_THROW(
      transport.registerIbBufferRange(rangePtr, 0), std::invalid_argument);
  EXPECT_THROW(
      transport.registerIbBufferRange(
          rangePtr, std::numeric_limits<std::size_t>::max()),
      std::invalid_argument);

  auto registration = transport.registerIbBufferRange(rangePtr, kRangeSize);
  EXPECT_TRUE(registration.valid());
  EXPECT_EQ(registration.localBuffer.ptr, rangePtr);
  EXPECT_EQ(registration.size, kRangeSize);

  transport.deregisterIbBufferRange(registration);
  EXPECT_FALSE(registration.valid());
  EXPECT_THROW(
      transport.deregisterIbBufferRange(registration), std::invalid_argument);
  CUDACHECK_TEST(cudaFree(allocation));
}

TEST_P(MultiSegmentRegistrationTest, ExactRangeReportsEffectiveStrictOrdering) {
  CUDACHECK_TEST(cudaSetDevice(0));

  auto config = makeConfig();
  config.enablePciRelaxedOrdering =
      MultipeerIbTransportConfig::PciRelaxedOrderingMode::Disabled;
  TransportHandle transport;
  try {
    transport = createTransport(GetParam(), config);
  } catch (const std::exception& e) {
    GTEST_SKIP() << backendName(GetParam())
                 << " transport not available: " << e.what();
  }

  constexpr std::size_t kSize = 2 * 1024 * 1024;
  void* allocation = nullptr;
  CUDACHECK_TEST(cudaMalloc(&allocation, kSize));

  auto registration = transport.registerIbBufferRange(allocation, kSize);
  EXPECT_FALSE(registration.relaxedOrdering);

  transport.deregisterIbBufferRange(registration);
  CUDACHECK_TEST(cudaFree(allocation));
}

TEST_P(MultiSegmentRegistrationTest, ExactRangeDoesNotReuseCachedRegistration) {
  CUDACHECK_TEST(cudaSetDevice(0));

  auto config = makeConfig();
  config.enablePciRelaxedOrdering =
      MultipeerIbTransportConfig::PciRelaxedOrderingMode::Enabled;
  TransportHandle transport;
  try {
    transport = createTransport(GetParam(), config);
  } catch (const std::exception& e) {
    GTEST_SKIP() << backendName(GetParam())
                 << " transport not available: " << e.what();
  }

  constexpr std::size_t kSize = 2 * 1024 * 1024;
  void* allocation = nullptr;
  CUDACHECK_TEST(cudaMalloc(&allocation, kSize));

  const auto cached =
      transport.registerBuffer(allocation, kSize, /*relaxedOrdering=*/false);
  constexpr std::size_t kRangeOffset = 17;
  constexpr std::size_t kRangeSize = kSize - 4099;
  auto* const range = static_cast<char*>(allocation) + kRangeOffset;
  auto exact = transport.registerIbBufferRange(range, kRangeSize);
  EXPECT_TRUE(exact.valid());
  EXPECT_EQ(exact.localBuffer.ptr, range);
  EXPECT_EQ(exact.size, kRangeSize);
  ASSERT_EQ(
      cached.lkey_per_device.size, exact.localBuffer.lkey_per_device.size);
  for (int nic = 0; nic < cached.lkey_per_device.size; ++nic) {
    EXPECT_NE(
        cached.lkey_per_device[nic], exact.localBuffer.lkey_per_device[nic]);
  }

  transport.deregisterIbBufferRange(exact);
  transport.deregisterBuffer(allocation);
  CUDACHECK_TEST(cudaFree(allocation));
}

TEST_P(MultiSegmentRegistrationTest, OverlappingExactRangesRemainIndependent) {
  CUDACHECK_TEST(cudaSetDevice(0));

  TransportHandle transport;
  try {
    transport = createTransport(GetParam());
  } catch (const std::exception& e) {
    GTEST_SKIP() << backendName(GetParam())
                 << " transport not available: " << e.what();
  }

  constexpr std::size_t kAllocationSize = 4 * 1024 * 1024;
  constexpr std::size_t kOuterSize = 3 * 1024 * 1024;
  constexpr std::size_t kInnerOffset = 1024 * 1024;
  constexpr std::size_t kInnerSize = 1024 * 1024;
  void* allocation = nullptr;
  CUDACHECK_TEST(cudaMalloc(&allocation, kAllocationSize));
  auto* const base = static_cast<char*>(allocation);

  auto outer = transport.registerIbBufferRange(base, kOuterSize);
  auto inner = transport.registerIbBufferRange(base + kInnerOffset, kInnerSize);
  EXPECT_TRUE(outer.valid());
  EXPECT_TRUE(inner.valid());
  EXPECT_EQ(outer.localBuffer.ptr, base);
  EXPECT_EQ(inner.localBuffer.ptr, base + kInnerOffset);

  transport.deregisterIbBufferRange(inner);
  EXPECT_FALSE(inner.valid());
  EXPECT_TRUE(outer.valid());
  transport.deregisterIbBufferRange(outer);
  CUDACHECK_TEST(cudaFree(allocation));
}

} // namespace comms::prims::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init follyInit(&argc, &argv);
  return RUN_ALL_TESTS();
}
