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

  IbgdaLocalBuffer registerBuffer(void* ptr, std::size_t size) {
    return ibgda ? ibgda->registerBuffer(ptr, size)
                 : ibrc->registerBuffer(ptr, size);
  }

  void deregisterBuffer(void* ptr) {
    if (ibgda) {
      ibgda->deregisterBuffer(ptr);
    } else {
      ibrc->deregisterBuffer(ptr);
    }
  }

  IbBufferRegistrationLease registerIbBulkBuffer(void* ptr, std::size_t size) {
    return ibgda ? ibgda->registerIbBulkBuffer(ptr, size)
                 : ibrc->registerIbBulkBuffer(ptr, size);
  }

  std::optional<IbBufferRegistrationView> lookupIbBulkBuffer(
      const IbBufferRegistrationLease& lease,
      void* ptr,
      std::size_t size) const {
    return ibgda ? ibgda->lookupIbBulkBuffer(lease, ptr, size)
                 : ibrc->lookupIbBulkBuffer(lease, ptr, size);
  }

  void deregisterIbBulkBuffer(IbBufferRegistrationLease& lease) {
    if (ibgda) {
      ibgda->deregisterIbBulkBuffer(lease);
    } else {
      ibrc->deregisterIbBulkBuffer(lease);
    }
  }

  bool isIbBulkBufferViewActive(const IbBufferRegistrationView& view) const {
    return ibgda ? ibgda->isIbBulkBufferViewActive(view)
                 : ibrc->isIbBulkBufferViewActive(view);
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

TEST_P(MultiSegmentRegistrationTest, BulkLeaseBoundsContainedViews) {
  CUDACHECK_TEST(cudaSetDevice(0));

  TransportHandle transport;
  try {
    transport = createTransport(GetParam());
  } catch (const std::exception& e) {
    GTEST_SKIP() << backendName(GetParam())
                 << " transport not available: " << e.what();
  }

  constexpr std::size_t kAllocationSize = 4 * 1024 * 1024;
  constexpr std::size_t kLeaseOffset = 512 * 1024;
  constexpr std::size_t kLeaseSize = 2 * 1024 * 1024;
  constexpr std::size_t kViewOffset = 128 * 1024;
  constexpr std::size_t kViewSize = 256 * 1024;
  void* allocation = nullptr;
  CUDACHECK_TEST(cudaMalloc(&allocation, kAllocationSize));
  auto* const leasePtr = static_cast<char*>(allocation) + kLeaseOffset;

  EXPECT_THROW(
      transport.registerIbBulkBuffer(leasePtr, 0), std::invalid_argument);
  auto lease = transport.registerIbBulkBuffer(leasePtr, kLeaseSize);
  EXPECT_THROW(
      transport.lookupIbBulkBuffer(lease, leasePtr, 0), std::invalid_argument);
  EXPECT_THROW(
      transport.lookupIbBulkBuffer(
          lease, leasePtr, std::numeric_limits<std::size_t>::max()),
      std::invalid_argument);
  auto exact = transport.lookupIbBulkBuffer(lease, leasePtr, kLeaseSize);
  auto contained =
      transport.lookupIbBulkBuffer(lease, leasePtr + kViewOffset, kViewSize);
  auto tail = transport.lookupIbBulkBuffer(lease, leasePtr + kLeaseSize - 1, 1);
  auto before = transport.lookupIbBulkBuffer(lease, leasePtr - 1, kViewSize);
  auto after = transport.lookupIbBulkBuffer(
      lease, leasePtr + kLeaseSize - kViewSize + 1, kViewSize);

  ASSERT_TRUE(exact.has_value());
  ASSERT_TRUE(contained.has_value());
  ASSERT_TRUE(tail.has_value());
  EXPECT_EQ(exact->localBuffer.ptr, leasePtr);
  EXPECT_EQ(contained->localBuffer.ptr, leasePtr + kViewOffset);
  EXPECT_EQ(exact->exchangeInfo.addr, reinterpret_cast<uint64_t>(leasePtr));
  EXPECT_EQ(
      contained->exchangeInfo.addr,
      reinterpret_cast<uint64_t>(leasePtr + kViewOffset));
  EXPECT_EQ(
      contained->exchangeInfo.numNics,
      contained->localBuffer.lkey_per_device.size);
  const auto remoteContained = contained->exchangeInfo.toRemoteBuffer();
  EXPECT_EQ(remoteContained.ptr, contained->localBuffer.ptr);
  EXPECT_EQ(
      remoteContained.rkey_per_device.size,
      contained->localBuffer.lkey_per_device.size);
  EXPECT_EQ(contained->size, kViewSize);
  EXPECT_EQ(contained->leaseGeneration, lease.generation());
  EXPECT_FALSE(before.has_value());
  EXPECT_FALSE(after.has_value());
  EXPECT_TRUE(transport.isIbBulkBufferViewActive(*contained));

  transport.deregisterIbBulkBuffer(lease);
  EXPECT_FALSE(lease.valid());
  EXPECT_FALSE(
      transport.lookupIbBulkBuffer(lease, leasePtr, kViewSize).has_value());
  EXPECT_FALSE(transport.isIbBulkBufferViewActive(*contained));
  EXPECT_THROW(transport.deregisterIbBulkBuffer(lease), std::invalid_argument);
  CUDACHECK_TEST(cudaFree(allocation));
}

TEST_P(MultiSegmentRegistrationTest, BulkLeaseReportsEffectiveStrictOrdering) {
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

  auto lease = transport.registerIbBulkBuffer(allocation, kSize);
  auto view = transport.lookupIbBulkBuffer(lease, allocation, kSize);
  ASSERT_TRUE(view.has_value());
  EXPECT_FALSE(view->relaxedOrdering);

  transport.deregisterIbBulkBuffer(lease);
  CUDACHECK_TEST(cudaFree(allocation));
}

TEST_P(MultiSegmentRegistrationTest, OverlappingBulkLeasesRemainDistinct) {
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

  auto outer = transport.registerIbBulkBuffer(base, kOuterSize);
  auto inner = transport.registerIbBulkBuffer(base + kInnerOffset, kInnerSize);
  auto outerView =
      transport.lookupIbBulkBuffer(outer, base + kInnerOffset, kInnerSize);
  auto innerView =
      transport.lookupIbBulkBuffer(inner, base + kInnerOffset, kInnerSize);

  ASSERT_TRUE(outerView.has_value());
  ASSERT_TRUE(innerView.has_value());
  EXPECT_NE(outer.generation(), inner.generation());
  EXPECT_EQ(outerView->leaseGeneration, outer.generation());
  EXPECT_EQ(innerView->leaseGeneration, inner.generation());

  transport.deregisterIbBulkBuffer(inner);
  EXPECT_FALSE(transport.isIbBulkBufferViewActive(*innerView));
  EXPECT_TRUE(transport.isIbBulkBufferViewActive(*outerView));
  transport.deregisterIbBulkBuffer(outer);
  CUDACHECK_TEST(cudaFree(allocation));
}

TEST_P(MultiSegmentRegistrationTest, ReregistrationChangesLeaseGeneration) {
  CUDACHECK_TEST(cudaSetDevice(0));

  TransportHandle transport;
  try {
    transport = createTransport(GetParam());
  } catch (const std::exception& e) {
    GTEST_SKIP() << backendName(GetParam())
                 << " transport not available: " << e.what();
  }

  constexpr std::size_t kSize = 2 * 1024 * 1024;
  void* allocation = nullptr;
  CUDACHECK_TEST(cudaMalloc(&allocation, kSize));

  auto first = transport.registerIbBulkBuffer(allocation, kSize);
  auto firstView = transport.lookupIbBulkBuffer(first, allocation, kSize);
  ASSERT_TRUE(firstView.has_value());
  const uint64_t firstGeneration = first.generation();
  transport.deregisterIbBulkBuffer(first);

  auto second = transport.registerIbBulkBuffer(allocation, kSize);
  auto secondView = transport.lookupIbBulkBuffer(second, allocation, kSize);
  ASSERT_TRUE(secondView.has_value());
  EXPECT_NE(firstGeneration, second.generation());
  EXPECT_FALSE(transport.isIbBulkBufferViewActive(*firstView));
  EXPECT_TRUE(transport.isIbBulkBufferViewActive(*secondView));

  transport.deregisterIbBulkBuffer(second);
  CUDACHECK_TEST(cudaFree(allocation));
}

} // namespace comms::prims::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init follyInit(&argc, &argv);
  return RUN_ALL_TESTS();
}
