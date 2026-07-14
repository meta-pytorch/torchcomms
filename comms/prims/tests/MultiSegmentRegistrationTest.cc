// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <cstddef>
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
};

TransportHandle createTransport(IbTestBackend backend) {
  auto bootstrap = std::make_shared<
      testing::NiceMock<meta::comms::testing::MockBootstrap>>();
  auto config = makeConfig();
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

} // namespace comms::prims::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init follyInit(&argc, &argv);
  return RUN_ALL_TESTS();
}
