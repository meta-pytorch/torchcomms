// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/memory/CuMulticastAllocation.h"
#include "comms/prims/memory/MultimemHandler.h"
#include "comms/prims/platform/CudaDriverLazy.h"

namespace comms::prims::tests {
namespace {

// Skip the test cleanly if multicast / CUDA 12.3+ is unavailable on this host.
// Returns the CUdevice for device 0 on success.
bool multicastDevice(CUdevice& cuDev) {
#if CUDART_VERSION < 12030
  (void)cuDev;
  return false;
#else
  if (cuda_driver_lazy_init() != 0) {
    return false;
  }
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
    return false;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    return false;
  }
  if (!MultimemHandler::isMultimemSupported(0)) {
    return false;
  }
  return pfn_cuDeviceGet(&cuDev, 0) == CUDA_SUCCESS;
#endif
}

unsigned int posixFdMask() {
  return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
}

// Builds a `numDevices`-team multicast prop sized to the recommended multicast
// granularity. Returns a prop with `size == 0` if the driver rejects the
// granularity query (the caller should skip in that case).
CUmulticastObjectProp makeProp(unsigned int numDevices, unsigned int mask) {
  CUmulticastObjectProp prop = {};
#if CUDART_VERSION >= 12030
  prop.numDevices = numDevices;
  prop.handleTypes = mask;
  prop.flags = 0;
  std::size_t gran = 0;
  if (pfn_cuMulticastGetGranularity(
          &gran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED) == CUDA_SUCCESS &&
      gran > 0) {
    prop.size = gran;
  }
#else
  (void)numDevices;
  (void)mask;
#endif
  return prop;
}

} // namespace

TEST(CuMulticastAllocationTest, CreateProducesUsableHandle) {
  CUdevice cuDev = 0;
  if (!multicastDevice(cuDev)) {
    GTEST_SKIP() << "multicast / CUDA 12.3+ unavailable";
  }
  auto prop = makeProp(/*numDevices=*/2, posixFdMask());
  if (prop.size == 0) {
    GTEST_SKIP() << "multicast granularity query failed";
  }

  auto alloc = CuMulticastAllocation::create(prop);

  EXPECT_NE(alloc.handle(), 0u);
  EXPECT_EQ(alloc.size(), prop.size);
  // device() is 0 until addDevice runs.
  EXPECT_EQ(alloc.device(), CUdevice{0});
}

TEST(CuMulticastAllocationTest, AddDeviceRecordsLocalDevice) {
  CUdevice cuDev = 0;
  if (!multicastDevice(cuDev)) {
    GTEST_SKIP() << "multicast / CUDA 12.3+ unavailable";
  }
  auto prop = makeProp(/*numDevices=*/2, posixFdMask());
  if (prop.size == 0) {
    GTEST_SKIP() << "multicast granularity query failed";
  }

  auto alloc = CuMulticastAllocation::create(prop);
  alloc.addDevice(cuDev);

  EXPECT_EQ(alloc.device(), cuDev);
}

// Aspirational single-process exercise of the bind / unbind lifecycle. The
// CUDA driver currently makes this untestable from a single process:
//   - numDevices=1 is rejected outright with "invalid argument".
//   - numDevices>=2 with only the local device added causes bindMem to BLOCK
//     waiting for the missing peer device(s) to be added, hanging the test.
// So this test reliably hits its GTEST_SKIP path on every host we run on
// today. It's left in the file as a documented placeholder: if the driver
// ever relaxes either constraint, the test starts exercising the bind/unbind
// destructor path automatically. Until then, the bind/unbind round-trip is
// covered indirectly by MultimemHandler's ppn>=3 success-path tests.
TEST(CuMulticastAllocationTest, BindMemUnbindsOnDestruction) {
  CUdevice cuDev = 0;
  if (!multicastDevice(cuDev)) {
    GTEST_SKIP() << "multicast / CUDA 12.3+ unavailable";
  }
  auto prop = makeProp(/*numDevices=*/1, posixFdMask());
  if (prop.size == 0) {
    GTEST_SKIP() << "multicast granularity query rejected numDevices=1";
  }

  const std::size_t backingFloor =
      MultimemHandler::backingGranularity(/*cudaDevice=*/0, /*nvlRanks=*/1);
  ASSERT_GT(backingFloor, 0u);
  auto backing =
      CuMemAllocation::create(cuDev, prop.size, posixFdMask(), backingFloor);
  ASSERT_NE(backing->handle(), 0u);

  std::unique_ptr<CuMulticastAllocation> alloc;
  try {
    alloc = std::make_unique<CuMulticastAllocation>(
        CuMulticastAllocation::create(prop));
  } catch (const std::runtime_error& ex) {
    GTEST_SKIP() << "cuMulticastCreate rejected numDevices=1 on this driver: "
                 << ex.what();
  }

  alloc->addDevice(cuDev);
  alloc->bindMem(
      backing->handle(),
      /*mcOffset=*/0,
      /*physOffset=*/0,
      prop.size);
  EXPECT_EQ(alloc->size(), prop.size);

  // Destruct the multicast allocation first, then the backing. The destructor
  // should cuMulticastUnbind + cuMemRelease; the subsequent backing release
  // would surface a driver error if unbind didn't happen.
  alloc.reset();
}

// A second addDevice() on the same object must throw. The destructor only
// unbinds the single recorded `device_`, so silently overwriting it would
// leak the earlier device's binding. The check fires before any driver call,
// so it's observable on any host that supports the initial create+addDevice
// (no `numDevices=1` constraint).
TEST(CuMulticastAllocationTest, RejectsDuplicateAddDevice) {
  CUdevice cuDev = 0;
  if (!multicastDevice(cuDev)) {
    GTEST_SKIP() << "multicast / CUDA 12.3+ unavailable";
  }
  auto prop = makeProp(/*numDevices=*/2, posixFdMask());
  if (prop.size == 0) {
    GTEST_SKIP() << "multicast granularity query failed";
  }

  auto alloc = CuMulticastAllocation::create(prop);
  alloc.addDevice(cuDev);
  EXPECT_EQ(alloc.device(), cuDev);

  try {
    alloc.addDevice(cuDev);
    FAIL() << "expected std::runtime_error on duplicate addDevice";
  } catch (const std::runtime_error& ex) {
    EXPECT_NE(
        std::string(ex.what()).find("device already added"), std::string::npos)
        << ex.what();
  }
  // device_ must still match the original successful addDevice; a guard that
  // ran the driver call before throwing would have overwritten it.
  EXPECT_EQ(alloc.device(), cuDev);
}

// A second bindMem() on the same object must throw. The destructor unbinds
// exactly one (mcOffset, size) range, so silently overwriting it would leak
// the earlier binding. Uses the same single-process bind path as
// `BindMemUnbindsOnDestruction` above and will skip on hosts where the driver
// rejects single-process bind setup (which is currently every host); the
// guard itself is also covered as a code-review item with the matching throw
// site in `CuMulticastAllocation::bindMem`.
TEST(CuMulticastAllocationTest, RejectsDuplicateBindMem) {
  CUdevice cuDev = 0;
  if (!multicastDevice(cuDev)) {
    GTEST_SKIP() << "multicast / CUDA 12.3+ unavailable";
  }
  auto prop = makeProp(/*numDevices=*/1, posixFdMask());
  if (prop.size == 0) {
    GTEST_SKIP() << "multicast granularity query rejected numDevices=1";
  }

  const std::size_t backingFloor =
      MultimemHandler::backingGranularity(/*cudaDevice=*/0, /*nvlRanks=*/1);
  ASSERT_GT(backingFloor, 0u);
  auto backing =
      CuMemAllocation::create(cuDev, prop.size, posixFdMask(), backingFloor);
  ASSERT_NE(backing->handle(), 0u);

  std::unique_ptr<CuMulticastAllocation> alloc;
  try {
    alloc = std::make_unique<CuMulticastAllocation>(
        CuMulticastAllocation::create(prop));
  } catch (const std::runtime_error& ex) {
    GTEST_SKIP() << "cuMulticastCreate rejected numDevices=1 on this driver: "
                 << ex.what();
  }

  alloc->addDevice(cuDev);
  alloc->bindMem(
      backing->handle(),
      /*mcOffset=*/0,
      /*physOffset=*/0,
      prop.size);

  try {
    alloc->bindMem(
        backing->handle(),
        /*mcOffset=*/0,
        /*physOffset=*/0,
        prop.size);
    FAIL() << "expected std::runtime_error on duplicate bindMem";
  } catch (const std::runtime_error& ex) {
    EXPECT_NE(std::string(ex.what()).find("already bound"), std::string::npos)
        << ex.what();
  }
}

TEST(CuMulticastAllocationTest, MoveConstructorTransfersOwnership) {
  CUdevice cuDev = 0;
  if (!multicastDevice(cuDev)) {
    GTEST_SKIP() << "multicast / CUDA 12.3+ unavailable";
  }
  auto prop = makeProp(/*numDevices=*/2, posixFdMask());
  if (prop.size == 0) {
    GTEST_SKIP() << "multicast granularity query failed";
  }

  auto src = CuMulticastAllocation::create(prop);
  const auto srcHandle = src.handle();
  CuMulticastAllocation dst(std::move(src));

  EXPECT_EQ(dst.handle(), srcHandle);
  EXPECT_EQ(dst.size(), prop.size);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentionally checking moved-from
  EXPECT_EQ(src.handle(), 0u);
}

// Move-assignment must release the destination's prior handle BEFORE adopting
// the source's. If it didn't, the prior handle would leak (no release) and
// later code that observes the FD count or cuMemRelease semantics would catch
// it; this test asserts the visible post-state -- handle and size correctly
// transferred, source inert.
TEST(CuMulticastAllocationTest, MoveAssignmentReleasesPriorHandleAndTransfers) {
  CUdevice cuDev = 0;
  if (!multicastDevice(cuDev)) {
    GTEST_SKIP() << "multicast / CUDA 12.3+ unavailable";
  }
  auto prop = makeProp(/*numDevices=*/2, posixFdMask());
  if (prop.size == 0) {
    GTEST_SKIP() << "multicast granularity query failed";
  }

  auto a = CuMulticastAllocation::create(prop);
  auto b = CuMulticastAllocation::create(prop);
  const auto bHandle = b.handle();

  a = std::move(b);

  EXPECT_EQ(a.handle(), bHandle);
  EXPECT_EQ(a.size(), prop.size);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentionally checking moved-from
  EXPECT_EQ(b.handle(), 0u);
}

// adopt(0, 0) constructs a zero-handle object. The destructor's release()
// must early-return on `handle_ == 0` and NOT call into the CUDA driver --
// this is the "I'm holding a moved-from / never-imported handle" case. A
// regression that called cuMemRelease(0) would surface as a CUDA error on
// destruction.
TEST(CuMulticastAllocationTest, AdoptWithZeroHandleHasInertDestruction) {
  auto inert = CuMulticastAllocation::adopt(/*handle=*/0, /*size=*/0);
  EXPECT_EQ(inert.handle(), 0u);
  EXPECT_EQ(inert.size(), 0u);
  // Destructor at scope exit must not call cuMemRelease on handle=0.
}

} // namespace comms::prims::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
