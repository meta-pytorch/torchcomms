// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include "comms/torchcomms/ncclx/TorchCommNCCLXCCA.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/CachingAllocatorHookMock.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/NcclxMock.hpp"

namespace torch::comms::test {

using ::testing::NiceMock;
using ::testing::Return;

// Lightweight test fixture for CCA memory hook — no communicator, no
// rank/size, no store. The CCA singleton starts uninitialized (nullptr).
class MemoryCCAHookTest : public ::testing::Test {
 protected:
  void SetUp() override {
    nccl_mock_ = std::make_shared<NiceMock<NcclxMock>>();
  }

  void TearDown() override {
    NcclxCachingAllocatorHook::setInstance(nullptr);
  }

  c10::cuda::CUDACachingAllocator::TraceEntry createAllocation(uintptr_t addr) {
    return c10::cuda::CUDACachingAllocator::TraceEntry(
        c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC,
        0, // device
        static_cast<size_t>(addr),
        1024, // size
        nullptr, // stream
        c10::cuda::MempoolId_t{0, 0},
        c10::approx_time_t{0});
  }

  c10::cuda::CUDACachingAllocator::TraceEntry createDeallocation(
      uintptr_t addr) {
    return c10::cuda::CUDACachingAllocator::TraceEntry(
        c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE,
        0, // device
        static_cast<size_t>(addr),
        1024, // size
        nullptr, // stream
        c10::cuda::MempoolId_t{0, 0},
        c10::approx_time_t{0});
  }

  std::shared_ptr<NiceMock<NcclxMock>> nccl_mock_;
};

TEST_F(MemoryCCAHookTest, RegDeregWorksAfterAttach) {
  // Use real NcclxCachingAllocatorHookImpl (not mock) with mock NCCL API
  auto hook = std::make_unique<NcclxCachingAllocatorHookImpl>();
  hook->setNcclApi(nccl_mock_);
  NcclxCachingAllocatorHook::setInstance(std::move(hook));

  EXPECT_NO_THROW(NcclxCachingAllocatorHook::getInstance());

  // Simulate allocation — should call globalRegisterWithPtr
  auto alloc_entry = createAllocation(0x9000);
  EXPECT_CALL(
      *nccl_mock_, globalRegisterWithPtr(reinterpret_cast<void*>(0x9000), 1024))
      .WillOnce(Return(ncclSuccess));
  NcclxCachingAllocatorHook::getInstance().regDeregMem(alloc_entry);

  // Simulate deallocation — should call globalDeregisterWithPtr
  auto dealloc_entry = createDeallocation(0x9000);
  EXPECT_CALL(
      *nccl_mock_,
      globalDeregisterWithPtr(reinterpret_cast<void*>(0x9000), 1024))
      .WillOnce(Return(ncclSuccess));
  NcclxCachingAllocatorHook::getInstance().regDeregMem(dealloc_entry);
}

} // namespace torch::comms::test
