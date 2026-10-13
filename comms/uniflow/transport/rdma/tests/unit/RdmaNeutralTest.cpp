// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/rdma/CopyEngine.h"
#include "comms/uniflow/transport/rdma/RdmaRegistrationHandle.h"
#include "comms/uniflow/transport/rdma/RdmaTransportProvider.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/uniflow/drivers/DeviceAdapterAccess.h"
#include "comms/uniflow/drivers/ibverbs/mock/MockIbvApi.h"

namespace uniflow {
namespace {

using ::testing::_;
using ::testing::Return;

class ManualEventBase final : public EventBase {
 public:
  void loop() override {
    while (runOne()) {
    }
  }
  void stop() noexcept override {}
  void dispatch(Func callback) override {
    callbacks_.push_back(std::move(callback));
  }
  void dispatchInline(Func callback) override {
    callback();
  }
  void dispatchAndWait(Func callback) override {
    callback();
  }
  [[noreturn]] void registerFd(int, uint32_t, IOCallback) override {
    throw std::logic_error("RDMA unit must not register an fd");
  }
  [[noreturn]] void unregisterFd(int) override {
    throw std::logic_error("RDMA unit must not unregister an fd");
  }
  bool inLoopThread() const noexcept override {
    return inCallback_;
  }
  bool isLoopRunning() const noexcept override {
    return false;
  }
  bool runOne() {
    if (callbacks_.empty()) {
      return false;
    }
    auto callback = std::move(callbacks_.front());
    callbacks_.pop_front();
    inCallback_ = true;
    callback();
    inCallback_ = false;
    return true;
  }

 private:
  std::deque<Func> callbacks_;
  bool inCallback_{false};
};

struct CompletionState {
  bool ready{false};
  bool queryError{false};
  unsigned destroyed{0};
};

class TestCompletion final : public DeviceCompletion {
 public:
  explicit TestCompletion(std::shared_ptr<CompletionState> state)
      : state_(std::move(state)) {}
  ~TestCompletion() override {
    ++state_->destroyed;
  }
  Result<bool> ready() override {
    if (state_->queryError) {
      return Err(ErrCode::DriverError, "mock query failure");
    }
    return state_->ready;
  }
  Status wait() override {
    return state_->ready ? Ok()
                         : Status(Err(ErrCode::DriverError, "mock not ready"));
  }

 private:
  std::shared_ptr<CompletionState> state_;
};

class TestDeviceAdapter final : public DeviceAdapter {
 public:
  enum class Failure {
    None,
    Copy,
    ThrowCopy,
    Signal,
    Record,
    Query,
    Drain,
  };

  DeviceCapabilities capabilities() const noexcept override {
    return caps;
  }
  Result<void*> pinnedHostAlloc(size_t size) override {
    return allocate(-1, size);
  }
  Result<void*> pinnedHostAlloc(int deviceId, size_t size) override {
    return allocate(deviceId, size);
  }
  Status pinnedHostFree(void* ptr) override {
    return free(-1, ptr);
  }
  Status pinnedHostFree(int deviceId, void* ptr) override {
    return free(deviceId, ptr);
  }
  Result<void*> hostGetDevicePointer(void* ptr) override {
    mappedDevices.push_back(-1);
    return ptr;
  }
  Result<void*> hostGetDevicePointer(int deviceId, void* ptr) override {
    mappedDevices.push_back(deviceId);
    return ptr;
  }
  Result<bool> isDmaBuffSupported(int deviceId) override {
    exportedDevice = deviceId;
    return false;
  }
  Result<DmaBuff> exportDmaBuff(int, void*, size_t, DmaBufMapping) override {
    return Err(ErrCode::NotImplemented);
  }
  Status closeDmaBuff(DmaBuff&) override {
    return Ok();
  }
  Status copyAsync(
      int deviceId,
      void* dst,
      const void* src,
      size_t size,
      DeviceCopyKind kind,
      DeviceStream stream) override {
    ++copies;
    copiedDevice = deviceId;
    copiedStream = stream;
    copiedKind = kind;
    pending_.emplace_back([dst, src, size] { std::memcpy(dst, src, size); });
    if (failure == Failure::ThrowCopy) {
      throw std::runtime_error("mock copy throw");
    }
    if (failure == Failure::Copy) {
      return Err(ErrCode::DriverError, "mock copy failure");
    }
    return Ok();
  }
  Status signal64(
      int deviceId,
      DeviceStream stream,
      uint64_t* address,
      uint64_t value) override {
    ++signals;
    signaledDevice = deviceId;
    signaledStream = stream;
    pending_.emplace_back([address, value] {
      std::atomic_ref<uint64_t>(*address).store(
          value, std::memory_order_release);
    });
    if (failure == Failure::Signal) {
      return Err(ErrCode::DriverError, "mock signal failure");
    }
    return Ok();
  }
  Result<std::unique_ptr<DeviceCompletion>> recordCompletion(
      int deviceId,
      DeviceStream stream) override {
    ++completions;
    completedDevice = deviceId;
    completedStream = stream;
    if (failure == Failure::Record) {
      return Err(ErrCode::DriverError, "mock record failure");
    }
    completionState = std::make_shared<CompletionState>();
    completionState->queryError = failure == Failure::Query;
    pending_.emplace_back([state = completionState] { state->ready = true; });
    return std::unique_ptr<DeviceCompletion>(
        std::make_unique<TestCompletion>(completionState));
  }
  Status synchronizeStream(int deviceId, DeviceStream stream) override {
    ++drains;
    drainedDevice = deviceId;
    drainedStream = stream;
    if (failure == Failure::Drain) {
      return Err(ErrCode::DriverError, "mock drain failure");
    }
    completePending();
    return Ok();
  }
  void completePending() {
    auto pending = std::move(pending_);
    pending_.clear();
    for (const auto& operation : pending) {
      operation();
    }
  }
  bool completeOne() {
    if (pending_.empty()) {
      return false;
    }
    auto operation = std::move(pending_.front());
    pending_.erase(pending_.begin());
    operation();
    return true;
  }
  size_t pendingOperations() const {
    return pending_.size();
  }

  DeviceCapabilities caps{
      .asyncCopy = true,
      .streamSignal64 = true,
      .completion = true,
  };
  Failure failure{Failure::None};
  unsigned copies{0};
  unsigned signals{0};
  unsigned completions{0};
  unsigned drains{0};
  int copiedDevice{-1};
  int signaledDevice{-1};
  int completedDevice{-1};
  int drainedDevice{-1};
  int exportedDevice{-1};
  DeviceStream copiedStream{nullptr};
  DeviceStream signaledStream{nullptr};
  DeviceStream completedStream{nullptr};
  DeviceStream drainedStream{nullptr};
  DeviceCopyKind copiedKind{DeviceCopyKind::HostToDevice};
  std::vector<int> allocatedDevices;
  std::vector<int> mappedDevices;
  std::vector<int> freedDevices;
  std::shared_ptr<CompletionState> completionState;

 private:
  Result<void*> allocate(int deviceId, size_t size) {
    auto* ptr = std::calloc(1, size);
    if (!ptr) {
      return Err(ErrCode::ResourceExhausted);
    }
    allocatedDevices.push_back(deviceId);
    return ptr;
  }
  Status free(int deviceId, void* ptr) {
    freedDevices.push_back(deviceId);
    std::free(ptr);
    return Ok();
  }
  std::vector<std::function<void()>> pending_;
};

class TestGpuDiscovery final : public TopologyDiscoveryBackend {
 public:
  Status discover(Topology& topology) override {
    topology.clear();
    for (int deviceId = 0; deviceId < 8; ++deviceId) {
      const auto nodeId = topology.addNode(
          TopoNode{
              .type = NodeType::GPU,
              .name = "mock:" + std::to_string(deviceId),
              .data = TopoNode::GpuData{.cudaDeviceId = deviceId},
          });
      topology.registerGpuNode(deviceId, nodeId);
    }
    topology.recomputePaths();
    topology.setStatus(Ok());
    return Ok();
  }
};

class RdmaNeutralTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ibv_ = std::make_shared<testing::NiceMock<MockIbvApi>>();
    adapter_ = std::make_shared<TestDeviceAdapter>();
    std::strcpy(device_.name, "mock_ib0");
    qp_.qp_num = 17;
    ON_CALL(*ibv_, getDeviceList(_)).WillByDefault([this](int* count) {
      *count = 1;
      return Result<ibv_device**>(devices_.data());
    });
    ON_CALL(*ibv_, freeDeviceList(_)).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, getDeviceName(_))
        .WillByDefault(Return(Result<const char*>("mock_ib0")));
    ON_CALL(*ibv_, openDevice(_))
        .WillByDefault(Return(Result<ibv_context*>(&context_)));
    ON_CALL(*ibv_, allocPd(_)).WillByDefault(Return(Result<ibv_pd*>(&pd_)));
    ON_CALL(*ibv_, queryDevice(_, _))
        .WillByDefault([](ibv_context*, ibv_device_attr* attributes) {
          attributes->phys_port_cnt = 1;
          return Ok();
        });
    ON_CALL(*ibv_, queryPort(_, _, _))
        .WillByDefault([](ibv_context*, uint8_t, ibv_port_attr* attributes) {
          attributes->state = IBV_PORT_ACTIVE;
          attributes->active_mtu = IBV_MTU_4096;
          attributes->link_layer = IBV_LINK_LAYER_ETHERNET;
          attributes->gid_tbl_len = 8;
          return Ok();
        });
    ON_CALL(*ibv_, queryGid(_, _, _, _)).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, isDmaBufSupported(_))
        .WillByDefault(Return(Result<bool>(false)));
    ON_CALL(*ibv_, deallocPd(_)).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, closeDevice(_)).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, regMr(_, _, _, _))
        .WillByDefault([this](ibv_pd*, void* ptr, size_t size, int) {
          auto mr = std::make_unique<ibv_mr>();
          mr->addr = ptr;
          mr->length = size;
          mr->lkey = 101 + static_cast<uint32_t>(mrs_.size());
          mr->rkey = 201 + static_cast<uint32_t>(mrs_.size());
          auto* result = mr.get();
          mrs_.push_back(std::move(mr));
          return Result<ibv_mr*>(result);
        });
    ON_CALL(*ibv_, deregMr(_)).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, createCq(_, _, _, _, _))
        .WillByDefault(Return(Result<ibv_cq*>(&cq_)));
    ON_CALL(*ibv_, createQp(_, _)).WillByDefault(Return(Result<ibv_qp*>(&qp_)));
    ON_CALL(*ibv_, modifyQp(_, _, _)).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, destroyQp(_)).WillByDefault(Return(Ok()));
    ON_CALL(*ibv_, destroyCq(_)).WillByDefault([this](ibv_cq*) {
      cqes_.clear();
      return Ok();
    });
    ON_CALL(*ibv_, pollCq(_, _, _))
        .WillByDefault([this](ibv_cq*, int count, ibv_wc* results) {
          int returned = 0;
          while (returned < count && !cqes_.empty()) {
            results[returned++] = cqes_.front();
            cqes_.pop_front();
          }
          return Result<int>(returned);
        });
  }

  std::shared_ptr<std::vector<NicResources>> makeNics() {
    auto nics = std::make_shared<std::vector<NicResources>>();
    nics->emplace_back(&device_, ibv_, -1, 3, uint8_t{1});
    return nics;
  }
  std::shared_ptr<RdmaSlabPool> makePool(
      int deviceId = -1,
      uint32_t slabNum = 1) {
    return std::make_shared<RdmaSlabPool>(
        adapter_,
        RdmaSlabPoolConfig{.slabNum = slabNum, .slabSize = 64},
        ibv_,
        makeNics(),
        deviceId);
  }
  DeviceStream stream() {
    return &streamToken_;
  }

  ManualEventBase eventBase_;
  std::shared_ptr<testing::NiceMock<MockIbvApi>> ibv_;
  std::shared_ptr<TestDeviceAdapter> adapter_;
  Topology topology_;
  ibv_device device_{};
  std::array<ibv_device*, 2> devices_{&device_, nullptr};
  ibv_context context_{.device = &device_};
  ibv_pd pd_{};
  ibv_cq cq_{};
  ibv_qp qp_{};
  std::vector<std::unique_ptr<ibv_mr>> mrs_;
  std::deque<ibv_wc> cqes_;
  std::array<char, 16> streamToken_{};
};

TEST_F(RdmaNeutralTest, HostCopyNeedsNeitherGpuAdapterNorStream) {
  auto pool = makePool();
  auto result = pool->acquire();
  ASSERT_TRUE(result);
  auto slab = std::move(result).value();
  std::array<char, 8> input{'r', 'd', 'm', 'a'};
  std::array<char, 8> output{};
  CopyEngine copy(MemoryType::DRAM, -1, std::nullopt, nullptr);

  ASSERT_TRUE(copy.copyToSlab(slab, input.data(), input.size()));
  ASSERT_TRUE(copy.copyFromSlab(output.data(), slab, output.size()));
  auto done = copy.isCopyDone(slab);
  ASSERT_TRUE(done);
  EXPECT_TRUE(done.value());
  EXPECT_EQ(output, input);
  EXPECT_EQ(adapter_->copies, 0);
  EXPECT_EQ(adapter_->drains, 0);
}

TEST_F(RdmaNeutralTest, CpuPoolUsesCompletionWithoutChangingAddressOrRkey) {
  auto pool = makePool();
  const auto address = pool->slabAddr(0);
  const auto rkey = pool->slabRkey(0);
  auto result = pool->acquire();
  ASSERT_TRUE(result);
  auto slab = std::move(result).value();
  std::array<char, 8> input{'g', 'p', 'u'};
  CopyEngine copy(MemoryType::VRAM, 7, stream(), adapter_);

  ASSERT_TRUE(copy.copyToSlab(slab, input.data(), input.size()));
  auto done = copy.isCopyDone(slab);
  ASSERT_TRUE(done);
  EXPECT_FALSE(done.value());
  EXPECT_FALSE(copy.resetState(slab));
  EXPECT_EQ(adapter_->signals, 0);
  EXPECT_EQ(adapter_->completions, 1);
  EXPECT_EQ(adapter_->completedDevice, 7);
  EXPECT_EQ(adapter_->completedStream, stream());

  adapter_->completePending();
  done = copy.isCopyDone(slab);
  ASSERT_TRUE(done);
  EXPECT_TRUE(done.value());
  ASSERT_TRUE(copy.resetState(slab));
  EXPECT_EQ(std::memcmp(slab.ptr(), input.data(), input.size()), 0);
  EXPECT_EQ(pool->slabAddr(0), address);
  EXPECT_EQ(pool->slabRkey(0), rkey);
  EXPECT_EQ(adapter_->allocatedDevices, (std::vector<int>{-1}));
  EXPECT_EQ(adapter_->drains, 0);
}

TEST_F(RdmaNeutralTest, ScopedPoolUsesOrderedMarkerOnItsActualDevice) {
  {
    auto pool = makePool(5);
    auto result = pool->acquire();
    ASSERT_TRUE(result);
    auto slab = std::move(result).value();
    std::array<char, 8> input{'m', 'a', 'r', 'k'};
    CopyEngine copy(MemoryType::VRAM, 5, stream(), adapter_);

    ASSERT_TRUE(copy.copyToSlab(slab, input.data(), input.size()));
    EXPECT_EQ(adapter_->signals, 1);
    EXPECT_EQ(adapter_->completions, 0);
    EXPECT_EQ(adapter_->copiedDevice, 5);
    EXPECT_EQ(adapter_->signaledDevice, 5);
    EXPECT_EQ(adapter_->signaledStream, stream());
    auto done = copy.isCopyDone(slab);
    ASSERT_TRUE(done);
    EXPECT_FALSE(done.value());
    adapter_->completePending();
    done = copy.isCopyDone(slab);
    ASSERT_TRUE(done);
    EXPECT_TRUE(done.value());
    EXPECT_EQ(std::memcmp(slab.ptr(), input.data(), input.size()), 0);
    ASSERT_TRUE(copy.resetState(slab));
  }
  EXPECT_EQ(adapter_->allocatedDevices, (std::vector<int>{5}));
  EXPECT_EQ(adapter_->mappedDevices, (std::vector<int>{5}));
  EXPECT_EQ(adapter_->freedDevices, (std::vector<int>{5}));
}

TEST_F(RdmaNeutralTest, OtherDeviceDoesNotUseThePoolsMappedMarker) {
  auto pool = makePool(3);
  auto result = pool->acquire();
  ASSERT_TRUE(result);
  auto slab = std::move(result).value();
  std::array<char, 8> input{};
  CopyEngine copy(MemoryType::VRAM, 8, stream(), adapter_);

  ASSERT_TRUE(copy.copyToSlab(slab, input.data(), input.size()));
  EXPECT_EQ(adapter_->signals, 0);
  EXPECT_EQ(adapter_->completions, 1);
  EXPECT_EQ(adapter_->completedDevice, 8);
}

TEST_F(RdmaNeutralTest, UnsupportedCopyAndMissingStreamFailBeforeEnqueue) {
  auto pool = makePool();
  auto result = pool->acquire();
  ASSERT_TRUE(result);
  auto slab = std::move(result).value();
  std::array<char, 8> input{};
  CopyEngine missingStream(MemoryType::VRAM, 7, std::nullopt, adapter_);
  EXPECT_FALSE(missingStream.copyToSlab(slab, input.data(), input.size()));

  adapter_->caps.asyncCopy = false;
  CopyEngine unsupported(MemoryType::VRAM, 7, stream(), adapter_);
  EXPECT_FALSE(unsupported.copyToSlab(slab, input.data(), input.size()));
  adapter_->caps.asyncCopy = true;
  adapter_->caps.completion = false;
  EXPECT_FALSE(unsupported.copyToSlab(slab, input.data(), input.size()));
  EXPECT_EQ(adapter_->copies, 0);
  EXPECT_EQ(adapter_->signals, 0);
  EXPECT_EQ(adapter_->drains, 0);
}

TEST_F(RdmaNeutralTest, AmbiguousCopyErrorDrainsTheExactStreamAndStaysAnError) {
  auto pool = makePool();
  auto result = pool->acquire();
  ASSERT_TRUE(result);
  auto slab = std::move(result).value();
  std::array<char, 8> input{'e', 'r', 'r'};
  CopyEngine copy(MemoryType::VRAM, 6, stream(), adapter_);
  adapter_->failure = TestDeviceAdapter::Failure::Copy;

  auto status = copy.copyToSlab(slab, input.data(), input.size());
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().message(), "mock copy failure");
  EXPECT_EQ(adapter_->drains, 1);
  EXPECT_EQ(adapter_->drainedDevice, 6);
  EXPECT_EQ(adapter_->drainedStream, stream());
  EXPECT_EQ(std::memcmp(slab.ptr(), input.data(), input.size()), 0);
  ASSERT_TRUE(copy.resetState(slab));
}

TEST_F(RdmaNeutralTest, SignalErrorDrainsAlreadyQueuedCopy) {
  auto pool = makePool(4);
  auto result = pool->acquire();
  ASSERT_TRUE(result);
  auto slab = std::move(result).value();
  std::array<char, 8> input{'s', 'i', 'g'};
  CopyEngine copy(MemoryType::VRAM, 4, stream(), adapter_);
  adapter_->failure = TestDeviceAdapter::Failure::Signal;

  auto status = copy.copyToSlab(slab, input.data(), input.size());
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().message(), "mock signal failure");
  EXPECT_EQ(adapter_->drains, 1);
  EXPECT_EQ(std::memcmp(slab.ptr(), input.data(), input.size()), 0);
}

TEST_F(RdmaNeutralTest, CompletionCreationAndQueryErrorsDrainQueuedCopies) {
  for (auto failure :
       {TestDeviceAdapter::Failure::Record,
        TestDeviceAdapter::Failure::Query}) {
    SCOPED_TRACE(static_cast<int>(failure));
    auto pool = makePool();
    auto result = pool->acquire();
    ASSERT_TRUE(result);
    auto slab = std::move(result).value();
    std::array<char, 8> input{};
    CopyEngine copy(MemoryType::VRAM, 2, stream(), adapter_);
    adapter_->failure = failure;
    auto status = copy.copyToSlab(slab, input.data(), input.size());
    if (failure == TestDeviceAdapter::Failure::Record) {
      EXPECT_FALSE(status);
    } else {
      ASSERT_TRUE(status);
      EXPECT_FALSE(copy.isCopyDone(slab));
      EXPECT_EQ(adapter_->completionState->destroyed, 1);
    }
    ASSERT_TRUE(copy.resetState(slab));
  }
  EXPECT_EQ(adapter_->drains, 2);
}

TEST_F(RdmaNeutralTest, ThrownCopyDrainsBeforeReleasingBuffers) {
  auto pool = makePool();
  auto result = pool->acquire();
  ASSERT_TRUE(result);
  auto slab = std::move(result).value();
  std::array<char, 8> input{};
  CopyEngine copy(MemoryType::VRAM, 2, stream(), adapter_);
  adapter_->failure = TestDeviceAdapter::Failure::ThrowCopy;

  EXPECT_FALSE(copy.copyToSlab(slab, input.data(), input.size()));
  EXPECT_EQ(adapter_->drains, 1);
  ASSERT_TRUE(copy.resetState(slab));
}

TEST_F(RdmaNeutralTest, DestructionDrainsPendingCompletionBeforeFree) {
  {
    auto pool = makePool();
    auto result = pool->acquire();
    ASSERT_TRUE(result);
    auto slab = std::move(result).value();
    std::array<char, 8> input{};
    {
      CopyEngine copy(MemoryType::VRAM, 2, stream(), adapter_);
      ASSERT_TRUE(copy.copyToSlab(slab, input.data(), input.size()));
      EXPECT_EQ(adapter_->drains, 0);
      EXPECT_TRUE(adapter_->freedDevices.empty());
    }
    EXPECT_EQ(adapter_->drains, 1);
    EXPECT_EQ(adapter_->completionState->destroyed, 1);
    EXPECT_TRUE(adapter_->freedDevices.empty());
  }
  EXPECT_EQ(adapter_->freedDevices, (std::vector<int>{-1}));
}

TEST_F(RdmaNeutralTest, FailedDrainDoesNotReleasePendingOwners) {
  auto pool = makePool();
  auto result = pool->acquire();
  ASSERT_TRUE(result);
  auto slab = std::move(result).value();
  std::array<char, 8> input{};
  CopyEngine copy(MemoryType::VRAM, 2, stream(), adapter_);
  ASSERT_TRUE(copy.copyToSlab(slab, input.data(), input.size()));
  adapter_->failure = TestDeviceAdapter::Failure::Drain;

  EXPECT_FALSE(copy.drain());
  EXPECT_EQ(adapter_->completionState->destroyed, 0);
  EXPECT_TRUE(adapter_->freedDevices.empty());
  adapter_->failure = TestDeviceAdapter::Failure::None;
  ASSERT_TRUE(copy.drain());
}

TEST_F(RdmaNeutralTest, CpuFactoryStaysUnlockedUntilActualVramRegistration) {
  unsigned adapterCreations = 0;
  GpuBackendSelector selector("nvidia", std::optional<std::string>{"nvidia"});
  PlatformRegistry platforms;
  ASSERT_TRUE(platforms.registerPlatform(
      PlatformRegistration{
          .name = "nvidia",
          .createDeviceAdapter =
              [&]() -> Result<std::shared_ptr<DeviceAdapter>> {
            ++adapterCreations;
            return std::shared_ptr<DeviceAdapter>(adapter_);
          },
          .createDiscoveryBackend =
              []() -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
            return std::unique_ptr<TopologyDiscoveryBackend>(
                std::make_unique<TestGpuDiscovery>());
          },
      }));
  auto access =
      std::make_shared<DeviceAdapterAccess>(std::nullopt, platforms, selector);
  RdmaTransportFactory factory(
      access, {"mock_ib0"}, &eventBase_, {}, ibv_, uint8_t{1}, -1, &topology_);
  std::array<char, 8> bytes{};
  Segment host(bytes.data(), bytes.size(), MemoryType::DRAM);
  auto hostRegistration = factory.registerSegment(host);
  ASSERT_TRUE(hostRegistration);
  EXPECT_EQ(adapterCreations, 0);
  EXPECT_FALSE(selector.selectedBackend());

  Segment device(bytes.data(), bytes.size(), MemoryType::VRAM, 7);
  auto deviceRegistration = factory.registerSegment(device);
  ASSERT_TRUE(deviceRegistration);
  EXPECT_EQ(adapterCreations, 1);
  EXPECT_EQ(selector.selectedBackend(), std::optional<std::string>{"nvidia"});
  EXPECT_EQ(adapter_->exportedDevice, 7);
}

TEST_F(RdmaNeutralTest, PassiveRemoteDecoderDoesNotRequireAnAdapter) {
  constexpr uint64_t base = 0x1000;
  unsigned calls = 0;
  RdmaRemoteRegistrationHandle handle(
      {31}, 41, base, nullptr, [&calls](const void* pointer) {
        ++calls;
        return reinterpret_cast<uint64_t>(pointer) & 0x0000ffffffffffffULL;
      });
  auto* encoded = reinterpret_cast<const void*>(0x0007000000001200ULL);
  EXPECT_EQ(handle.toWireAddr(encoded), 0x200);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(handle.rkey(0), 31);
}

} // namespace
} // namespace uniflow
