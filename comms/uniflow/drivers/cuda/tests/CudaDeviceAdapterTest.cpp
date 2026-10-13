// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaDeviceAdapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>

#include "comms/uniflow/drivers/cuda/CudaDevicePtr.h"
#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/drivers/cuda/mock/MockCudaDriverApi.h"

namespace uniflow {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

namespace {

constexpr size_t kSize = 4096;

class LegacyDeviceAdapter : public DeviceAdapter {
 public:
  MOCK_METHOD(Result<void*>, pinnedHostAlloc, (size_t), (override));
  MOCK_METHOD(Status, pinnedHostFree, (void*), (override));
  MOCK_METHOD(Result<void*>, hostGetDevicePointer, (void*), (override));
  MOCK_METHOD(Result<bool>, isDmaBuffSupported, (int), (override));
  MOCK_METHOD(
      Result<DmaBuff>,
      exportDmaBuff,
      (int, void*, size_t, DmaBufMapping),
      (override));
  MOCK_METHOD(Status, closeDmaBuff, (DmaBuff&), (override));
};

class CudaDeviceTransferTest : public ::testing::Test {
 protected:
  static constexpr int kPrevious = 1;
  static constexpr int kDevice = 2;

  DeviceStream stream() const {
    return static_cast<DeviceStream>(stream_);
  }

  void enter() {
    EXPECT_CALL(*api_, getDevice())
        .InSequence(sequence_)
        .WillOnce(Return(kPrevious));
    EXPECT_CALL(*api_, setDevice(kDevice))
        .InSequence(sequence_)
        .WillOnce(Return(Ok()));
  }

  void leave(Status result = Ok()) {
    EXPECT_CALL(*api_, setDevice(kPrevious))
        .InSequence(sequence_)
        .WillOnce(Return(std::move(result)));
  }

  void expectRecord(Status result = Ok()) {
    enter();
    EXPECT_CALL(*api_, eventCreate(_))
        .InSequence(sequence_)
        .WillOnce([&](auto* event) {
          *event = event_;
          return Ok();
        });
    EXPECT_CALL(*api_, eventRecord(event_, stream_))
        .InSequence(sequence_)
        .WillOnce(Return(std::move(result)));
    leave();
  }

  void expectEventDestroy(Status result = Ok()) {
    enter();
    EXPECT_CALL(*api_, eventDestroy(event_))
        .InSequence(sequence_)
        .WillOnce(Return(std::move(result)));
    leave();
  }

  std::shared_ptr<StrictMock<MockCudaApi>> api_ =
      std::make_shared<StrictMock<MockCudaApi>>();
  std::shared_ptr<StrictMock<MockCudaDriverApi>> driver_ =
      std::make_shared<StrictMock<MockCudaDriverApi>>();
  ::testing::Sequence sequence_;
  char streamTag_{0};
  char eventTag_{0};
  MockStream stream_ = reinterpret_cast<MockStream>(&streamTag_);
  MockEvent event_ = reinterpret_cast<MockEvent>(&eventTag_);
  int src_{1};
  int dst_{0};
};

class CudaDeviceCopyTest
    : public CudaDeviceTransferTest,
      public ::testing::WithParamInterface<DeviceCopyKind> {};

class CudaDmaBuffTest : public CudaDeviceTransferTest {
 protected:
  void SetUp() override {
    fd_ = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
    ASSERT_GE(fd_, 0);
  }

  void TearDown() override {
    if (fd_ >= 0 && ::fcntl(fd_, F_GETFD) >= 0) {
      ::close(fd_);
    }
  }

  void expectExport(Status result = Ok()) {
    EXPECT_CALL(
        *driver_,
        memGetHandleForAddressRange(
            _, _, _, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0))
        .InSequence(sequence_)
        .WillOnce([this, result = std::move(result)](
                      void* handle, auto, auto, auto, auto) mutable {
          *static_cast<int*>(handle) = fd_;
          return std::move(result);
        });
  }

  void expectClosed() {
    errno = 0;
    const int descriptor = ::fcntl(fd_, F_GETFD);
    const int error = errno;
    if (descriptor == -1 && error == EBADF) {
      fd_ = -1;
    }
    EXPECT_EQ(descriptor, -1);
    EXPECT_EQ(error, EBADF);
  }

  int fd_{-1};
};

} // namespace

TEST(DeviceAdapterTest, UnsupportedDefaultsDoNotPretendToTransferOrComplete) {
  StrictMock<LegacyDeviceAdapter> adapter;
  auto capabilities = adapter.capabilities();
  EXPECT_FALSE(capabilities.asyncCopy);
  EXPECT_FALSE(capabilities.streamSignal64);
  EXPECT_FALSE(capabilities.nonBlockingStreams);
  EXPECT_FALSE(capabilities.completion);
  auto unsupported = [](const auto& result) {
    EXPECT_TRUE(result.hasError());
    if (result.hasError()) {
      EXPECT_EQ(result.error().code(), ErrCode::NotImplemented);
    }
  };
  int src = 1;
  int dst = 0;
  uint64_t signal = 0;
  unsupported(adapter.copyAsync(
      0, &dst, &src, sizeof(src), DeviceCopyKind::HostToDevice, nullptr));
  unsupported(adapter.signal64(0, nullptr, &signal, 1));
  unsupported(adapter.createNonBlockingStream(0));
  unsupported(adapter.synchronizeStream(0, nullptr));
  unsupported(adapter.destroyStream(0, nullptr));
  unsupported(adapter.recordCompletion(0, nullptr));
}

TEST_F(CudaDeviceTransferTest, CapabilitiesArePassiveAndSignalNeedsDriver) {
  CudaDeviceAdapter adapter(api_, driver_);
  auto capabilities = adapter.capabilities();
  EXPECT_TRUE(capabilities.asyncCopy);
  EXPECT_TRUE(capabilities.streamSignal64);
  EXPECT_TRUE(capabilities.nonBlockingStreams);
  EXPECT_TRUE(capabilities.completion);
  CudaDeviceAdapter withoutDriver(api_);
  EXPECT_FALSE(withoutDriver.capabilities().streamSignal64);
  uint64_t signal = 0;
  auto result = withoutDriver.signal64(kDevice, stream(), &signal, 1);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::NotImplemented);
}

TEST_P(CudaDeviceCopyTest, CopiesUseRequestedKindDeviceAndStream) {
  auto kind = GetParam();
  auto cudaKind = kind == DeviceCopyKind::HostToDevice ? kMockMemcpyH2D
      : kind == DeviceCopyKind::DeviceToHost           ? kMockMemcpyD2H
                                                       : kMockMemcpyD2D;
  enter();
  EXPECT_CALL(*api_, memcpyAsync(&dst_, &src_, sizeof(src_), cudaKind, stream_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);
  EXPECT_TRUE(
      adapter.copyAsync(kDevice, &dst_, &src_, sizeof(src_), kind, stream()));
}

INSTANTIATE_TEST_SUITE_P(
    CopyDirections,
    CudaDeviceCopyTest,
    ::testing::Values(
        DeviceCopyKind::HostToDevice,
        DeviceCopyKind::DeviceToHost,
        DeviceCopyKind::DeviceToDevice));

TEST_F(CudaDeviceTransferTest, NullCopyStreamStillUsesRequestedDevice) {
  enter();
  EXPECT_CALL(
      *api_, memcpyAsync(&dst_, &src_, sizeof(src_), kMockMemcpyH2D, nullptr))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);
  EXPECT_TRUE(adapter.copyAsync(
      kDevice,
      &dst_,
      &src_,
      sizeof(src_),
      DeviceCopyKind::HostToDevice,
      nullptr));
}

TEST_F(CudaDeviceTransferTest, CurrentDeviceIsBoundBeforeCopy) {
  EXPECT_CALL(*api_, getDevice())
      .InSequence(sequence_)
      .WillOnce(Return(kDevice));
  EXPECT_CALL(*api_, setDevice(kDevice))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  EXPECT_CALL(
      *api_, memcpyAsync(&dst_, &src_, sizeof(src_), kMockMemcpyH2D, stream_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*api_, setDevice(kDevice))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  CudaDeviceAdapter adapter(api_, driver_);
  EXPECT_TRUE(adapter.copyAsync(
      kDevice,
      &dst_,
      &src_,
      sizeof(src_),
      DeviceCopyKind::HostToDevice,
      stream()));
}

TEST_F(CudaDeviceTransferTest, InvalidArgumentsDoNotTouchTheGpu) {
  CudaDeviceAdapter adapter(api_, driver_);
  auto invalidDevice = adapter.copyAsync(
      -1, &dst_, &src_, sizeof(src_), DeviceCopyKind::HostToDevice, stream());
  ASSERT_TRUE(invalidDevice.hasError());
  EXPECT_EQ(invalidDevice.error().code(), ErrCode::InvalidArgument);
  auto invalidKind = adapter.copyAsync(
      kDevice,
      &dst_,
      &src_,
      sizeof(src_),
      static_cast<DeviceCopyKind>(99),
      stream());
  ASSERT_TRUE(invalidKind.hasError());
  EXPECT_EQ(invalidKind.error().code(), ErrCode::InvalidArgument);
  auto invalidSignal = adapter.signal64(kDevice, stream(), nullptr, 1);
  ASSERT_TRUE(invalidSignal.hasError());
  EXPECT_EQ(invalidSignal.error().code(), ErrCode::InvalidArgument);
  auto invalidDestroy = adapter.destroyStream(kDevice, nullptr);
  ASSERT_TRUE(invalidDestroy.hasError());
  EXPECT_EQ(invalidDestroy.error().code(), ErrCode::InvalidArgument);
}

TEST_F(CudaDeviceTransferTest, CurrentDeviceFailurePreventsCopy) {
  EXPECT_CALL(*api_, getDevice())
      .WillOnce(Return(Err(ErrCode::DriverError, "get device failed")));
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.copyAsync(
      kDevice,
      &dst_,
      &src_,
      sizeof(src_),
      DeviceCopyKind::HostToDevice,
      stream());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().message(), "get device failed");
}

TEST_F(CudaDeviceTransferTest, DeviceSwitchFailureRestoresWithoutCopy) {
  EXPECT_CALL(*api_, getDevice())
      .InSequence(sequence_)
      .WillOnce(Return(kPrevious));
  EXPECT_CALL(*api_, setDevice(kDevice))
      .InSequence(sequence_)
      .WillOnce(Return(Err(ErrCode::DriverError, "set device failed")));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.copyAsync(
      kDevice,
      &dst_,
      &src_,
      sizeof(src_),
      DeviceCopyKind::HostToDevice,
      stream());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().message(), "set device failed");
}

TEST_F(CudaDeviceTransferTest, CopyAndRestorationFailuresRemainErrors) {
  enter();
  EXPECT_CALL(
      *api_, memcpyAsync(&dst_, &src_, sizeof(src_), kMockMemcpyH2D, stream_))
      .InSequence(sequence_)
      .WillOnce(Return(Err(ErrCode::DriverError, "enqueue failed")));
  leave(Err(ErrCode::DriverError, "restore failed"));
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.copyAsync(
      kDevice,
      &dst_,
      &src_,
      sizeof(src_),
      DeviceCopyKind::HostToDevice,
      stream());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(
      result.error().message(),
      "enqueue failed; device restoration failed: restore failed");
}

TEST_F(CudaDeviceTransferTest, RestorationFailureAfterEnqueueIsNotSuccess) {
  enter();
  EXPECT_CALL(
      *api_, memcpyAsync(&dst_, &src_, sizeof(src_), kMockMemcpyH2D, stream_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave(Err(ErrCode::DriverError, "restore failed"));
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.copyAsync(
      kDevice,
      &dst_,
      &src_,
      sizeof(src_),
      DeviceCopyKind::HostToDevice,
      stream());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().message(), "restore failed");
}

TEST_F(CudaDeviceTransferTest, CopyExceptionAttemptsNonThrowingRestoration) {
  enter();
  EXPECT_CALL(
      *api_, memcpyAsync(&dst_, &src_, sizeof(src_), kMockMemcpyH2D, stream_))
      .InSequence(sequence_)
      .WillOnce(testing::Throw(std::runtime_error("enqueue threw")));
  leave(Err(ErrCode::DriverError, "restore failed"));
  CudaDeviceAdapter adapter(api_, driver_);
  EXPECT_THROW(
      adapter.copyAsync(
          kDevice,
          &dst_,
          &src_,
          sizeof(src_),
          DeviceCopyKind::HostToDevice,
          stream()),
      std::runtime_error);
}

TEST_F(CudaDeviceTransferTest, SignalIsOrderedOnTheSuppliedStream) {
  uint64_t signal = 0;
  enter();
  EXPECT_CALL(*driver_, streamWriteValue64(_, _, 7, 0))
      .InSequence(sequence_)
      .WillOnce([&](auto actualStream, auto address, auto, auto) {
        EXPECT_EQ(static_cast<DeviceStream>(actualStream), stream());
        EXPECT_EQ(address, toDevicePtr(reinterpret_cast<uint64_t>(&signal)));
        return Ok();
      });
  leave();
  CudaDeviceAdapter adapter(api_, driver_);
  EXPECT_TRUE(adapter.signal64(kDevice, stream(), &signal, 7));
}

TEST_F(CudaDmaBuffTest, ExportUsesRequestedDeviceAndRestoresCaller) {
  enter();
  expectExport();
  leave();
  CudaDeviceAdapter adapter(api_, driver_);

  auto result = adapter.exportDmaBuff(kDevice, &src_, sizeof(src_));

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->fd, fd_);
  EXPECT_GE(::fcntl(fd_, F_GETFD), 0);
  EXPECT_TRUE(adapter.closeDmaBuff(result.value()));
  expectClosed();
}

TEST_F(CudaDmaBuffTest, CurrentDeviceIsBoundBeforeExport) {
  EXPECT_CALL(*api_, getDevice())
      .InSequence(sequence_)
      .WillOnce(Return(kDevice));
  EXPECT_CALL(*api_, setDevice(kDevice))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  expectExport();
  EXPECT_CALL(*api_, setDevice(kDevice))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  CudaDeviceAdapter adapter(api_, driver_);

  auto result = adapter.exportDmaBuff(kDevice, &src_, sizeof(src_));

  ASSERT_TRUE(result.hasValue());
  EXPECT_TRUE(adapter.closeDmaBuff(result.value()));
  expectClosed();
}

TEST_F(CudaDmaBuffTest, RestorationFailureClosesUnpublishedDescriptor) {
  enter();
  expectExport();
  leave(Err(ErrCode::DriverError, "restore failed"));
  CudaDeviceAdapter adapter(api_, driver_);

  auto result = adapter.exportDmaBuff(kDevice, &src_, sizeof(src_));

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().message(), "restore failed");
  expectClosed();
}

TEST_F(CudaDmaBuffTest, DriverFailureClosesPartialDescriptorAndRestoresDevice) {
  enter();
  expectExport(Err(ErrCode::DriverError, "export failed"));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);

  auto result = adapter.exportDmaBuff(kDevice, &src_, sizeof(src_));

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().message(), "export failed");
  expectClosed();
}

TEST_F(CudaDeviceTransferTest, PrivateStreamIsNonBlockingAndOwnedByCaller) {
  enter();
  EXPECT_CALL(*api_, streamCreateNonBlocking(_))
      .InSequence(sequence_)
      .WillOnce([&](auto* output) {
        *output = stream_;
        return Ok();
      });
  leave();
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.createNonBlockingStream(kDevice);
  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result.value(), stream());
  enter();
  EXPECT_CALL(*api_, streamSynchronize(stream_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  EXPECT_TRUE(adapter.synchronizeStream(kDevice, result.value()));
  enter();
  EXPECT_CALL(*api_, streamDestroy(stream_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  EXPECT_TRUE(adapter.destroyStream(kDevice, result.value()));
}

TEST_F(
    CudaDeviceTransferTest,
    UnpublishedStreamIsDestroyedOnRestorationFailure) {
  enter();
  EXPECT_CALL(*api_, streamCreateNonBlocking(_))
      .InSequence(sequence_)
      .WillOnce([&](auto* output) {
        *output = stream_;
        return Ok();
      });
  leave(Err(ErrCode::DriverError, "restore failed"));
  EXPECT_CALL(*api_, getDevice())
      .InSequence(sequence_)
      .WillOnce(Return(kDevice));
  EXPECT_CALL(*api_, setDevice(kDevice))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*api_, streamDestroy(stream_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*api_, setDevice(kDevice))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.createNonBlockingStream(kDevice);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().message(), "restore failed");
}

TEST_F(CudaDeviceTransferTest, FailedPartialStreamCreationCleansTheHandle) {
  enter();
  EXPECT_CALL(*api_, streamCreateNonBlocking(_))
      .InSequence(sequence_)
      .WillOnce([&](auto* output) -> Status {
        *output = stream_;
        return Err(ErrCode::DriverError, "create failed");
      });
  leave();
  enter();
  EXPECT_CALL(*api_, streamDestroy(stream_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.createNonBlockingStream(kDevice);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().message(), "create failed");
}

TEST_F(CudaDeviceTransferTest, StreamCreationExceptionStillCleansTheHandle) {
  enter();
  EXPECT_CALL(*api_, streamCreateNonBlocking(_))
      .InSequence(sequence_)
      .WillOnce([&](auto* output) -> Status {
        *output = stream_;
        throw std::runtime_error("create threw");
      });
  leave();
  enter();
  EXPECT_CALL(*api_, streamDestroy(stream_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);
  EXPECT_THROW(adapter.createNonBlockingStream(kDevice), std::runtime_error);
}

TEST_F(
    CudaDeviceTransferTest,
    DefaultStreamDrainIsDeviceScopedAndErrorsPropagate) {
  enter();
  EXPECT_CALL(*api_, streamSynchronize(nullptr))
      .InSequence(sequence_)
      .WillOnce(Return(Err(ErrCode::DriverError, "drain failed")));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.synchronizeStream(kDevice, nullptr);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().message(), "drain failed");
}

TEST_F(CudaDeviceTransferTest, DefaultStreamCompletionIsDeviceScoped) {
  enter();
  EXPECT_CALL(*api_, eventCreate(_))
      .InSequence(sequence_)
      .WillOnce([&](auto* output) {
        *output = event_;
        return Ok();
      });
  EXPECT_CALL(*api_, eventRecord(event_, nullptr))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.recordCompletion(kDevice, nullptr);
  ASSERT_TRUE(result.hasValue());
  expectEventDestroy();
  result.value().reset();
}

TEST_F(CudaDeviceTransferTest, CompletionReadinessAndWaitAreEventSpecific) {
  CudaDeviceAdapter adapter(api_, driver_);
  expectRecord();
  auto result = adapter.recordCompletion(kDevice, stream());
  ASSERT_TRUE(result.hasValue());
  auto completion = std::move(result).value();
  enter();
  EXPECT_CALL(*api_, eventQuery(event_))
      .InSequence(sequence_)
      .WillOnce(Return(false));
  leave();
  auto ready = completion->ready();
  ASSERT_TRUE(ready.hasValue());
  EXPECT_FALSE(ready.value());
  enter();
  EXPECT_CALL(*api_, eventQuery(event_))
      .InSequence(sequence_)
      .WillOnce(Return(true));
  leave();
  ready = completion->ready();
  ASSERT_TRUE(ready.hasValue());
  EXPECT_TRUE(ready.value());
  enter();
  EXPECT_CALL(*api_, eventSynchronize(event_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  EXPECT_TRUE(completion->wait());
  expectEventDestroy();
  completion.reset();
}

TEST_F(CudaDeviceTransferTest, CompletionErrorsAreNotReadinessOrSuccess) {
  CudaDeviceAdapter adapter(api_, driver_);
  expectRecord();
  auto result = adapter.recordCompletion(kDevice, stream());
  ASSERT_TRUE(result.hasValue());
  auto completion = std::move(result).value();
  enter();
  EXPECT_CALL(*api_, eventQuery(event_))
      .InSequence(sequence_)
      .WillOnce(Return(Err(ErrCode::DriverError, "query failed")));
  leave();
  auto ready = completion->ready();
  ASSERT_TRUE(ready.hasError());
  EXPECT_EQ(ready.error().message(), "query failed");
  enter();
  EXPECT_CALL(*api_, eventSynchronize(event_))
      .InSequence(sequence_)
      .WillOnce(Return(Err(ErrCode::DriverError, "wait failed")));
  leave();
  auto waited = completion->wait();
  ASSERT_TRUE(waited.hasError());
  EXPECT_EQ(waited.error().message(), "wait failed");
  expectEventDestroy(Err(ErrCode::DriverError, "destroy failed"));
  completion.reset();
}

TEST_F(CudaDeviceTransferTest, FailedEventRecordDestroysTheCreatedEvent) {
  expectRecord(Err(ErrCode::DriverError, "record failed"));
  expectEventDestroy();
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.recordCompletion(kDevice, stream());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().message(), "record failed");
}

TEST_F(
    CudaDeviceTransferTest,
    FailedEventCreationCleansAPartiallyCreatedEvent) {
  enter();
  EXPECT_CALL(*api_, eventCreate(_))
      .InSequence(sequence_)
      .WillOnce([&](auto* event) -> Status {
        *event = event_;
        return Err(ErrCode::DriverError, "create failed");
      });
  leave();
  expectEventDestroy();
  CudaDeviceAdapter adapter(api_, driver_);
  auto result = adapter.recordCompletion(kDevice, stream());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().message(), "create failed");
}

TEST_F(CudaDeviceTransferTest, CompletionKeepsApiAliveAfterAdapterDestruction) {
  auto adapter = std::make_unique<CudaDeviceAdapter>(api_, driver_);
  expectRecord();
  auto result = adapter->recordCompletion(kDevice, stream());
  ASSERT_TRUE(result.hasValue());
  auto completion = std::move(result).value();
  enter();
  EXPECT_CALL(*api_, eventSynchronize(event_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  expectEventDestroy();
  std::weak_ptr<CudaApi> weakApi = api_;
  adapter.reset();
  api_.reset();
  EXPECT_FALSE(weakApi.expired());
  EXPECT_TRUE(completion->wait());
  completion.reset();
  EXPECT_TRUE(weakApi.expired());
}

TEST_F(CudaDeviceTransferTest, ScopedPinnedStorageUsesAllocationDevice) {
  enter();
  EXPECT_CALL(
      *api_, hostAlloc(kSize, cudaHostAllocMapped | cudaHostAllocPortable))
      .InSequence(sequence_)
      .WillOnce(Return(Result<void*>(&dst_)));
  leave();
  enter();
  EXPECT_CALL(*api_, hostGetDevicePointer(&dst_))
      .InSequence(sequence_)
      .WillOnce(Return(Result<void*>(&src_)));
  leave();
  enter();
  EXPECT_CALL(*api_, hostFree(&dst_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);

  auto allocation = adapter.pinnedHostAlloc(kDevice, kSize);
  ASSERT_TRUE(allocation);
  EXPECT_EQ(allocation.value(), &dst_);
  auto mapped = adapter.hostGetDevicePointer(kDevice, allocation.value());
  ASSERT_TRUE(mapped);
  EXPECT_EQ(mapped.value(), &src_);
  EXPECT_TRUE(adapter.pinnedHostFree(kDevice, allocation.value()));
}

TEST_F(
    CudaDeviceTransferTest,
    ScopedAllocationRestorationErrorFreesUnpublishedStorage) {
  enter();
  EXPECT_CALL(
      *api_, hostAlloc(kSize, cudaHostAllocMapped | cudaHostAllocPortable))
      .InSequence(sequence_)
      .WillOnce(Return(Result<void*>(&dst_)));
  leave(Err(ErrCode::DriverError, "restore failed"));
  enter();
  EXPECT_CALL(*api_, hostFree(&dst_))
      .InSequence(sequence_)
      .WillOnce(Return(Ok()));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);

  auto result = adapter.pinnedHostAlloc(kDevice, kSize);

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message(), "restore failed");
}

TEST_F(CudaDeviceTransferTest, ScopedAllocationFailureStillRestoresDevice) {
  enter();
  EXPECT_CALL(*api_, hostAlloc(kSize, _))
      .InSequence(sequence_)
      .WillOnce(
          Return(Err(ErrCode::ResourceExhausted, "host allocation failed")));
  leave();
  CudaDeviceAdapter adapter(api_, driver_);

  auto result = adapter.pinnedHostAlloc(kDevice, kSize);

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrCode::ResourceExhausted);
  EXPECT_EQ(result.error().message(), "host allocation failed");
}

TEST(DeviceAdapterTest, ScopedHostMethodsPreserveLegacyOverrides) {
  StrictMock<LegacyDeviceAdapter> legacy;
  DeviceAdapter& adapter = legacy;
  int storage{0};
  EXPECT_CALL(legacy, pinnedHostAlloc(kSize))
      .WillOnce(Return(Result<void*>(&storage)));
  EXPECT_CALL(legacy, hostGetDevicePointer(&storage))
      .WillOnce(Return(Result<void*>(&storage)));
  EXPECT_CALL(legacy, pinnedHostFree(&storage)).WillOnce(Return(Ok()));

  auto allocation = adapter.pinnedHostAlloc(0, kSize);
  ASSERT_TRUE(allocation);
  auto mapped = adapter.hostGetDevicePointer(0, allocation.value());
  ASSERT_TRUE(mapped);
  EXPECT_EQ(mapped.value(), &storage);
  EXPECT_TRUE(adapter.pinnedHostFree(0, allocation.value()));
}

TEST(CudaDeviceAdapterTest, PinnedHostAllocPassesMappedPortableFlags) {
  // pinnedHostAlloc must request mapped + portable so the returned host
  // pointer is usable for both DMA from a NIC and host-mapped device
  // access from any CUDA context. Verifying the flag argument is the
  // observable contract that callers (RdmaSlabPool, RdmaTransport) rely
  // on -- a bare delegation/round-trip check would not catch a regression
  // in those flags.
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int sentinel = 0;
  EXPECT_CALL(
      *mock, hostAlloc(kSize, cudaHostAllocMapped | cudaHostAllocPortable))
      .WillOnce(Return(Result<void*>(&sentinel)));

  CudaDeviceAdapter adapter(mock);

  auto res = adapter.pinnedHostAlloc(kSize);
  ASSERT_FALSE(res.hasError());
}

TEST(CudaDeviceAdapterTest, PinnedHostAllocPropagatesError) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  EXPECT_CALL(*mock, hostAlloc(kSize, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "oom")));

  CudaDeviceAdapter adapter(mock);

  auto res = adapter.pinnedHostAlloc(kSize);
  ASSERT_TRUE(res.hasError());
  EXPECT_EQ(res.error().code(), ErrCode::DriverError);
}

TEST(CudaDeviceAdapterTest, PinnedHostFreeDelegatesToCudaApi) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int sentinel = 0;
  void* fakePtr = &sentinel;
  EXPECT_CALL(*mock, hostFree(fakePtr)).WillOnce(Return(Ok()));

  CudaDeviceAdapter adapter(mock);

  EXPECT_FALSE(adapter.pinnedHostFree(fakePtr).hasError());
}

TEST(CudaDeviceAdapterTest, IsDmaBuffSupportedDelegatesToDriverApi) {
  auto cudaApi = std::make_shared<NiceMock<MockCudaApi>>();
  auto driverApi = std::make_shared<NiceMock<MockCudaDriverApi>>();
  EXPECT_CALL(*driverApi, isDmaBufSupported(3)).WillOnce(Return(true));

  CudaDeviceAdapter adapter(cudaApi, driverApi);

  auto res = adapter.isDmaBuffSupported(3);
  ASSERT_FALSE(res.hasError());
  EXPECT_TRUE(res.value());
}

TEST(CudaDeviceAdapterTest, ExportDmaBuffPopulatesFdOffsetAndIova) {
  auto cudaApi = std::make_shared<NiceMock<MockCudaApi>>();
  auto driverApi = std::make_shared<NiceMock<MockCudaDriverApi>>();
  EXPECT_CALL(*cudaApi, getDevice()).WillOnce(Return(0));
  EXPECT_CALL(*cudaApi, setDevice(0)).Times(2).WillRepeatedly(Return(Ok()));
  constexpr int kFakeFd = 42;
  EXPECT_CALL(
      *driverApi,
      memGetHandleForAddressRange(
          _, _, _, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0))
      .WillOnce([&](void* handle,
                    CUdeviceptr,
                    size_t,
                    CUmemRangeHandleType,
                    unsigned long long) {
        *static_cast<int*>(handle) = kFakeFd;
        return Ok();
      });

  CudaDeviceAdapter adapter(cudaApi, driverApi);

  // Use a real heap-aligned allocation as the base, then offset into it to
  // get a non-page-aligned pointer. posix_memalign + free keeps the
  // allocate/free pair symmetric (mixing aligned `new[]` with default
  // `delete[]` would be UB).
  const size_t pageSize = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  void* backingRaw = nullptr;
  ASSERT_EQ(::posix_memalign(&backingRaw, pageSize, pageSize * 2), 0);
  std::unique_ptr<void, decltype(&::free)> backing(backingRaw, &::free);
  void* ptr = static_cast<std::byte*>(backing.get()) + 128;

  auto res = adapter.exportDmaBuff(/*deviceId=*/0, ptr, /*len=*/4096);
  ASSERT_FALSE(res.hasError());
  EXPECT_EQ(res->fd, kFakeFd);
  EXPECT_EQ(res->offset, 128u);
  EXPECT_EQ(res->len, 4096u);
  EXPECT_EQ(res->iova, reinterpret_cast<uint64_t>(ptr));
}

TEST(CudaDeviceAdapterTest, ExportDmaBuffDataDirectSelectsPcieMapping) {
  /*
   * The DmaBufMapping::DataDirect route must map the buffer over PCIe BAR1 so
   * the exported fd can back an mlx5dv Data-Direct MR. The observable contract
   * is the range flag handed to the driver: DataDirect => the PCIe flag,
   * Default => 0 (asserted by ExportDmaBuffPopulatesFdOffsetAndIova). On CUDA
   * older than 12.8 the PCIe mapping cannot be expressed, so DataDirect must
   * fail cleanly rather than silently fall back to the C2C mapping.
   */
  auto cudaApi = std::make_shared<NiceMock<MockCudaApi>>();
  auto driverApi = std::make_shared<NiceMock<MockCudaDriverApi>>();
  CudaDeviceAdapter adapter(cudaApi, driverApi);

  const size_t pageSize = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  void* backingRaw = nullptr;
  ASSERT_EQ(::posix_memalign(&backingRaw, pageSize, pageSize), 0);
  std::unique_ptr<void, decltype(&::free)> backing(backingRaw, &::free);

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  EXPECT_CALL(*cudaApi, getDevice()).WillOnce(Return(0));
  EXPECT_CALL(*cudaApi, setDevice(0)).Times(2).WillRepeatedly(Return(Ok()));
  constexpr int kFakeFd = 7;
  EXPECT_CALL(
      *driverApi,
      memGetHandleForAddressRange(
          _,
          _,
          _,
          CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
          static_cast<unsigned long long>(
              CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE)))
      .WillOnce([&](void* handle,
                    CUdeviceptr,
                    size_t,
                    CUmemRangeHandleType,
                    unsigned long long) {
        *static_cast<int*>(handle) = kFakeFd;
        return Ok();
      });
  auto res = adapter.exportDmaBuff(
      /*deviceId=*/0, backing.get(), pageSize, DmaBufMapping::DataDirect);
  ASSERT_FALSE(res.hasError());
  EXPECT_EQ(res->fd, kFakeFd);
#else
  auto res = adapter.exportDmaBuff(
      /*deviceId=*/0, backing.get(), pageSize, DmaBufMapping::DataDirect);
  EXPECT_TRUE(res.hasError());
#endif
}

TEST(CudaDeviceAdapterTest, ExportDmaBuffRejectsNullPtr) {
  auto cudaApi = std::make_shared<NiceMock<MockCudaApi>>();
  auto driverApi = std::make_shared<NiceMock<MockCudaDriverApi>>();
  CudaDeviceAdapter adapter(cudaApi, driverApi);

  auto res = adapter.exportDmaBuff(/*deviceId=*/0, nullptr, /*len=*/4096);
  EXPECT_TRUE(res.hasError());
}

TEST(CudaDeviceAdapterTest, CloseDmaBuffIsNoopForUnsetFd) {
  auto cudaApi = std::make_shared<NiceMock<MockCudaApi>>();
  auto driverApi = std::make_shared<NiceMock<MockCudaDriverApi>>();
  CudaDeviceAdapter adapter(cudaApi, driverApi);

  DmaBuff buff;
  EXPECT_FALSE(adapter.closeDmaBuff(buff).hasError());
  EXPECT_EQ(buff.fd, -1);
}

} // namespace uniflow
