// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

/// Helpers shared by the single-process and cross-process P2P VMM integration
/// tests: VMM allocations shareable as POSIX fds, device buffer I/O, and the
/// checks that prove a VMM segment travels over the P2P tier.

#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDevicePtr.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/executor/EventBase.h"
#include "comms/uniflow/transport/p2p/P2pRegistrationHandle.h"
#include "comms/uniflow/transport/p2p/P2pTransport.h"

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

#include <cstdint>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

#include <gtest/gtest.h>

// ASSERT-based check so a failed CUDA setup call produces a clean test failure
// rather than a silently-ignored error code.
#ifndef ASSERT_CUDA
#define ASSERT_CUDA(expr) ASSERT_EQ((expr), cudaSuccess) << #expr
#endif

namespace uniflow {

/// Friend wrapper to assemble RegisteredSegment / RemoteRegisteredSegment with
/// handles. The name must be exactly "SegmentTest" to match Segment.h.
class SegmentTest {
 public:
  static RegisteredSegment makeRegistered(
      Segment& segment,
      std::unique_ptr<RegistrationHandle> handle) {
    RegisteredSegment reg(segment);
    reg.handles_.push_back(std::move(handle));
    return reg;
  }

  static RemoteRegisteredSegment makeRemote(
      void* buf,
      size_t len,
      std::unique_ptr<RemoteRegistrationHandle> handle) {
    RemoteRegisteredSegment remote(buf, len);
    remote.handles_.push_back(std::move(handle));
    return remote;
  }

  static RegisteredSegment makeUnregistered(Segment& segment) {
    return RegisteredSegment(segment);
  }

  static const RegistrationHandle* findHandle(
      const RegisteredSegment& reg,
      TransportType type) {
    for (const auto& handle : reg.handles_) {
      if (handle->transportType() == type) {
        return handle.get();
      }
    }
    return nullptr;
  }
};

inline int gpuCount() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return 0;
  }
  return count;
}

// Fails every IPC call. VMM tests register through it so a VMM export failure
// surfaces as a test failure instead of an IPC fallback: IPC on VMM memory is
// unsafe on some ROCm 7.0 runtimes.
class IpcDisabledCudaApi : public CudaApi {
 public:
  Result<IpcMemHandle> ipcGetMemHandle(void* /*devPtr*/) override {
    return Err(ErrCode::NotImplemented, "IPC is disabled in VMM tests");
  }
  Result<void*> ipcOpenMemHandle(const IpcMemHandle& /*handle*/) override {
    return Err(ErrCode::NotImplemented, "IPC is disabled in VMM tests");
  }
};

// Physical chunks of one VMM allocation each, mapped back to back into one
// reservation and shareable as POSIX fds, like PyTorch expandable segments.
class VmmBuffer {
 public:
  explicit VmmBuffer(std::shared_ptr<CudaDriverApi> driver)
      : driver_{std::move(driver)} {}

  ~VmmBuffer() {
    for (size_t i = 0; i < mapped_; ++i) {
      (void)driver_->cuMemUnmap(chunkPtr(i), chunkSize_);
    }
    if (reserved_) {
      (void)driver_->cuMemAddressFree(base_, size());
    }
    for (const auto handle : handles_) {
      (void)driver_->cuMemRelease(handle);
    }
  }

  VmmBuffer(const VmmBuffer&) = delete;
  VmmBuffer& operator=(const VmmBuffer&) = delete;

  // Chunks live on device; each device in accessDevices gets read-write access.
  Status
  alloc(int device, size_t chunkCount, const std::vector<int>& accessDevices) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t granularity = 0;
    CHECK_EXPR(driver_->cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    chunkSize_ = (kMinChunkSize + granularity - 1) / granularity * granularity;
    chunkCount_ = chunkCount;
    CHECK_EXPR(driver_->cuMemAddressReserve(
        &base_, size(), chunkSize_, CUdeviceptr{}, 0));
    reserved_ = true;
    for (size_t i = 0; i < chunkCount; ++i) {
      CUmemGenericAllocationHandle handle{};
      CHECK_EXPR(driver_->cuMemCreate(&handle, chunkSize_, &prop, 0));
      handles_.push_back(handle);
      CHECK_EXPR(driver_->cuMemMap(chunkPtr(i), chunkSize_, 0, handle, 0));
      ++mapped_;
    }
    return driver_->cuMemSetAccess(
        base_, size(), accessDescs(accessDevices).data(), accessDevices.size());
  }

  void* ptr() const {
    return reinterpret_cast<void*>(base_);
  }
  size_t chunkSize() const {
    return chunkSize_;
  }
  size_t size() const {
    return chunkSize_ * chunkCount_;
  }

 private:
  static constexpr size_t kMinChunkSize = size_t{2} << 20;

  static std::vector<CUmemAccessDesc> accessDescs(
      const std::vector<int>& devices) {
    std::vector<CUmemAccessDesc> descs(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
      descs[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      descs[i].location.id = devices[i];
      descs[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    return descs;
  }

  CUdeviceptr chunkPtr(size_t index) const {
    return toDevicePtr(reinterpret_cast<uintptr_t>(ptr()) + index * chunkSize_);
  }

  std::shared_ptr<CudaDriverApi> driver_;
  CUdeviceptr base_{};
  size_t chunkSize_{0};
  size_t chunkCount_{0};
  bool reserved_{false};
  size_t mapped_{0};
  std::vector<CUmemGenericAllocationHandle> handles_;
};

inline void allocVmm(
    VmmBuffer& buffer,
    int device,
    size_t chunkCount,
    const std::vector<int>& accessDevices) {
  const auto status = buffer.alloc(device, chunkCount, accessDevices);
  ASSERT_FALSE(status.hasError()) << status.error().message();
}

inline std::vector<uint32_t> iotaWords(size_t bytes, uint32_t first = 1) {
  std::vector<uint32_t> words(bytes / sizeof(uint32_t));
  std::iota(words.begin(), words.end(), first);
  return words;
}

// Synchronizes the device so another process sees the words once this returns.
inline void
writeWords(int device, void* dst, const std::vector<uint32_t>& words) {
  ASSERT_CUDA(cudaSetDevice(device));
  ASSERT_CUDA(cudaMemcpy(
      dst,
      words.data(),
      words.size() * sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  ASSERT_CUDA(cudaDeviceSynchronize());
}

inline void zeroDevice(int device, void* dst, size_t bytes) {
  ASSERT_CUDA(cudaSetDevice(device));
  ASSERT_CUDA(cudaMemset(dst, 0, bytes));
  ASSERT_CUDA(cudaDeviceSynchronize());
}

inline std::vector<uint32_t>
readWords(int device, const void* src, size_t bytes) {
  std::vector<uint32_t> words(bytes / sizeof(uint32_t));
  EXPECT_EQ(cudaSetDevice(device), cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(words.data(), src, bytes, cudaMemcpyDeviceToHost),
      cudaSuccess);
  return words;
}

inline void assertPosixFd(const RegistrationHandle* handle) {
  ASSERT_NE(handle, nullptr) << "no P2P handle";
  const auto* p2p = dynamic_cast<const P2pRegistrationHandle*>(handle);
  ASSERT_NE(p2p, nullptr);
  ASSERT_EQ(p2p->sharingMode(), P2pSharingMode::PosixFd);
}

inline std::vector<TransferRequest> wholeSegment(
    RegisteredSegment& local,
    RemoteRegisteredSegment& remote,
    size_t len) {
  std::vector<TransferRequest> reqs;
  reqs.push_back(
      TransferRequest{
          .local = local.span(size_t{0}, len),
          .remote = remote.span(size_t{0}, len),
      });
  return reqs;
}

inline void registerVmm(
    P2pTransportFactory& factory,
    Segment& segment,
    std::unique_ptr<RegistrationHandle>& handle) {
  auto reg = factory.registerSegment(segment);
  ASSERT_TRUE(reg.hasValue()) << reg.error().message();
  handle = std::move(reg.value());
  ASSERT_NO_FATAL_FAILURE(assertPosixFd(handle.get()));
}

// Registers a segment through an IPC-disabled P2P factory, so a VMM export
// failure fails the test here instead of reaching a MultiTransport, whose P2P
// tier would fall back to IPC: unsafe on VMM memory on some ROCm 7.0 runtimes.
inline void assertVmmExportable(Segment& segment, EventBase* evb) {
  P2pTransportFactory factory(
      segment.deviceId(), evb, std::make_shared<IpcDisabledCudaApi>());
  std::unique_ptr<RegistrationHandle> handle;
  registerVmm(factory, segment, handle);
}

// Loopback TCP stands in for the next tier; an exact-match NIC filter naming no
// device keeps RDMA out on any host, so a transfer the P2P tier cannot carry is
// counted on TCP.
inline MultiTransportFactoryOptions multiTransportOptions() {
  MultiTransportFactoryOptions options;
  options.nicFilter = NicFilter("=uniflow_p2p_vmm_test_nonic");
  options.enableTcp = true;
  options.tcpBindHost = "127.0.0.1";
  return options;
}

struct TransferCounts {
  uint64_t p2p{0};
  uint64_t rdma{0};
  uint64_t tcp{0};
  bool operator==(const TransferCounts&) const = default;
  friend std::ostream& operator<<(std::ostream& os, const TransferCounts& c) {
    return os << "{p2p=" << c.p2p << ", rdma=" << c.rdma << ", tcp=" << c.tcp
              << "}";
  }
};

inline TransferCounts transferCounts(const MultiTransport& transport) {
  return TransferCounts{
      .p2p = transport.transferCount(TransportType::NVLink),
      .rdma = transport.transferCount(TransportType::RDMA),
      .tcp = transport.transferCount(TransportType::TCP),
  };
}

} // namespace uniflow
