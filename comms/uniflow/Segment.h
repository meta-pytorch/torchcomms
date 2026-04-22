// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/transport/TransportType.h"

namespace uniflow {

/// Type of memory segment
enum class MemoryType : uint8_t {
  DRAM, // Host memory (CPU RAM)
  VRAM, // GPU memory (HBM/GDDR)
  NVME, // NVMe storage
};

class Segment;
class RegisteredSegment;
class RemoteRegisteredSegment;

/// Handle for a locally registered memory segment. Each transport backend
/// (e.g., RDMA, NVLink) provides its own subclass that holds backend-specific
/// registration state (e.g., ibv_mr for RDMA, IPC handle for NVLink).
/// Destroying the handle deregisters the segment from that backend.
class RegistrationHandle {
 public:
  virtual ~RegistrationHandle() = default;

  /// Returns the transport backend type for this handle.
  virtual TransportType transportType() const noexcept = 0;

  /// Serializes backend-specific state into an opaque byte vector
  /// for export to remote peers via exportId().
  virtual std::vector<uint8_t> serialize() const = 0;
};

/// Handle for a remotely registered memory segment. Represents registration
/// state for a peer's memory that has been imported into the local process,
/// enabling one-sided operations (put/get) to target that remote memory.
/// Destroying the handle releases the imported remote registration.
class RemoteRegistrationHandle {
 public:
  virtual ~RemoteRegistrationHandle() = default;

  /// Returns the transport backend type for this handle.
  virtual TransportType transportType() const noexcept = 0;
};

template <typename Derived>
class SegmentBase {
 public:
  class TSpan {
   public:
    TSpan(const Derived& segment, size_t offset, size_t length)
        : buf_(static_cast<uint8_t*>(segment.buf_) + offset),
          length_(length),
          memType_(segment.memType_),
          deviceId_(segment.deviceId_) {
      CHECK_THROW_EXCEPTION(offset <= segment.len_, std::invalid_argument);
      // Check for overflow and bounds in a single safe comparison
      CHECK_THROW_EXCEPTION(
          length <= segment.len_ && offset <= segment.len_ - length,
          std::invalid_argument);
    }

    const void* data() const noexcept {
      return buf_;
    }

    void* mutable_data() const noexcept {
      return buf_;
    }

    size_t size() const noexcept {
      return length_;
    }

    MemoryType memType() const noexcept {
      return memType_;
    }

    int deviceId() const noexcept {
      return deviceId_;
    }

   private:
    void* buf_;
    const size_t length_;
    const MemoryType memType_;
    const int deviceId_;
  };

  using Span = TSpan;

  template <typename S = Derived>
  typename S::Span span() {
    S& segment = static_cast<S&>(*this);
    return typename S::Span(segment, 0, segment.len_);
  }

  template <typename S = Derived>
  typename S::Span span(size_t offset, size_t length) {
    return typename S::Span(static_cast<S&>(*this), offset, length);
  }

  template <typename S = Derived>
  typename S::Span span(const void* buf, size_t len) {
    S& segment = static_cast<S&>(*this);
    CHECK_THROW_EXCEPTION((segment.buf_ <= buf), std::invalid_argument);
    const size_t offset = (uintptr_t)buf - (uintptr_t)segment.buf_;
    return typename S::Span(segment, offset, len);
  }

  MemoryType memType() const noexcept {
    return memType_;
  }

  int deviceId() const noexcept {
    return deviceId_;
  }

  const void* data() const noexcept {
    return buf_;
  }

  void* mutable_data() noexcept {
    return const_cast<void*>(data());
  }

  size_t len() const noexcept {
    return len_;
  }

  SegmentBase(
      void* buf,
      size_t len,
      MemoryType memType = MemoryType::DRAM,
      int deviceId = -1)
      : memType_(memType), deviceId_(deviceId), buf_(buf), len_(len) {}

  ~SegmentBase() = default;
  SegmentBase(const SegmentBase&) = default;
  SegmentBase& operator=(const SegmentBase&) = default;

  SegmentBase(SegmentBase&&) = default;
  SegmentBase& operator=(SegmentBase&&) = default;

 protected:
  MemoryType memType_;
  int deviceId_{-1};
  void* buf_{nullptr};
  size_t len_{0};
};

class Segment : public SegmentBase<Segment> {
 public:
  Segment(
      void* buf,
      size_t len,
      MemoryType memType = MemoryType::DRAM,
      int deviceId = -1)
      : SegmentBase(buf, len, memType, deviceId) {}

  ~Segment() = default;
  Segment(const Segment&) = default;
  Segment& operator=(const Segment&) = default;

  Segment(Segment&&) = default;
  Segment& operator=(Segment&&) = default;
};

class RegisteredSegment : public SegmentBase<RegisteredSegment> {
 public:
  virtual ~RegisteredSegment() = default;

  RegisteredSegment(const RegisteredSegment&) = delete;
  RegisteredSegment& operator=(const RegisteredSegment&) = delete;

  RegisteredSegment(RegisteredSegment&&) = default;
  RegisteredSegment& operator=(RegisteredSegment&&) = default;

  class Span : public TSpan {
   public:
    Span(RegisteredSegment& segment, size_t offset, size_t length)
        : TSpan(segment, offset, length), handles_(segment.handles_) {}

    friend class MultiTransport;
    friend class NvLinkTransport;
    friend class RdmaTransport;

   private:
    std::span<const std::unique_ptr<RegistrationHandle>> handles_;
  };

  Result<std::vector<uint8_t>> exportId() const;

  /// Construct from a Segment and a single backend registration handle.
  static RegisteredSegment fromHandle(
      Segment& segment,
      std::unique_ptr<RegistrationHandle> handle) {
    RegisteredSegment reg(segment);
    reg.handles_.push_back(std::move(handle));
    return reg;
  }

  friend class SegmentTest;
  friend class MultiTransportFactory;

 private:
  explicit RegisteredSegment(Segment& segment)
      : SegmentBase(
            segment.mutable_data(),
            segment.len(),
            segment.memType(),
            segment.deviceId()) {}

  explicit RegisteredSegment(
      void* buf,
      size_t len,
      MemoryType memType = MemoryType::DRAM,
      int deviceId = -1)
      : SegmentBase(buf, len, memType, deviceId) {}

  std::vector<std::unique_ptr<RegistrationHandle>> handles_;
};

class RemoteRegisteredSegment : public SegmentBase<RemoteRegisteredSegment> {
 public:
  virtual ~RemoteRegisteredSegment() = default;

  RemoteRegisteredSegment(const RemoteRegisteredSegment&) = delete;
  RemoteRegisteredSegment& operator=(const RemoteRegisteredSegment&) = delete;

  RemoteRegisteredSegment(RemoteRegisteredSegment&&) = default;
  RemoteRegisteredSegment& operator=(RemoteRegisteredSegment&&) = default;

  class Span : public TSpan {
   public:
    Span(RemoteRegisteredSegment& segment, size_t offset, size_t length)
        : TSpan(segment, offset, length),
          handles_(segment.handles_),
          nvlinkOffset_(offset) {}

    friend class MultiTransport;
    friend class NVLinkTransport;
    friend class RdmaTransport;

   private:
    std::span<const std::unique_ptr<RemoteRegistrationHandle>> handles_;
    size_t nvlinkOffset_;
  };

  /// Construct from a remote buffer address and a single backend handle.
  static RemoteRegisteredSegment fromHandle(
      void* buf,
      size_t len,
      std::unique_ptr<RemoteRegistrationHandle> handle) {
    RemoteRegisteredSegment remote(buf, len);
    remote.handles_.push_back(std::move(handle));
    return remote;
  }

  friend class SegmentTest;
  friend class MultiTransportFactory;
  friend class MultiTransportFactoryTest;

 private:
  using remoteHandleT = Result<std::unique_ptr<RemoteRegistrationHandle>>;
  static Result<RemoteRegisteredSegment> from(
      std::span<const uint8_t> exportId,
      const std::function<
          remoteHandleT(TransportType, size_t, std::span<const uint8_t>)>&
          getHandle);

  explicit RemoteRegisteredSegment(Segment& segment)
      : SegmentBase(
            segment.mutable_data(),
            segment.len(),
            segment.memType(),
            segment.deviceId()) {}

  explicit RemoteRegisteredSegment(
      void* buf,
      size_t len,
      MemoryType memType = MemoryType::DRAM,
      int deviceId = -1)
      : SegmentBase(buf, len, memType, deviceId) {}

  std::vector<std::unique_ptr<RemoteRegistrationHandle>> handles_;
};

} // namespace uniflow
