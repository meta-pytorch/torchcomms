// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <deque>
#include <memory>
#include <string>

#include <folly/Synchronized.h>
#include <folly/futures/Future.h>
#include <folly/io/async/AsyncTimeout.h>
#include <folly/io/async/EventBase.h>

#include <comms/utils/commSpecs.h>

// Forward declaration
class CtranIb;

namespace torch::comms {

/*
 * RDMA Transport needs access to the memory buffer in order to transmit or
 * receive. To do you, user need to register the memory. This class provides
 * a convenient RAII style wrapper so that memory can be freed automatically.
 *
 * The memory is directly registered to the IB-Device and not specific
 * to an instance of a transport. And user can use any sub-range of this
 * registered memory for I/O APIs on RdmaTransport.
 *
 * Gotcha - Minimum memory block to be registered must be > 4097 bytes.
 */
class RdmaMemory : folly::MoveOnly {
 public:
  class View {
   public:
    /*
     * Create a view of a subset of RdmaMemory with bounds checking.
     * Asserts that the view is within the bounds of the parent memory.
     */
    View(const RdmaMemory& parent, size_t offset, size_t length)
        : parent_(parent), offset_(offset), length_(length) {
      CHECK_THROW(offset <= parent_.len_, std::invalid_argument);
      CHECK_THROW(offset + length <= parent_.len_, std::invalid_argument);
    }

    /*
     * Get pointer to the start of the view within the parent memory
     */
    const void* data() const {
      return static_cast<const uint8_t*>(parent_.buf_) + offset_;
    }

    /*
     * Get the length of this view
     */
    size_t size() const {
      return length_;
    }

    const RdmaMemory* operator->() const {
      return &parent_;
    }

   protected:
    const RdmaMemory& parent_;
    const size_t offset_;
    const size_t length_;
  };

  class MutableView : public View {
   public:
    /*
     * Create a writable view of a subset of RdmaMemory with bounds checking.
     * Asserts that the view is within the bounds of the parent memory.
     */
    MutableView(const RdmaMemory& parent, size_t offset, size_t length)
        : View(parent, offset, length) {}

    /*
     * Get mutable pointer to the start of the view within the parent memory
     */
    void* mutable_data() const {
      return const_cast<void*>(View::data());
    }
  };

  RdmaMemory(const void* buf, size_t len, int cudaDev);
  RdmaMemory(RdmaMemory&& other) noexcept;
  ~RdmaMemory();

  View createView() const {
    return View(*this, 0, len_);
  }

  View createView(size_t offset, size_t length) const {
    return View(*this, offset, length);
  }

  View createView(const void* buf, size_t length) const {
    const size_t offset = (uintptr_t)buf - (uintptr_t)buf_;
    return View(*this, offset, length);
  }

  MutableView createMutableView() const {
    return MutableView(*this, 0, len_);
  }

  MutableView createMutableView(size_t offset, size_t length) const {
    return MutableView(*this, offset, length);
  }

  MutableView createMutableView(const void* buf, size_t length) const {
    const size_t offset = (uintptr_t)buf - (uintptr_t)buf_;
    return MutableView(*this, offset, length);
  }

  /*
   * Local key associated with this buffer
   */
  void* localKey() const {
    return regHdl_;
  }

  /*
   * Get the access key for the registered buffer, that can be
   * used by the remote side to access the buffer.
   */
  std::string remoteKey() const {
    return remoteKey_;
  }

  int getDevice() const {
    return cudaDev_;
  }

  size_t length() const {
    return len_;
  }

  const void* data() const {
    return buf_;
  }

  /*
   * Check if the given buffer and length are contained within this memory
   * region.
   */
  bool contains(const void* buf, const size_t len) const;

 private:
  const void* buf_{nullptr};
  const size_t len_{0};
  const int cudaDev_{-1};

  void* regHdl_{nullptr};
  std::string remoteKey_;
};

/**
 * Remote RDMA Buffer defining a pointer address and its associated
 * accessKey
 */
struct RdmaRemoteBuffer {
  void* ptr{nullptr};
  const size_t len{0};
  const std::string accessKey;
};

/*
 * RDMA Transport that provides easy to use APIs for transferring data
 * from memory of one host to another.
 *
 * Expected Usage:
 * - Endpoint-A:
 *   1. auto transport = std::make_unique<RdmaTransport>(cudaDev, true);
 *   2. auto serverUrl = transport->bind();
 *   3. transport->connect(clientUrl);
 *   4. Use APIs for memory registration and data transfer
 *
 * - Endpoint-B:
 *   1. auto transport = std::make_unique<RdmaTransport>(cudaDev, false);
 *   2. auto clientUrl = transport->bind();
 *   3. transport->connect(serverUrl);
 *   4. Use APIs for memory registration and data transfer
 *
 * folly::EventBase is used to drive the underlying RDMA operations. User
 * should have a dedicated EventBase for for transport operations and can
 * be shared across all transport instances. When requests are pending, this
 * will likely keep EventBase thread pretty busy to minimize latency.
 *
 * Supported RDMA APIs
 * - `write` -> RDMA write to a remote memory
 * - `waitForWrite` -> Wait for a remote write operation
 *
 * Future APIs that can be supported as per use-case. Given this framework
 * adding new APIs should be relatively straightforward.
 * - Send - RDMA Send (needs matching Recv on other end)
 * - Recv - RDMA Receive (needs mactching Send on other end)
 * - Read - RDMA Read from a remote memory
 * - waitForRead - Wait for a remote read operation
 * - <Atomic APIs>
 */
class __attribute__((visibility("default"))) RdmaTransport {
 public:
  /*
   * Constructor for RdmaTransport.
   * cudaDev - Transport needs to use NIC for I/O. It does so by identifying
   *           the NIC associated with specified cudaDevice.
   * evb - EventLoop to drive the RDMA operations.
   */
  explicit RdmaTransport(int cudaDev, folly::EventBase* evb = nullptr);

  ~RdmaTransport();

  /* Query whether RDMA is supported on the platform.
   * If not, it is likely that the platform does not have backend NIC or no
   * proper driver installed.
   */
  static bool supported();

  /*
   * Bind the transport and retrieve the unique identifier that can be used to
   * connect from the other end. Throws exception on error.
   */
  std::string bind();

  /*
   * Connect to the peer transport given a peerUrl.
   */
  commResult_t connect(const std::string& peerUrl);

  /*
   * Check if the transport has been connected.
   * If not, indicates it is a local transport, and can use only the local
   * operations.
   */
  bool connected() const;

  /*
   * [Remote Op] Transfer data from local buffer to remote buffer on the peer
   * rank via RDMA. The remote side can use the `checkNotify` API to wait for
   * the completion of the transfer for every iput call with notify=true.
   */
  folly::SemiFuture<commResult_t> write(
      RdmaMemory::View localBuffer,
      const RdmaRemoteBuffer& remoteBuffer,
      bool notify);

  /*
   * [Remote Op] Check the arrival of incoming put transfer from the remote
   * rank.
   */
  folly::SemiFuture<commResult_t> waitForWrite();

  /*
   * [Remote Op] Transfer data from remote buffer on the peer rank to local
   * buffer via RDMA.
   */
  folly::SemiFuture<commResult_t> read(
      RdmaMemory::MutableView& localBuffer,
      const RdmaRemoteBuffer& remoteBuffer);

 private:
  /*
   * Drive the IB progress loop and drive completion of pending requests.
   */
  void progress();

  std::unique_ptr<CtranIb> ib_;
  int cudaDev_{-1};
  folly::EventBase* evb_{nullptr};

  struct Work;
  folly::Synchronized<std::deque<std::unique_ptr<Work>>> pendingWorks_;
  std::unique_ptr<folly::AsyncTimeout> progressTimeout_;
};

} // namespace torch::comms
