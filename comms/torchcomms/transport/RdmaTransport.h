// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
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
 */
class RdmaMemory : folly::MoveOnly {
 public:
  /**
   * Read-only view into a subset of registered RDMA memory.
   *
   * LIFETIME SAFETY WARNING:
   * View holds a reference to the parent RdmaMemory object. The parent
   * MUST remain valid (not destroyed or moved) for the entire lifetime
   * of any View objects. Accessing a View after its parent has been
   * destroyed results in undefined behavior.
   *
   * Safe usage patterns:
   *   - Use View only within the scope where parent RdmaMemory is valid
   *   - Do not store View objects beyond the parent's lifetime
   *   - Do not return View from functions that return by value if the
   *     parent is a local variable
   *
   * Example of UNSAFE code:
   *   RdmaMemory::View getView() {
   *     RdmaMemory mem(buf, len, dev);  // local variable
   *     return mem.createView(0, len);  // DANGER: mem destroyed here!
   *   }  // View now holds dangling reference
   */
  class View {
   public:
    /*
     * Create a view of a subset of RdmaMemory with bounds checking.
     * Asserts that the view is within the bounds of the parent memory.
     *
     * @param parent The parent RdmaMemory - must outlive this View
     * @param offset Byte offset from start of parent buffer
     * @param length Length of the view in bytes
     * @throws std::invalid_argument if offset + length exceeds parent bounds
     */
    View(const RdmaMemory& parent, size_t offset, size_t length)
        : parent_(parent), offset_(offset), length_(length) {
      CHECK_THROW(offset <= parent_.len_, std::invalid_argument);
      // Check for overflow and bounds in a single safe comparison
      CHECK_THROW(
          length <= parent_.len_ && offset <= parent_.len_ - length,
          std::invalid_argument);
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

    /**
     * Check if the parent memory is still valid (non-null buffer).
     * Note: This is a best-effort check - if the parent has been destroyed,
     * calling this method is already undefined behavior.
     */
    bool isParentValid() const {
      return parent_.buf_ != nullptr && parent_.regHdl_ != nullptr;
    }

   protected:
    const RdmaMemory& parent_;
    const size_t offset_;
    const size_t length_;
  };

  /**
   * Mutable view into a subset of registered RDMA memory.
   *
   * See View class for lifetime safety warnings - all the same caveats apply.
   */
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
  RdmaMemory& operator=(RdmaMemory&& other) = delete;
  ~RdmaMemory() noexcept;

  View createView() const {
    return View(*this, 0, len_);
  }

  View createView(size_t offset, size_t length) const {
    return View(*this, offset, length);
  }

  /**
   * Create a view from an arbitrary pointer within this registered memory.
   *
   * @param buf Pointer that must be within the bounds of this RdmaMemory
   * @param length Length of the view in bytes
   * @throws std::out_of_range if buf is not within this memory region
   * @throws std::invalid_argument if the resulting view exceeds bounds
   */
  View createView(const void* buf, size_t length) const {
    if (!contains(buf, length)) {
      throw std::out_of_range(
          "Pointer is not within the bounds of this registered memory region");
    }
    const size_t offset = (uintptr_t)buf - (uintptr_t)buf_;
    return View(*this, offset, length);
  }

  MutableView createMutableView() const {
    return MutableView(*this, 0, len_);
  }

  MutableView createMutableView(size_t offset, size_t length) const {
    return MutableView(*this, offset, length);
  }

  /**
   * Create a mutable view from an arbitrary pointer within this registered
   * memory.
   *
   * @param buf Pointer that must be within the bounds of this RdmaMemory
   * @param length Length of the view in bytes
   * @throws std::out_of_range if buf is not within this memory region
   * @throws std::invalid_argument if the resulting view exceeds bounds
   */
  MutableView createMutableView(const void* buf, size_t length) const {
    if (!contains(buf, length)) {
      throw std::out_of_range(
          "Pointer is not within the bounds of this registered memory region");
    }
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
  // These members are logically const after construction, but need to be
  // mutable for move semantics to properly invalidate moved-from objects.
  const void* buf_{nullptr};
  size_t len_{0};
  int cudaDev_{-1};

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
 * should have a dedicated EventBase for transport operations and can
 * be shared across all transport instances. When requests are pending, this
 * will likely keep EventBase thread pretty busy to minimize latency.
 *
 * Supported RDMA APIs
 * - `write` -> RDMA write to a remote memory
 * - `read`  -> RDMA read from a remote memory
 * - `waitForWrite` -> Wait for a remote write operation
 *
 * Future APIs that can be supported as per use-case. Given this framework
 * adding new APIs should be relatively straightforward.
 * - Send - RDMA Send (needs matching Recv on other end)
 * - Recv - RDMA Receive (needs matching Send on other end)
 * - waitForRead - Wait for a remote read operation
 * - <Atomic APIs>
 *
 * API return value contracts (commResult_t):
 * All async APIs (write, read, waitForWrite) are guaranteed to eventually
 * return a commResult_t value — they must never hang indefinitely. Use the
 * timeout parameter or abort() to ensure bounded completion.
 *
 *   commSuccess — normal completion:
 *     write(), read(), waitForWrite(), connect()
 *
 *   commTimeout — operation exceeded its timeout duration:
 *     write()
 *     (read and waitForWrite do not yet support timeout)
 *
 *   commInternalError — IB / transport-level failure:
 *     write(), read(), waitForWrite()
 *
 *   commUserAbort — abort() was called while the operation was pending:
 *     write(), read(), waitForWrite()
 *
 *   Throws (no commResult_t) — unrecoverable setup error:
 *     bind(), connect()
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

  // Non-copyable and non-movable
  RdmaTransport(const RdmaTransport&) = delete;
  RdmaTransport& operator=(const RdmaTransport&) = delete;
  RdmaTransport(RdmaTransport&&) = delete;
  RdmaTransport& operator=(RdmaTransport&&) = delete;

  /* Query whether RDMA is supported on the platform.
   * If not, it is likely that the platform does not have backend NIC or no
   * proper driver installed.
   */
  static bool supported();

  /*
   * Bind the transport and retrieve the unique identifier that can be used to
   * connect from the other end. Throws on failure.
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
   *
   * @param timeout Optional timeout duration for the write operation. When
   *                specified, the operation will complete with commTimeout if
   *                the RDMA write does not finish within this duration. If not
   *                specified (nullopt), the write waits indefinitely.
   */
  folly::SemiFuture<commResult_t> write(
      RdmaMemory::View localBuffer,
      const RdmaRemoteBuffer& remoteBuffer,
      bool notify,
      std::optional<std::chrono::milliseconds> timeout = std::nullopt);

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

  /*
   * Mock type for testing RDMA transport error scenarios
   */
  enum class MockType {
    None, // No mock, normal operation
    Timeout, // Works complete with timeout error after specified duration
    Failure, // Works complete immediately with commInternalError
  };

  /*
   * Context for mock behavior, set via setMockForTest().
   */
  struct MockContext {
    MockType type{MockType::None};
  };

  /*
   * Inject software mock for testing. Any write while mock is enabled
   * will behave according to the mock configuration:
   * - Timeout: works stay pending until timeout fires (requires a timeout
   *            to be specified in the write() call) or abort() is called
   * - Failure: works complete immediately with commInternalError
   * - None: reset to disable mock
   *
   * The mock type is captured when operations are created, not when they
   * complete. Changing the mock config after calling write does not affect
   * already-created operations.
   *
   * The control is per RdmaTransport instance, and the state is shared with
   * all threads accessing the instance.
   */
  void setMockForTest(MockContext config);

  /*
   * Abort all pending works and complete them with commUserAbort status.
   * This is called by upper layer to cancel pending operations.
   *
   * Currently cleans up pending works at RDMA Transport level. This API is
   * preserved for future cleanup extensions.
   */
  void abort();

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
  // Mock configuration for testing; updated by setMockForTest
  folly::Synchronized<MockContext> mockContext_;
};

} // namespace torch::comms
