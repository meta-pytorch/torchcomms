// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <vector>
#include "comms/uniflow/Result.h"

namespace uniflow::controller {

/// A framed, bidirectional connection to one peer.
///
/// Not safe for concurrent use: at most one send() and one recv() may be in
/// flight at a time, and callers must serialize their own access. Overlapping
/// sends do not merely scramble bytes -- each send() writes its own length
/// prefix, so two senders interleave frames and the peer reassembles a payload
/// from both.
///
/// Enforcement is uneven, so do not rely on a diagnostic. TcpConn<AsyncIO>
/// rejects an overlapping call with ErrCode::ResourceExhausted, but
/// TcpConn<SyncIO> -- what accept() and connect() return, and therefore what
/// every control connection is -- has no such check and corrupts the stream
/// silently.
class Conn {
 public:
  virtual ~Conn() = default;

  /// The data is owned by the caller, not the connection. The caller
  /// must ensure the buffer outlives the returned future.
  [[nodiscard]] virtual std::future<Result<size_t>> send(
      std::span<const uint8_t> data) = 0;

  /// Allocating recv: reads length prefix, allocates buffer, fills it.
  [[nodiscard]] virtual std::future<Result<size_t>> recv(
      std::vector<uint8_t>& data) = 0;

  /// Zero-copy recv: reads length prefix, fills caller's pre-allocated buffer.
  /// Error if payload exceeds buf.size(). The buffer is owned by the
  /// caller, not the connection. The caller must ensure the buffer
  /// outlives the returned future.
  [[nodiscard]] virtual std::future<Result<size_t>> recv(
      std::span<uint8_t> buf) = 0;

  /// Interrupt any blocked recv(). After close(), recv() must return an error.
  /// Called by Connection::shutdown() to release whatever thread is parked in
  /// recv(); that thread belongs to the caller, not to Connection.
  virtual void close() {}

  /// Splits recv() time into "waiting for the next frame to start" and
  /// "draining a frame that has started". For a length-prefixed stream those
  /// are the block on the length prefix and the block on the payload, which is
  /// the only place the two can be told apart -- above this layer a recv() is
  /// one opaque wait.
  ///
  /// headerWaitNs is a first-byte latency: for a get it covers the network
  /// round trip plus everything the peer did before it started replying, so it
  /// separates "the remote is slow" from "our drain is slow". Relaxed atomics,
  /// written only by the reader thread; a torn read across a reset costs a
  /// misattributed sample, never correctness.
  struct RecvPhaseStats {
    std::atomic<uint64_t> headerWaitNs{0};
    std::atomic<uint64_t> payloadDrainNs{0};
    std::atomic<uint64_t> frames{0};
    std::atomic<uint64_t> payloadBytes{0};

    void reset() {
      headerWaitNs.store(0, std::memory_order_relaxed);
      payloadDrainNs.store(0, std::memory_order_relaxed);
      frames.store(0, std::memory_order_relaxed);
      payloadBytes.store(0, std::memory_order_relaxed);
    }
  };

  RecvPhaseStats& recvPhaseStats() {
    return recvPhaseStats_;
  }

 private:
  RecvPhaseStats recvPhaseStats_;
};

class Server {
 public:
  virtual ~Server() = default;

  virtual Status init() = 0;

  virtual const std::string& getId() const = 0;

  [[nodiscard]] virtual std::future<std::unique_ptr<Conn>> accept() = 0;
};

class Client {
 public:
  Client() = default;
  virtual ~Client() = default;

  [[nodiscard]] virtual std::future<std::unique_ptr<Conn>> connect(
      std::string id) = 0;
};

} // namespace uniflow::controller
