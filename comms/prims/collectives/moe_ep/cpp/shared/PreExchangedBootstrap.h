// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include <folly/futures/Future.h>

#include "comms/common/bootstrap/IBootstrap.h"

namespace comms::prims::moe_ep {

/**
 * PreExchangedBootstrap — IBootstrap impl that returns pre-gathered
 * allGather results in FIFO order.
 *
 * Use case: the LL runtime needs `MultipeerIbgdaTransport::exchange()` and
 * `exchangeBuffer()` to succeed, both of which call `bootstrap->allGather`.
 * On our backend we don't have a runtime collective layer plumbed into the
 * C++ Buffer ctor — Python-side `dist.all_gather_object` is the only
 * collective available. This bootstrap bridges the two: Python pre-gathers
 * the relevant byte arrays in advance and queues them; the transport's
 * sequential `allGather` calls pop from the queue.
 *
 * Ordering contract: caller MUST queue allGather results in the exact
 * order that `MultipeerIbgdaTransport::exchange()` and the subsequent
 * `exchangeBuffer()` calls will consume them. Mismatched sizes throw.
 *
 * `barrier` is a no-op; assumes Python-side `dist.barrier()` was called
 * before construction. `send/recv` are unsupported (throw).
 */
class PreExchangedBootstrap : public meta::comms::IBootstrap {
 public:
  PreExchangedBootstrap(int rank, int nRanks) : rank_(rank), nRanks_(nRanks) {}

  /**
   * Push a pre-gathered allGather result. `data.size()` must equal
   * `nRanks * len` for the corresponding allGather call. Each rank's slot
   * occupies `len` bytes.
   */
  void pushGather(std::vector<uint8_t> data) {
    std::lock_guard<std::mutex> lock(mu_);
    queue_.push(std::move(data));
  }

  folly::SemiFuture<int> allGather(void* buf, int len, int rank, int nranks)
      override {
    if (rank != rank_ || nranks != nRanks_) {
      return folly::makeSemiFuture<int>(
          std::runtime_error("PreExchangedBootstrap: rank/nranks mismatch"));
    }
    std::vector<uint8_t> data;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (queue_.empty()) {
        return folly::makeSemiFuture<int>(std::runtime_error(
            "PreExchangedBootstrap: no pre-gathered data queued for this "
            "allGather call (rank " +
            std::to_string(rank_) + ")"));
      }
      data = std::move(queue_.front());
      queue_.pop();
    }
    const std::size_t expected = static_cast<std::size_t>(nranks) * len;
    if (data.size() != expected) {
      return folly::makeSemiFuture<int>(std::runtime_error(
          "PreExchangedBootstrap: gathered size mismatch (got " +
          std::to_string(data.size()) + ", expected " +
          std::to_string(expected) + ")"));
    }
    std::memcpy(buf, data.data(), expected);
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int> barrier(int /*rank*/, int /*nranks*/) override {
    // No-op; assume Python pre-barriered before passing data to C++.
    return folly::makeSemiFuture(0);
  }

  folly::SemiFuture<int>
  send(void* /*buf*/, int /*len*/, int /*peer*/, int /*tag*/) override {
    return folly::makeSemiFuture<int>(
        std::runtime_error("PreExchangedBootstrap: send not supported"));
  }

  folly::SemiFuture<int>
  recv(void* /*buf*/, int /*len*/, int /*peer*/, int /*tag*/) override {
    return folly::makeSemiFuture<int>(
        std::runtime_error("PreExchangedBootstrap: recv not supported"));
  }

 private:
  const int rank_;
  const int nRanks_;
  std::mutex mu_;
  std::queue<std::vector<uint8_t>> queue_;
};

} // namespace comms::prims::moe_ep
