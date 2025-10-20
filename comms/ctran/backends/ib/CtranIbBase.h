// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IB_BASE_H_
#define CTRAN_IB_BASE_H_

#include <folly/String.h>
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogUtils.h"

struct CtranIbRemoteAccessKey {
  std::array<uint32_t, CTRAN_MAX_IB_DEVICES_PER_RANK> rkeys{};
  int nKeys{0};

  std::string toString() const {
    std::vector<uint32_t> keys(rkeys.begin(), rkeys.begin() + nKeys);
    return folly::join(", ", keys);
  }

  static CtranIbRemoteAccessKey fromString(const std::string& str) {
    CtranIbRemoteAccessKey key;
    std::vector<folly::StringPiece> tokens;
    if (folly::trimWhitespace(str).empty()) {
      return key;
    }
    folly::split(',', str, tokens, true /* ignoreEmpty */);
    CHECK_THROW(
        tokens.size() <= CTRAN_MAX_IB_DEVICES_PER_RANK, std::invalid_argument);
    for (const auto& token : tokens) {
      key.rkeys[key.nKeys++] = folly::to<uint32_t>(token);
    }
    return key;
  }
};

// Ib Device info
struct CtranIbDevice {
  ibverbx::IbvDevice* ibvDevice{nullptr};
  const ibverbx::IbvPd* ibvPd{nullptr};
  ibverbx::IbvCq* ibvCq;
  uint8_t port{0};
  std::string devName;
};

/**
 * Class of request to track progress of a isendCtrl, irecvCtrl, or iput IB
 * operation.
 */
class CtranIbRequest {
 public:
  CtranIbRequest(){};
  ~CtranIbRequest(){};

  // Mark the number of expected references associated with the request (e.g.,
  // multiple references if a data is internally chunked to multiple IB packets
  // and issued via multiple QPs). Default set refCount to 1 when creating the
  // request.
  void setRefCount(int refCount);

  // Mark completion of a reference
  inline commResult_t complete() {
    this->refCount_--;
    if (this->refCount_ < 0) {
      CLOGF(ERR, "CTRAN-IB: req {} refCount_ < 0", (void*)this);
      return commInternalError;
    }
    if (this->refCount_ == 0) {
      this->state_ = CtranIbRequest::COMPLETE;
    }
    return commSuccess;
  }

  // Return true if all references have been completed. Otherwise false.
  inline bool isComplete() const {
    return this->state_ == CtranIbRequest::COMPLETE;
  }

  // Used to mark need remote notify at completion of numQp packets.
  // Used by IBVC internally.
  bool notify{false};

  // Repost the request with specified number of references.
  // It sets the state to INCOMPLETE.
  void repost(int refCount);

 private:
  enum {
    INCOMPLETE,
    COMPLETE,
  } state_{INCOMPLETE};
  int refCount_{1};
  CtranIbConfig config_{};
};

/**
 * Structure to describe a pending control operation
 */
struct PendingOp {
  enum PendingOpType {
    UNDEFINED,
    ISEND_CTRL,
    IRECV_CTRL,
  };

 public:
  PendingOp(
      PendingOp::PendingOpType opType,
      std::optional<const int> type,
      void* payload,
      size_t size,
      int peerRank,
      CtranIbRequest& req)
      : opType(opType),
        type(type),
        payload(payload),
        size(size),
        peerRank(peerRank),
        req(req) {}
  ~PendingOp() {}

  PendingOpType opType{UNDEFINED};

  std::optional<const int> type{std::nullopt};
  void* payload{nullptr};
  size_t size{0};

  int peerRank{-1};
  CtranIbRequest& req;

  std::string toString() const {
    return fmt::format(
        "opType {} type {} payload {} size {} peerRank {} ibReq {}",
        opType == PendingOpType::ISEND_CTRL ? "ISEND_CTRL" : "IRECV_CTRL",
        type.value_or(-1),
        payload,
        size,
        peerRank,
        (void*)&req);
  }
};

#endif
