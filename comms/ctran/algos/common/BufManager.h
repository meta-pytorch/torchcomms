// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <sstream>
#include <vector>
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/TmpBufSegManager.h"

using ::ctran::utils::TmpBufSegManager;

/* Convenient types to manage temporary buffers used in algorithms */

namespace ctran::algos::bufmanager {
// FIXME: We should consolidate the enum with DevMemType. BufManager currently
// uses the enum value as the array index: https://fburl.com/code/xrf2jnmu, so
// we would need to remove the index dependency on enum value first.
enum class MemType {
  kDevice = 0,
  kHostPinned,
  kNumBufTypes,
};

inline std::string memTypeToStr(const MemType type) {
  switch (type) {
    case MemType::kDevice:
      return "DEVICE";
    case MemType::kHostPinned:
      return "HOST_PINNED";
    default:
      return "UNKNOWN";
  };
}

struct MemBase {
  MemType type{MemType::kDevice};
  void* ptr{nullptr};
  size_t size{0};
  void* segHdl{nullptr};
  void* regHdl{nullptr};
  std::string memKey;
  const std::vector<int> peerRanks;

  // indexed by peerRank
  std::vector<void*> remotePtrs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;

  std::string toString() const {
    std::stringstream ss;
    ss << "[" << memTypeToStr(type) << "] ptr: " << ptr << ", size: " << size
       << ", segHdl: " << segHdl << ", regHdl: " << regHdl;
    return ss.str();
  }
};

struct BasicBuf {
  void* ptr{nullptr};
  size_t size{0};

  std::string toString() const {
    std::stringstream ss;
    ss << "[BASIC_BUF] ptr: " << ptr << ", size: " << size;
    return ss.str();
  }
};

struct RegBuf {
  void* ptr{nullptr};
  size_t size{0};
  void* regHdl{nullptr};

  std::string toString() const {
    std::stringstream ss;
    ss << "[REG_BUF] ptr: " << ptr << ", size: " << size
       << ", regHdl: " << regHdl;
    return ss.str();
  }
};

struct RemRegBuf {
  int peerRank{-1};
  void* ptr{nullptr};
  CtranMapperRemoteAccessKey rkey;

  std::string toString() const {
    std::stringstream ss;
    ss << "[REM_REG_BUF] peerRank: " << peerRank << " ptr: " << ptr
       << ", backend: " << CtranMapper::backendToStr(rkey.backend);
    if (rkey.backend == CtranMapperBackend::IB) {
      for (int i = 0; i < rkey.ibKey.nKeys; i++) {
        ss << ", rkey: " << rkey.ibKey.rkeys[i];
      }
    }
    return ss.str();
  }
};

commResult_t commitBase(
    MemBase& base,
    const ncclx::CommStateX* statex,
    const CommLogData* logMetaData);
commResult_t releaseBase(
    MemBase& base,
    const ncclx::CommStateX* statex,
    CtranMapper* mapper);
commResult_t exchangeBase(
    MemBase& base,
    const std::vector<int>& peerRanks,
    const int maxNumRanks,
    const ncclx::CommStateX* statex,
    CtranMapper* mapper);
}; // namespace ctran::algos::bufmanager

namespace ctran::algos {
template <typename T, T MaxNumBufs>
struct BufSnapshot {
  std::array<bufmanager::BasicBuf, static_cast<size_t>(MaxNumBufs)> bufs;
  std::array<bufmanager::RegBuf, static_cast<size_t>(MaxNumBufs)> regBufs;
  std::
      array<std::vector<bufmanager::RemRegBuf>, static_cast<size_t>(MaxNumBufs)>
          remRegBufs;

  inline bufmanager::BasicBuf& getBuf(const T id) {
    return bufs[(size_t)id];
  }

  inline bufmanager::RegBuf& getRegBuf(const T id) {
    return regBufs[(size_t)id];
  }

  inline std::vector<bufmanager::RemRegBuf>& getRemRegBufs(const T id) {
    return remRegBufs[(size_t)id];
  }
};

template <typename T, T MaxNumBufs>
class BufManager {
 public:
  BufManager(
      const ncclx::CommStateX* statex,
      CtranMapper* mapper,
      const CommLogData* logMetaData,
      const std::string& memKey)
      : mapper_(mapper),
        statex_(statex),
        logMetaData_(logMetaData),
        memKey_(memKey) {};
  ~BufManager() {
    // It is recommended to call release() explicitly to catch any error
    FB_COMMCHECKIGNORE(release());
  }

  // Check if the buffer is pre-inserted
  inline bool contains(const bufmanager::MemType memType, const T id) {
    const auto memType_ = (size_t)memType;
    return memSegMgrs_[memType_].contains(id);
  }

  // Check if user has called exchange to register memory and shared with other
  // ranks in the communicator
  inline bool isExchanged() const {
    return isExchanged_;
  }

  // Check if user has called commit to allocate memory
  inline bool isCommitted() const {
    return isCommitted_;
  }

  // Assign a basic buffer (no registration) specified by id.
  // It requires user to 1) insert the id, and 2) call commit() to allocate the
  // memory.
  // - If the id is not pre-inserted or the memory is not commited, return false
  //   and the buffer is not assigned.
  // - Otherwise, return true and the buffer is assigned from the internal
  //   memory base based on tracked offset and length.
  inline bool assignBuf(
      const bufmanager::MemType memType,
      const T id,
      bufmanager::BasicBuf& buf) {
    if (!isCommitted() || !contains(memType, id)) {
      return false;
    }
    const auto memType_ = (size_t)memType;
    const auto& segInfo = memSegMgrs_[memType_].getSegInfo(id);
    const auto& base = memBases_[memType_];
    buf.ptr = BUFOFFSET(base.ptr, segInfo.offset);
    buf.size = segInfo.len;
    return true;
  }

  // Assign a locally registered buffer specified by id.
  // It requires user to 1) insert the id, 2) call commit() to allocate the
  // memory, and 3) call exchange() to register the memory with other ranks.
  // - If any of the above conditions are not met, return false and the buffer
  //   is not assigned.
  // - Otherwise, return true and the buffer is assigned from the internal
  //   memory base based on tracked offset and length, and with the regHdl of
  //   the base.
  inline bool assignRegBuf(
      const bufmanager::MemType memType,
      const T id,
      bufmanager::RegBuf& regBuf) {
    if (!isCommitted() || !isExchanged() || !contains(memType, id)) {
      return false;
    }
    const auto memType_ = (size_t)memType;
    const auto& segInfo = memSegMgrs_[memType_].getSegInfo(id);
    const auto& base = memBases_[memType_];
    regBuf.ptr = BUFOFFSET(base.ptr, segInfo.offset);
    regBuf.size = segInfo.len;
    regBuf.regHdl = base.regHdl;
    return true;
  }

  // Assign a vector of remotely imported buffers specified by id.
  // It requires user to 1) insert the id, 2) call commit() to allocate the
  // memory, and 3) call exchange() to register the memory with other ranks.
  // - If any of the above conditions are not met, return false and the buffer
  //   is not assigned.
  // - Otherwise, return true and the remote buffers are assigned from the
  //   remote ptrs of the internal memory base based on tracked offset, and with
  //   the rkey of the base.
  // - It would hit either out-of-range access exception or get a nullptr if any
  //   of the required peerRanks were not provided in exchange(). Callsite is
  //   required to ensure the correctness of peerRanks.
  inline bool assignRemRegBuf(
      const bufmanager::MemType memType,
      const T id,
      std::vector<int>& peerRanks,
      std::vector<bufmanager::RemRegBuf>& remRegBufs) {
    if (!isCommitted() || !isExchanged() || !contains(memType, id)) {
      return false;
    }

    const auto memType_ = (size_t)memType;
    const auto& base = memBases_[memType_];
    const auto& segInfo = memSegMgrs_[memType_].getSegInfo(id);

    remRegBufs.resize(peerRanks.size());
    for (auto i = 0; i < peerRanks.size(); i++) {
      const auto peerRank = peerRanks[i];
      remRegBufs[i].peerRank = peerRank;
      remRegBufs[i].ptr =
          BUFOFFSET(base.remotePtrs.at(peerRank), segInfo.offset);
      remRegBufs[i].rkey = base.remoteAccessKeys.at(peerRank);
    }
    return true;
  }

  // Insert a buffer specified by id and length. The
  // buffer is not allocated nor assigned in this call, but only tracks the
  // length and offset of each buffer, so that callsite can assign the buffer
  // after commit(). See commit() for allocation, and assign*() APIs for
  // assignment.
  inline commResult_t
  insert(const bufmanager::MemType type, const T id, const size_t len) {
    auto res = memSegMgrs_[(size_t)type].insert(id, len);
    return res ? commSuccess : commInternalError;
  }

  // Allocate the memory bases based on the inserted buffers. Each memType will
  // be allocated as separate base. After this call, callsite can assign
  // local buffers. See assignBuf().
  inline commResult_t commit() {
    for (auto type = 0; type < memBases_.size(); type++) {
      const auto totalLen = memSegMgrs_[type].totalLen;
      // Initialize base if any buffers inserted with this memType
      if (totalLen > 0) {
        // Add suffix to separate multiple bases with different memType and
        // sizes
        std::string memBaseKey = memKey_ + "_" + std::to_string(type) + "_" +
            std::to_string(totalLen);

        auto& base = memBases_[type];
        base.memKey = memBaseKey;
        base.size = totalLen;
        base.type = (bufmanager::MemType)type;
        FB_COMMCHECK(commitBase(base, statex_, logMetaData_));
      }
    }
    isCommitted_ = true;
    return commSuccess;
  }

  // Register the memory bases and exchange the registration with other ranks
  // specified in peerRanks. After this call, callsite can assign registered
  // buffers and remote buffers. See assignRegBuf() and assignRemRegBuf().
  inline commResult_t exchange(
      const std::vector<int>& peerRanks,
      const int maxNumRanks) {
    for (auto& base : memBases_) {
      // zero-sized base is skipped internally
      FB_COMMCHECK(
          exchangeBase(base, peerRanks, maxNumRanks, statex_, mapper_));
    }
    isExchanged_ = true;
    return commSuccess;
  }

  // Release the memory bases. It is recommended to call release() explicitly to
  // catch any error.
  inline commResult_t release() {
    for (auto& base : memBases_) {
      // zero-sized base is skipped internally
      FB_COMMCHECK(releaseBase(base, statex_, mapper_));
    }
    isCommitted_ = false;
    isExchanged_ = false;
    return commSuccess;
  }

 private:
  // Actual allocated memory bases, each element is a different memtype
  std::array<bufmanager::MemBase, (size_t)bufmanager::MemType::kNumBufTypes>
      memBases_;
  // Segment manager helpers for each memtype to track requested buffers and
  // their offsets, and totalLen for allocation in commit()
  std::array<
      TmpBufSegManager<T, MaxNumBufs>,
      (size_t)bufmanager::MemType::kNumBufTypes>
      memSegMgrs_;

  bool isCommitted_{false};
  bool isExchanged_{false};

  CtranMapper* mapper_{nullptr};
  const ncclx::CommStateX* statex_{nullptr};
  const CommLogData* logMetaData_;
  const std::string memKey_;
};

} // namespace ctran::algos
