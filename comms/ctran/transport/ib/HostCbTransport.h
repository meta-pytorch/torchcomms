// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/transport/IP2pHostTransport.h"
#include "comms/ctran/transport/ib/ChunkHooks.h"
#include "comms/ctran/transport/ib/HostTransportDev.cuh"
#include "comms/ctran/transport/ib/HostTransportImpl.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/commSpecs.h"

class CtranIb;
class CtranIbVirtualConn;
struct CommLogData;

namespace ctran::transport::ib {

using ctran::algos::GpeKernelSync;

constexpr int kMaxPipelineDepth = 8;

// --- CB chunk state machine. Each in-flight chunk on a CB staging slot
// walks through these states once. ---

enum class SendChunkStatus {
  IDLE,
  PREPARE_DATA,
  WAIT_PREPARE,
  WAIT_REMOTE_READY,
  WAIT_IPUT,
};

enum class RecvChunkStatus {
  IDLE,
  WAIT_RECV,
  PROCESS_DATA,
  WAIT_PROCESS,
  SIGNAL_READY,
};

// --- Per-chunk descriptors recorded into a CB staging-slot record at
// issue time. ---

struct SendChunkInfo {
  const void* userBuf{nullptr};
  size_t offset{0};
  size_t len{0};
  int vcIdx{0};
  int stagingSlot{kNoStagingSlot}; // 0..pipelineDepth-1
  uint64_t round{0}; // GpeKernelSync round for this chunk.
  SendChunkHooks hooks;
};

struct RecvChunkInfo {
  void* userBuf{nullptr};
  size_t offset{0};
  size_t len{0};
  int vcIdx{0};
  int stagingSlot{kNoStagingSlot};
  uint64_t round{0};
  RecvChunkHooks hooks;
};

// --- CB per-staging-slot records. One in-flight chunk at a time per
// slot. issued/completed counters drive testChunkDone for CB. ---

struct SendStagingRecord {
  SendChunkStatus status{SendChunkStatus::IDLE};
  CtranIbRequest iputReq;
  std::optional<SendChunkInfo> chunk;
  uint64_t issued{0}; // bumped on each iSendChunk acquiring this slot
  uint64_t completed{0}; // bumped when state machine finishes this chunk
};

struct RecvStagingRecord {
  RecvChunkStatus status{RecvChunkStatus::IDLE};
  std::optional<RecvChunkInfo> chunk;
  uint64_t issued{0};
  uint64_t completed{0};
};

// Pure copy-based host-side IB transport.
//
// Owns:
//   - pipelineDepth_ CB send/recv staging-slot records (each bound 1:1
//     to a staging buffer + GpeKernelSync + flow-control triple).
//   - The peer staging-buffer + per-slot remoteReady info exchanged
//     via the one-time postResourceExchange/testResourceExchange
//     handshake the first time a chunk is issued.
//   - The device-side HostTransportDev mirror used by the CB copy
//     kernels; lazily built on first getDeviceTransport() call.
//
// Shared plumbing (lazy connectVcs, ctrl-msg primitives, mem
// export/import) lives in HostTransportImpl.h as
// ctran::transport::ib::impl::* free functions — this class delegates
// to them rather than inheriting.
//
// Allocations all flow through comms/ctran/utils/Alloc.h
// (commCudaMalloc / commCudaHostAlloc) so they show up in NCCL
// memtrace and can be pooled in a future iteration.
class HostCbTransport : public IP2pHostTransport {
 public:
  // pool is borrowed (owned by CtranGpe). The CB transport pops
  // pipelineDepth_ entries for sends and recvs each, and reset()s
  // them in its destructor.
  // logMetaData is borrowed (owned by CtranMapper / comm) and used by
  // the ctran::utils allocator wrappers to attribute each allocation
  // in NCCL memtrace.
  HostCbTransport(
      int peerRank,
      CtranIb* ctranIb,
      GpeKernelSyncPool* pool,
      int pipelineDepth,
      size_t chunkSize,
      int myRank,
      int cudaDev,
      const CommLogData* logMetaData);

  ~HostCbTransport() override;

  inline HostTransportMode mode() const override {
    return HostTransportMode::kCopyBased;
  }

  inline int peerRank() const override {
    return peerRank_;
  }
  inline int numVcs() const {
    return static_cast<int>(vcs_.size());
  }
  inline bool vcsReady() const {
    return !vcs_.empty();
  }
  inline int myRank() const {
    return myRank_;
  }
  inline int cudaDev() const {
    return cudaDev_;
  }

  inline int pipelineDepth() const override {
    return pipelineDepth_;
  }
  inline size_t chunkSize() const override {
    return chunkSize_;
  }

  // CB chunking driven by chunkSize_.
  inline int computeTotalChunks(size_t totalSize) const override {
    return static_cast<int>((totalSize + chunkSize_ - 1) / chunkSize_);
  }
  inline size_t computeChunkOffset(int chunkIdx, size_t /*totalSize*/)
      const override {
    return static_cast<size_t>(chunkIdx) * chunkSize_;
  }
  inline size_t computeChunkLen(int chunkIdx, size_t totalSize) const override {
    const size_t off = static_cast<size_t>(chunkIdx) * chunkSize_;
    return std::min(chunkSize_, totalSize - off);
  }

  // === Resource lifecycle ===
  inline commResult_t progress() override;

  // === Per-operation ctrl exchange ===
  // Pure ctrl-msg primitives. CB callers typically build a SYNC msg
  // (msg.setType(ControlMsgType::SYNC)) in their own buffer before
  // iSendCtrlMsg; the ZC-handshake helpers
  // ctran::transport::ib::impl::exportRecvBuf /
  // ctran::transport::ib::impl::importRemoteInfo are CB-irrelevant.
  inline commResult_t iSendCtrlMsg(
      ControlMsgType type,
      const void* payload,
      size_t len,
      CtrlRequest* out) override;
  inline commResult_t iRecvCtrlMsg(void* payload, size_t len, CtrlRequest* out)
      override;

  // === Ctrl-request poll/wait ===
  inline commResult_t testCtrlMsgDone(CtrlRequest& req, bool* done) override;
  inline commResult_t waitCtrlMsgDone(CtrlRequest& req) override;

  inline bool isReadyForSend(int vcIdx, int stagingSlot = kNoStagingSlot)
      override;
  inline bool isReadyForRecv(int vcIdx, int stagingSlot = kNoStagingSlot)
      override;

  // CB iSendChunk: enqueues into the per-slot state machine.
  // Asserts ZC-specific fields are nullptr; asserts stagingSlot is in
  // range and that the slot is IDLE (caller must have drained via
  // testChunkDone).
  inline commResult_t iSendChunk(const SendChunkArgs& args) override;

  inline commResult_t iRecvChunk(const RecvChunkArgs& args) override;

  inline commResult_t testChunkDone(const ChunkRequest& req, bool* done)
      override;

  // Lazily build (and cache) the device-side mirror used by the CB
  // copy kernels. First call cudaMallocs the device struct; subsequent
  // calls return the cached pointer.
  HostTransportDev* getDeviceTransport();

  // Set the per-slot GpeKernelSync `nworkers` for all send and recv
  // staging slots. Must be called BEFORE the first iSendChunk /
  // iRecvChunk that schedules a kernel against this transport, and
  // must match the number of GPU blocks the kernel actually launches
  // (gpeSyncComplete writes completeFlag[myBlockIdx], and
  // GpeKernelSync::isComplete(round) walks completeFlag[0..nworkers);
  // mismatched nworkers either hangs or under-syncs).
  //
  // `sendNumBlocks` <= 0 → leave send-side nworkers unchanged.
  // `recvNumBlocks` <= 0 → leave recv-side nworkers unchanged.
  // Default ctor value is CTRAN_ALGO_MAX_THREAD_BLOCKS — algorithms
  // that launch the maximum-block kernel can skip this call.
  void setKernelNumBlocks(int sendNumBlocks, int recvNumBlocks);

  // === Per-transport caller-lock (see IP2pHostTransport.h) ===
  //
  // Aborts on double-lock-from-same-thread and unlock-without-lock
  // (mirrors CtranIb::epochLock / epochUnlock at
  // fbcode/comms/ctran/backends/ib/CtranIb.cc:741-794).
  inline void lock() override {
    FB_CHECKABORT(
        !::ctran::transport::impl::p2pTransportLockFlagIsSet(this),
        "CTRAN-IB: HostCbTransport::lock() called twice on the same thread "
        "without an intervening unlock(). Likely a missing "
        "P2pTransportLockGuard scope.");
    transportMutex_.lock();
    ::ctran::transport::impl::p2pTransportLockFlagSet(this, true);
  }
  inline void unlock() override {
    FB_CHECKABORT(
        ::ctran::transport::impl::p2pTransportLockFlagIsSet(this),
        "CTRAN-IB: HostCbTransport::unlock() called without a matching "
        "lock() on this thread.");
    ::ctran::transport::impl::p2pTransportLockFlagSet(this, false);
    transportMutex_.unlock();
  }

  // === Test-only accessors ===
  char* sendStagingBase() const {
    return sendStaging_;
  }
  char* recvStagingBase() const {
    return recvStaging_;
  }
  char* sendStagingSlot(int slot) const {
    return sendStaging_ + static_cast<size_t>(slot) * chunkSize_;
  }
  char* recvStagingSlot(int slot) const {
    return recvStaging_ + static_cast<size_t>(slot) * chunkSize_;
  }
  void* sendStagingRegElem() const {
    return sendStagingRegElem_;
  }
  void* recvStagingRegElem() const {
    return recvStagingRegElem_;
  }
  void* peerStagingBuf() const {
    return peerStagingBuf_;
  }
  const CtranIbRemoteAccessKey& peerStagingKey() const {
    return peerStagingKey_;
  }

 private:
  struct ResourceExchangeMsg {
    ControlMsg staging;
    ControlMsg counter;
  };
  struct ResourceExchangeState {
    ResourceExchangeMsg outMsg{};
    ResourceExchangeMsg inMsg{};
    CtranIbRequest sendReq{};
    CtranIbRequest recvReq{};
  };

  // Aborts if stagingSlot is outside [0, pipelineDepth_). Used by
  // every API that takes a stagingSlot.
  inline void ensureValidStagingSlot(int stagingSlot);

  inline commResult_t postResourceExchange();
  inline bool testResourceExchange();
  inline void signalSlotReady(int physicalSlot);

  // Chunk pipeline helpers
  inline ChunkContext makeSendChunkContext(
      int physicalSlot,
      const SendChunkInfo& chunk) const;
  inline ChunkContext makeRecvChunkContext(
      int physicalSlot,
      const RecvChunkInfo& chunk) const;

  inline void progressSendPipeline();
  inline void progressRecvPipeline();
  inline void advanceSendStagingRecord(
      SendStagingRecord& slot,
      int physicalSlot);
  inline void advanceRecvStagingRecord(
      RecvStagingRecord& slot,
      int physicalSlot);
  inline void pollRecvNotifications();

  int peerRank_;
  int myRank_;
  int cudaDev_;
  CtranIb* ctranIb_;

  std::vector<std::shared_ptr<CtranIbVirtualConn>> vcs_;

  int pipelineDepth_;
  size_t chunkSize_;
  const CommLogData* logMetaData_{nullptr};
  GpeKernelSyncPool* gpeKernelSyncPool_{nullptr};

  HostTransportDev* devTransport_{nullptr};

  // Staging buffers (owned, GPU memory, IB-registered)
  char* sendStaging_{nullptr};
  char* recvStaging_{nullptr};
  void* sendStagingRegElem_{nullptr};
  void* recvStagingRegElem_{nullptr};

  // Per-slot GpeKernelSync (borrowed from gpeKernelSyncPool_;
  // returned via reset() in dtor)
  std::vector<GpeKernelSync*> sendSyncs_;
  std::vector<GpeKernelSync*> recvSyncs_;

  // Per-slot flow control counters (never reset, monotonically increasing).
  uint64_t* remoteReady_{nullptr};
  void* remoteReadyRegElem_{nullptr};
  uint64_t slotGeneration_[kMaxPipelineDepth]{};

  // Dummy buffer for ifetchAndAdd return value (receiver side, pinned + IB-reg)
  uint64_t* fetchAddDiscardBuf_{nullptr};
  void* fetchAddDiscardRegElem_{nullptr};
  CtranIbRequest atomicReqs_[kMaxPipelineDepth];

  // One-time resource exchange results (peer staging + peer remoteReady)
  void* peerStagingBuf_{nullptr};
  CtranIbRemoteAccessKey peerStagingKey_{};
  void* peerRemoteReadyAddr_{nullptr};
  CtranIbRemoteAccessKey peerRemoteReadyKey_{};
  bool resourcesExchanged_{false};
  std::optional<ResourceExchangeState> resExchangeState_;

  // CB staging-slot records.
  SendStagingRecord sendStagingSlots_[kMaxPipelineDepth];
  RecvStagingRecord recvStagingSlots_[kMaxPipelineDepth];

  // Number of staging-slot records currently not in IDLE. Maintained
  // by iSendChunk/iRecvChunk (++ on issue) and by
  // advanceSendStagingRecord/advanceRecvStagingRecord (-- on the final
  // transition back to IDLE). When zero, progressSendPipeline /
  // progressRecvPipeline skip the per-slot scan entirely.
  int cbSendActive_{0};
  int cbRecvActive_{0};

  // Per-transport caller-must-lock mutex. Algorithms must hold this
  // (typically via P2pTransportLockGuard) for the duration of every
  // hot-path call into the transport — see checkLocked() in
  // IP2pHostTransport.h. Read-only trivial accessors do NOT require
  // this lock.
  std::mutex transportMutex_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Inline definitions for the critical-path overrides and the helpers
// the progress()/iSendChunk/iRecvChunk hot paths reach into.
// ─────────────────────────────────────────────────────────────────────────────

inline void HostCbTransport::ensureValidStagingSlot(int stagingSlot) {
  FB_CHECKABORT(
      stagingSlot >= 0 && stagingSlot < pipelineDepth_,
      "CTRAN-IB: CB stagingSlot=" + std::to_string(stagingSlot) +
          " out of range [0, " + std::to_string(pipelineDepth_) + ")");
}

inline bool HostCbTransport::isReadyForSend(int vcIdx, int stagingSlot) {
  ::ctran::transport::checkLocked(this);
  impl::checkValidVc(vcs_, vcIdx);
  ensureValidStagingSlot(stagingSlot);
  return sendStagingSlots_[stagingSlot].status == SendChunkStatus::IDLE;
}

inline bool HostCbTransport::isReadyForRecv(int vcIdx, int stagingSlot) {
  ::ctran::transport::checkLocked(this);
  impl::checkValidVc(vcs_, vcIdx);
  ensureValidStagingSlot(stagingSlot);
  return recvStagingSlots_[stagingSlot].status == RecvChunkStatus::IDLE;
}

inline ChunkContext HostCbTransport::makeSendChunkContext(
    int /*physicalSlot*/,
    const SendChunkInfo& chunk) const {
  return ChunkContext{
      .slotIdx = chunk.stagingSlot,
      .round = static_cast<int>(chunk.round),
      .offset = chunk.offset,
      .len = chunk.len,
      .stagingSlot =
          sendStaging_ + static_cast<size_t>(chunk.stagingSlot) * chunkSize_,
      .userBuf = chunk.userBuf,
      .sync = sendSyncs_[chunk.stagingSlot],
      .remoteReady = &remoteReady_[chunk.stagingSlot],
      .slotGeneration =
          const_cast<uint64_t*>(&slotGeneration_[chunk.stagingSlot]),
  };
}

inline ChunkContext HostCbTransport::makeRecvChunkContext(
    int /*physicalSlot*/,
    const RecvChunkInfo& chunk) const {
  return ChunkContext{
      .slotIdx = chunk.stagingSlot,
      .round = static_cast<int>(chunk.round),
      .offset = chunk.offset,
      .len = chunk.len,
      .stagingSlot =
          recvStaging_ + static_cast<size_t>(chunk.stagingSlot) * chunkSize_,
      .userBuf = chunk.userBuf,
      .sync = recvSyncs_[chunk.stagingSlot],
      .signalSlotReady =
          +[](void* ctx, int s) {
            static_cast<HostCbTransport*>(ctx)->signalSlotReady(s);
          },
      .signalCtx = const_cast<HostCbTransport*>(this),
  };
}

inline commResult_t HostCbTransport::postResourceExchange() {
  if (resExchangeState_ || resourcesExchanged_) {
    return commSuccess;
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: HostCbTransport resource exchange with peer {} (myRank={})",
      peerRank_,
      myRank_);

  FB_COMMCHECK(checkEpochLock(ctranIb_));
  impl::checkValidVc(vcs_, impl::kCtrlMsgVc);

  resExchangeState_ = ResourceExchangeState{};
  auto& st = *resExchangeState_;

  FB_COMMCHECK(
      CtranIb::exportMem(recvStaging_, recvStagingRegElem_, st.outMsg.staging));
  FB_COMMCHECK(
      CtranIb::exportMem(remoteReady_, remoteReadyRegElem_, st.outMsg.counter));

  CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs_[impl::kCtrlMsgVc]->mutex, {
    FB_COMMCHECK(
        vcs_[impl::kCtrlMsgVc]->isendCtrlMsg(
            st.outMsg.staging.type, &st.outMsg, sizeof(st.outMsg), st.sendReq));
  });

  CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs_[impl::kCtrlMsgVc]->mutex, {
    FB_COMMCHECK(
        vcs_[impl::kCtrlMsgVc]->irecvCtrlMsg(
            &st.inMsg, sizeof(st.inMsg), st.recvReq));
  });

  return commSuccess;
}

inline bool HostCbTransport::testResourceExchange() {
  if (resourcesExchanged_) {
    return true;
  }
  if (!resExchangeState_) {
    return false;
  }
  auto& st = *resExchangeState_;
  if (!st.sendReq.isComplete() || !st.recvReq.isComplete()) {
    return false;
  }
  FB_COMMCHECK(
      CtranIb::importMem(&peerStagingBuf_, &peerStagingKey_, st.inMsg.staging));
  FB_COMMCHECK(
      CtranIb::importMem(
          &peerRemoteReadyAddr_, &peerRemoteReadyKey_, st.inMsg.counter));

  resExchangeState_.reset();
  resourcesExchanged_ = true;
  return true;
}

inline void HostCbTransport::signalSlotReady(int physicalSlot) {
  auto* slotAddr = static_cast<char*>(peerRemoteReadyAddr_) +
      static_cast<size_t>(physicalSlot) * sizeof(uint64_t);
  atomicReqs_[physicalSlot] = CtranIbRequest{};
  const auto& slot = recvStagingSlots_[physicalSlot];
  const int vcIdx = slot.chunk ? slot.chunk->vcIdx : 0;
  CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs_[vcIdx]->mutex, {
    (void)vcs_[vcIdx]->ifetchAndAdd(
        fetchAddDiscardBuf_,
        slotAddr,
        1,
        fetchAddDiscardRegElem_,
        peerRemoteReadyKey_,
        &atomicReqs_[physicalSlot]);
  });
}

inline commResult_t HostCbTransport::iSendCtrlMsg(
    ControlMsgType type,
    const void* payload,
    size_t len,
    CtrlRequest* out) {
  ::ctran::transport::checkLocked(this);
  FB_CHECKABORT(
      out != nullptr, "CTRAN-IB: iSendCtrlMsg requires non-null out pointer");
  FB_COMMCHECK(checkEpochLock(ctranIb_));
  impl::checkValidVc(vcs_, impl::kCtrlMsgVc);
  return impl::iSendCtrlMsgImpl(vcs_, type, payload, len, out->ctrlReq_);
}

inline commResult_t
HostCbTransport::iRecvCtrlMsg(void* payload, size_t len, CtrlRequest* out) {
  ::ctran::transport::checkLocked(this);
  FB_CHECKABORT(
      out != nullptr, "CTRAN-IB: iRecvCtrlMsg requires non-null out pointer");
  FB_COMMCHECK(checkEpochLock(ctranIb_));
  impl::checkValidVc(vcs_, impl::kCtrlMsgVc);
  return impl::iRecvCtrlMsgImpl(vcs_, payload, len, out->ctrlReq_);
}

inline commResult_t HostCbTransport::testCtrlMsgDone(
    CtrlRequest& req,
    bool* done) {
  ::ctran::transport::checkLocked(this);
  FB_CHECKABORT(
      done != nullptr, "CTRAN-IB: testCtrlMsgDone requires non-null done");
  return impl::testCtrlMsgDoneImpl(
      req.complete_, req.ctrlReq_, [this]() { return progress(); }, done);
}

inline commResult_t HostCbTransport::waitCtrlMsgDone(CtrlRequest& req) {
  ::ctran::transport::checkLocked(this);
  return impl::waitCtrlMsgDoneImpl(
      req.complete_, req.ctrlReq_, [this]() { return progress(); });
}

inline commResult_t HostCbTransport::iSendChunk(const SendChunkArgs& args) {
  ::ctran::transport::checkLocked(this);
  FB_COMMCHECK(checkEpochLock(ctranIb_));
  FB_CHECKABORT(
      args.len <= chunkSize_,
      "CTRAN-IB: CB iSendChunk len=" + std::to_string(args.len) +
          " exceeds chunkSize=" + std::to_string(chunkSize_));
  SendChunkHooks hooks = args.hooks;
  if (!hooks.prepareData) {
    hooks = makeCopyBasedSendHooks();
  }
  FB_COMMCHECK(postResourceExchange());

  impl::checkValidVc(vcs_, args.vcIdx);
  ensureValidStagingSlot(args.stagingSlot);
  auto& slot = sendStagingSlots_[args.stagingSlot];
  FB_CHECKABORT(
      slot.status == SendChunkStatus::IDLE && !slot.chunk,
      "CTRAN-IB: CB iSendChunk stagingSlot=" +
          std::to_string(args.stagingSlot) +
          " is not IDLE; caller must drain via testChunkDone before reissue");

  slot.chunk = SendChunkInfo{
      .userBuf = args.userBuf,
      .offset = args.offset,
      .len = args.len,
      .vcIdx = args.vcIdx,
      .stagingSlot = args.stagingSlot,
      .round = static_cast<uint64_t>(args.round),
      .hooks = std::move(hooks),
  };
  slot.status = SendChunkStatus::PREPARE_DATA;
  ++cbSendActive_;
  const uint64_t mySeq = ++slot.issued;
  if (args.req != nullptr) {
    *args.req = ChunkRequest{
        .kind = ChunkKind::kSend,
        .vcIdx = static_cast<int16_t>(args.vcIdx),
        .stagingSlot = static_cast<int16_t>(args.stagingSlot),
        .mySeq = mySeq,
    };
  }
  return commSuccess;
}

inline commResult_t HostCbTransport::iRecvChunk(const RecvChunkArgs& args) {
  ::ctran::transport::checkLocked(this);
  FB_COMMCHECK(checkEpochLock(ctranIb_));
  FB_CHECKABORT(
      args.len <= chunkSize_,
      "CTRAN-IB: CB iRecvChunk len=" + std::to_string(args.len) +
          " exceeds chunkSize=" + std::to_string(chunkSize_));
  RecvChunkHooks hooks = args.hooks;
  if (!hooks.processData) {
    hooks = makeCopyBasedRecvHooks();
  }
  FB_COMMCHECK(postResourceExchange());

  impl::checkValidVc(vcs_, args.vcIdx);
  ensureValidStagingSlot(args.stagingSlot);
  auto& slot = recvStagingSlots_[args.stagingSlot];
  FB_CHECKABORT(
      slot.status == RecvChunkStatus::IDLE && !slot.chunk,
      "CTRAN-IB: CB iRecvChunk stagingSlot=" +
          std::to_string(args.stagingSlot) +
          " is not IDLE; caller must drain via testChunkDone before reissue");

  slot.chunk = RecvChunkInfo{
      .userBuf = args.userBuf,
      .offset = args.offset,
      .len = args.len,
      .vcIdx = args.vcIdx,
      .stagingSlot = args.stagingSlot,
      .round = static_cast<uint64_t>(args.round),
      .hooks = std::move(hooks),
  };
  slot.status = RecvChunkStatus::WAIT_RECV;
  ++cbRecvActive_;
  const uint64_t mySeq = ++slot.issued;
  if (args.req != nullptr) {
    *args.req = ChunkRequest{
        .kind = ChunkKind::kRecv,
        .vcIdx = static_cast<int16_t>(args.vcIdx),
        .stagingSlot = static_cast<int16_t>(args.stagingSlot),
        .mySeq = mySeq,
    };
  }
  return commSuccess;
}

inline commResult_t HostCbTransport::progress() {
  ::ctran::transport::checkLocked(this);
  FB_COMMCHECK(impl::progressImpl(ctranIb_));
  if (!vcsReady()) {
    return commSuccess;
  }
  progressSendPipeline();
  progressRecvPipeline();
  return commSuccess;
}

inline void HostCbTransport::progressSendPipeline() {
  if (cbSendActive_ == 0) {
    return;
  }
  if (!testResourceExchange()) {
    return;
  }
  for (int s = 0; s < pipelineDepth_; ++s) {
    advanceSendStagingRecord(sendStagingSlots_[s], s);
  }
}

inline void HostCbTransport::advanceSendStagingRecord(
    SendStagingRecord& slot,
    int physicalSlot) {
  if (!slot.chunk) {
    return;
  }
  auto& chunk = *slot.chunk;
  auto ctx = makeSendChunkContext(physicalSlot, chunk);

  switch (slot.status) {
    case SendChunkStatus::PREPARE_DATA:
      chunk.hooks.prepareData(ctx);
      slot.status = SendChunkStatus::WAIT_PREPARE;
      break;

    case SendChunkStatus::WAIT_PREPARE:
      if (chunk.hooks.isDataReady(ctx)) {
        slot.status = SendChunkStatus::WAIT_REMOTE_READY;
      }
      break;

    case SendChunkStatus::WAIT_REMOTE_READY:
      if (chunk.hooks.isRemoteReady(ctx)) {
        const void* src = chunk.hooks.getLocalSrc(ctx);
        // CB destination is always the peer's staging slot — no
        // alternate path now that mixed-mode is dropped.
        void* dst = static_cast<char*>(peerStagingBuf_) +
            static_cast<size_t>(chunk.stagingSlot) * chunkSize_;
        void* regElem = sendStagingRegElem_;

        slot.iputReq.repost(1);
        CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs_[chunk.vcIdx]->mutex, {
          (void)vcs_[chunk.vcIdx]->iput(
              src,
              dst,
              chunk.len,
              regElem,
              peerStagingKey_,
              /*notify=*/true,
              /*config=*/nullptr,
              &slot.iputReq,
              /*fast=*/false);
        });
        slot.status = SendChunkStatus::WAIT_IPUT;
      }
      break;

    case SendChunkStatus::WAIT_IPUT:
      if (slot.iputReq.isComplete()) {
        chunk.hooks.onSendDone(ctx);
        ++slot.completed;
        slot.chunk.reset();
        slot.status = SendChunkStatus::IDLE;
        --cbSendActive_;
      }
      break;

    case SendChunkStatus::IDLE:
      break;
  }
}

inline void HostCbTransport::progressRecvPipeline() {
  pollRecvNotifications();
  if (cbRecvActive_ == 0) {
    return;
  }
  if (!testResourceExchange()) {
    return;
  }
  for (int s = 0; s < pipelineDepth_; ++s) {
    advanceRecvStagingRecord(recvStagingSlots_[s], s);
  }
}

inline void HostCbTransport::pollRecvNotifications() {
  // impl::progressImpl(ctranIb_) (called from progress()) already
  // drained notify CQEs into per-VC notifyCount_. Walk each VC and try
  // to attribute each notify to a CB recv slot in WAIT_RECV. Unmatched
  // notifies stay accumulated for a future progress() pass — ZC recvs
  // are handled by HostZcTransport, not here.
  if (cbRecvActive_ == 0) {
    return;
  }
  for (int i = 0; i < numVcs(); ++i) {
    while (true) {
      RecvStagingRecord* match = nullptr;
      for (int s = 0; s < pipelineDepth_; ++s) {
        auto& slot = recvStagingSlots_[s];
        if (slot.status != RecvChunkStatus::WAIT_RECV || !slot.chunk ||
            slot.chunk->vcIdx != i) {
          continue;
        }
        if (match == nullptr || slot.chunk->round < match->chunk->round) {
          match = &slot;
        }
      }
      if (!match) {
        break;
      }
      bool notified = false;
      CTRAN_IB_PER_OBJ_LOCK_GUARD(
          vcs_[i]->mutex, { (void)vcs_[i]->checkNotify(&notified); });
      if (!notified) {
        break;
      }
      match->status = RecvChunkStatus::PROCESS_DATA;
    }
  }
}

inline void HostCbTransport::advanceRecvStagingRecord(
    RecvStagingRecord& slot,
    int physicalSlot) {
  if (!slot.chunk) {
    return;
  }
  auto& chunk = *slot.chunk;
  auto ctx = makeRecvChunkContext(physicalSlot, chunk);

  switch (slot.status) {
    case RecvChunkStatus::PROCESS_DATA: {
      chunk.hooks.processData(ctx);
      slot.status = RecvChunkStatus::WAIT_PROCESS;
      break;
    }

    case RecvChunkStatus::WAIT_PROCESS:
      if (chunk.hooks.isProcessDone(ctx)) {
        slot.status = RecvChunkStatus::SIGNAL_READY;
      }
      break;

    case RecvChunkStatus::SIGNAL_READY: {
      chunk.hooks.signalReady(ctx);
      chunk.hooks.onRecvDone(ctx);
      ++slot.completed;
      slot.chunk.reset();
      slot.status = RecvChunkStatus::IDLE;
      --cbRecvActive_;
      break;
    }

    case RecvChunkStatus::IDLE:
    case RecvChunkStatus::WAIT_RECV:
      break;
  }
}

inline commResult_t HostCbTransport::testChunkDone(
    const ChunkRequest& req,
    bool* done) {
  ::ctran::transport::checkLocked(this);
  FB_CHECKABORT(
      done != nullptr, "CTRAN-IB: testChunkDone requires non-null done");
  FB_COMMCHECK(progress());
  if (req.stagingSlot < 0 || req.stagingSlot >= pipelineDepth_) {
    *done = false;
    return commSuccess;
  }
  if (req.kind == ChunkKind::kSend) {
    *done = sendStagingSlots_[req.stagingSlot].completed >= req.mySeq;
  } else {
    *done = recvStagingSlots_[req.stagingSlot].completed >= req.mySeq;
  }
  return commSuccess;
}

} // namespace ctran::transport::ib
