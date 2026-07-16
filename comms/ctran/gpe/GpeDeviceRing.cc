// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/gpe/GpeDeviceRing.h"

#include <chrono>
#include <cstdint>
#include <memory>

#include "comms/ctran/gpe/CtranGpeImpl.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

ctran::gpe::GpeCmdId GpeDeviceRingCmdRegistry::registerCmd(CtranGpeCmd* cmd) {
  ctran::gpe::GpeCmdId id = nextId_.fetch_add(1, std::memory_order_relaxed);
  cmd->cmdId = id;
  cmd->inDeviceRingRegistry = true;
  map_.wlock()->insert_or_assign(id, cmd);
  return id;
}

CtranGpeCmd* GpeDeviceRingCmdRegistry::lookup(ctran::gpe::GpeCmdId id) const {
  auto locked = map_.rlock();
  auto it = locked->find(id);
  return it == locked->end() ? nullptr : it->second;
}

CtranGpeCmd* GpeDeviceRingCmdRegistry::lookupAndFire(ctran::gpe::GpeCmdId id) {
  // Hold the map lock across BOTH the lookup and recordRingReplay so the pair
  // is atomic with erase(): once erase() removes the entry, no further fire can
  // be applied, so the cmd's inFlight can only fall to 0 and stay there. This
  // makes the erase-then-wait teardown in cmdDestroy() safe — otherwise a fire
  // looked up but not yet counted could bump inFlight after cmdDestroy's wait
  // passed, and the worker would touch freed memory.
  auto locked = map_.rlock();
  auto it = locked->find(id);
  if (it == locked->end()) {
    return nullptr;
  }
  recordRingReplay(it->second);
  return it->second;
}

void GpeDeviceRingCmdRegistry::erase(ctran::gpe::GpeCmdId id) {
  map_.wlock()->erase(id);
}

size_t GpeDeviceRingCmdRegistry::size() const {
  return map_.rlock()->size();
}

void GpeDeviceRingCmdRegistry::recordRingReplay(CtranGpeCmd* cmd) {
  // Mirror cmdCb's per-replay bookkeeping: each ring entry is one fire (one
  // graph replay), so bump inFlight (paired with the worker's fetch_sub and
  // awaited by cmdDestroy) and advance each op's opCount.
  if (!cmd->persistent) {
    return;
  }
  cmd->inFlight.fetch_add(1, std::memory_order_release);
  for (auto& op : cmd->coll.opGroup) {
    ++op->opCount;
  }
}

void CtranGpe::Impl::initDeviceRing() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  // System-scope 128b atomic exchange (the ring's device write) requires sm_90+
  // NVIDIA hardware; on HIP the write traps. Keep the ring disabled so the
  // host-node path is used, matching NVIDIA pre-Hopper behavior.
  return;
#else
  if (!NCCL_CTRAN_GPE_DEVICE_RING) {
    return;
  }
  // The ring's device write is a System-scope 128b atomic exchange, which
  // requires sm_90+ (Hopper). On older NVIDIA GPUs that instruction traps, so
  // gate on the actual compute capability and fall back to the host-node path.
  // Queried once per process (-1 = query failed).
  static const int ccMajor = [] {
    int dev = 0;
    int major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
      return -1;
    }
    return major;
  }();
  if (ccMajor < 0) {
    CLOGF(
        WARN,
        "CTRAN-GPE: could not query compute capability; leaving device ring disabled");
    return;
  }
  if (ccMajor < 9) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-GPE: device ring requires sm_90+ (found compute capability {}.x); using host-node dispatch",
        ccMajor);
    return;
  }
  // Validate the (signed) cvar before casting to uint32_t: a non-positive
  // value would otherwise wrap to a huge size and attempt an absurd pinned
  // allocation. Reject it with a clear diagnostic and fall back to host nodes.
  const int configuredRingSize = NCCL_CTRAN_GPE_DEVICE_RING_SIZE;
  if (configuredRingSize <= 0) {
    CLOGF(
        WARN,
        "CTRAN-GPE: NCCL_CTRAN_GPE_DEVICE_RING_SIZE={} must be > 0; using host-node dispatch",
        configuredRingSize);
    return;
  }
  auto ring = std::make_unique<ctran::gpe::GpeRing>(
      static_cast<uint32_t>(configuredRingSize));
  if (!ring->valid()) {
    CLOGF(
        WARN,
        "CTRAN-GPE: device ring allocation failed; falling back to host-node dispatch");
    return;
  }
  deviceRingReader_ = std::make_unique<ctran::gpe::GpeRingReader>(*ring);
  deviceRing_ = std::move(ring);
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-GPE: device-ring dispatch enabled (ring size {})",
      deviceRing_->size());
#endif
}

CtranGpeCmd* CtranGpe::Impl::acquireNextCmd() {
  if (!deviceRingEnabled()) {
    // Unchanged legacy path: block on the CPU FIFO.
    return cmdDequeue();
  }

  // Ring path: consume from two sources without blocking on either. The CPU
  // FIFO carries TERMINATE, eager submits, and captured cmds on the host-node
  // path; the device ring carries captured ring cmds in GPU execution order.
  //
  // Block on the FIFO condvar so a control/eager submit (or TERMINATE/abort)
  // wakes us immediately, but time out every poll interval to drain the ring,
  // which cannot signal the condvar. This keeps an idle worker off-core (there
  // is one GPE thread per comm) while bounding ring dispatch latency.
  const auto pollInterval = std::chrono::microseconds(
      NCCL_CTRAN_GPE_DEVICE_RING_POLL_US > 0
          ? NCCL_CTRAN_GPE_DEVICE_RING_POLL_US
          : 50);
  for (;;) {
    // Priority 1: control/eager cmds on the FIFO. Only wait when nothing is
    // already pending from a prior ring poll. ringPending_ is owned solely by
    // this worker thread, so reading it under the FIFO lock is race-free.
    {
      auto locked = cmdQueue_.lock();
      if (locked->queue.empty() && ringPending_.empty()) {
        cmdQueueCv_.wait_for(locked.as_lock(), pollInterval);
      }
      if (!locked->queue.empty()) {
        auto* cmd = locked->queue.front();
        locked->queue.pop();
        return cmd;
      }
    }

    // Priority 2: cmds drained from the ring on a prior poll, in exec order.
    if (!ringPending_.empty()) {
      auto* cmd = ringPending_.front();
      ringPending_.pop();
      return cmd;
    }

    // Poll the ring (FIFO lock released) for newly started kernels. poll()
    // delivers every entry published since the last poll, in slot order.
    auto pollResult = deviceRingReader_->poll([this](
                                                  const auto& entry,
                                                  uint64_t /*slot*/) {
      // Look up and fire under the registry lock so the pair is atomic with
      // cmdDestroy()'s erase (see GpeDeviceRingCmdRegistry::lookupAndFire).
      CtranGpeCmd* cmd = deviceRingCmdRegistry_.lookupAndFire(entry.data);
      if (cmd != nullptr) {
        ringPending_.push(cmd);
      } else {
        // The cmd was destroyed (graph teardown) after publishing, or an id
        // was never registered. Drop it — safe because a destroyed cmd's
        // registry entry is erased before free.
        CLOGF(
            WARN,
            "CTRAN-GPE: device ring entry cmd_id {} has no live registry entry; skipping",
            entry.data);
      }
    });

    // The ring was full when we polled: a kernel is stalled in write_blocking()
    // waiting for us to drain, so device work is being throttled by the GPE
    // consumer. Rate-limited so a sustained stall logs periodically.
    if (pollResult.writerThrottled) {
      CLOGF_EVERY_MS(
          WARN,
          1000,
          "CTRAN-GPE: device ring full (size {}); kernel writes throttled — GPE consumer not draining fast enough",
          deviceRing_->size());
    }
  }
}
