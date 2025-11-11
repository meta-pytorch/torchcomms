// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/mapper/CtranMapper.h"
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/tracing/MapperTrace.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/alloc.h"

#ifdef ENABLE_META_COMPRESSION
#include <comms/ctran/mapper/fb/IcompressionManager.h>
#endif

using namespace ncclx;
namespace {
std::vector<CtranMapperBackend> getToEnableBackends(
    const std::vector<CommBackend>& overrideBackend) {
  const std::unordered_map<enum CommBackend, CtranMapperBackend>
      CommBackendMap = {
          {CommBackend::UNSET, CtranMapperBackend::UNSET},
          {CommBackend::IB, CtranMapperBackend::IB},
          {CommBackend::NVL, CtranMapperBackend::NVL},
          {CommBackend::SOCKET, CtranMapperBackend::SOCKET},
          {CommBackend::TCPDM, CtranMapperBackend::TCPDM}};

  const std::unordered_map<enum NCCL_CTRAN_BACKENDS, CtranMapperBackend>
      NCCLCtranBackendMap = {
          {NCCL_CTRAN_BACKENDS::ib, CtranMapperBackend::IB},
          {NCCL_CTRAN_BACKENDS::nvl, CtranMapperBackend::NVL},
          {NCCL_CTRAN_BACKENDS::socket, CtranMapperBackend::SOCKET},
          {NCCL_CTRAN_BACKENDS::tcpdm, CtranMapperBackend::TCPDM}};

  std::vector<CtranMapperBackend> enableBackends;

  if (overrideBackend.size() == 0 ||
      (overrideBackend.size() == 1 &&
       overrideBackend[0] == CommBackend::UNSET)) {
    for (auto& b : NCCL_CTRAN_BACKENDS) {
      enableBackends.emplace_back(NCCLCtranBackendMap.at(b));
    }
  } else {
    CLOGF(
        WARN,
        "CTRAN-MAPPER: Try to override backends through Ctran Config. Currently it is specific config for MCCL. If you are using NCCL with NCCL_CTRAN_BACKENDS, please report this to MCCL team");
    for (auto& b : overrideBackend) {
      if (b == CommBackend::UNSET) {
        FB_ERRORTHROW(
            commInvalidUsage, "CTRAN-MAPPER: Invalid override backend UNSET");
      }
      enableBackends.emplace_back(CommBackendMap.at(b));
    }
  }

  return enableBackends;
}
} // namespace

CtranMapper::CtranMapper(CtranComm* comm) {
  const auto statex = comm->statex_.get();
  if (NCCL_MAPPERTRACE_ENABLE) {
    this->mapperTrace = std::make_unique<ncclx::colltrace::MapperTrace>();
  }
  this->logMetaData_ = CommLogData{
      0,
      statex->commHash(),
      statex->commDesc(),
      statex->rank(),
      statex->nRanks()};

  this->comm = comm;
  auto backendsToEnable = getToEnableBackends(comm->config_.backends);
  /* check user preference for backends */
  std::vector<bool> enableBackends(CtranMapperBackend::NUM_BACKENDS, false);
  iPutCount = std::vector<int>(CtranMapperBackend::NUM_BACKENDS, 0);
  iGetCount = std::vector<int>(CtranMapperBackend::NUM_BACKENDS, 0);
  std::vector<std::string> enableBackendsStrs;
  for (auto b : backendsToEnable) {
    enableBackends.at(b) = true;
    enableBackendsStrs.push_back(backendToStr(b));
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-MAPPER: configure NCCL_CTRAN_BACKENDS [{}]",
      vecToStr(enableBackendsStrs));

  if ((enableBackends[CtranMapperBackend::IB] ||
       enableBackends[CtranMapperBackend::NVL] ||
       enableBackends[CtranMapperBackend::SOCKET]) &&
      enableBackends[CtranMapperBackend::TCPDM]) {
    FB_ERRORTHROW(
        commInvalidArgument,
        "CTRAN-MAPPER: TCPDM can not be enabled with IB, NVL or Socket backends");
  }

  this->ctrlMgr = std::make_unique<CtranCtrlManager>();

  /* enable available backends */
  if (enableBackends[CtranMapperBackend::IB]) {
    try {
      this->ctranIb =
          std::make_unique<class CtranIb>(comm, this->ctrlMgr.get());
      this->ctranIb->regCtrlCb(this->ctrlMgr);
    } catch (const std::bad_alloc& e) {
      ctranIb = nullptr;
      CLOGF(WARN, "CTRAN-MAPPER: IB backend not enabled");
    }
  }
  if (enableBackends[CtranMapperBackend::SOCKET]) {
    if (!this->ctranIb) {
      this->ctranSock =
          std::make_unique<class CtranSocket>(comm, this->ctrlMgr.get());
    } else {
      CLOGF_SUBSYS(
          INFO,
          INIT,
          "CTRAN-MAPPER: SOCKET backend not enabled, since IB backend is enabled");
    }
  }
  if (enableBackends[CtranMapperBackend::TCPDM]) {
    this->ctranTcpDm =
        std::make_unique<class ctran::CtranTcpDm>(comm, this->ctrlMgr.get());
    CLOGF(WARN, "CTRAN-MAPPER: TCPDM backend is enabled");
  }

  if (enableBackends[CtranMapperBackend::NVL]) {
    // NVL backend depends on IB backend for control msg exchange
    if (this->ctranIb || this->ctranSock || this->ctranTcpDm) {
      try {
        this->ctranNvl = std::make_unique<class CtranNvl>(comm);
        this->ctranNvl->regCtrlCb(this->ctrlMgr);
      } catch (const std::bad_alloc& e) {
        // FIXME: give more specific exception + error message
        CLOGF(
            WARN, "CTRAN-MAPPER: NVL backend not enabled. Error {}", e.what());
      }
    } else {
      CLOGF(
          WARN,
          "CTRAN-MAPPER: NVL backend not enabled. Require valid IB, TCPDM or Socket backend");
    }
  }

  std::vector<int> nvlRanks{};
  std::vector<int> ibRanks{};
  std::vector<int> sockRanks{};
  std::vector<int> tcpRanks{};
  for (int i = 0; i < statex->nRanks(); i++) {
    // Each rank may be connected via multiple backends. Decide which backend to
    // use based on buffer registration at collective time
    if (this->ctranNvl && this->ctranNvl->isSupported(i)) {
      nvlRanks.push_back(i);
    }
    if (this->ctranIb) {
      ibRanks.push_back(i);
    }
    if (this->ctranSock) {
      sockRanks.push_back(i);
    }
    if (this->ctranTcpDm) {
      tcpRanks.push_back(i);
    }
  }
  const auto nvlRankRangesStr =
      ::ctran::utils::rangesToStr(::ctran::utils::getRanges(nvlRanks));
  const auto ibRanksRangesStr =
      ::ctran::utils::rangesToStr(::ctran::utils::getRanges(ibRanks));
  const auto socketRanksRangesStr =
      ::ctran::utils::rangesToStr(::ctran::utils::getRanges(sockRanks));
  const auto tcpRanksRangesStr =
      ::ctran::utils::rangesToStr(::ctran::utils::getRanges(tcpRanks));
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-MAPPER: NVL ranks: {}, IB ranks: {}, SOCKET ranks: {}, TCPDM ranks: {}",
      nvlRankRangesStr.c_str(),
      ibRanksRangesStr.c_str(),
      socketRanksRangesStr.c_str(),
      tcpRanksRangesStr.c_str());

  this->rank = statex->rank();

  return;
}

void CtranMapper::reportProfiling(bool flush) {
  /* flush timestamps */
  if (!this->timestamps.empty() &&
      ((this->timestamps.size() > NCCL_CTRAN_PROFILING_REPORT_COUNT ||
        flush))) {
    if (NCCL_CTRAN_PROFILING == NCCL_CTRAN_PROFILING::stdout ||
        NCCL_CTRAN_PROFILING == NCCL_CTRAN_PROFILING::info) {
      std::stringstream ss;
      ss << "[CTRAN-MAPPER] Communication Profiling:" << std::endl;
      for (auto& ts : this->timestamps) {
        ss << "    collective=" << ts->algo << std::endl;
        ss << "    startTime="
           << std::chrono::duration_cast<std::chrono::nanoseconds>(
                  ts->start.time_since_epoch())
                  .count()
           << std::endl;
        for (auto& tsp : ts->recvCtrl) {
          ss << "        recvCtrl[" << tsp.peer << "]="
             << std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tsp.now.time_since_epoch())
                    .count()
             << std::endl;
        }
        for (auto& tsp : ts->putIssued) {
          ss << "        putIssued[" << tsp.peer << "]="
             << std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tsp.now.time_since_epoch())
                    .count()
             << std::endl;
        }
        for (auto& tsp : ts->putComplete) {
          ss << "        putComplete[" << tsp.peer << "]="
             << std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tsp.now.time_since_epoch())
                    .count()
             << std::endl;
        }
        for (auto& tsp : ts->kernelPost) {
          ss << "        kernelPost[" << tsp.peer << "]="
             << std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tsp.now.time_since_epoch())
                    .count()
             << std::endl;
        }
        for (auto& tsp : ts->kernelWait) {
          ss << "        kernelWait[" << tsp.peer << "]="
             << std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tsp.now.time_since_epoch())
                    .count()
             << std::endl;
        }
        for (auto& tsp : ts->kernelWaitComplete) {
          ss << "        kernelWaitComplete[" << tsp.peer << "]="
             << std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tsp.now.time_since_epoch())
                    .count()
             << std::endl;
        }
        if (NCCL_CTRAN_PROFILING == NCCL_CTRAN_PROFILING::info) {
          CLOGF(INFO, "{}", ss.str());
          ss.str("");
          ss.clear();
        }
      }
      if (NCCL_CTRAN_PROFILING == NCCL_CTRAN_PROFILING::stdout) {
        std::cout << ss.str() << std::flush;
      }
    } else if (NCCL_CTRAN_PROFILING == NCCL_CTRAN_PROFILING::kineto) {
      auto pid = getpid();
      static uint64_t reportCnt = 0;
      std::stringstream stream;
      std::string hostname = ctran::utils::getFullHostname();
      std::string filename(
          NCCL_CTRAN_KINETO_PROFILE_DIR + std::string("/nccl_ctran_log.") +
          std::to_string(pid) + std::string(".rank") +
          std::to_string(this->rank) + "." + hostname + std::string(".comm") +
          hashToHexStr(this->logMetaData_.commHash) + std::string(".") +
          std::to_string(reportCnt++) + std::string(".json"));
      CLOGF(INFO, "Dumping ctran profile to {}", filename);

      int id = 0;
      stream << "[" << std::endl;
      for (auto& ts : this->timestamps) {
        int collId = id;
        stream << "{\"name\": \"" << ts->algo << "\", " << "\"cat\": \"COL\", "
               << "\"id\": \"" << id++ << "\", " << "\"ph\": \"b\", "
               << "\"pid\": \"" << this->rank << "\", " << "\"tid\": \"-1\", "
               << "\"ts\": \""
               << std::chrono::duration_cast<std::chrono::microseconds>(
                      ts->start.time_since_epoch())
                      .count()
               << "\"}," << std::endl;
        CtranMapperTimestampPoint last(0);
        for (auto& tsp : ts->recvCtrl) {
          stream << "{\"name\": \"recvCtrl\", " << "\"cat\": \"NET\", "
                 << "\"id\": \"" << id++ << "\", " << "\"ph\": \"X\", "
                 << "\"pid\": \"" << this->rank << "\", " << "\"tid\": \""
                 << tsp.peer << "\", " << "\"ts\": \""
                 << std::chrono::duration_cast<std::chrono::microseconds>(
                        tsp.now.time_since_epoch())
                        .count()
                 << "\", \"dur\": \"0\"" << "}," << std::endl;
        }
        for (auto& tsp : ts->putIssued) {
          stream << "{\"name\": \"put\", " << "\"cat\": \"NET\", "
                 << "\"id\": \"" << id++ << "\", " << "\"ph\": \"b\", "
                 << "\"pid\": \"" << this->rank << "\", " << "\"tid\": \""
                 << tsp.peer << "\", " << "\"ts\": \""
                 << std::chrono::duration_cast<std::chrono::microseconds>(
                        tsp.now.time_since_epoch())
                        .count()
                 << "\"}," << std::endl;
        }
        id -= ts->putIssued.size();
        for (auto& tsp : ts->putComplete) {
          stream << "{\"name\": \"put\", " << "\"cat\": \"NET\", "
                 << "\"id\": \"" << id++ << "\", " << "\"ph\": \"e\", "
                 << "\"pid\": \"" << this->rank << "\", " << "\"tid\": \""
                 << tsp.peer << "\", " << "\"ts\": \""
                 << std::chrono::duration_cast<std::chrono::microseconds>(
                        tsp.now.time_since_epoch())
                        .count()
                 << "\"}," << std::endl;
          last = tsp;
        }
        for (auto& tsp : ts->kernelPost) {
          stream << "{\"name\": \"kernelPost\", " << "\"cat\": \"NET\", "
                 << "\"id\": \"" << id++ << "\", " << "\"ph\": \"X\", "
                 << "\"pid\": \"" << this->rank << "\", " << "\"tid\": \""
                 << tsp.peer << "\", " << "\"ts\": \""
                 << std::chrono::duration_cast<std::chrono::microseconds>(
                        tsp.now.time_since_epoch())
                        .count()
                 << "\", \"dur\": \"0\"" << "}," << std::endl;
        }
        for (auto& tsp : ts->kernelWait) {
          stream << "{\"name\": \"kernelWait\", " << "\"cat\": \"NET\", "
                 << "\"id\": \"" << id++ << "\", " << "\"ph\": \"b\", "
                 << "\"pid\": \"" << this->rank << "\", " << "\"tid\": \""
                 << tsp.peer << "\", " << "\"ts\": \""
                 << std::chrono::duration_cast<std::chrono::microseconds>(
                        tsp.now.time_since_epoch())
                        .count()
                 << "\"}," << std::endl;
        }
        id -= ts->kernelWait.size();
        for (auto& tsp : ts->kernelWaitComplete) {
          stream << "{\"name\": \"kernelWait\", " << "\"cat\": \"NET\", "
                 << "\"id\": \"" << id++ << "\", " << "\"ph\": \"e\", "
                 << "\"pid\": \"" << this->rank << "\", " << "\"tid\": \""
                 << tsp.peer << "\", " << "\"ts\": \""
                 << std::chrono::duration_cast<std::chrono::microseconds>(
                        tsp.now.time_since_epoch())
                        .count()
                 << "\"}," << std::endl;
          last = tsp;
        }
        stream << "{\"name\": \"" << ts->algo << "\", " << "\"cat\": \"COL\", "
               << "\"id\": \"" << collId << "\", " << "\"ph\": \"e\", "
               << "\"pid\": \"" << this->rank << "\", " << "\"tid\": \"-1\", "
               << "\"ts\": \""
               << std::chrono::duration_cast<std::chrono::microseconds>(
                      last.now.time_since_epoch())
                      .count()
               << "\"}," << std::endl;
      }
      // cut off trailing comma if any event is generated
      if (this->timestamps.size()) {
        stream.seekp(stream.str().length() - 2);
      }
      stream << "]" << std::endl;
      std::ofstream f(filename);
      f << stream.str();
      f.close();
    }
    this->timestamps.clear();
  }
}

void CtranMapper::setAtDestruction() {
  this->atDestruction = true;
}

CtranMapper::~CtranMapper() {
#ifdef ENABLE_META_COMPRESSION
  compressionManagerDestroy(this);
#endif

  // Should already be set in ~Ctran()
  // Mark again for sure
  setAtDestruction();

  this->reportProfiling(true);

  // Release any pending CB_CTRL requests;
  // intentionally avoid progress polling in destructor to ensure it is never
  // blocked.
  this->postedCbCtrlReqs_.clear();

  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

commResult_t CtranMapper::epochLock() {
  if (ctranIb) {
    return ctranIb->epochLock();
  }
  return commSuccess;
}

commResult_t CtranMapper::epochUnlock() {
  if (ctranIb) {
    return ctranIb->epochUnlock();
  }
  return commSuccess;
}

commResult_t CtranMapper::remReleaseMem(CtranMapperRegElem* regElem) {
  if (!this->atDestruction) {
    // Notify remote rank to release previous imported memory via NVL backend.
    // Skip if deregMem is called at destruction, since remote rank will release
    // any remaining imported memory at destruction.
    auto exportedNvlRanks = exportRegCache_.wlock()->remove(regElem);
    for (auto peerRank : exportedNvlRanks) {
      // We ensure the remote rank always release before next import because
      // all control messages to the given peer are transferred in order via a
      // single vc's control channel. This guarantee can prevent the remote rank
      // misuse a previously imported and cached segment if the same vaddr of
      // the segment is reused in a future importing segment but mapped to a
      // different range.

      auto backend =
          ctranIb ? CtranMapperBackend::IB : CtranMapperBackend::SOCKET;
      std::unique_ptr<CbCtrlRequest> req =
          std::make_unique<CbCtrlRequest>(peerRank, backend);

      FB_COMMCHECK(this->ctranNvl->remReleaseMem(
          regElem->nvlRegElem, peerRank, req->msg));
      if (this->ctranIb) {
        FB_COMMCHECK(this->ctranIb->isendCtrlMsg(
            req->msg.type,
            &req->msg,
            sizeof(ControlMsg),
            peerRank,
            req->ibReq));
      } else if (this->ctranSock) {
        FB_COMMCHECK(
            this->ctranSock->isendCtrlMsg(req->msg, peerRank, req->sockReq));
      }
      // TCPDM does not share local memory registration with the remote
      // and does not need to release it.

      CLOGF_TRACE(
          COLL,
          "CTRAN-MAPPER: Posted CB ctrlmsg to rank {}: {}",
          peerRank,
          req->msg.toString());

      // cbCtrl requests will be checked in progress and erase & free at
      // completion. mapper needs to free up all cbCtrl requests at destruction.
      this->postedCbCtrlReqs_.push_back(std::move(req));
    }
  }

  return commSuccess;
}

commResult_t CtranMapper::regMem(
    const void* buf,
    std::size_t len,
    void** segHdl,
    bool forceRegist,
    bool ncclManaged,
    void** regHdl) {
  auto regCache = CtranMapperRegCache::getInstance();
  CHECK_VALID_REGCACHE(regCache);

  const int cudaDev = comm->statex_->cudaDev();
  // Cache segment.
  // regCache either returns an already cached handle or create a
  // new entry if the segment is not yet cached.
  CtranMapperSegment* segment = nullptr;
  void* segHdl_ = nullptr;

  FB_COMMCHECK(regCache->cacheSegment(
      buf,
      len,
      cudaDev,
      ncclManaged,
      logMetaData_.commHash,
      &segment,
      &segHdl_));

  // Register the segment only if in Eager mode or forced by caller
  CtranMapperRegElem* regHdl_ = nullptr;
  if (NCCL_CTRAN_REGISTER == NCCL_CTRAN_REGISTER::eager || forceRegist) {
    bool didRegister = false;
    FB_COMMCHECK(regCache->regRange(
        segment->range.buf,
        segment->range.len,
        cudaDev,
        "eagerRegMem",
        logMetaData_,
        didRegister,
        &regHdl_));
  }

  *segHdl = segHdl_;
  if (regHdl) {
    *regHdl = regHdl_;
  }
  return commSuccess;
}

DevMemType CtranMapper::segmentType(void* segHdl) {
  auto regCache = CtranMapperRegCache::getInstance();
  CHECK_VALID_REGCACHE(regCache);

  CtranMapperSegment* segment = regCache->getSegment(segHdl);
  return segment->getType();
}

commResult_t CtranMapper::deregMem(void* segHdl, const bool skipRemRelease) {
  auto regCache = CtranMapperRegCache::getInstance();
  CHECK_VALID_REGCACHE(regCache);

  // Fast return for nullptr handle
  if (segHdl == nullptr) {
    return commSuccess;
  }

  auto timerBegin = std::chrono::steady_clock::now();
  auto regElems = regCache->getRegElems(segHdl);
  if (!skipRemRelease) {
    // Release remote registration associated with each regElem within this
    // communicator.

    // Acquire epoch lock for thread-safe ctrl msg exchange via backend.
    // Require the caller not call it within an existing epoch lock.
    // NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK is enabled in UT to check misuse
    // within Ctran.
    epochLock();
    for (auto& regElem : regElems) {
      FB_COMMCHECK(remReleaseMem(regElem));
    }
    epochUnlock();
  } else {
    // Skip remote release, just remove the regElems from local exportRegCache_.
    // The caller is responsible to release all remote registration (e.g., in
    // winFree)
    for (auto& regElem : regElems) {
      exportRegCache_.wlock()->remove(regElem);
    }
  }

  // Remove segment from cache
  // - No-op if the segment is still in-use.
  // - Only the last communicator holding the segment returns freed = true. It
  //   also deregisters all associated registrations.
  // - Ownership of registrations are transferred to the last communicator to
  //   release remote registration.
  // - If the segment no longer exists in cache, likely double dereg.
  //   ncclInvaidUsage is returned.
  bool freed = false, ncclManaged = false;
  std::vector<std::unique_ptr<CtranMapperRegElem>> regElemsFreed;
  FB_COMMCHECK(
      regCache->freeSegment(segHdl, freed, ncclManaged, regElemsFreed));

  // freed is true means all segments are deregistered
  if (freed) {
    for (auto& regElem : regElemsFreed) {
      logMemoryEvent(
          logMetaData_,
          "",
          "deregMem",
          reinterpret_cast<uintptr_t>(regElem->buf),
          regElem->len,
          regElem->numSegments(),
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - timerBegin)
              .count(),
          true /* isRegMemEvent */);
    }
  }
  return commSuccess;
}

commResult_t CtranMapper::deregDynamic(void* regHdl) {
  auto regCache = CtranMapperRegCache::getInstance();
  CHECK_VALID_REGCACHE(regCache);

  auto* regElem = reinterpret_cast<CtranMapperRegElem*>(regHdl);

  // Remote Release
  //
  // Epoch lock acquired by caller for thread-safe ctrl msg exchange via
  // backend. Requires the caller to call it within an existing epoch lock.
  FB_COMMCHECK(this->remReleaseMem(regElem));

  // Deregister dynamic registration
  FB_COMMCHECK(regCache->deregDynamic(regElem));

  return commSuccess;
}

commResult_t CtranMapper::deregRemReg(struct CtranMapperRemoteAccessKey* rkey) {
  switch (rkey->backend) {
    case CtranMapperBackend::NVL:
      FB_CHECKABORT(
          ctranNvl != nullptr,
          "Unexpected rkey with NVL backend but ctranNvl is not initialized");
      FB_COMMCHECK(ctranNvl->releaseMem(&rkey->nvlKey));
      break;
    default:
      // no-op for other backends
      break;
  }
  return commSuccess;
}

commResult_t CtranMapper::regAsync(const void* buf, const size_t len) {
  // No-op if not in async mode
  if (NCCL_CTRAN_REGISTER != NCCL_CTRAN_REGISTER::async) {
    return commSuccess;
  }

  auto regCache = CtranMapperRegCache::getInstance();
  CHECK_VALID_REGCACHE(regCache);

  // Already registered
  if (regCache->isRegistered(buf, len)) {
    return commSuccess;
  }

  // Not yet registered, submit the registration to be performed by background
  // thread.
  const int cudaDev = comm->statex_->cudaDev();

  if (NCCL_CTRAN_REGISTER_ERROR_ON_DYNAMIC) {
    // Optionally fail on dynamic registration for user buffer registration
    // debugging. Intentionally fail on user main thread to capture the user
    // level stack trace properly.
    // If the buffer is pre-registered, it enforces the registration at submit
    // time, and later searchRegHandle on GPE thread would hit fast path.
    void* regHdl = nullptr;
    bool dynamicRegist = false;
    FB_COMMCHECK(searchRegHandle(
        buf, len, &regHdl, &dynamicRegist, false /* allowDynamic */));
    CLOGF_TRACE(COLL, "regAsync registered buf {} len {}", buf, len);
    return commSuccess;
  } else {
    FB_COMMCHECK(regCache->asyncRegRange(buf, len, cudaDev, logMetaData_));
    CLOGF_TRACE(COLL, "regAsync submitted buf {} len {}", buf, len);
    return commInProgress;
  }
}

commResult_t CtranMapper::searchRegHandle(
    const void* buf,
    std::size_t len,
    void** regHdl,
    bool* dynamicRegist,
    bool allowDynamic) {
  auto regCache = CtranMapperRegCache::getInstance();
  CHECK_VALID_REGCACHE(regCache);

  const int cudaDev = comm->statex_->cudaDev();
  *dynamicRegist = false;

  // First try find whether the given <buf, len> range has been covered by an
  // existing registration so the registration can be reused.
  // If not yet registered, but all underlying segments have ben pre-registered
  // (i.e., cached) by user, regRange will internally perform the
  // registration and cache it. All logic should be handled within a single
  // global lock.
  CtranMapperRegElem* regHdl_ = nullptr;
  bool didRegister = false;
  FB_COMMCHECK(regCache->regRange(
      buf, len, cudaDev, "regMem", logMetaData_, didRegister, &regHdl_));

  if (!regHdl_) {
    if (!allowDynamic) {
      CLOGF(
          ERR,
          "CTRAN-MAPPER: buffer {} len {} is not pre-registered by user. ",
          buf,
          len);
      return commInvalidUsage;
    }

    // Oops, this is not a known cached segment, we have to do dynamic
    // registration with the given buf, len range. Caller is responsible for
    // immediate deregisgration after current use.
    FB_COMMCHECK(
        regCache->regDynamic(buf, len, comm->statex_->cudaDev(), &regHdl_));
    *dynamicRegist = true;
    CLOGF(
        WARN,
        "CTRAN-MAPPER: buffer {} len {} is not pre-registered by user. "
        "We have to one-time register it and deregister immediately after "
        "current collective, which may likely cause a performance slowdown.",
        (void*)buf,
        len);
  }

  CLOGF_TRACE(
      COLL,
      "searchRegHandle buf {} len {} returned handle (regElem) {} dynamicRegist {}",
      buf,
      len,
      (void*)regHdl_,
      *dynamicRegist);

  *regHdl = regHdl_;
  return commSuccess;
}

commResult_t CtranMapper::icopy(
    void* dbuf,
    const void* sbuf,
    std::size_t len,
    cudaStream_t stream,
    CtranMapperRequest** req) {
  CtranMapperRequest* traceReq = nullptr;
  if (req) {
    *req = new CtranMapperRequest(CtranMapperRequest::ReqType::COPY, rank);
    (*req)->workStream = stream;
    traceReq = *req;
  }
  FB_CUDACHECK(cudaMemcpyAsync(dbuf, sbuf, len, cudaMemcpyDefault, stream));

  iCopyCount++;

  if (this->mapperTrace) {
    this->mapperTrace->recordMapperEvent(
        ncclx::colltrace::CopyStart{
            .sourceBuffer = const_cast<void*>(sbuf),
            .destBuffer = dbuf,
            .length = len,
            .stream = stream,
            .req = traceReq});
  }

  return commSuccess;
}

commResult_t CtranMapper::preConnect(const std::unordered_set<int>& peerRanks) {
  if (this->ctranIb != nullptr) {
    FB_COMMCHECK(this->ctranIb->preConnect(peerRanks));
  } else if (ctranSock != nullptr) {
    FB_COMMCHECK(this->ctranSock->preConnect(peerRanks));
  } else if (this->ctranTcpDm != nullptr) {
    FB_COMMCHECK(this->ctranTcpDm->preConnect(peerRanks));
  }
  return commSuccess;
}

CtranMapperBackend CtranMapper::getBackend(int rank) {
  if (this->ctranNvl && this->ctranNvl->isSupported(rank)) {
    return CtranMapperBackend::NVL;
  } else if (this->ctranIb) {
    return CtranMapperBackend::IB;
  } else if (this->ctranSock) {
    return CtranMapperBackend::SOCKET;
  } else if (this->ctranTcpDm) {
    return CtranMapperBackend::TCPDM;
  }
  return CtranMapperBackend::UNSET;
}

bool CtranMapper::hasBackend(int rank, CtranMapperBackend specified) {
  if (specified == CtranMapperBackend::NVL) {
    return this->ctranNvl && this->ctranNvl->isSupported(rank);
  } else if (specified == CtranMapperBackend::IB) {
    return this->ctranIb != nullptr;
  } else if (specified == CtranMapperBackend::TCPDM) {
    return this->ctranTcpDm != nullptr;
  } else if (specified == CtranMapperBackend::SOCKET) {
    return this->ctranSock != nullptr;
  } else {
    return false;
  }
}

bool CtranMapper::hasBackend() {
  const auto& statex = comm->statex_;
  const auto nRanks = statex->nRanks();
  const auto rank = statex->rank();
  for (int peer = 0; peer < nRanks; peer++) {
    if (peer != rank && getBackend(peer) == CtranMapperBackend::UNSET) {
      return false;
    }
  }
  return true;
}

CtranIb* CtranMapper::ctranIbPtr() {
  return ctranIb.get();
}

CtranSocket* CtranMapper::ctranSockPtr() {
  return ctranSock.get();
}

commResult_t CtranMapper::intraAllGatherCtrl(
    const void* buf,
    void* hdl,
    std::vector<void*>& remoteBufs,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
    CtranMapperBackend backend) {
  const auto& statex = comm->statex_;
  const int nLocalRanks = statex->nLocalRanks();

  std::vector<int> ranks(nLocalRanks);
  for (int i = 0; i < nLocalRanks; i++) {
    ranks[i] = statex->localRankToRank(i);
  }

  return this->allGatherCtrl(
      buf, hdl, ranks, remoteBufs, remoteAccessKeys, backend);
}

commResult_t CtranMapper::allGatherCtrl(
    const void* buf,
    void* hdl,
    std::vector<void*>& remoteBufs,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
    CtranMapperBackend backend) {
  const int nRanks = comm->statex_->nRanks();
  std::vector<int> ranks(nRanks);
  for (int i = 0; i < nRanks; i++) {
    ranks[i] = i;
  }
  return this->allGatherCtrl(
      buf, hdl, ranks, remoteBufs, remoteAccessKeys, backend);
}

commResult_t CtranMapper::allGatherCtrl(
    const void* buf,
    void* hdl,
    const std::vector<int>& ranks,
    std::vector<void*>& remoteBufs,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
    CtranMapperBackend backend) {
  const int rank = comm->statex_->rank();

  // Skip if rank is not in the ranks list
  if (std::find(ranks.begin(), ranks.end(), rank) == ranks.end()) {
    return commSuccess;
  }

  std::vector<CtranMapperRequest> reqs(ranks.size() * 2);
  auto numReqs = 0;
  // If rank is in the ranks list, exchange with all others
  for (auto peer : ranks) {
    // Direct fill rank itself's buf
    if (rank == peer) {
      remoteBufs[peer] = const_cast<void*>(buf);
      remoteAccessKeys[peer].backend = CtranMapperBackend::UNSET;
      continue;
    }

    // Exchange with other ranks
    FB_COMMCHECK(irecvCtrlImpl(
        &remoteBufs[peer], &remoteAccessKeys[peer], peer, &reqs[numReqs++]));
    FB_COMMCHECK(isendCtrlImpl(buf, hdl, peer, &reqs[numReqs++], backend));
  }

  // TODO: completes all send and recv requests for now. May expose requests to
  // algorithm for more flexible completion control
  for (auto i = 0; i < numReqs; i++) {
    FB_COMMCHECK(waitRequest(&reqs[i]));
  }

  return commSuccess;
}

commResult_t CtranMapper::barrier() {
  const auto& statex = comm->statex_;
  const int nRanks = statex->nRanks();
  const int myRank = statex->rank();

  std::vector<CtranMapperRequest> reqs(nRanks * 2 - 2);

  int idx = 0;
  // Sync ctrl to all ranks except itself
  for (int r = 1; r < nRanks; r++) {
    // Shift from my rank by 1 to avoid receiver side congestion
    const int peerRank = (myRank + r) % nRanks;

    FB_COMMCHECK(irecvCtrlImpl(peerRank, &reqs[idx++]));
    FB_COMMCHECK(isendCtrlImpl(peerRank, &reqs[idx++]));
  }

  // TODO: completes all send and recv requests for now. May expose requests to
  // algorithm for more flexible completion control
  for (auto& req : reqs) {
    FB_COMMCHECK(waitRequest(&req));
  }
  return commSuccess;
}

commResult_t CtranMapper::intraBarrier() {
  const auto& statex = comm->statex_;
  const int nLocalRanks = statex->nLocalRanks();
  const int myLocalRank = statex->localRank();

  if (nLocalRanks == 1) {
    return commSuccess;
  }

  std::vector<CtranMapperRequest> reqs(nLocalRanks * 2 - 2);

  int idx = 0;
  // Sync ctrl to all local ranks except itself
  // TODO: get rid of heavy mapperRequest
  for (int r = 1; r < nLocalRanks; r++) {
    // Shift from my local rank by 1 to avoid receiver side congestion
    const int peerLocalRank = (myLocalRank + r) % nLocalRanks;
    int peerRank = statex->localRankToRank(peerLocalRank);

    FB_COMMCHECK(irecvCtrlImpl(peerRank, &reqs[idx++]));
    FB_COMMCHECK(isendCtrlImpl(peerRank, &reqs[idx++]));
  }

  // TODO: completes all send and recv requests for now. May expose requests to
  // algorithm for more flexible completion control
  for (auto& req : reqs) {
    FB_COMMCHECK(waitRequest(&req));
  }
  return commSuccess;
}

std::unordered_map<CtranMapperRegElem*, std::unordered_set<int>>
CtranMapper::dumpExportRegCache() const {
  return exportRegCache_.rlock()->dump();
}

std::string CtranMapperNotify::toString() const {
  std::stringstream ss;
  ss << "peer=" << peer << ", backend=" << CtranMapper::backendToStr(backend)
     << ", notifyCnt=" << notifyCnt;
  if (backend == CtranMapperBackend::NVL) {
    ss << ", kernElem=" << kernElem;
  }
  return ss.str();
}

std::string CtranMapperRemoteAccessKey::toString() const {
  std::stringstream ss;
  ss << "backend=" << CtranMapper::backendToStr(backend);
  switch (backend) {
    case CtranMapperBackend::NVL:
      ss << ", nvlKey=" << "[" << nvlKey.toString() << "]";
      break;
    case CtranMapperBackend::IB:
      ss << ", ibKey=" << "[" << ibKey.toString() << "]";
      break;
    default:
      break;
  }

  return ss.str();
}
