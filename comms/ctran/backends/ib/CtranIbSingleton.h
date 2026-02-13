// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IB_SINGLETON_H_
#define CTRAN_IB_SINGLETON_H_

#include <memory>
#include <vector>

#include <folly/Singleton.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/ib/ibutils.h"
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/commSpecs.h"

/**
 * Singleton class to hold the IB network resources that are reused by all
 * communicators in the lifetime of program.
 */
class CtranIbSingleton {
 public:
  CtranIbSingleton(const CtranIbSingleton& obj) = delete;
  CtranIbSingleton& operator=(const CtranIbSingleton&) = delete;

  // Returns a shared_ptr to the singleton instance. Using shared_ptr ensures
  // that callers who hold a reference will keep the singleton alive.
  static std::shared_ptr<CtranIbSingleton> getInstance();
  std::vector<ibverbx::IbvDevice> ibvDevices;

  const ibverbx::IbvPd& getIbvPd(size_t idx) const;

  // Record traffic in bytes whenever IB data transfer happens.
  // Accumulate per device (identified by cudaDev), QP, or both.
  void recordDeviceTraffic(
      ibverbx::ibv_context* ctx,
      const int cudaDev,
      size_t nbytes);

  // Record CtranComm reference to track missed comm destruction which may
  // cause failed CtranIbSingleton destruction (e.g., wrap_ibv_dealloc_pd
  // failed due to resource being used).
  void commRef(CtranComm* comm);
  void commDeref(CtranComm* comm);

  // Track number of active IB registrations.
  size_t getActiveRegCount();
  void incActiveRegCount();
  void decActiveRegCount();

  // startIbAsyncEventHandler provided for testing
  void startIbAsyncEventHandler(const int cudaDev);
  void stopIbAsyncEventHandler();
  static bool getStopIbAsyncEventHandlerFlag();

  static void ibAsyncEventHandler(const int cudaDev);

  bool getDevToDmaBufSupport(int cudaDev);

  size_t getDeviceTrafficSnapshot(const int cudaDev);

  static IVerbsWrapper* getVerbsPtr();

  // Implement actual destroy logic here and called by destructor.
  // This allows test to check destroy() if needed.
  commResult_t destroy();
  std::atomic_bool stopIbAsyncEventHandlerFlag = false;
  std::unique_ptr<VerbsUtils> verbsUtils = std::make_unique<VerbsUtils>();

 private:
  // Allow folly::Singleton to construct the singleton
  friend class folly::Singleton<CtranIbSingleton>;

  CtranIbSingleton();
  ~CtranIbSingleton();
  std::vector<ibverbx::IbvPd> ibvPds_;
  std::vector<std::unique_ptr<std::atomic<size_t>>> devBytes_;

  std::unordered_map<int, bool> devToDmaBufSupport;
  std::mutex dmaBufSupportMutex_;

  std::atomic<std::size_t> activeRegCount_ = 0;
  folly::Synchronized<std::unordered_set<CtranComm*>> comms_;

  std::thread ibAsyncEventThread_;

  // Track whether destroy has been triggered but skipped due to
  // COMM_ABORT_SCOPE setting
  bool destroySkipped_ = false;
};

static inline void CHECK_VALID_IB_SINGLETON(
    std::shared_ptr<CtranIbSingleton> singleton) {
  FB_CHECKABORT(
      singleton != nullptr, "Failed to get CtranIbSingleton instance");
}

#endif
