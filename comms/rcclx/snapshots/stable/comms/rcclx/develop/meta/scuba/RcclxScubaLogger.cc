#include "RcclxScubaLogger.h"

RcclxScubaLogger::RcclxScubaLogger() {}

void RcclxScubaLogger::InitializeInitScubaEvents(
    uint64_t commIdHash,
    uint64_t commHash,
    std::string commDesc,
    int rank,
    int nRanks) {
  if (_initialized) {
    return;
  }

  CommLogData commLogData{
      commIdHash, commHash, std::move(commDesc), rank, nRanks};

  // Initialize _commInitFuncEvent using placement new with CommLogData
  // constructor
  _commInitFuncEvent =
      std::make_unique<NcclScubaEvent>(NcclScubaEvent(&commLogData));

  // Initialize _initBootstrapEvent using placement new with CommLogData
  // constructor
  _initBootstrapEvent =
      std::make_unique<NcclScubaEvent>(NcclScubaEvent(&commLogData));

  // Initialize all other events
  _initTransportsRankEvent =
      std::make_unique<NcclScubaEvent>(NcclScubaEvent(&commLogData));

  _commInitRankConfigEvent =
      std::make_unique<NcclScubaEvent>(NcclScubaEvent(&commLogData));

  _commFinalizeEvent =
      std::make_unique<NcclScubaEvent>(NcclScubaEvent(&commLogData));

  _commDestroyEvent =
      std::make_unique<NcclScubaEvent>(NcclScubaEvent(&commLogData));

  _commAbortEvent =
      std::make_unique<NcclScubaEvent>(NcclScubaEvent(&commLogData));

  _commSplitEvent =
      std::make_unique<NcclScubaEvent>(NcclScubaEvent(&commLogData));

  _p2pPreconnectEvent =
      std::make_unique<NcclScubaEvent>(NcclScubaEvent(&commLogData));

  _collPreconnectEvent =
      std::make_unique<NcclScubaEvent>(NcclScubaEvent(&commLogData));

  _initialized = true;
}
