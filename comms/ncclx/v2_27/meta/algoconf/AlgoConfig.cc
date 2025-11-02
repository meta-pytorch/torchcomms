// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/algoconf/AlgoConfig.h"
#include <folly/Singleton.h>
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/GlobalHints.h"

using ncclx::GlobalHints;
namespace {

// TODO: populate the helper functions from CVAR auto-generated code
inline const std::string algoValToStr(enum NCCL_SENDRECV_ALGO val) {
  switch (val) {
    case NCCL_SENDRECV_ALGO::ctran:
      return "ctran";
    case NCCL_SENDRECV_ALGO::orig:
      return "orig";
  }
}

inline void algoStrToVal(const std::string& str, enum NCCL_SENDRECV_ALGO& val) {
  if (str == "ctran") {
    val = NCCL_SENDRECV_ALGO::ctran;
  } else {
    val = NCCL_SENDRECV_ALGO::orig;
  }
}

inline const std::string algoValToStr(enum NCCL_ALLGATHER_ALGO val) {
  switch (val) {
    case NCCL_ALLGATHER_ALGO::orig:
      return "orig";
    case NCCL_ALLGATHER_ALGO::ctran:
      return "ctran";
    case NCCL_ALLGATHER_ALGO::ctdirect:
      return "ctdirect";
    case NCCL_ALLGATHER_ALGO::ctring:
      return "ctring";
    case NCCL_ALLGATHER_ALGO::ctrd:
      return "ctrd";
    case NCCL_ALLGATHER_ALGO::ctbrucks:
      return "ctbrucks";
      break;
  }
}

inline void algoStrToVal(
    const std::string& str,
    enum NCCL_ALLGATHER_ALGO& val) {
  if (str == "ctran") {
    val = NCCL_ALLGATHER_ALGO::ctran;
  } else if (str == "ctdirect") {
    val = NCCL_ALLGATHER_ALGO::ctdirect;
  } else if (str == "ctring") {
    val = NCCL_ALLGATHER_ALGO::ctring;
  } else if (str == "ctrd") {
    val = NCCL_ALLGATHER_ALGO::ctrd;
  } else if (str == "ctbrucks") {
    val = NCCL_ALLGATHER_ALGO::ctbrucks;
  } else {
    val = NCCL_ALLGATHER_ALGO::orig;
  }
}

inline const std::string algoValToStr(enum NCCL_ALLREDUCE_ALGO val) {
  switch (val) {
    case NCCL_ALLREDUCE_ALGO::orig:
      return "orig";
    case NCCL_ALLREDUCE_ALGO::ctran:
      return "ctran";
    case NCCL_ALLREDUCE_ALGO::ctdirect:
      return "ctdirect";
    case NCCL_ALLREDUCE_ALGO::ctarg:
      return "ctarg";
    case NCCL_ALLREDUCE_ALGO::ctring:
      return "ctring";
  }
}

inline void algoStrToVal(
    const std::string& str,
    enum NCCL_ALLREDUCE_ALGO& val) {
  if (str == "ctran") {
    val = NCCL_ALLREDUCE_ALGO::ctran;
  } else if (str == "ctdirect") {
    val = NCCL_ALLREDUCE_ALGO::ctdirect;
  } else if (str == "ctarg") {
    val = NCCL_ALLREDUCE_ALGO::ctarg;
  } else if (str == "ctring") {
    val = NCCL_ALLREDUCE_ALGO::ctring;
  } else {
    val = NCCL_ALLREDUCE_ALGO::orig;
  }
}

inline const std::string algoValToStr(enum NCCL_ALLTOALLV_ALGO val) {
  switch (val) {
    case ::NCCL_ALLTOALLV_ALGO::orig:
      return "orig";
    case NCCL_ALLTOALLV_ALGO::ctran:
      return "ctran";
    case NCCL_ALLTOALLV_ALGO::compCtran:
      return "compCtran";
    case NCCL_ALLTOALLV_ALGO::bsCompCtran:
      return "bsCompCtran";
  }
}

inline void algoStrToVal(
    const std::string& str,
    enum NCCL_ALLTOALLV_ALGO& val) {
  if (str == "ctran") {
    val = NCCL_ALLTOALLV_ALGO::ctran;
  } else if (str == "compCtran") {
    val = NCCL_ALLTOALLV_ALGO::compCtran;
  } else if (str == "bsCompCtran") {
    val = NCCL_ALLTOALLV_ALGO::bsCompCtran;
  } else {
    val = NCCL_ALLTOALLV_ALGO::orig;
  }
}

class AlgoConfig {
 public:
  AlgoConfig();
  ~AlgoConfig() {}
  static std::shared_ptr<AlgoConfig> getInstance();

  // allow testOnlyResetAlgoConfig to reset the state for testing
  void reset();

  std::atomic<enum NCCL_SENDRECV_ALGO> sendrecv = NCCL_SENDRECV_ALGO_DEFAULT;
  std::atomic<enum NCCL_ALLGATHER_ALGO> allgather = NCCL_ALLGATHER_ALGO_DEFAULT;
  std::atomic<enum NCCL_ALLREDUCE_ALGO> allreduce = NCCL_ALLREDUCE_ALGO_DEFAULT;
  std::atomic<enum NCCL_ALLTOALLV_ALGO> alltoallv = NCCL_ALLTOALLV_ALGO_DEFAULT;
};

template <typename T>
void setAlgo(const std::string& key, const std::string& valStr);
template <typename T>
void resetAlgo(const std::string& key);

AlgoConfig::AlgoConfig() {
  reset();
};

void AlgoConfig::reset() {
  // Ensure CVAR is initialized before initializing algo hints default values
  ncclCvarInit();
  // Update default algo value from CVAR
  sendrecv.store(NCCL_SENDRECV_ALGO);
  allgather.store(NCCL_ALLGATHER_ALGO);
  allreduce.store(NCCL_ALLREDUCE_ALGO);
  alltoallv.store(NCCL_ALLTOALLV_ALGO);

  // Register algo hints
  auto hintsMngr = GlobalHints::getInstance();
  ncclx::GlobalHintEntry sendrecvEntry = {
      .setHook = setAlgo<enum NCCL_SENDRECV_ALGO>,
      .resetHook = resetAlgo<enum NCCL_SENDRECV_ALGO>};
  hintsMngr->regHintEntry("algo_sendrecv", sendrecvEntry);

  ncclx::GlobalHintEntry allgatherEntry = {
      .setHook = setAlgo<enum NCCL_ALLGATHER_ALGO>,
      .resetHook = resetAlgo<enum NCCL_ALLGATHER_ALGO>};
  hintsMngr->regHintEntry("algo_allgather", allgatherEntry);

  ncclx::GlobalHintEntry allreduceEntry = {
      .setHook = setAlgo<enum NCCL_ALLREDUCE_ALGO>,
      .resetHook = resetAlgo<enum NCCL_ALLREDUCE_ALGO>};
  hintsMngr->regHintEntry("algo_allreduce", allreduceEntry);

  ncclx::GlobalHintEntry alltoallvEntry = {
      .setHook = setAlgo<enum NCCL_ALLTOALLV_ALGO>,
      .resetHook = resetAlgo<enum NCCL_ALLTOALLV_ALGO>};
  hintsMngr->regHintEntry("algo_alltoallv", alltoallvEntry);
}

folly::Singleton<AlgoConfig> algoConfigSingleton;

std::shared_ptr<AlgoConfig> AlgoConfig::getInstance() {
  auto algoConfig = algoConfigSingleton.try_get();
  if (!algoConfig) {
    throw std::runtime_error("AlgoConfig singleton is not initialized");
  }
  return algoConfig;
}

template <typename T>
void setAlgo(
    const std::string& key __attribute__((unused)),
    const std::string& valStr) {
  auto algoConfig = AlgoConfig::getInstance();
  if (std::is_same<T, enum NCCL_SENDRECV_ALGO>::value) {
    enum NCCL_SENDRECV_ALGO val;
    algoStrToVal(valStr, val);
    algoConfig->sendrecv.store(val);
  } else if (std::is_same<T, enum NCCL_ALLGATHER_ALGO>::value) {
    enum NCCL_ALLGATHER_ALGO val;
    algoStrToVal(valStr, val);
    algoConfig->allgather.store(val);
  } else if (std::is_same<T, enum NCCL_ALLREDUCE_ALGO>::value) {
    enum NCCL_ALLREDUCE_ALGO val;
    algoStrToVal(valStr, val);
    algoConfig->allreduce.store(val);
  } else if (std::is_same<T, enum NCCL_ALLTOALLV_ALGO>::value) {
    enum NCCL_ALLTOALLV_ALGO val;
    algoStrToVal(valStr, val);
    algoConfig->alltoallv.store(val);
  }
}

template <typename T>
void resetAlgo(const std::string& key __attribute__((unused))) {
  auto algoConfig = AlgoConfig::getInstance();
  if (std::is_same<T, enum NCCL_SENDRECV_ALGO>::value) {
    algoConfig->sendrecv.store(NCCL_SENDRECV_ALGO);
  } else if (std::is_same<T, enum NCCL_ALLGATHER_ALGO>::value) {
    algoConfig->allgather.store(NCCL_ALLGATHER_ALGO);
  } else if (std::is_same<T, enum NCCL_ALLREDUCE_ALGO>::value) {
    algoConfig->allreduce.store(NCCL_ALLREDUCE_ALGO);
  } else if (std::is_same<T, enum NCCL_ALLTOALLV_ALGO>::value) {
    algoConfig->alltoallv.store(NCCL_ALLTOALLV_ALGO);
  }
}
} // namespace

namespace ncclx::algoconf {
enum NCCL_SENDRECV_ALGO getSendRecvAlgo() {
  auto algoConfig = AlgoConfig::getInstance();
  return algoConfig->sendrecv.load();
}

enum NCCL_ALLGATHER_ALGO getAllGatherAlgo() {
  auto algoConfig = AlgoConfig::getInstance();
  return algoConfig->allgather.load();
}

enum NCCL_ALLREDUCE_ALGO getAllReduceAlgo() {
  auto algoConfig = AlgoConfig::getInstance();
  return algoConfig->allreduce.load();
}

enum NCCL_ALLTOALLV_ALGO getAllToAllVAlgo() {
  auto algoConfig = AlgoConfig::getInstance();
  return algoConfig->alltoallv.load();
}

std::string getAlgoHintValue(enum NCCL_SENDRECV_ALGO algo) {
  return algoValToStr(algo);
}
std::string getAlgoHintValue(enum NCCL_ALLGATHER_ALGO algo) {
  return algoValToStr(algo);
}
std::string getAlgoHintValue(enum NCCL_ALLREDUCE_ALGO algo) {
  return algoValToStr(algo);
}
std::string getAlgoHintValue(enum NCCL_ALLTOALLV_ALGO algo) {
  return algoValToStr(algo);
}

void testOnlyResetAlgoConfig() {
  auto algoConfig = AlgoConfig::getInstance();
  // It may cause hint entry to be registerd twice if this is the first time to
  // initialize algoConfigSingleton. I.e., algoConfigSingleton will register all
  // entries in its constructor. It should be OK as this function is called only
  // in tests.
  algoConfig->reset();
}

void testOnlySetAlgo(enum NCCL_SENDRECV_ALGO algo) {
  auto algoConfig = AlgoConfig::getInstance();
  algoConfig->sendrecv.store(algo);
}

void testOnlySetAlgo(enum NCCL_ALLGATHER_ALGO algo) {
  auto algoConfig = AlgoConfig::getInstance();
  algoConfig->allgather.store(algo);
}

void testOnlySetAlgo(enum NCCL_ALLREDUCE_ALGO algo) {
  auto algoConfig = AlgoConfig::getInstance();
  algoConfig->allreduce.store(algo);
}

void testOnlySetAlgo(enum NCCL_ALLTOALLV_ALGO algo) {
  auto algoConfig = AlgoConfig::getInstance();
  algoConfig->alltoallv.store(algo);
}

void setupGlobalHints() {
  // AlgoConfig constructor will register the hint entries to GlobalHints
  auto algoConfig = AlgoConfig::getInstance();
}
} // namespace ncclx::algoconf
