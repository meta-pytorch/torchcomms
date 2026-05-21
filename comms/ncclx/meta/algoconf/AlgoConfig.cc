// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/algoconf/AlgoConfig.h"
#include <folly/Singleton.h>
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/GlobalHints.h"

using ncclx::GlobalHints;
namespace {

inline const std::string algoValToStr(enum NCCL_SENDRECV_ALGO val) {
  switch (val) {
    case NCCL_SENDRECV_ALGO::ctran:
      return "ctran";
    case NCCL_SENDRECV_ALGO::orig:
      return "orig";
    case NCCL_SENDRECV_ALGO::ctzcopy:
      return "ctzcopy";
    case NCCL_SENDRECV_ALGO::ctp2p:
      return "ctp2p";
    case NCCL_SENDRECV_ALGO::ctgraph:
      return "ctgraph";
  }
}

inline void algoStrToVal(const std::string& str, enum NCCL_SENDRECV_ALGO& val) {
  if (str == "ctran") {
    val = NCCL_SENDRECV_ALGO::ctran;
  } else if (str == "ctzcopy") {
    val = NCCL_SENDRECV_ALGO::ctzcopy;
  } else if (str == "ctp2p") {
    val = NCCL_SENDRECV_ALGO::ctp2p;
  } else if (str == "ctgraph") {
    val = NCCL_SENDRECV_ALGO::ctgraph;
  } else {
    val = NCCL_SENDRECV_ALGO::orig;
  }
}

class AlgoConfig {
 public:
  AlgoConfig();
  ~AlgoConfig() {}
  static std::shared_ptr<AlgoConfig> getInstance();

  void reset();

  std::atomic<enum NCCL_SENDRECV_ALGO> sendrecv =
      NCCL_SENDRECV_ALGO_DEFAULTCVARVALUE;
};

template <typename T>
void setAlgo(const std::string& key, const std::string& valStr);
template <typename T>
void resetAlgo(const std::string& key);

AlgoConfig::AlgoConfig() {
  reset();
};

void AlgoConfig::reset() {
  ncclCvarInit();
  sendrecv.store(NCCL_SENDRECV_ALGO);

  auto hintsMngr = GlobalHints::getInstance();
  ncclx::GlobalHintEntry sendrecvEntry = {
      .setHook = setAlgo<enum NCCL_SENDRECV_ALGO>,
      .resetHook = resetAlgo<enum NCCL_SENDRECV_ALGO>};
  hintsMngr->regHintEntry("algo_sendrecv", sendrecvEntry);
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
  }
}

template <typename T>
void resetAlgo(const std::string& key __attribute__((unused))) {
  auto algoConfig = AlgoConfig::getInstance();
  if (std::is_same<T, enum NCCL_SENDRECV_ALGO>::value) {
    algoConfig->sendrecv.store(NCCL_SENDRECV_ALGO);
  }
}
} // namespace

namespace ncclx::algoconf {
enum NCCL_SENDRECV_ALGO getSendRecvAlgo() {
  auto algoConfig = AlgoConfig::getInstance();
  return algoConfig->sendrecv.load();
}

std::string getAlgoHintValue(enum NCCL_SENDRECV_ALGO algo) {
  return algoValToStr(algo);
}

void testOnlyResetAlgoConfig() {
  auto algoConfig = AlgoConfig::getInstance();
  algoConfig->reset();
}

void testOnlySetAlgo(enum NCCL_SENDRECV_ALGO algo) {
  auto algoConfig = AlgoConfig::getInstance();
  algoConfig->sendrecv.store(algo);
}

void setupGlobalHints() {
  auto algoConfig = AlgoConfig::getInstance();
}
} // namespace ncclx::algoconf
