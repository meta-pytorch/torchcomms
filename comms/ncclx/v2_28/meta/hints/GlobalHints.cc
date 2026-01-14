// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string>
#include <unordered_map>

#include <folly/Singleton.h>
#include <folly/Synchronized.h>
#include "comms/utils/logger/LogUtils.h"

#include "meta/hints/GlobalHints.h"
#include "nccl.h"

namespace ncclx {

folly::Singleton<GlobalHints> globalHintsMngrSingleton;

GlobalHints::GlobalHints() {
  for (auto& key : kHintKeysArray) {
    // Register default hints without hooks
    regHintEntry(std::string(key));
  }
}

// Storage for basic hints that require only string key and value management
bool GlobalHints::setHint(std::string key, std::string val) {
  auto entry = getHintEntry(key);
  if (!entry) {
    CLOGF(
        WARN,
        "GLOBAL-HINTS: Hint key {} is not registered. Ignore setHint.",
        key);
    return false; // not registered key
  }
  // call optional setHook before setting the value
  // NOTE: setHook should manage atomicity by itself if needed. We don't
  // guarantee consistency between any internal state updated by setHook and the
  // value in hintVals_.
  // hintVals_ is used only for tracking the hint value set/reset by user, not
  // for any internal state associated with the hint
  auto& setHook = entry->setHook;
  if (setHook) {
    setHook(key, val);
  }
  hintVals_.wlock()->insert_or_assign(std::move(key), std::move(val));
  return true; // always success; if key exists, overwrite
}

std::optional<std::string> GlobalHints::getHint(const std::string& key) {
  auto hints = hintVals_.rlock();
  auto it = hints->find(key);
  if (it == hints->end()) {
    return std::nullopt;
  } else {
    return it->second;
  }
}

bool GlobalHints::resetHint(const std::string& key) {
  auto entry = getHintEntry(key);
  if (!entry) {
    CLOGF(
        WARN,
        "GLOBAL-HINTS: Hint key {} is not registered. Ignore resetHint.",
        key);
    return false; // not registered key
  }
  // call optional resetHook before reseting the hint
  // NOTE: similar to setHoot, resetHook should manage atomicity by itself if
  // needed. We don't guarantee consistency between any internal state updated
  // by resetHook and the value in hintVals_.
  // hintVals_ is used only for tracking the hint value set/reset by user, not
  // for any internal state associated with the hint.
  auto& resetHook = entry->resetHook;
  if (resetHook) {
    resetHook(key);
  }

  hintVals_.wlock()->erase(key);
  return true; // always success; if hint is yet not set, erase() is noop
}

void GlobalHints::testOnlyReset() {
  hintVals_.wlock()->clear();
  hintEntries_.wlock()->clear();
}

std::shared_ptr<GlobalHints> GlobalHints::getInstance() {
  auto mngr = globalHintsMngrSingleton.try_get();
  if (!mngr) {
    throw std::runtime_error("GlobalHints singleton is not initialized");
  }
  return mngr;
}

bool GlobalHints::regHintEntry(std::string key, GlobalHintEntry& entry) {
  auto entries = hintEntries_.wlock();
  // If entry is not provided, create a default entry with no hooks
  const auto& res = entries->insert({std::move(key), std::move(entry)});
  return res.second; // insert success or already exists
}

bool GlobalHints::regHintEntry(std::string key) {
  auto entries = hintEntries_.wlock();
  // If entry is not provided, create a default entry with no hooks
  auto entry_ = GlobalHintEntry{};
  CLOGF(INFO, "GLOBAL-HINTS: Hint key {} is registered.", key);
  const auto& res = entries->insert({std::move(key), std::move(entry_)});
  return res.second; // insert success or already exists
}

const GlobalHintEntry* GlobalHints::getHintEntry(const std::string& key) {
  auto entries = hintEntries_.rlock();

  auto it = entries->find(key);
  if (it == entries->end()) {
    return nullptr;
  }
  return &it->second;
}

__attribute__((visibility("default"))) ncclResult_t
setGlobalHint(std::string key, std::string val) {
  auto mngr = GlobalHints::getInstance();
  auto res = mngr->setHint(std::move(key), std::move(val));
  return res ? ncclSuccess : ncclInternalError;
}

__attribute__((visibility("default"))) std::optional<std::string> getGlobalHint(
    std::string_view key) {
  // TODO: use const string to avoid string creation cost per get, if callsite
  // is not string_view
  const std::string keyStr(key);
  auto mngr = GlobalHints::getInstance();
  return mngr->getHint(keyStr);
}

__attribute__((visibility("default"))) bool resetGlobalHint(
    const std::string& key) {
  auto mngr = GlobalHints::getInstance();
  return mngr->resetHint(key);
}

__attribute__((visibility("default"))) bool setGlobalHint(
    const char* key,
    const char* val) {
  auto mngr = GlobalHints::getInstance();
  return mngr->setHint(std::string(key), std::string(val));
}

__attribute__((visibility("default"))) bool resetGlobalHint(const char* key) {
  auto mngr = GlobalHints::getInstance();
  return mngr->resetHint(std::string(key));
}

void testOnlyResetGlobalHints() {
  auto mngr = GlobalHints::getInstance();
  mngr->testOnlyReset();
}

template <typename T>
std::optional<T> getTypedGlobalHint(std::string_view key) {
  auto val = getGlobalHint(key);
  if (!val) {
    return std::nullopt;
  }

  T valT;
  try {
    valT = folly::to<T>(*val);
  } catch (const std::exception& e) {
    return std::nullopt;
  }
  return valT;
}

template std::optional<bool> getTypedGlobalHint<bool>(std::string_view key);

} // namespace ncclx
