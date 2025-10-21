// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <unordered_map>

#include "comms/utils/commSpecs.h"

namespace meta::comms {

using kvType = std::unordered_map<std::string, std::string>;
/*
 * COMM hints object - Allows us to set hints (key, value) in the form
 * of strings, to be used by other operations.
 *
 * Ctran internal counterpart to ncclx::Hints, enabling hint support without
 * dependency on nccl.h. See comms/ncclx/v2_25/meta/wrapper/MetaFactory.h for
 * translation from ncclx::Hints to this class.
 */
class Hints {
 public:
  Hints();
  commResult_t set(const std::string& key, const std::string& val);
  commResult_t get(const std::string& key, std::string& val) const;

 private:
  kvType kv;
};

} // namespace meta::comms
