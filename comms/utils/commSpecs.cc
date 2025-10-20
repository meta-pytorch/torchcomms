// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/commSpecs.h"

std::size_t CommLogData::hash() const noexcept {
  return folly::hash::hash_combine(commId, commHash, commDesc, rank, nRanks);
}

namespace meta::comms {

bool CommsError::operator==(const CommsError& other) const {
  return message == other.message && errorCode == other.errorCode;
}

} // namespace meta::comms
