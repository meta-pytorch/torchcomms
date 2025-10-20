// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/GenericMetadata.h"

#include <folly/logging/xlog.h>

#include "comms/utils/Conversion.h"

namespace meta::comms::colltrace {

std::string_view CollectiveMetadata::getMetadataType() noexcept {
  return "CollectiveMetadata";
}

std::size_t CollectiveMetadata::hash() const noexcept {
  // Start with the hash from the parent class
  std::size_t seed = folly::hash::hash_combine(
      opName,
      algoName,
      opCount,
      sendbuff,
      recvbuff,
      static_cast<int>(dataType));

  // Handle optional count
  if (count.has_value()) {
    seed = folly::hash::hash_combine(seed, *count);
  }

  return seed;
}

folly::dynamic CollectiveMetadata::toDynamic() const noexcept {
  folly::dynamic result = folly::dynamic::object();
  // Add collective name
  result["opName"] = opName;
  result["algoName"] = algoName;
  result["opCount"] = opCount;

  // Add the collective attributes
  result["sendbuff"] = sendbuff;
  result["recvbuff"] = recvbuff;
  result["dataType"] = getCommsDatatypeStr(dataType);

  // Handle optional count
  if (count.has_value()) {
    result["count"] = count.value();
  }

  return result;
}

CollectiveMetadata CollectiveMetadata::fromDynamic(
    const folly::dynamic& d) noexcept {
  return CollectiveMetadata{
      .opName = d["opName"].asString(),
      .algoName = d["algoName"].asString(),
      .opCount = static_cast<uint64_t>(d["opCount"].asInt()),
      .sendbuff = static_cast<uintptr_t>(d["sendbuff"].asInt()),
      .recvbuff = static_cast<uintptr_t>(d["recvbuff"].asInt()),
      .dataType = stringToCommsDatatype(d["dataType"].asString()),
      .count = d["count"].isNull() ? std::nullopt
                                   : std::optional<size_t>{d["count"].asInt()}};
}

bool CollectiveMetadata::operator==(const CollectiveMetadata& other) const {
  return opName == other.opName && algoName == other.algoName &&
      opCount == other.opCount && sendbuff == other.sendbuff &&
      recvbuff == other.recvbuff && dataType == other.dataType &&
      count == other.count;
}

bool CollectiveMetadata::operator!=(const CollectiveMetadata& other) const {
  return !(*this == other);
}

std::string_view GroupedP2PMetaData::getMetadataType() noexcept {
  return "GroupedP2PMetaData";
}

std::size_t GroupedP2PMetaData::hash() const noexcept {
  // Start with the hash from the parent class
  std::size_t seed = folly::hash::hash_combine(
      opName, algoName, opCount, static_cast<int>(dataType), count);

  // Handle ranksInGroupedP2P
  for (const auto& rank : ranksInGroupedP2P) {
    seed = folly::hash::hash_combine(seed, rank);
  }

  return seed;
}

folly::dynamic GroupedP2PMetaData::toDynamic() const noexcept {
  folly::dynamic result = folly::dynamic::object();
  // Add operation name and algorithm name
  result["opName"] = opName;
  result["algoName"] = algoName;
  result["opCount"] = opCount;

  // Add the dataType and count
  result["dataType"] = getCommsDatatypeStr(dataType);
  result["count"] = count;

  // Handle ranksInGroupedP2P
  result["ranksInGroupedP2P"] =
      folly::DynamicConstructor<decltype(ranksInGroupedP2P)>{}.construct(
          ranksInGroupedP2P);

  return result;
}

GroupedP2PMetaData GroupedP2PMetaData::fromDynamic(
    const folly::dynamic& d) noexcept {
  GroupedP2PMetaData metadata{
      .opName = d["opName"].asString(),
      .algoName = d["algoName"].asString(),
      .opCount = static_cast<uint64_t>(d["opCount"].asInt()),
      .dataType = stringToCommsDatatype(d["dataType"].asString()),
      .count = static_cast<size_t>(d["count"].asInt())};

  // Handle ranksInGroupedP2P
  metadata.ranksInGroupedP2P =
      folly::DynamicConverter<decltype(ranksInGroupedP2P)>{}.convert(
          d["ranksInGroupedP2P"]);

  return metadata;
}

bool GroupedP2PMetaData::operator==(const GroupedP2PMetaData& other) const {
  return opName == other.opName && algoName == other.algoName &&
      opCount == other.opCount &&
      ranksInGroupedP2P == other.ranksInGroupedP2P &&
      dataType == other.dataType && count == other.count;
}

bool GroupedP2PMetaData::operator!=(const GroupedP2PMetaData& other) const {
  return !(*this == other);
}

std::string_view GroupedCollP2PMetaData::getMetadataType() noexcept {
  return "GroupedCollP2PMetaData";
}

std::size_t GroupedCollP2PMetaData::hash() const noexcept {
  // Start with a base seed value
  std::size_t seed = 0;

  // Handle optional p2p metadata
  if (p2p.has_value()) {
    seed = folly::hash::hash_combine(seed, p2p->hash());
  }

  // Hash all collective metadata
  for (const auto& coll : colls) {
    seed = folly::hash::hash_combine(seed, coll.hash());
  }

  return seed;
}

folly::dynamic GroupedCollP2PMetaData::toDynamic() const noexcept {
  // TODO: Add better serialization for the grouped metadata. For now, we
  // simply use the first collective or p2p metadata.
  if (!colls.empty()) {
    return colls.front().toDynamic();
  }
  if (p2p.has_value()) {
    return p2p->toDynamic();
  }
  // Return empty object if neither colls nor p2p has data
  return folly::dynamic::object();
}

GroupedCollP2PMetaData GroupedCollP2PMetaData::fromDynamic(
    const folly::dynamic& d) noexcept {
  // TODO: Currently the from dynamic function is not functioning as expected.
  GroupedCollP2PMetaData metadata;

  XLOG_FIRST_N(ERR, 1)
      << "GroupedCollP2PMetaData::fromDynamic is not supported yet";

  return metadata;
}

bool GroupedCollP2PMetaData::operator==(
    const GroupedCollP2PMetaData& other) const {
  return p2p == other.p2p && colls == other.colls;
}

bool GroupedCollP2PMetaData::operator!=(
    const GroupedCollP2PMetaData& other) const {
  return !(*this == other);
}

} // namespace meta::comms::colltrace
