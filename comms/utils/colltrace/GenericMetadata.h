// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <folly/dynamic.h>

#include "comms/utils/colltrace/CollMetadata.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

// Metadata for collectives.
struct CollectiveMetadata {
  std::string opName;
  std::string algoName;
  uint64_t opCount;
  uintptr_t sendbuff; // Use uintptr_t as we only need the address.
  uintptr_t recvbuff;
  commDataType_t dataType{commDataType_t::commNumTypes};
  std::optional<std::size_t>
      count; // For some collectives, there is no single count

  // Custom comparison operators since C++20 default comparisons are not fully
  // supported
  bool operator==(const CollectiveMetadata& other) const;
  bool operator!=(const CollectiveMetadata& other) const;

  std::size_t hash() const noexcept;
  folly::dynamic toDynamic() const noexcept;
  static CollectiveMetadata fromDynamic(const folly::dynamic& d) noexcept;
  static std::string_view getMetadataType() noexcept;
};

static_assert(
    GenericMetadataComponent<CollectiveMetadata>,
    "CollectiveMetadata should conform to GenericMetadataComponent concept");

struct GroupedP2PMetaData {
  std::string opName;
  std::string algoName;
  uint64_t opCount;

  std::vector<int> ranksInGroupedP2P;
  commDataType_t dataType{
      commDataType_t::commNumTypes}; // Should always be initialized as int8 as
  // we are counting bytes
  std::size_t count;

  // Custom comparison operators since C++20 default comparisons are not fully
  // supported
  bool operator==(const GroupedP2PMetaData& other) const;
  bool operator!=(const GroupedP2PMetaData& other) const;

  std::size_t hash() const noexcept;
  folly::dynamic toDynamic() const noexcept;
  static GroupedP2PMetaData fromDynamic(const folly::dynamic& d) noexcept;
  static std::string_view getMetadataType() noexcept;
};

static_assert(
    GenericMetadataComponent<GroupedP2PMetaData>,
    "GroupedP2PMetaData should conform to GenericMetadataComponent concept");

struct GroupedCollP2PMetaData {
  std::vector<CollectiveMetadata> colls;
  std::optional<GroupedP2PMetaData> p2p;

  // Custom comparison operators since C++20 default comparisons are not fully
  // supported
  bool operator==(const GroupedCollP2PMetaData& other) const;
  bool operator!=(const GroupedCollP2PMetaData& other) const;

  std::size_t hash() const noexcept;
  folly::dynamic toDynamic() const noexcept;
  static GroupedCollP2PMetaData fromDynamic(const folly::dynamic& d) noexcept;
  static std::string_view getMetadataType() noexcept;
};

static_assert(
    GenericMetadataComponent<GroupedCollP2PMetaData>,
    "GroupedCollP2PMetaData should conform to GenericMetadataComponent concept");

} // namespace meta::comms::colltrace
