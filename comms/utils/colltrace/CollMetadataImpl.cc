// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CollMetadataImpl.h"

#include "comms/utils/Conversion.h"

namespace meta::comms::colltrace {

std::size_t BaselineMetadata::hash() const noexcept {
  // Combine all fields into a hash
  return folly::hash::hash_combine(
      reinterpret_cast<uintptr_t>(stream),
      static_cast<int>(coll),
      static_cast<int>(algorithm),
      static_cast<int>(protocol),
      static_cast<int>(redOp),
      root);
}

folly::dynamic BaselineMetadata::toDynamic() const noexcept {
  folly::dynamic result = folly::dynamic::object();

  // Add all fields to the dynamic object
  result["stream"] = reinterpret_cast<uintptr_t>(stream);
  result["coll"] = commFuncToString(coll);
  result["algorithm"] = commAlgoToString(algorithm);
  result["protocol"] = commProtocolToString(protocol);
  result["redOp"] = commRedOpToString(redOp);
  result["root"] = root;

  return result;
}

BaselineMetadata BaselineMetadata::fromDynamic(
    const folly::dynamic& d) noexcept {
  return BaselineMetadata{
      .stream = reinterpret_cast<cudaStream_t>(
          static_cast<uintptr_t>(d["stream"].asInt())),
      .coll = stringToCommFunc(d["coll"].asString()),
      .algorithm = stringToCommAlgo(d["algorithm"].asString()),
      .protocol = stringToCommProtocol(d["protocol"].asString()),
      .redOp = stringToCommRedOp(d["redOp"].asString()),
      .root = static_cast<int>(d["root"].asInt())};
}

bool BaselineMetadata::operator==(const BaselineMetadata& other) const {
  return stream == other.stream && coll == other.coll &&
      algorithm == other.algorithm && protocol == other.protocol &&
      redOp == other.redOp && root == other.root;
}

bool BaselineMetadata::operator!=(const BaselineMetadata& other) const {
  return !(*this == other);
}

std::size_t CtranMetadata::hash() const noexcept {
  // Hash only the stream field
  return folly::hash::hash_combine(reinterpret_cast<uintptr_t>(stream));
}

folly::dynamic CtranMetadata::toDynamic() const noexcept {
  folly::dynamic result = folly::dynamic::object();

  // Add only the stream field to the dynamic object
  result["stream"] = reinterpret_cast<uintptr_t>(stream);

  return result;
}

CtranMetadata CtranMetadata::fromDynamic(const folly::dynamic& d) noexcept {
  return CtranMetadata{
      .stream = reinterpret_cast<cudaStream_t>(
          static_cast<uintptr_t>(d["stream"].asInt()))};
}

bool CtranMetadata::operator==(const CtranMetadata& other) const {
  return stream == other.stream;
}

bool CtranMetadata::operator!=(const CtranMetadata& other) const {
  return !(*this == other);
}

} // namespace meta::comms::colltrace
