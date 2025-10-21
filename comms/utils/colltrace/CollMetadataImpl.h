// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/utils/colltrace/CollMetadata.h"
#include "comms/utils/colltrace/CommLogDataSerialize.h"
#include "comms/utils/colltrace/GenericMetadata.h" // IWYU pragma: keep
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

struct BaselineMetadata {
  cudaStream_t stream{nullptr};
  CommFunc coll{CommFunc::NumFuncs};
  CommAlgo algorithm{CommAlgo::NumAlgorithms};
  CommProtocol protocol{CommProtocol::NumProtocols};
  commRedOp_t redOp{commRedOp_t::commNumOps};
  int root; // peer for p2p operations

  // Custom comparison operators since C++20 default comparisons are not fully
  // supported
  bool operator==(const BaselineMetadata& other) const;
  bool operator!=(const BaselineMetadata& other) const;

  std::size_t hash() const noexcept;
  folly::dynamic toDynamic() const noexcept;
  static BaselineMetadata fromDynamic(const folly::dynamic& d) noexcept;
};

static_assert(
    MetadataComponent<BaselineMetadata>,
    "BaselineMetadata should conform to MetadataComponent concept");

struct CtranMetadata {
  cudaStream_t stream{nullptr};

  // Custom comparison operators since C++20 default comparisons are not fully
  // supported
  bool operator==(const CtranMetadata& other) const;
  bool operator!=(const CtranMetadata& other) const;

  std::size_t hash() const noexcept;
  folly::dynamic toDynamic() const noexcept;
  static CtranMetadata fromDynamic(const folly::dynamic& d) noexcept;
};

static_assert(
    MetadataComponent<CtranMetadata>,
    "CtranMetadata should conform to MetadataComponent concept");

template <
    MetadataComponent BackendSpecificMetadata,
    GenericMetadataComponent GenericMetadata>
class CollMetadataImpl : public ICollMetadata {
 public:
  CollMetadataImpl(
      CommLogData commMetadata,
      BackendSpecificMetadata backendSpecificMetadata,
      GenericMetadata genericMetadata)
      : commMetadata_(std::move(commMetadata)),
        backendSpecificMetadata_(std::move(backendSpecificMetadata)),
        genericMetadata_(std::move(genericMetadata)) {}

  std::size_t hash() const override {
    return folly::hash::hash_combine(
        commMetadata_.hash(),
        backendSpecificMetadata_.hash(),
        genericMetadata_.hash());
  }
  bool equals(const ICollMetadata& other) const noexcept override {
    if (getMetadataType() != other.getMetadataType()) {
      return false;
    }

    const auto& otherMetadata = static_cast<decltype(*this)>(other);
    return commMetadata_ == otherMetadata.commMetadata_ &&
        backendSpecificMetadata_ == otherMetadata.backendSpecificMetadata_ &&
        genericMetadata_ == otherMetadata.genericMetadata_;
  }

  std::string_view getMetadataType() const noexcept override {
    return GenericMetadata::getMetadataType();
  }

  folly::dynamic toDynamic() const noexcept override {
    folly::dynamic result = folly::dynamic::object();

    result.update(
        folly::DynamicConstructor<CommLogData>::construct(commMetadata_));
    result.update(backendSpecificMetadata_.toDynamic());
    result.update(genericMetadata_.toDynamic());
    result["MetadataType"] = std::string(GenericMetadata::getMetadataType());

    return result;
  }
  void fromDynamic(const folly::dynamic& d) noexcept override {
    // Use the fromDynamic methods for the other components
    commMetadata_ = folly::DynamicConverter<CommLogData>::convert(d);
    backendSpecificMetadata_ = BackendSpecificMetadata::fromDynamic(d);
    genericMetadata_ = GenericMetadata::fromDynamic(d);
  }

 private:
  CommLogData commMetadata_;
  BackendSpecificMetadata backendSpecificMetadata_;
  GenericMetadata genericMetadata_;
};

template <
    MetadataComponent BackendSpecificMetadata,
    GenericMetadataComponent GenericMetadata>
std::unique_ptr<ICollMetadata> makeCollMetadata(
    CommLogData commMetadata,
    BackendSpecificMetadata backendSpecificMetadata,
    GenericMetadata genericMetadata) {
  return std::make_unique<
      CollMetadataImpl<BackendSpecificMetadata, GenericMetadata>>(
      std::move(commMetadata),
      std::move(backendSpecificMetadata),
      std::move(genericMetadata));
}

} // namespace meta::comms::colltrace
