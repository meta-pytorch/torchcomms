// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/DynamicConverter.h>
#include <folly/dynamic.h>

namespace meta::comms::colltrace {

// For components that can be used as metadata, we don't want to use inheritance
// as we want be able to consturct the metadata components with designated
// initializer for clearity. But we still want them to be hashable and
// comparable. For usage, you should statically check if the component conforms
// to the concept. For example:
//
// struct CollectiveMetadata {
//   ...fields
// }
// static_assert(MetadataComponent<CollectiveMetadata>);
//
// This will make sure that the component is hashable and comparable.
template <typename T>
concept MetadataComponent = requires(T a, const T b, const folly::dynamic& j) {
  { b.hash() } -> std::convertible_to<std::size_t>;
  { b == a } -> std::convertible_to<bool>;
  { b.toDynamic() } -> std::same_as<folly::dynamic>;
  { T::fromDynamic(j) } -> std::same_as<T>;
};

template <typename T>
concept GenericMetadataComponent = MetadataComponent<T> && requires {
  { T::getMetadataType() } -> std::same_as<std::string_view>;
};

// For the interface of the metadata, the only requirement is that
// 1. It can be converted to/from folly::dynamic for serialization.
// 2. We can get the metadata type from the metadata object.
// 3. The metadata type is hashable and comparable.
class ICollMetadata {
 public:
  virtual ~ICollMetadata() = default;

  // Enables hashing and equality comparison for the metadata.
  virtual std::size_t hash() const = 0;
  virtual bool equals(const ICollMetadata& other) const noexcept = 0;

  virtual std::string_view getMetadataType() const noexcept = 0;

  virtual folly::dynamic toDynamic() const noexcept = 0;
  virtual void fromDynamic(const folly::dynamic& d) noexcept = 0;
};

} // namespace meta::comms::colltrace

/*********** Define construction/convertion of folly::dynamic ***********/
namespace folly {

// Defines dynamic constructor for ICollMetadata so we can use it during
// serialization.
template <>
struct DynamicConstructor<meta::comms::colltrace::ICollMetadata> {
  static dynamic construct(const meta::comms::colltrace::ICollMetadata& m) {
    return m.toDynamic();
  }
};

template <meta::comms::colltrace::MetadataComponent T>
struct DynamicConstructor<T> {
  static dynamic construct(const T& m) {
    return m.toDynamic();
  }
};

template <meta::comms::colltrace::MetadataComponent T>
struct DynamicConverter<T> {
  static T convert(const dynamic& d) {
    return T::fromDynamic();
  }
};

} // namespace folly

/*********** Define std::hash ***********/
namespace std {

template <>
struct hash<meta::comms::colltrace::ICollMetadata> {
  size_t operator()(
      const meta::comms::colltrace::ICollMetadata& metadata) const noexcept {
    return metadata.hash();
  };
};

// Defines the hash function for CommLogData on the fly
template <meta::comms::colltrace::MetadataComponent T>
struct hash<T> {
  size_t operator()(const T& metadataComp) const noexcept {
    return metadataComp.hash();
  }
};

} // namespace std
