// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <type_traits>
#include <variant>

// Check if a type is in a variant type. Got the code from:
// https://stackoverflow.com/questions/45892170/how-do-i-check-if-an-stdvariant-can-hold-a-certain-type
template <typename T, typename VARIANT_T>
struct isVariantMember;

template <typename T, typename... ALL_T>
struct isVariantMember<T, std::variant<ALL_T...>>
    : public std::disjunction<std::is_same<T, ALL_T>...> {};

template <typename VariantType, typename T, std::size_t index = 0>
constexpr std::size_t variant_index() {
  static_assert(
      std::variant_size_v<VariantType> > index, "Type not found in variant");
  if constexpr (index == std::variant_size_v<VariantType>) {
    return index;
  } else if constexpr (std::is_same_v<
                           std::variant_alternative_t<index, VariantType>,
                           T>) {
    return index;
  } else {
    return variant_index<VariantType, T, index + 1>();
  }
}
