// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/variadic/types.hpp"

#include <tuple>

namespace tim
{
namespace mpl
{
namespace impl
{
//======================================================================================//
//
//      filter if predicate evaluates to false (result)
//
//======================================================================================//

template <bool, typename T>
struct filter_if_false_result;

//--------------------------------------------------------------------------------------//

template <typename T>
struct filter_if_false_result<true, T>
{
    using type = std::tuple<T>;
    template <template <typename...> class Op>
    using operation_type = Op<T>;
};

//--------------------------------------------------------------------------------------//

template <typename T>
struct filter_if_false_result<false, T>
{
    using type = std::tuple<>;
    template <template <typename...> class Op>
    using operation_type = std::tuple<>;
};

//======================================================================================//
//
//      filter if predicate evaluates to true (result)
//
//======================================================================================//

template <bool, typename T>
struct filter_if_true_result;

template <typename T>
struct filter_if_true_result<false, T>
{
    using type = std::tuple<T>;
    template <template <typename...> class Op>
    using operation_type = Op<T>;
};

//--------------------------------------------------------------------------------------//

template <typename T>
struct filter_if_true_result<true, T>
{
    using type = std::tuple<>;
    template <template <typename...> class Op>
    using operation_type = std::tuple<>;
};

//======================================================================================//
//
//      filter if predicate evaluates to false (operator)
//
//======================================================================================//

template <template <typename> class Predicate, typename Sequence>
struct filter_if_false;

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_false<Predicate, std::tuple<Ts...>>
{
    using type = tuple_concat_t<
        conditional_t<Predicate<Ts>::value, std::tuple<Ts>, std::tuple<>>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_false<Predicate, type_list<Ts...>>
{
    using type = convert_t<tuple_concat_t<conditional_t<Predicate<Ts>::value,
                                                        std::tuple<Ts>, std::tuple<>>...>,
                           type_list<>>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_false<Predicate, component_list<Ts...>>
{
    using type = convert_t<tuple_concat_t<conditional_t<Predicate<Ts>::value,
                                                        std::tuple<Ts>, std::tuple<>>...>,
                           component_list<>>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_false<Predicate, component_tuple<Ts...>>
{
    using type = convert_t<tuple_concat_t<conditional_t<Predicate<Ts>::value,
                                                        std::tuple<Ts>, std::tuple<>>...>,
                           component_tuple<>>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_false<Predicate, auto_list<Ts...>>
{
    using type = convert_t<tuple_concat_t<conditional_t<Predicate<Ts>::value,
                                                        std::tuple<Ts>, std::tuple<>>...>,
                           auto_list<>>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_false<Predicate, auto_tuple<Ts...>>
{
    using type = convert_t<tuple_concat_t<conditional_t<Predicate<Ts>::value,
                                                        std::tuple<Ts>, std::tuple<>>...>,
                           auto_tuple<>>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename Sequence>
using filter_false = typename filter_if_false<Predicate, Sequence>::type;

//======================================================================================//
//
/// \struct tim::impl::filter_if_false_after_decay
/// \brief Removes types if predicate evaluates to false. Applies
/// decay_t<remove_optional_t<T>> before evaluating predicate
//
//======================================================================================//

template <template <typename> class Predicate, typename Sequence>
struct filter_if_false_after_decay;

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_false_after_decay<Predicate, std::tuple<Ts...>>
{
    using type =
        tuple_concat_t<conditional_t<Predicate<decay_t<remove_optional_t<Ts>>>::value,
                                     std::tuple<Ts>, std::tuple<>>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_false_after_decay<Predicate, type_list<Ts...>>
{
    using type = convert_t<
        tuple_concat_t<conditional_t<Predicate<decay_t<remove_optional_t<Ts>>>::value,
                                     std::tuple<Ts>, std::tuple<>>...>,
        type_list<>>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename Sequence>
using filter_false_after_decay_t =
    typename filter_if_false_after_decay<Predicate, Sequence>::type;

//======================================================================================//
//
//      filter if predicate evaluates to true (operator)
//
//======================================================================================//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_true
{
    using type = tuple_concat_t<
        conditional_t<Predicate<Ts>::value, std::tuple<>, std::tuple<Ts>>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_true<Predicate, std::tuple<Ts...>>
{
    using type = tuple_concat_t<
        conditional_t<Predicate<Ts>::value, std::tuple<>, std::tuple<Ts>>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_true<Predicate, type_list<Ts...>>
{
    using type =
        type_concat_t<conditional_t<Predicate<Ts>::value, type_list<>, type_list<Ts>>...>;
};

template <template <typename> class Predicate, typename Sequence>
using filter_true = typename filter_if_true<Predicate, Sequence>::type;

//======================================================================================//
//
}  // namespace impl

//======================================================================================//

template <template <typename> class Predicate, typename Sequence>
using filter_true_t = impl::filter_true<Predicate, Sequence>;

template <template <typename> class Predicate, typename Sequence>
using filter_false_t = impl::filter_false<Predicate, Sequence>;

///
///  generic alias for extracting all types with a specified trait enabled
///
template <template <typename> class Predicate, typename... Sequence>
struct get_true_types
{
    using type = impl::filter_false<Predicate, std::tuple<Sequence...>>;
};

template <template <typename> class Predicate, typename... Sequence>
struct get_true_types<Predicate, std::tuple<Sequence...>>
{
    using type = impl::filter_false<Predicate, std::tuple<Sequence...>>;
};

template <template <typename> class Predicate, typename... Sequence>
struct get_true_types<Predicate, type_list<Sequence...>>
{
    using type =
        convert_t<impl::filter_false<Predicate, std::tuple<Sequence...>>, type_list<>>;
};

template <template <typename> class Predicate, typename... Sequence>
using get_true_types_t = typename get_true_types<Predicate, Sequence...>::type;

///
///  generic alias for extracting all types with a specified trait disabled
///
template <template <typename> class Predicate, typename... Sequence>
struct get_false_types
{
    using type = impl::filter_true<Predicate, std::tuple<Sequence...>>;
};

template <template <typename> class Predicate, typename... Sequence>
struct get_false_types<Predicate, std::tuple<Sequence...>>
{
    using type = impl::filter_true<Predicate, std::tuple<Sequence...>>;
};

template <template <typename> class Predicate, typename... Sequence>
struct get_false_types<Predicate, type_list<Sequence...>>
{
    using type =
        convert_t<impl::filter_true<Predicate, std::tuple<Sequence...>>, type_list<>>;
};

template <template <typename> class Predicate, typename... Sequence>
using get_false_types_t = typename get_false_types<Predicate, Sequence...>::type;

//======================================================================================//
//
//      determines if storage should be implemented
//
//======================================================================================//

template <typename T>
using non_quirk_t = impl::filter_true<concepts::is_quirk_type, T>;

template <typename T>
using non_placeholder_t = impl::filter_true<concepts::is_placeholder, T>;

/// filter out any types that are not available
template <typename... Types>
using implemented_t =
    impl::filter_false_after_decay_t<trait::is_available, type_list<Types...>>;

template <typename T>
using implemented_list_t = impl::filter_false_after_decay_t<trait::is_available, T>;

template <typename T>
using available_t = impl::filter_false<trait::is_available, T>;

//--------------------------------------------------------------------------------------//
}  // namespace mpl
//
template <typename... T>
using stl_tuple_t = convert_t<mpl::available_t<concat<T...>>, std::tuple<>>;

template <typename... T>
using type_list_t = convert_t<mpl::available_t<concat<T...>>, type_list<>>;

template <typename Tag, typename... T>
using component_bundle_t =
    convert_t<mpl::available_t<type_list<T...>>, component_bundle<Tag>>;

template <typename... T>
using component_tuple_t = convert_t<mpl::available_t<concat<T...>>, component_tuple<>>;

template <typename... T>
using component_list_t = convert_t<mpl::available_t<concat<T...>>, component_list<>>;

template <typename... T>
using auto_tuple_t = convert_t<mpl::available_t<concat<T...>>, auto_tuple<>>;

template <typename... T>
using auto_list_t = convert_t<mpl::available_t<concat<T...>>, auto_list<>>;

template <typename Tag, typename... T>
using auto_bundle_t = convert_t<mpl::available_t<type_list<T...>>, auto_bundle<Tag>>;

template <typename... T>
using lightweight_tuple_t =
    convert_t<mpl::available_t<concat<T...>>, lightweight_tuple<>>;
//
}  // namespace tim
