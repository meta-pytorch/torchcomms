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

#include "timemory/mpl/available.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/transient_function.hpp"
#include "timemory/variadic/types.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tim
{
namespace variadic
{
//
namespace impl
{
template <typename T>
struct add_pointer_if_not
{
    using type = conditional_t<std::is_pointer<T>::value, T, std::add_pointer_t<T>>;
};
//
template <typename... T>
struct add_pointer_if_not<type_list<T...>>
{
    using type = type_list<typename add_pointer_if_not<T>::type...>;
};
//
}  // namespace impl
//
template <typename T>
using add_pointer_if_not_t = typename impl::add_pointer_if_not<T>::type;
//
template <typename... T>
struct heap_wrapper_types
{
    TIMEMORY_DELETED_OBJECT(heap_wrapper_types)

    /// the set of types, unaltered, in a type_list
    using type_list_type = type_list<T...>;

    /// the set of types without any pointers
    using reference_type = type_list<std::remove_pointer_t<T>...>;

    /// type list of the available types
    using available_type = type_list_t<reference_type>;

    /// the original bundle type
    template <typename BundleT>
    using this_type = convert_t<type_list<T...>, BundleT>;

    /// the type after available_t<concat<...>>
    template <typename BundleT>
    using type = convert_t<available_type, BundleT>;

    /// conversion to equivalent wrapper requiring explicit start/stop
    template <typename BundleT>
    using component_type = convert_t<type_list<T...>, BundleT>;

    /// conversion to equivalent wrapper which automatically starts/stops
    template <typename BundleT>
    using auto_type = concepts::auto_type_t<convert_t<type_list_type, BundleT>>;

    /// the valid types to instantiate in a tuple
    template <typename ApiT = TIMEMORY_API>
    using data_type = conditional_t<trait::is_available<ApiT>::value,
                                    convert_t<add_pointer_if_not_t<mpl::non_placeholder_t<
                                                  mpl::non_quirk_t<type_list_t<T...>>>>,
                                              std::tuple<>>,
                                    std::tuple<>>;
};

template <typename T>
struct optional_identity
{
    using type = T;
};

template <typename T>
struct optional_identity<T*>
{
    using type = std::conditional_t<trait::is_available<T>::value, std::optional<T>, T*>;
};

template <typename T>
using optional_identity_t = typename optional_identity<T>::type;

template <typename... T>
struct stack_wrapper_types
{
    TIMEMORY_DELETED_OBJECT(stack_wrapper_types)

    /// the set of types, unaltered, in a type_list
    using type_list_type = type_list<T...>;

    /// the set of types without any pointers
    using reference_type = type_list<std::remove_pointer_t<T>...>;

    /// type list of the available types
    using available_type = type_list_t<reference_type>;

    /// the original bundle type
    template <typename BundleT>
    using this_type = convert_t<type_list_type, BundleT>;

    /// the type after available_t<concat<...>>
    template <typename BundleT>
    using type = convert_t<available_type, BundleT>;

    /// conversion to equivalent wrapper requiring explicit start/stop
    template <typename BundleT>
    using component_type = convert_t<type_list<T...>, BundleT>;

    /// conversion to equivalent wrapper which automatically starts/stops
    template <typename BundleT>
    using auto_type = concepts::auto_type_t<convert_t<type_list_type, BundleT>>;

    /// the valid types to instantiate in a tuple
    template <typename ApiT = TIMEMORY_API>
    using data_type =
        conditional_t<trait::is_available<ApiT>::value,
                      convert_t<mpl::non_placeholder_t<mpl::non_quirk_t<
                                    type_list_t<optional_identity_t<T>...>>>,
                                std::tuple<>>,
                      std::tuple<>>;
};

template <typename... T>
struct mixed_wrapper_types
{
    TIMEMORY_DELETED_OBJECT(mixed_wrapper_types)

    /// the set of types, unaltered, in a type_list
    using type_list_type = type_list<T...>;

    /// the set of types without any pointers
    using reference_type = type_list<std::remove_pointer_t<T>...>;

    /// type list of the available types
    using available_type = type_list_t<std::remove_pointer_t<T>...>;

    /// the original bundle type
    template <typename BundleT>
    using this_type = convert_t<type_list_type, BundleT>;

    /// the type after available_t<concat<...>>
    template <typename BundleT>
    using type = convert_t<type_list_t<T...>, BundleT>;

    /// conversion to equivalent wrapper requiring explicit start/stop
    template <typename BundleT>
    using component_type = convert_t<type_list<T...>, BundleT>;

    /// conversion to equivalent wrapper which automatically starts/stops
    template <typename BundleT>
    using auto_type = concepts::auto_type_t<convert_t<type_list_type, BundleT>>;

    /// the valid types to instantiate in a tuple
    template <typename ApiT = TIMEMORY_API>
    using data_type =
        conditional_t<trait::is_available<ApiT>::value,
                      convert_t<mpl::non_placeholder_t<mpl::non_quirk_t<
                                    type_list_t<optional_identity_t<T>...>>>,
                                std::tuple<>>,
                      std::tuple<>>;
};
//
template <typename... T>
struct heap_wrapper_types<type_list<T...>> : heap_wrapper_types<T...>
{};

template <typename... T>
struct stack_wrapper_types<type_list<T...>> : stack_wrapper_types<T...>
{};

template <typename... T>
struct mixed_wrapper_types<type_list<T...>> : mixed_wrapper_types<T...>
{};
//
template <typename... T>
struct heap_wrapper_types<std::tuple<T...>> : heap_wrapper_types<T...>
{};

template <typename... T>
struct stack_wrapper_types<std::tuple<T...>> : stack_wrapper_types<T...>
{};

template <typename... T>
struct mixed_wrapper_types<std::tuple<T...>> : mixed_wrapper_types<T...>
{};
//
namespace impl
{
struct internal_tag
{};

template <typename... T>
struct bundle;

template <typename C, typename A, typename T>
struct bundle_definitions;

using EmptyT = std::tuple<>;

template <typename U>
using sample_type_t =
    conditional_t<trait::sampler<U>::value, operation::sample<U>, EmptyT>;

template <typename... T>
struct bundle
{
    using tuple_type     = std::tuple<T...>;
    using reference_type = std::tuple<T...>;
    using sample_type    = std::tuple<sample_type_t<T>...>;
    using print_type     = std::tuple<operation::print<T>...>;
};

template <typename... T>
struct bundle<std::tuple<T...>> : bundle<T...>
{};

template <typename... T>
struct bundle<type_list<T...>> : bundle<T...>
{};

template <template <typename...> class CompL, template <typename...> class AutoL,
          template <typename...> class DataL, typename... L, typename... T>
struct bundle_definitions<CompL<L...>, AutoL<L...>, DataL<T...>>
{
    using component_type = CompL<T...>;
    using auto_type      = AutoL<T...>;
};

template <template <typename> class Op, typename TagT, typename... T>
struct generic_operation;

template <template <typename> class Op, typename... T>
struct custom_operation;

template <template <typename> class Op, typename TagT, typename... T>
struct generic_operation
{
    using type =
        std::tuple<operation::generic_operator<remove_pointer_t<T>,
                                               Op<remove_pointer_t<T>>, TagT>...>;
};

template <template <typename> class Op, typename TagT, typename... T>
struct generic_operation<Op, TagT, std::tuple<T...>> : generic_operation<Op, TagT, T...>
{};

template <template <typename> class Op, typename... T>
struct custom_operation
{
    using type = std::tuple<Op<T>...>;
};

template <template <typename> class Op, typename... T>
struct custom_operation<Op, std::tuple<T...>> : custom_operation<Op, T...>
{};

template <typename... U>
struct quirk_config;

template <typename T, typename... F, typename... U>
struct quirk_config<T, type_list<F...>, U...>
{
    static constexpr bool value =
        is_one_of<T, type_list<F..., U...>>::value ||
        is_one_of<T, contains_one_of_t<quirk::is_config, concat<F..., U...>>>::value;
};

template <typename Tp, size_t N, size_t... Idx>
auto
to_tuple_pointer(std::array<Tp, N>& _data, std::index_sequence<Idx...>,
                 enable_if_t<!std::is_pointer<Tp>::value, int> = 0)
{
    return std::make_tuple(std::addressof(std::get<Idx>(_data))...);
}

template <typename Tp, size_t N, size_t... Idx>
auto
to_tuple_pointer(std::array<Tp, N>& _data, std::index_sequence<Idx...>,
                 enable_if_t<std::is_pointer<Tp>::value, int> = 0)
{
    return std::make_tuple(std::get<Idx>(_data)...);
}

TIMEMORY_INLINE
bool
global_enabled()
{
    static bool& _value = settings::enabled();
    return _value;
}

//----------------------------------------------------------------------------------//
//  type is not explicitly listed so redirect to opaque search
//
template <typename ApiT = TIMEMORY_API, template <typename...> class TupleT,
          typename... Types>
void
get(const TupleT<Types...>& m_data, void*& ptr, size_t _hash)
{
    // using data_type = type_list<Types...>;
    // TIMEMORY_FOLD_EXPRESSION(
    //    operation::generic_operator<Types, operation::get<Types>, ApiT>{
    //        std::get<index_of<Types, data_type>::value>(m_data), ptr, _hash });

    using get_type = std::tuple<operation::generic_operator<
        decay_t<remove_optional_t<Types>>,
        operation::get<decay_t<remove_optional_t<Types>>>, ApiT>...>;
    mpl::apply<void>::access<get_type>(m_data, ptr, _hash);
}

//----------------------------------------------------------------------------------//
//  exact type available
//
template <typename U, typename ApiT = TIMEMORY_API, typename DataT,
          typename T = decay_t<U>>
decltype(auto)
get(DataT&& m_data)
{
    using type             = remove_optional_t<T>;
    using data_type        = decay_t<DataT>;
    using unqual_data_type = concepts::unqualified_type_t<data_type>;

    type* _val = nullptr;
    if constexpr(is_one_of<type, data_type>::value)
    {
        _val = &(std::get<index_of<type, data_type>::value>(std::forward<DataT>(m_data)));
    }
    else if constexpr(is_one_of<type*, data_type>::value)
    {
        _val = std::get<index_of<type*, data_type>::value>(std::forward<DataT>(m_data));
    }
    else if constexpr(is_one_of<std::unique_ptr<type>, data_type>::value)
    {
        _val = std::get<index_of<std::unique_ptr<type>, data_type>::value>(
                   std::forward<DataT>(m_data))
                   .get();
    }
    else if constexpr(is_one_of<std::shared_ptr<type>, data_type>::value)
    {
        _val = std::get<index_of<std::shared_ptr<type>, data_type>::value>(
                   std::forward<DataT>(m_data))
                   .get();
    }
    else if constexpr(is_one_of<std::optional<type>, data_type>::value)
    {
        auto& _v = std::get<index_of<std::optional<type>, data_type>::value>(
            std::forward<DataT>(m_data));
        if(_v)
            _val = &*_v;
    }
    else if constexpr(concepts::is_tuple<unqual_data_type>::value)
    {
        void* _ptr = nullptr;
        get(std::forward<DataT>(m_data), _ptr, typeid_hash<type>());
        if(_ptr)
            _val = static_cast<type*>(_ptr);
    }
    else if constexpr(concepts::is_wrapper<unqual_data_type>::value)
    {
        void* _ptr = nullptr;
        std::forward<DataT>(m_data).get(_val, typeid_hash<type>());
        if(_ptr)
            _val = static_cast<type*>(_ptr);
    }
    else
    {
        static_assert(std::is_empty<DataT>::value, "Error! unsupported data type");
    }

    return _val;
}

//----------------------------------------------------------------------------------//
/// this is a simple alternative to get<T>() when used from SFINAE in operation
/// namespace which has a struct get also templated. Usage there can cause error
/// with older compilers
template <typename U, typename data_type, typename T = std::remove_pointer_t<decay_t<U>>>
auto
get_component(
    data_type&& m_data,
    enable_if_t<trait::is_available<T>::value && is_one_of<T, decay_t<data_type>>::value,
                int> = 0)
{
    return get<T>(std::forward<data_type>(m_data));
}

template <typename U, typename data_type, typename T = std::remove_pointer_t<decay_t<U>>>
auto
get_component(
    data_type&& m_data,
    enable_if_t<trait::is_available<T>::value && is_one_of<T*, decay_t<data_type>>::value,
                int> = 0)
{
    return get<T>(std::forward<data_type>(m_data));
}

template <typename U, typename data_type, typename T = remove_optional_t<decay_t<U>>>
auto
get_component(data_type&& m_data,
              enable_if_t<trait::is_available<T>::value &&
                              is_one_of<std::optional<T>, decay_t<data_type>>::value,
                          int> = 0)
{
    return get<std::optional<T>>(std::forward<data_type>(m_data));
}

/// returns a reference from a stack component instead of a pointer
template <typename U, typename data_type, typename T = std::remove_pointer_t<decay_t<U>>>
auto&
get_reference(
    data_type& m_data,
    enable_if_t<trait::is_available<T>::value && is_one_of<T, decay_t<data_type>>::value,
                int> = 0)
{
    return std::get<index_of<T, decay_t<data_type>>::value>(m_data);
}

/// returns a reference from a heap component instead of a pointer
template <typename U, typename data_type, typename T = std::remove_pointer_t<decay_t<U>>>
auto&
get_reference(
    data_type& m_data,
    enable_if_t<trait::is_available<T>::value && is_one_of<T*, decay_t<data_type>>::value,
                int> = 0)
{
    return std::get<index_of<T*, decay_t<data_type>>::value>(m_data);
}

/// returns a reference from a heap component instead of a pointer
template <typename U, typename data_type, typename T = remove_optional_t<decay_t<U>>>
auto&
get_reference(data_type& m_data,
              enable_if_t<trait::is_available<T>::value &&
                              is_one_of<std::optional<T>, decay_t<data_type>>::value,
                          int> = 0)
{
    return std::get<index_of<std::optional<T>, decay_t<data_type>>::value>(m_data);
}

}  // namespace impl
}  // namespace variadic
}  // namespace tim
