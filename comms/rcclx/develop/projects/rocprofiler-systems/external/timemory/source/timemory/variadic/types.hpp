//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#pragma once

#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/macros.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

///
/// \macro TSTAG
/// \brief for tuple_size overloads, clang uses 'class tuple_size' while GCC uses
/// 'struct tuple_size'... which results in a lot of mismatches-tag warnings
///
#if !defined(TSTAG)
#    if defined(TIMEMORY_CLANG_COMPILER)
#        define TSTAG(X) class
#    else
#        define TSTAG(X) X
#    endif
#endif

//======================================================================================//
//
namespace tim
{
//--------------------------------------------------------------------------------------//
//
//  Forward declaration of variadic wrapper types
//
//--------------------------------------------------------------------------------------//

template <typename... Types>
class lightweight_tuple;

template <typename... Types>
class component_bundle;

template <typename... Types>
class component_tuple;

template <typename... Types>
class component_list;

template <typename... Types>
class auto_base_bundle;

template <typename... Types>
class auto_bundle;

template <typename... Types>
class auto_tuple;

template <typename... Types>
class auto_list;

template <typename TupleT, typename ListT>
class component_hybrid;

template <typename TupleT, typename ListT>
class auto_hybrid;

//
// actual definition of various bundles
//
template <typename Tag, typename CompT, typename BundleT>
class auto_base_bundle<Tag, CompT, BundleT>;

template <typename ApiT, typename... Types>
class auto_bundle<ApiT, Types...>;

template <typename ApiT, typename... Types>
class component_bundle<ApiT, Types...>;

//
//  concepts for conversion
//
namespace concepts
{
//
template <typename... Types>
struct component_type<auto_tuple<Types...>>
{
    using type = component_tuple<Types...>;
};
//
template <typename... Types>
struct component_type<auto_list<Types...>>
{
    using type = component_list<Types...>;
};
//
template <typename... Types>
struct component_type<auto_bundle<Types...>>
{
    using type = component_bundle<Types...>;
};
//
template <typename... Types>
struct component_type<auto_hybrid<Types...>>
{
    using type = component_hybrid<Types...>;
};
//
}  // namespace concepts
//
namespace mpl
{
#if !defined(CXX17)
template <typename F, typename... Args>
struct is_invocable
: std::is_constructible<std::function<void(Args...)>,
                        std::reference_wrapper<std::remove_reference_t<F>>>
{};

template <typename R, typename F, typename... Args>
struct is_invocable_r
: std::is_constructible<std::function<R(Args...)>,
                        std::reference_wrapper<std::remove_reference_t<F>>>
{};
#else
template <typename F, typename... Args>
using is_invocable = std::is_invocable<F, Args...>;

template <typename R, typename F, typename... Args>
using is_invocable_r = std::is_invocable<R, F, Args...>;
#endif
/// \class execution_handler
/// \tparam BundleT A component bundler, e.g. component_bundle
/// \tparam DataT The data type returned from a function that was executed inside
/// the chained member functions calls of BundleT
///
/// \brief This is an intermediate type that permits operations such as:
///
/// \code{.cpp}
/// long fibonacci(long);
///
/// long run(long n)
/// {
///     using bundle_t = tim::component_tuple<wall_clock>;
///
///     return bundle_t{ "run" }.start().execute(fibonacci, n).stop().return_result();
/// }
///
/// long fibonacci(long n)
/// {
///     using bundle_t = tim::component_tuple<wall_clock>;
///
///     return (n < 2) ? n :
/// }
/// \endcode
template <typename BundleT, typename DataT>
class execution_handler;

template <typename BundleT, typename FuncT, typename... Args>
auto
execute(BundleT&& _bundle, FuncT&& _func, Args&&... _args,
        enable_if_t<is_invocable<FuncT, Args...>::value &&
                        !std::is_void<std::result_of_t<FuncT(Args...)>>::value,
                    int> = 0);
//
template <typename BundleT, typename FuncT, typename... Args>
auto
execute(BundleT&& _bundle, FuncT&& _func, Args&&... _args,
        enable_if_t<is_invocable<FuncT, Args...>::value &&
                        std::is_void<std::result_of_t<FuncT(Args...)>>::value,
                    int> = 0);
//
//
template <typename BundleT, typename ValueT>
auto
execute(BundleT&& _bundle, ValueT&& _value,
        enable_if_t<!is_invocable<ValueT>::value, long> = 0);
//
}  // namespace mpl
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//                                  IS VARIADIC / IS WRAPPER
//
//--------------------------------------------------------------------------------------//
// these are variadic types used to bundle components together
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_variadic, std::tuple, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_variadic, type_list, true_type, typename)

// there are timemory-specific variadic wrappers
#if defined(TIMEMORY_USE_DEPRECATED)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_wrapper, auto_hybrid, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_wrapper, component_hybrid, true_type, typename)
#endif

// {auto,component}_bundle are empty if one template is supplied
TIMEMORY_DEFINE_TEMPLATE_CONCEPT(is_empty, auto_bundle, true_type, typename)
TIMEMORY_DEFINE_TEMPLATE_CONCEPT(is_empty, component_bundle, true_type, typename)

//======================================================================================//

TIMEMORY_DEFINE_VARIADIC_CONCEPT_TYPE(tuple_type, std::tuple, typename, std::tuple<T...>)
TIMEMORY_DEFINE_VARIADIC_CONCEPT_TYPE(auto_type, std::tuple, typename, auto_bundle<T...>)
TIMEMORY_DEFINE_VARIADIC_CONCEPT_TYPE(component_type, std::tuple, typename,
                                      component_bundle<T...>)

TIMEMORY_DEFINE_VARIADIC_CONCEPT_TYPE(tuple_type, type_list, typename, std::tuple<T...>)
TIMEMORY_DEFINE_VARIADIC_CONCEPT_TYPE(auto_type, type_list, typename, auto_bundle<T...>)
TIMEMORY_DEFINE_VARIADIC_CONCEPT_TYPE(component_type, type_list, typename,
                                      component_bundle<T...>)

//======================================================================================//

namespace tim
{
namespace concepts
{
template <typename T, typename... Types>
struct tuple_type<component_bundle<T, Types...>>
{
    using type = std::tuple<Types...>;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Types>
struct tuple_type<auto_bundle<T, Types...>>
{
    using type = std::tuple<Types...>;
};
}  // namespace concepts
}  // namespace tim
//======================================================================================//

namespace tim
{
//--------------------------------------------------------------------------------------//
//
//      convert all variadic wrappers into type lists
//
//--------------------------------------------------------------------------------------//
//
//              Final result
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
struct type_list_only
{
    using type = type_list<Types...>;
};
//
//--------------------------------------------------------------------------------------//
//
//              Second to last result
//
//--------------------------------------------------------------------------------------//
//
template <typename... Lhs, typename... Rhs>
struct type_list_only<type_list<Lhs...>, type_list<Rhs...>>
{
    using type = typename type_list_only<Lhs..., Rhs...>::type;
};
//
//--------------------------------------------------------------------------------------//
//
//              Encountered variadic while reducing but tail still exists
//
//--------------------------------------------------------------------------------------//
//
template <typename... Lhs, typename... Types, typename... Rhs>
struct type_list_only<type_list<Lhs...>, std::tuple<Types...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Lhs..., Types...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

template <typename... Lhs, typename... Types, typename... Rhs>
struct type_list_only<type_list<Lhs...>, component_tuple<Types...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Lhs..., Types...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

template <typename... Lhs, typename... Types, typename... Rhs>
struct type_list_only<type_list<Lhs...>, component_list<Types...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Lhs..., Types...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

#if defined(TIMEMORY_USE_DEPRECATED)
template <typename Tup, typename Lst, typename... Lhs, typename... Rhs>
struct type_list_only<type_list<Lhs...>, component_hybrid<Tup, Lst>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Lhs..., Tup, Lst>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};
#endif

template <typename... Lhs, typename... Types, typename... Rhs>
struct type_list_only<type_list<Lhs...>, auto_tuple<Types...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Lhs..., Types...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

template <typename... Lhs, typename... Types, typename... Rhs>
struct type_list_only<type_list<Lhs...>, auto_list<Types...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Lhs..., Types...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

#if defined(TIMEMORY_USE_DEPRECATED)
template <typename Tup, typename Lst, typename... Lhs, typename... Rhs>
struct type_list_only<type_list<Lhs...>, auto_hybrid<Tup, Lst>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Lhs..., Tup, Lst>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};
#endif
//
//--------------------------------------------------------------------------------------//
//
//              Listed first
//
//--------------------------------------------------------------------------------------//
//
template <typename... Lhs, typename... Rhs>
struct type_list_only<type_list<Lhs...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Lhs...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

template <typename... Types, typename... Rhs>
struct type_list_only<std::tuple<Types...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Types...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

template <typename... Types, typename... Rhs>
struct type_list_only<component_tuple<Types...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Types...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

template <typename... Types, typename... Rhs>
struct type_list_only<component_list<Types...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Types...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

template <typename... Types, typename... Rhs>
struct type_list_only<auto_tuple<Types...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Types...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

template <typename... Types, typename... Rhs>
struct type_list_only<auto_list<Types...>, Rhs...>
{
    using type = typename type_list_only<typename type_list_only<Types...>::type,
                                         typename type_list_only<Rhs...>::type>::type;
};

#if defined(TIMEMORY_USE_DEPRECATED)
template <typename Tup, typename Lst, typename... Rhs>
struct type_list_only<component_hybrid<Tup, Lst>, Rhs...>
{
    using tup_types = typename type_list_only<Tup>::type;
    using lst_types = typename type_list_only<Lst>::type;
    using type      = component_hybrid<convert_t<tup_types, component_tuple<>>,
                                  convert_t<lst_types, component_list<>>>;
};

template <typename Tup, typename Lst, typename... Rhs>
struct type_list_only<auto_hybrid<Tup, Lst>, Rhs...>
{
    using tup_types = typename type_list_only<Tup>::type;
    using lst_types = typename type_list_only<Lst>::type;
    using type      = auto_hybrid<convert_t<tup_types, component_tuple<>>,
                             convert_t<lst_types, component_list<>>>;
};
#endif
//
//--------------------------------------------------------------------------------------//
//
//              Concatenation of variadic wrappers
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
template <typename... Types>
struct concat
{
    using type = std::tuple<Types...>;
};

template <typename... Types>
struct concat<type_list<Types...>>
{
    using type = std::tuple<Types...>;
};

template <typename... Types>
struct concat<std::tuple<Types...>>
{
    using type = std::tuple<Types...>;
};

template <typename... Types>
struct concat<component_tuple<Types...>>
{
    using type = typename concat<Types...>::type;
};

template <typename... Types>
struct concat<component_list<Types...>>
{
    using type = typename concat<Types...>::type;
};

template <typename... Types>
struct concat<component_list<Types*...>> : concat<component_list<Types...>>
{};

template <typename... Types>
struct concat<auto_tuple<Types...>>
{
    using type = typename concat<Types...>::type;
};

template <typename... Types>
struct concat<auto_list<Types...>>
{
    using type = typename concat<Types...>::type;
};

template <typename... Types>
struct concat<auto_list<Types*...>> : concat<auto_list<Types...>>
{};

template <typename... Lhs, typename... Rhs>
struct concat<std::tuple<Lhs...>, std::tuple<Rhs...>>
{
    using type = typename concat<Lhs..., Rhs...>::type;
};

//--------------------------------------------------------------------------------------//
//      component_hybrid
//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_DEPRECATED)
template <typename... TupTypes, typename... LstTypes>
struct concat<component_hybrid<std::tuple<TupTypes...>, std::tuple<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<component_hybrid<component_tuple<TupTypes...>, std::tuple<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<component_hybrid<std::tuple<TupTypes...>, component_list<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<component_hybrid<type_list<TupTypes...>, type_list<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<component_hybrid<component_tuple<TupTypes...>, type_list<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<component_hybrid<type_list<TupTypes...>, component_list<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

//--------------------------------------------------------------------------------------//
//      auto_hybrid
//--------------------------------------------------------------------------------------//

template <typename... TupTypes, typename... LstTypes>
struct concat<auto_hybrid<std::tuple<TupTypes...>, std::tuple<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<auto_hybrid<component_tuple<TupTypes...>, std::tuple<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<auto_hybrid<std::tuple<TupTypes...>, component_list<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<auto_hybrid<type_list<TupTypes...>, type_list<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<auto_hybrid<component_tuple<TupTypes...>, type_list<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<auto_hybrid<type_list<TupTypes...>, component_list<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};
#endif
//--------------------------------------------------------------------------------------//
//
//      Combine
//
//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename... Rhs>
struct concat<std::tuple<Lhs...>, Rhs...>
{
    using type = typename concat<typename concat<Lhs...>::type,
                                 typename concat<Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      component_tuple
//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename... Rhs>
struct concat<component_tuple<Lhs...>, Rhs...>
{
    using type = typename concat<typename concat<Lhs...>::type,
                                 typename concat<Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      component_list
//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename... Rhs>
struct concat<component_list<Lhs...>, Rhs...>
{
    using type = typename concat<typename concat<Lhs...>::type,
                                 typename concat<Rhs...>::type>::type;
};

template <typename... Lhs, typename... Rhs>
struct concat<component_list<Lhs...>*, Rhs*...>
: public concat<component_list<Lhs...>, Rhs...>
{};

template <typename... Lhs, typename... Rhs>
struct concat<component_list<Lhs...>*, Rhs...>
: public concat<component_list<Lhs...>, Rhs...>
{};

//--------------------------------------------------------------------------------------//
//      auto_tuple
//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename... Rhs>
struct concat<auto_tuple<Lhs...>, Rhs...>
{
    using type = typename concat<typename concat<Lhs...>::type,
                                 typename concat<Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      auto_list
//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename... Rhs>
struct concat<auto_list<Lhs...>, Rhs...>
{
    using type = typename concat<typename concat<Lhs...>::type,
                                 typename concat<Rhs...>::type>::type;
};

template <typename... Lhs, typename... Rhs>
struct concat<auto_list<Lhs...>*, Rhs*...> : public concat<auto_list<Lhs...>, Rhs...>
{};

template <typename... Lhs, typename... Rhs>
struct concat<auto_list<Lhs...>*, Rhs...> : public concat<auto_list<Lhs...>, Rhs...>
{};

//--------------------------------------------------------------------------------------//
//      component_hybrid
//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_DEPRECATED)
template <typename Tup, typename Lst, typename... Rhs, typename... Tail>
struct concat<component_hybrid<Tup, Lst>, component_tuple<Rhs...>, Tail...>
{
    using type =
        typename concat<component_hybrid<typename concat<Tup, Rhs...>::type, Lst>,
                        Tail...>::type;
    using tuple_type = typename type::tuple_t;
    using list_type  = typename type::list_t;
};

template <typename Tup, typename Lst, typename... Rhs, typename... Tail>
struct concat<component_hybrid<Tup, Lst>, component_list<Rhs...>, Tail...>
{
    using type =
        typename concat<component_hybrid<Tup, typename concat<Lst, Rhs...>::type>,
                        Tail...>::type;
    using tuple_type = typename type::tuple_t;
    using list_type  = typename type::list_t;
};

//--------------------------------------------------------------------------------------//
//      auto_hybrid
//--------------------------------------------------------------------------------------//

template <typename Tup, typename Lst, typename... Rhs, typename... Tail>
struct concat<auto_hybrid<Tup, Lst>, component_tuple<Rhs...>, Tail...>
{
    using type = typename concat<auto_hybrid<typename concat<Tup, Rhs...>::type, Lst>,
                                 Tail...>::type;
    using tuple_type = typename type::tuple_t;
    using list_type  = typename type::list_t;
};

template <typename Tup, typename Lst, typename... Rhs, typename... Tail>
struct concat<auto_hybrid<Tup, Lst>, component_list<Rhs...>, Tail...>
{
    using type = typename concat<auto_hybrid<Tup, typename concat<Lst, Rhs...>::type>,
                                 Tail...>::type;
    using tuple_type = typename type::tuple_t;
    using list_type  = typename type::list_t;
};
#endif
}  // namespace impl

template <typename... Types>
using concat = typename impl::concat<Types...>::type;

// template <typename... Types>
// using concat = typename type_list_only<Types...>::type;

}  // namespace tim

//======================================================================================//

namespace std
{
//
//--------------------------------------------------------------------------------------//
//
//   Forward declare intent to define these once all the type-traits have been set
//                    after including "timemory/components/types.hpp"
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
TSTAG(struct)
tuple_size<tim::lightweight_tuple<Types...>>;

template <typename Tag, typename... Types>
TSTAG(struct)
tuple_size<tim::component_bundle<Tag, Types...>>;

template <typename... Types>
TSTAG(struct)
tuple_size<tim::component_tuple<Types...>>;

template <typename... Types>
TSTAG(struct)
tuple_size<tim::component_list<Types...>>;

template <typename... Types>
TSTAG(struct)
tuple_size<tim::auto_bundle<Types...>>;

template <typename... Types>
TSTAG(struct)
tuple_size<tim::auto_tuple<Types...>>;

template <typename... Types>
TSTAG(struct)
tuple_size<tim::auto_list<Types...>>;

#if defined(TIMEMORY_USE_DEPRECATED)
template <typename TupleT, typename ListT>
TSTAG(struct)
tuple_size<tim::component_hybrid<TupleT, ListT>>;

template <typename TupleT, typename ListT>
TSTAG(struct)
tuple_size<tim::auto_hybrid<TupleT, ListT>>;
#endif
//
//--------------------------------------------------------------------------------------//
//
}  // namespace std
