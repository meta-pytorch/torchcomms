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

#ifndef TIMEMORY_VARIADIC_FUNCTIONAL_CPP_
#define TIMEMORY_VARIADIC_FUNCTIONAL_CPP_

#include "timemory/api.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/operations/types/cache.hpp"
#include "timemory/operations/types/generic.hpp"
#include "timemory/utility/types.hpp"

#include <functional>   // std::ref
#include <ostream>      // std::ostream
#include <tuple>        // std::get, std::tuple
#include <type_traits>  // std::remove_pointer_t
#include <utility>      // std::forward

namespace tim
{
namespace invoke
{
namespace invoke_impl
{
//
//--------------------------------------------------------------------------------------//
//
template <template <typename> class OpT, typename Tag,
          template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
invoke(TupleT<Tp...>& _obj, Args&&... _args)
{
    using data_type = std::tuple<decay_t<Tp>...>;
    TIMEMORY_FOLD_EXPRESSION(
        operation::generic_operator<remove_optional_t<decay_t<Tp>>,
                                    OpT<remove_optional_t<decay_t<Tp>>>, Tag>(
            std::get<index_of<decay_t<Tp>, data_type>::value>(_obj),
            std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename, typename> class OpT, typename Tag,
          template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
invoke(TupleT<Tp...>& _obj, Args&&... _args)
{
    using data_type = std::tuple<decay_t<Tp>...>;
    TIMEMORY_FOLD_EXPRESSION(
        operation::generic_operator<remove_optional_t<decay_t<Tp>>,
                                    OpT<remove_optional_t<decay_t<Tp>>, Tag>, Tag>(
            std::get<index_of<decay_t<Tp>, data_type>::value>(_obj),
            std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename> class OpT, typename Tag,
          template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
invoke(TupleT<Tp&...>&& _obj, Args&&... _args)
{
    using data_type = std::tuple<decay_t<Tp>...>;
    TIMEMORY_FOLD_EXPRESSION(
        operation::generic_operator<remove_optional_t<decay_t<Tp>>,
                                    OpT<remove_optional_t<decay_t<Tp>>>, Tag>(
            std::get<index_of<decay_t<Tp>, data_type>::value>(_obj),
            std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename, typename> class OpT, typename Tag,
          template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
invoke(TupleT<Tp&...>&& _obj, Args&&... _args)
{
    using data_type = std::tuple<decay_t<Tp>...>;
    TIMEMORY_FOLD_EXPRESSION(
        operation::generic_operator<remove_optional_t<decay_t<Tp>>,
                                    OpT<remove_optional_t<decay_t<Tp>>, Tag>, Tag>(
            std::get<index_of<decay_t<Tp>, data_type>::value>(_obj),
            std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename> class OpT, typename Tag,
          template <typename...> class TupleT, typename... Tp,
          template <typename...> class ValueT, typename... Vp, typename... Args>
TIMEMORY_INLINE void
invoke_data(TupleT<Tp...>& _obj, ValueT<Vp...>& _val, Args&&... _args)
{
    using data_type = std::tuple<decay_t<Tp>...>;
    TIMEMORY_FOLD_EXPRESSION(
        operation::generic_operator<remove_optional_t<decay_t<Tp>>,
                                    OpT<remove_optional_t<decay_t<Tp>>>, Tag>(
            std::get<index_of<decay_t<Tp>, data_type>::value>(_obj),
            std::get<index_of<decay_t<Tp>, data_type>::value>(_val),
            std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename> class OpT, typename Tag,
          template <typename...> class TupleT, typename... Tp,
          template <typename...> class ValueT, typename... Vp, typename... Args>
TIMEMORY_INLINE void
invoke_data(TupleT<Tp&...>&& _obj, ValueT<Vp...>& _val, Args&&... _args)
{
    using data_type = std::tuple<decay_t<Tp>...>;
    TIMEMORY_FOLD_EXPRESSION(
        operation::generic_operator<remove_optional_t<decay_t<Tp>>,
                                    OpT<remove_optional_t<decay_t<Tp>>>, Tag>(
            std::get<index_of<decay_t<Tp>, data_type>::value>(_obj),
            std::get<index_of<decay_t<Tp>, data_type>::value>(_val),
            std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
construct(TupleT<Tp...>& _obj, Args&&... _args)
{
    using data_type = std::tuple<decay_t<Tp>...>;
    TIMEMORY_FOLD_EXPRESSION(
        std::get<index_of<decay_t<Tp>, data_type>::value>(_obj) =
            operation::construct<Tp>{}(std::forward<Args>(_args)...));
}
//
}  // namespace invoke_impl
//
//======================================================================================//
//
template <typename... Args>
void
print(std::ostream& os, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(os << args << "\n");
}
//
template <typename... Args>
void
print(std::ostream& os, const std::string& delim, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(os << args << delim);
}
//
//--------------------------------------------------------------------------------------//
//                                  invoke
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class OpT, typename ApiT,
          template <typename...> class TupleT, typename... Tp, typename... Args>
void
invoke(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<OpT, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class OpT, template <typename...> class TupleT,
          typename... Tp, typename... Args>
void
invoke(TupleT<Tp...>& obj, Args&&... args)
{
    invoke<OpT, TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class OpT, typename ApiT,
          template <typename...> class TupleT, typename... Tp, typename... Args>
void
invoke(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<OpT, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                   std::forward<Args>(args)...);
}
//
template <template <typename...> class OpT, template <typename...> class TupleT,
          typename... Tp, typename... Args>
void
invoke(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke<OpT, TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj),
                              std::forward<Args>(args)...);
}
//
template <template <typename...> class OpT, typename ApiT, typename... Up,
          template <typename...> class TupleT, typename... Tp, typename... Args>
void
invoke(mpl::piecewise_select<Up...>, TupleT<Tp...>& obj, Args&&... args)
{
    using data_type = TupleT<Tp...>;
    invoke_impl::invoke<OpT, ApiT>(
        std::forward_as_tuple(std::get<index_of<Up, data_type>::value>(obj)...),
        std::forward<Args>(args)...);
}
//
template <template <typename...> class OpT, typename ApiT, typename... Up,
          template <typename...> class TupleT, typename... Tp, typename... Args>
void
invoke(mpl::piecewise_select<Up...>, TupleT<Tp&...>& obj, Args&&... args)
{
    using data_type = TupleT<Tp...>;
    invoke_impl::invoke<OpT, ApiT>(
        std::tie(std::get<index_of<Up, data_type>::value>(obj)...),
        std::forward<Args>(args)...);
}
//
template <template <typename...> class OpT, typename ApiT, typename... Up,
          template <typename...> class TupleT, typename... Tp, typename... Args>
void
invoke(mpl::piecewise_ignore<Up...>, TupleT<Tp...>& obj, Args&&... args)
{
    using data_type = mpl::subtract_t<type_list<Tp...>, type_list<Up...>>;
    invoke(convert_t<data_type, mpl::piecewise_select<>>{}, obj,
           std::forward<Args>(args)...);
}
//
template <template <typename...> class OpT, typename ApiT, typename... Up,
          template <typename...> class TupleT, typename... Tp, typename... Args>
void
invoke(mpl::piecewise_ignore<Up...>, TupleT<Tp&...>& obj, Args&&... args)
{
    using data_type = mpl::subtract_t<type_list<Tp...>, type_list<Up...>>;
    invoke(convert_t<data_type, mpl::piecewise_select<>>{}, obj,
           std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  construct
//--------------------------------------------------------------------------------------//
//
template <typename TupleT, typename ApiT, typename... Args>
auto
construct(Args&&... args)
{
    IF_CONSTEXPR(trait::is_available<ApiT>::value)
    {
        TupleT obj{};
        invoke_impl::construct(std::ref(obj).get(), std::forward<Args>(args)...);
        return obj;
    }
    return TupleT{};
}
//
//
template <typename TupleT, typename... Args>
auto
construct(Args&&... args)
{
    return construct<TupleT, TIMEMORY_API>(std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  destroy
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp>
auto
destroy(TupleT<Tp...>& obj)
{
    invoke_impl::invoke<operation::generic_deleter, ApiT>(obj);
}
//
template <template <typename...> class TupleT, typename... Tp>
auto
destroy(TupleT<Tp...>& obj)
{
    destroy<TIMEMORY_API>(obj);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp>
auto
destroy(TupleT<Tp&...>&& obj)
{
    invoke_impl::invoke<operation::generic_deleter, ApiT>(
        std::forward<TupleT<Tp&...>>(obj));
}
//
template <template <typename...> class TupleT, typename... Tp>
auto
destroy(TupleT<Tp&...>&& obj)
{
    destroy<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj));
}
//
//--------------------------------------------------------------------------------------//
//                                  start
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
start(TupleT<Tp...>& obj, Args&&... args)
{
    using data_type        = std::tuple<remove_optional_t<decay_t<Tp>>...>;
    using priority_types_t = mpl::filter_false_t<mpl::negative_start_priority, data_type>;
    using priority_tuple_t = mpl::sort<trait::start_priority, priority_types_t>;
    using delayed_types_t  = mpl::filter_false_t<mpl::positive_start_priority, data_type>;
    using delayed_tuple_t  = mpl::sort<trait::start_priority, delayed_types_t>;

    // start high priority components
    auto&& _priority_start = mpl::get_reference_tuple<priority_tuple_t>(obj);
    invoke_impl::invoke<operation::priority_start, ApiT>(_priority_start,
                                                         std::forward<Args>(args)...);
    // start non-prioritized components
    invoke_impl::invoke<operation::standard_start, ApiT>(obj,
                                                         std::forward<Args>(args)...);
    // start low prioritized components
    auto&& _delayed_start = mpl::get_reference_tuple<delayed_tuple_t>(obj);
    invoke_impl::invoke<operation::delayed_start, ApiT>(_delayed_start,
                                                        std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
start(TupleT<Tp...>& obj, Args&&... args)
{
    start<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
start(TupleT<Tp&...>&& obj, Args&&... args)
{
    using data_type        = std::tuple<remove_optional_t<decay_t<Tp>>...>;
    using priority_types_t = mpl::filter_false_t<mpl::negative_start_priority, data_type>;
    using priority_tuple_t = mpl::sort<trait::start_priority, priority_types_t>;
    using delayed_types_t  = mpl::filter_false_t<mpl::positive_start_priority, data_type>;
    using delayed_tuple_t  = mpl::sort<trait::start_priority, delayed_types_t>;

    // start high priority components
    auto&& _priority_start =
        mpl::get_reference_tuple<priority_tuple_t>(std::forward<TupleT<Tp&...>>(obj));
    invoke_impl::invoke<operation::priority_start, ApiT>(_priority_start,
                                                         std::forward<Args>(args)...);

    // start non-prioritized components
    invoke_impl::invoke<operation::standard_start, ApiT>(
        std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);

    // start low prioritized components
    auto&& _delayed_start =
        mpl::get_reference_tuple<delayed_tuple_t>(std::forward<TupleT<Tp&...>>(obj));
    invoke_impl::invoke<operation::delayed_start, ApiT>(_delayed_start,
                                                        std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
start(TupleT<Tp&...>&& obj, Args&&... args)
{
    start<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  stop
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
stop(TupleT<Tp...>& obj, Args&&... args)
{
    using data_type        = std::tuple<remove_optional_t<decay_t<Tp>>...>;
    using priority_types_t = mpl::filter_false_t<mpl::negative_stop_priority, data_type>;
    using priority_tuple_t = mpl::sort<trait::stop_priority, priority_types_t>;
    using delayed_types_t  = mpl::filter_false_t<mpl::positive_stop_priority, data_type>;
    using delayed_tuple_t  = mpl::sort<trait::stop_priority, delayed_types_t>;

    // stop high priority components
    auto&& _priority_stop = mpl::get_reference_tuple<priority_tuple_t>(obj);
    invoke_impl::invoke<operation::priority_stop, ApiT>(_priority_stop,
                                                        std::forward<Args>(args)...);

    // stop non-prioritized components
    invoke_impl::invoke<operation::standard_stop, ApiT>(obj, std::forward<Args>(args)...);

    // stop low prioritized components
    auto&& _delayed_stop = mpl::get_reference_tuple<delayed_tuple_t>(obj);
    invoke_impl::invoke<operation::delayed_stop, ApiT>(_delayed_stop,
                                                       std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
stop(TupleT<Tp...>& obj, Args&&... args)
{
    stop<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
stop(TupleT<Tp&...>&& obj, Args&&... args)
{
    using data_type        = std::tuple<remove_optional_t<decay_t<Tp>>...>;
    using priority_types_t = mpl::filter_false_t<mpl::negative_stop_priority, data_type>;
    using priority_tuple_t = mpl::sort<trait::stop_priority, priority_types_t>;
    using delayed_types_t  = mpl::filter_false_t<mpl::positive_stop_priority, data_type>;
    using delayed_tuple_t  = mpl::sort<trait::stop_priority, delayed_types_t>;

    // stop high priority components
    auto&& _priority_stop =
        mpl::get_reference_tuple<priority_tuple_t>(std::forward<TupleT<Tp&...>>(obj));
    invoke_impl::invoke<operation::priority_stop, ApiT>(_priority_stop,
                                                        std::forward<Args>(args)...);

    // stop non-prioritized components
    invoke_impl::invoke<operation::standard_stop, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                        std::forward<Args>(args)...);

    // stop low prioritized components
    auto&& _delayed_stop =
        mpl::get_reference_tuple<delayed_tuple_t>(std::forward<TupleT<Tp&...>>(obj));
    invoke_impl::invoke<operation::delayed_stop, ApiT>(_delayed_stop,
                                                       std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
stop(TupleT<Tp&...>&& obj, Args&&... args)
{
    stop<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  mark
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
mark(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::mark, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
mark(TupleT<Tp...>& obj, Args&&... args)
{
    mark<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
mark(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::mark, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                               std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
mark(TupleT<Tp&...>&& obj, Args&&... args)
{
    mark<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  mark_begin
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
mark_begin(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::mark_begin, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
mark_begin(TupleT<Tp...>& obj, Args&&... args)
{
    mark_begin<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
mark_begin(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::mark_begin, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                     std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
mark_begin(TupleT<Tp&...>&& obj, Args&&... args)
{
    mark_begin<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj),
                             std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  mark_end
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
mark_end(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::mark_end, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
mark_end(TupleT<Tp...>& obj, Args&&... args)
{
    mark_end<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
mark_end(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::mark_end, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                   std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
mark_end(TupleT<Tp&...>&& obj, Args&&... args)
{
    mark_end<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj),
                           std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  store
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
store(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::store, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
store(TupleT<Tp...>& obj, Args&&... args)
{
    store<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
store(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::store, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
store(TupleT<Tp&...>&& obj, Args&&... args)
{
    store<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  reset
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
reset(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::reset, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
reset(TupleT<Tp...>& obj, Args&&... args)
{
    reset<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
reset(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::reset, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
reset(TupleT<Tp&...>&& obj, Args&&... args)
{
    reset<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  record
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
record(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::record, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
record(TupleT<Tp...>& obj, Args&&... args)
{
    record<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
record(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::record, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                 std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
record(TupleT<Tp&...>&& obj, Args&&... args)
{
    record<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  measure
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
measure(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::measure, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
measure(TupleT<Tp...>& obj, Args&&... args)
{
    measure<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
measure(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::measure, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                  std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
measure(TupleT<Tp&...>&& obj, Args&&... args)
{
    measure<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  push
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
push(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::push_node, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
push(TupleT<Tp...>& obj, Args&&... args)
{
    push<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
push(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::push_node, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                    std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
push(TupleT<Tp&...>&& obj, Args&&... args)
{
    push<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  pop
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
pop(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::pop_node, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
pop(TupleT<Tp...>& obj, Args&&... args)
{
    pop<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
pop(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::pop_node, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                   std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
pop(TupleT<Tp&...>&& obj, Args&&... args)
{
    pop<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  set_prefix
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
set_prefix(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::set_prefix, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
set_prefix(TupleT<Tp...>& obj, Args&&... args)
{
    set_prefix<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
set_prefix(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::set_prefix, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                     std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
set_prefix(TupleT<Tp&...>&& obj, Args&&... args)
{
    set_prefix<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj),
                             std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  set_scope
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
set_scope(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::set_scope, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
set_scope(TupleT<Tp...>& obj, Args&&... args)
{
    set_scope<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
set_scope(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::set_scope, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                    std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
set_scope(TupleT<Tp&...>&& obj, Args&&... args)
{
    set_scope<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj),
                            std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  set_state
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
set_state(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke_data<operation::set_state, ApiT>(obj,
                                                         std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
set_state(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke_data<operation::set_state, ApiT>(
        std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  assemble
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
assemble(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::assemble, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
assemble(TupleT<Tp...>& obj, Args&&... args)
{
    assemble<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
assemble(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::assemble, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                   std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
assemble(TupleT<Tp&...>&& obj, Args&&... args)
{
    assemble<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj),
                           std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  derive
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
derive(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::derive, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
derive(TupleT<Tp...>& obj, Args&&... args)
{
    derive<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
derive(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::derive, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                 std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
derive(TupleT<Tp&...>&& obj, Args&&... args)
{
    derive<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  audit
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
audit(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::audit, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
audit(TupleT<Tp...>& obj, Args&&... args)
{
    audit<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
audit(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::audit, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
audit(TupleT<Tp&...>&& obj, Args&&... args)
{
    audit<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  add_secondary
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
add_secondary(TupleT<Tp...>& obj, Args&&... args)
{
    invoke_impl::invoke<operation::add_secondary, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
add_secondary(TupleT<Tp...>& obj, Args&&... args)
{
    add_secondary<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
add_secondary(TupleT<Tp&...>&& obj, Args&&... args)
{
    invoke_impl::invoke<operation::add_secondary, ApiT>(std::forward<TupleT<Tp&...>>(obj),
                                                        std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
add_secondary(TupleT<Tp&...>&& obj, Args&&... args)
{
    add_secondary<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj),
                                std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  get
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
auto
get(TupleT<Tp...>& obj, Args&&... args)
{
    using data_type         = TupleT<remove_optional_t<Tp>...>;
    using data_collect_type = mpl::get_data_type_t<data_type>;
    using data_value_type   = mpl::get_data_value_t<data_type>;

    data_value_type _data{};
    auto&&          _obj = mpl::get_reference_tuple<data_collect_type>(obj);
    invoke_impl::invoke_data<operation::get_data, ApiT>(_obj, _data,
                                                        std::forward<Args>(args)...);
    return _data;
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
auto
get(TupleT<Tp...>& obj, Args&&... args)
{
    return ::tim::invoke::get<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
auto
get(TupleT<Tp&...>&& obj, Args&&... args)
{
    using data_type         = TupleT<remove_optional_t<Tp>...>;
    using data_collect_type = mpl::get_data_type_t<data_type>;
    using data_value_type   = mpl::get_data_value_t<data_type>;

    data_value_type _data{};
    auto&&          _obj = mpl::get_reference_tuple<data_collect_type>(obj);
    invoke_impl::invoke_data<operation::get_data, ApiT>(_obj, _data,
                                                        std::forward<Args>(args)...);
    return _data;
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
auto
get(TupleT<Tp&...>&& obj, Args&&... args)
{
    return ::tim::invoke::get<TIMEMORY_API>(std::forward<TupleT<Tp&...>>(obj),
                                            std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp>
auto
get(TupleT<Tp...>& obj, void*& _ptr, size_t _hash)
{
    invoke_impl::invoke<operation::get, ApiT>(obj, _ptr, _hash);
}
//
template <template <typename...> class TupleT, typename... Tp>
auto
get(TupleT<Tp...>& obj, void*& _ptr, size_t _hash)
{
    return ::tim::invoke::get<TIMEMORY_API>(obj, _ptr, _hash);
}
//
//--------------------------------------------------------------------------------------//
//                                  get_labeled
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
auto
get_labeled(TupleT<Tp...>& obj, Args&&... args)
{
    using data_type         = TupleT<remove_optional_t<Tp>...>;
    using data_collect_type = mpl::get_data_type_t<data_type>;
    using data_label_type   = mpl::get_data_label_t<data_type>;

    data_label_type _data{};
    auto&&          _obj = mpl::get_reference_tuple<data_collect_type>(obj);
    invoke_impl::invoke_data<operation::get_labeled_data, ApiT>(
        _obj, _data, std::forward<Args>(args)...);
    return _data;
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
auto
get_labeled(TupleT<Tp...>& obj, Args&&... args)
{
    return get_labeled<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
auto
get_labeled(TupleT<Tp&...>&& obj, Args&&... args)
{
    using data_type         = TupleT<remove_optional_t<Tp>...>;
    using data_collect_type = mpl::get_data_type_t<data_type>;
    using data_label_type   = mpl::get_data_label_t<data_type>;

    data_label_type _data{};
    auto&&          _obj = mpl::get_reference_tuple<data_collect_type>(obj);
    invoke_impl::invoke_data<operation::get_labeled_data, ApiT>(
        _obj, _data, std::forward<Args>(args)...);
    return _data;
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
auto
get_labeled(TupleT<Tp&...>&& obj, Args&&... args)
{
    return get_labeled<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//                                  get_labeled
//--------------------------------------------------------------------------------------//
//
namespace impl
{
template <typename ArchiveT, typename TupleT, size_t... Idx>
auto
serialize(ArchiveT& ar, TupleT& obj, std::index_sequence<Idx...>)
{
    auto _serialize = [&ar](auto& _obj) {
        auto _label = _obj.label();
        ar(cereal::make_nvp(_label.c_str(), _obj));
    };
    TIMEMORY_FOLD_EXPRESSION(_serialize(std::get<Idx>(obj)));
}
//
}  // namespace impl
//
template <typename ArchiveT, template <typename...> class TupleT, typename... Tp>
auto
serialize(ArchiveT& ar, TupleT<Tp...>& obj)
{
    impl::serialize(ar, obj, std::make_index_sequence<sizeof...(Tp)>{});
}
//
template <typename ArchiveT, template <typename...> class TupleT, typename... Tp>
auto
serialize(ArchiveT& ar, TupleT<Tp&...>&& obj)
{
    impl::serialize(ar, obj, std::make_index_sequence<sizeof...(Tp)>{});
}
//
//--------------------------------------------------------------------------------------//
//                                  get_cache
//--------------------------------------------------------------------------------------//
//
template <typename... BundleT>
auto
get_cache()
{
    return operation::construct_cache<std::tuple<BundleT...>>{}();
}
//
//--------------------------------------------------------------------------------------//
//
//      This is for multiple component bundlers, e.g.
//          start(comp_tuple<A...>, comp_tuple<B...>)
//
//      MUST USE INDEX SEQUENCE. W/o index sequence multiple instances of same bundle
//      will give weird results
//
//--------------------------------------------------------------------------------------//
//
namespace disjoint
{
//
#define TIMEMORY_DEFINE_DISJOINT_FUNCTION(FUNC)                                          \
    namespace disjoint_impl                                                              \
    {                                                                                    \
    template <template <typename...> class TupleT, typename... Tp, size_t... Idx,        \
              typename... Args>                                                          \
    void FUNC(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)               \
    {                                                                                    \
        TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).FUNC(std::forward<Args>(args)...));  \
    }                                                                                    \
    }                                                                                    \
                                                                                         \
    template <template <typename...> class TupleT, typename... Tp, typename... Args>     \
    void FUNC(TupleT<Tp...>&& obj, Args&&... args)                                       \
    {                                                                                    \
        disjoint_impl::FUNC(std::forward<TupleT<Tp...>>(obj),                            \
                            make_index_sequence<sizeof...(Tp)>{},                        \
                            std::forward<Args>(args)...);                                \
    }

TIMEMORY_DEFINE_DISJOINT_FUNCTION(push)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(pop)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(start)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(stop)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(mark)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(mark_begin)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(mark_end)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(store)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(reset)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(record)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(measure)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(set_prefix)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(set_scope)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(assemble)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(derive)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(audit)
TIMEMORY_DEFINE_DISJOINT_FUNCTION(add_secondary)

#undef TIMEMORY_DEFINE_DISJOINT_FUNCTION

// invoke is slightly different than the others
namespace disjoint_impl
{
template <template <typename...> class TupleT, typename... Tp, typename FuncT,
          size_t... Idx, typename... Args>
void
invoke(TupleT<Tp...>&& obj, FuncT&& func, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(
        std::forward<FuncT>(func)(std::get<Idx>(obj), std::forward<Args>(args)...));
}
}  // namespace disjoint_impl
//
template <template <typename...> class TupleT, typename... Tp, typename FuncT,
          typename... Args>
void
invoke(TupleT<Tp...>&& obj, FuncT&& func, Args&&... args)
{
    disjoint_impl::invoke(std::forward<TupleT<Tp...>>(obj), std::forward<FuncT>(func),
                          make_index_sequence<sizeof...(Tp)>{},
                          std::forward<Args>(args)...);
}
}  // namespace disjoint
}  // namespace invoke
}  // namespace tim

#endif
