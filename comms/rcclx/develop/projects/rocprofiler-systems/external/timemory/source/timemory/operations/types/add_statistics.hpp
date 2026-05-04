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

/**
 * \file timemory/operations/types/add_statistics.hpp
 * \brief Definition for various functions for add_statistics in operations
 */

#pragma once

#include "timemory/mpl/policy.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/utility/demangle.hpp"

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace policy
{
//
//--------------------------------------------------------------------------------------//
//
//          Configure the default method for policy::record_statistics
//
//--------------------------------------------------------------------------------------//
//
template <typename CompT, typename Tp>
inline void
record_statistics<CompT, Tp>::operator()(statistics<Tp>& _stats, const CompT& _obj,
                                         bool _last)
{
    using component_type = std::remove_pointer_t<decay_t<CompT>>;
    using result_type    = decltype(std::declval<CompT>().get());
    static_assert(std::is_same<result_type, Tp>::value,
                  "Error! The default implementation of "
                  "'policy::record_statistics<Component, T>::operator()' requires 'T' "
                  "to be the same type as the return type from 'Component::get()'");

    if(!_last)
    {
        if(_obj.get_laps() < 2)
        {
            _stats += _obj.get();
        }
        else
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(
                settings::debug(),
                "Updating statistics<%s> skipped for %s. Laps: %lu > "
                "1",
                demangle<Tp>().c_str(), demangle<component_type>().c_str(),
                (unsigned long) _obj.get_laps());
        }
    }
    else
    {
        constexpr bool has_accum = trait::base_has_accum<component_type>::value;
        // make a copy and replace the accumulation with last measurement
        auto _tmp = _obj;
        IF_CONSTEXPR(has_accum) { _tmp.set_accum(_tmp.get_last()); }
        else { _tmp.set_value(_tmp.get_last()); }
        _stats += _tmp.get();
    }
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace policy
//
//--------------------------------------------------------------------------------------//
//
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::add_statistics
/// \brief
///     Enabling statistics in timemory has two parts:
///     1. tim::trait::record_statistics must be set to true for component
///     2. tim::trait::statistics must set the data type of the statistics
///         - this is usually set to the data type returned from get()
///         - tuple<> is the default and will fully disable statistics unless changed
///
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct add_statistics
{
    using type = T;

    TIMEMORY_DEFAULT_OBJECT(add_statistics)

    //----------------------------------------------------------------------------------//
    // if statistics is not enabled
    //
    template <typename StatsT>
    TIMEMORY_INLINE add_statistics(const type& _obj, StatsT& _stats, bool _last = false)
    {
        (*this)(_obj, _stats, _last);
    }

    //----------------------------------------------------------------------------------//
    // if statistics is not enabled
    //
    TIMEMORY_INLINE add_statistics(const type& _obj, bool _last = false)
    {
        (*this)(_obj, _last);
    }

    //----------------------------------------------------------------------------------//
    // generic operator
    //
    template <typename U>
    TIMEMORY_INLINE auto operator()(const U& rhs, bool _last = false) const
    {
        return sfinae(rhs, 0, 0, _last);
    }

    //----------------------------------------------------------------------------------//
    // if statistics is enabled
    //
    template <typename StatsT, typename U = type>
    TIMEMORY_INLINE void operator()(
        const U& rhs, StatsT& stats, bool _last = false,
        enable_if_t<stats_enabled<U, StatsT>::value, int> = 0) const;

    //----------------------------------------------------------------------------------//
    // if statistics is not enabled
    //
    template <typename StatsT, typename U = type>
    TIMEMORY_INLINE void operator()(
        const U&, StatsT&, bool = true,
        enable_if_t<!stats_enabled<U, StatsT>::value, int> = 0) const
    {}

private:
    template <typename U>
    TIMEMORY_INLINE auto sfinae(const U& rhs, int, int, bool _last) const
        -> decltype(rhs.update_statistics(_last))
    {
        return rhs.update_statistics(_last);
    }

    template <typename U>
    TIMEMORY_INLINE auto sfinae(const U& rhs, int, long, bool _last) const
        -> decltype(rhs.get_iterator()->stats(),
                    decay_t<decltype(rhs.get_iterator()->stats())>{})
    {
        using stats_type = decay_t<decltype(rhs.get_iterator()->stats())>;
        auto itr         = rhs.get_iterator();
        if(itr)
        {
            stats_type& _stats = itr->stats();
            (*this)(rhs, _stats, _last);
            return _stats;
        }
        return stats_type{};
    }

    template <typename U>
    TIMEMORY_INLINE void sfinae(const U&, long, long, bool) const
    {}
};
//
template <typename T>
template <typename StatsT, typename U>
void
add_statistics<T>::operator()(const U& rhs, StatsT& stats, bool _last,
                              enable_if_t<stats_enabled<U, StatsT>::value, int>) const
{
    // for type comparison
    using incoming_t = decay_t<typename StatsT::value_type>;
    using expected_t = decay_t<typename trait::statistics<U>::type>;
    // check the incomming stat type against declared stat type
    // but allow for permissive_statistics when there is an acceptable
    // implicit conversion
    static_assert(trait::permissive_statistics<U>::value ||
                      std::is_same<incoming_t, expected_t>::value,
                  "add_statistics was passed a data type different than declared "
                  "trait::statistics type. To disable this error, e.g. permit "
                  "implicit conversion, set trait::permissive_statistics "
                  "to true_type for component");
    using stats_policy_type = policy::record_statistics<U>;
    stats_policy_type{}(stats, rhs, _last);
}
//
}  // namespace operation
}  // namespace tim
