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
 * \file timemory/operations/types/print_statistics.hpp
 * \brief Definition for various functions for print_statistics in operations
 */

#pragma once

#include "timemory/data/statistics.hpp"
#include "timemory/data/stream.hpp"
#include "timemory/environment/declaration.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::print_statistics
/// \brief prints the statistics for a type
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct print_statistics : public common_utils
{
public:
    using type     = Tp;
    using widths_t = std::vector<int64_t>;

public:
    TIMEMORY_DEFAULT_OBJECT(print_statistics)

    template <typename Vp>
    constexpr bool operator()(const Vp&)
    {
        return stats_enabled<decay_t<Tp>, decay_t<Vp>>::value;
    }

    template <typename Self, template <typename> class Sp, typename Vp, typename Up = Tp,
              enable_if_t<stats_enabled<Up, Vp>::value, int> = 0>
    TIMEMORY_NOINLINE TIMEMORY_COLD print_statistics(const type&, utility::stream& _os,
                                                     const Self&, const Sp<Vp>&    _stats,
                                                     uint64_t)
    {
        if(!trait::report_statistics<type>::value || !trait::report<type>::stats())
            return;

        if(trait::report<type>::min())
            utility::write_entry(_os, "MIN", _stats.get_min());
        if(trait::report<type>::max())
            utility::write_entry(_os, "MAX", _stats.get_max());
        if(trait::report<type>::variance())
            utility::write_entry(_os, "VAR", _stats.get_variance());
        if(trait::report<type>::stddev())
            utility::write_entry(_os, "STDDEV", _stats.get_stddev());
    }

    template <typename Self, typename Vp, typename Up = Tp,
              enable_if_t<!stats_enabled<Up, Vp>::value, int> = 0>
    print_statistics(const type&, utility::stream&, const Self&, const Vp&, uint64_t)
    {}

    template <typename Self>
    print_statistics(const type&, utility::stream&, const Self&,
                     const statistics<std::tuple<>>&, uint64_t)
    {}

public:
    template <template <typename> class Sp, typename Vp, typename Up = Tp,
              enable_if_t<stats_enabled<Up, Vp>::value, int> = 0>
    static TIMEMORY_NOINLINE TIMEMORY_COLD void get_header(utility::stream& _os,
                                                           const Sp<Vp>&)
    {
        if(!trait::report_statistics<type>::value || !trait::report<type>::stats())
            return;

        auto _flags = Tp::get_format_flags();
        auto _width = Tp::get_width();
        auto _prec  = Tp::get_precision();

        if(trait::report<type>::min())
            utility::write_header(_os, "MIN", _flags, _width, _prec);
        if(trait::report<type>::max())
            utility::write_header(_os, "MAX", _flags, _width, _prec);
        if(trait::report<type>::variance())
            utility::write_header(_os, "VAR", _flags, _width, _prec);
        if(trait::report<type>::stddev())
            utility::write_header(_os, "STDDEV", _flags, _width, _prec);
    }

    template <typename Vp, typename Up = Tp,
              enable_if_t<!stats_enabled<Up, Vp>::value, int> = 0>
    static void get_header(utility::stream&, Vp&)
    {}

    static void get_header(utility::stream&, const statistics<std::tuple<>>&) {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
