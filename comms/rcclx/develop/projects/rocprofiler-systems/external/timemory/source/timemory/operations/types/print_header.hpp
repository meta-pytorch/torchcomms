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
 * \file timemory/operations/types/print_header.hpp
 * \brief Definition for various functions for print_header in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/print_statistics.hpp"

#include <iomanip>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct print_header : public common_utils
{
    using type     = Tp;
    using widths_t = std::vector<int64_t>;

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Statp, typename Up = Tp,
              enable_if_t<is_enabled<Up>::value, char> = 0>
    TIMEMORY_COLD print_header(const type& _obj, utility::stream& _os,
                               const Statp& _stats)
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        auto _labels = get_labels(_obj);

        _os.set_prefix_begin();
        utility::write_header(_os, "LABEL");

        if(trait::report<type>::count())
            utility::write_header(_os, "COUNT");

        if(trait::report<type>::depth())
            utility::write_header(_os, "DEPTH");
        _os.set_prefix_end();

        auto ios_fixed = std::ios_base::fixed;
        auto ios_dec   = std::ios_base::dec;
        auto ios_showp = std::ios_base::showpoint;
        auto f_self    = ios_fixed | ios_dec | ios_showp;
        int  w_self    = 8;
        int  p_self    = 1;
        auto f_value   = Tp::get_format_flags();
        auto w_value   = Tp::get_width();
        auto p_value   = Tp::get_precision();

        if(trait::report<type>::metric())
            utility::write_header(_os, "METRIC");

        if(trait::report<type>::units())
            utility::write_header(_os, "UNITS");

        if(trait::report<type>::sum())
            utility::write_header(_os, "SUM", f_value, w_value, p_value);

        if(trait::report<type>::mean())
            utility::write_header(_os, "MEAN", f_value, w_value, p_value);

        if(trait::report<type>::stats())
            print_statistics<Tp>::get_header(_os, _stats);

        if(trait::report<type>::self())
            utility::write_header(_os, "% SELF", f_self, w_self, p_self);

        _os.insert_break();
        if(!_labels.empty())
        {
            for(size_t i = 0; i < _labels.size() - 1; ++i)
            {
                if(trait::report<type>::metric())
                    utility::write_header(_os, "METRIC");

                if(trait::report<type>::units())
                    utility::write_header(_os, "UNITS");

                if(trait::report<type>::sum())
                    utility::write_header(_os, "SUM", f_value, w_value, p_value);

                if(trait::report<type>::mean())
                    utility::write_header(_os, "MEAN", f_value, w_value, p_value);

                if(trait::report<type>::stats())
                    print_statistics<Tp>::get_header(_os, _stats);

                if(trait::report<type>::self())
                    utility::write_header(_os, "% SELF", f_self, w_self, p_self);

                _os.insert_break();
            }
        }
    }

    template <typename... Args, typename Up = Tp,
              enable_if_t<!is_enabled<Up>::value, char> = 0>
    print_header(const type&, utility::stream&, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
