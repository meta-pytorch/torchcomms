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

#include "timemory/components/papi/papi_config.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/settings/settings.hpp"

#include <vector>

namespace tim
{
namespace component
{
template <typename Tp>
struct papi_config_factory;

template <>
struct papi_config_factory<void>
{
    auto operator()() const
    {
        return new papi_config{ [](papi_config& _cfg) {
            if(_cfg.config_string.empty() && !_cfg.fixed)
                _cfg.config_string = settings::papi_events();
        } };
    }
};

template <int... EventTypes>
struct papi_config_factory<papi_tuple<EventTypes...>>
{
    auto operator()() const
    {
        auto* _cfg  = new papi_config{ std::set<int>{ EventTypes... } };
        _cfg->fixed = true;
        return _cfg;
    }
};

template <typename Tp>
struct papi_config_factory<papi_common<Tp>> : papi_config_factory<Tp>
{
    using papi_config_factory<Tp>::operator();
};
}  // namespace component
}  // namespace tim
