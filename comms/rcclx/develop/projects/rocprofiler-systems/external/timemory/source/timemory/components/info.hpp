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

#include "timemory/enum.h"
#include "timemory/macros/language.hpp"
#include "timemory/utility/macros.hpp"

#include <set>
#include <string>

namespace tim
{
namespace component
{
/// \struct tim::component::info
/// \brief Contains index and name information. An entry within runtime::info
/// can be created and runtime::enumerate will utilize this info
///
struct info
{
    using match_functor_t = bool (*)(const char*);

    TIMEMORY_COMPONENT    index         = TIMEMORY_COMPONENTS_END;
    std::string           name          = {};
    std::set<std::string> identifiers   = {};
    match_functor_t       match_functor = nullptr;

    TIMEMORY_DEFAULT_OBJECT(info)

    info(TIMEMORY_COMPONENT _index, std::string _name, std::set<std::string> _ids = {},
         match_functor_t _match = nullptr)
    : index{ _index }
    , name{ std::move(_name) }
    , identifiers{ std::move(_ids) }
    , match_functor{ _match }
    {
        identifiers.emplace(name);
    }

    void operator()(std::string _name, const std::set<std::string>& _ids = {},
                    match_functor_t _match = nullptr)
    {
        identifiers.emplace(_name);
        for(auto&& itr : _ids)
            identifiers.emplace(itr);
        if(!match_functor)
            match_functor = _match;
    }

    bool matches(tim::string_view_cref_t _v) const
    {
        if(!match_functor)
            return false;
        return (*match_functor)(_v.data());
    }
};
}  // namespace component
}  // namespace tim
