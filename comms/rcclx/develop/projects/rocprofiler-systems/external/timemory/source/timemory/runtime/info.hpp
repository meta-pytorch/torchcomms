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

#include "timemory/components/info.hpp"
#include "timemory/defines.h"
#include "timemory/macros/language.hpp"

#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace tim
{
namespace runtime
{
struct info
{
    using enum_key_map_t = std::map<int, component::info>;
    using value_type     = typename enum_key_map_t::value_type;
    using key_type       = typename enum_key_map_t::key_type;
    using mapped_type    = typename enum_key_map_t::mapped_type;
    using iterator       = typename enum_key_map_t::const_iterator;

    static auto begin() { return _get().cbegin(); }
    static auto end() { return _get().cend(); }
    static auto empty() { return _get().empty(); }
    static auto size() { return _get().size(); }
    static auto find(int _v) { return _get().find(_v); }
    static auto find(tim::string_view_cref_t);
    static auto count(int _v) { return _get().count(_v); }
    static auto at(int _idx) { return _get().at(_idx); }
    static auto error_message(tim::string_view_cref_t);

    template <typename... Args>
    static auto emplace(Args&&... args);

    template <typename... Args>
    static auto insert(Args&&... args);

    auto operator[](int _idx) const { return _get()[_idx]; }

private:
    static enum_key_map_t& _get();
};

template <typename... Args>
inline auto
info::emplace(Args&&... args)
{
    return _get().emplace(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto
info::insert(Args&&... args)
{
    return _get().insert(std::forward<Args>(args)...);
}

inline auto
info::error_message(tim::string_view_cref_t _name)
{
    std::stringstream _msg;
    _msg << "Valid choices are: [";
    for(auto& itr : _get())
    {
        std::stringstream _ss{};
        for(const auto& iitr : itr.second.identifiers)
            _ss << ", "
                << "'" << iitr << "'";
        auto&& _c = _ss.str();
        if(_c.length() > 2)
            _msg << _c.substr(2);
    }
    _msg << ']';
    auto _choices = _msg.str();
    fprintf(stderr, "[%s] Unknown component: '%s'. %s\n", TIMEMORY_PROJECT_NAME,
            _name.data(), _choices.c_str());
}

inline auto
info::find(tim::string_view_cref_t _v)
{
    info _this{};
    for(const auto& itr : _this)
    {
        if(itr.second.matches(_v))
            return std::make_pair(itr.second, true);
    }
    return std::make_pair(component::info{}, false);
}

inline info::enum_key_map_t&
info::_get()
{
    static enum_key_map_t _v = {};
    return _v;
}
}  // namespace runtime
}  // namespace tim
