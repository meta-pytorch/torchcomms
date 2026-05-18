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

#include "timemory/macros/language.hpp"
#include "timemory/utility/types.hpp"

#include <cstdint>
#include <functional>
#include <set>
#include <string>

namespace tim
{
//
namespace scope
{
struct config;
}
//
namespace component
{
//
struct opaque
{
    using init_func_t   = std::function<void()>;
    using setup_func_t  = std::function<void*(void*, string_view_cref_t, scope::config)>;
    using push_func_t   = std::function<void(void*&, string_view_cref_t, scope::config)>;
    using start_func_t  = std::function<void(void*)>;
    using stop_func_t   = std::function<void(void*)>;
    using pop_func_t    = std::function<void(void*)>;
    using get_func_t    = std::function<void(void*, void*&, size_t)>;
    using delete_func_t = std::function<void(void*)>;
    using sample_func_t = std::function<void(void*)>;
    using stats_func_t  = std::function<void(void*, bool)>;

    template <typename InitF, typename StartF, typename StopF, typename GetF,
              typename DelF, typename SetupF, typename PushF, typename PopF,
              typename SampleF, typename UpdateStatsF>
    opaque(bool _valid, size_t _typeid, InitF&& _init, StartF&& _start, StopF&& _stop,
           GetF&& _get, DelF&& _del, SetupF&& _setup, PushF&& _push, PopF&& _pop,
           SampleF&& _sample, UpdateStatsF&& _stats);

    opaque() = default;
    ~opaque();
    opaque(const opaque&) = default;
    opaque(opaque&&)      = default;
    opaque& operator=(const opaque&) = default;
    opaque& operator=(opaque&&) = default;

    explicit operator bool() const { return m_valid; }

    void init() const;
    void setup(string_view_cref_t _prefix, scope::config _scope);
    void push(string_view_cref_t _prefix, scope::config _scope);
    void sample() const;
    void start() const;
    void stop() const;
    void pop() const;
    void cleanup();
    void get(void*& ptr, size_t _hash) const;
    void update_statistics(bool) const;
    void set_copy(bool val);

    bool         m_valid  = false;
    bool         m_copy   = false;
    size_t       m_typeid = 0;
    void*        m_data   = nullptr;
    init_func_t  m_init   = []() {};
    setup_func_t m_setup  = [](void*, string_view_cref_t, scope::config) {
        return nullptr;
    };
    push_func_t   m_push         = [](void*&, string_view_cref_t, scope::config) {};
    start_func_t  m_start        = [](void*) {};
    stop_func_t   m_stop         = [](void*) {};
    pop_func_t    m_pop          = [](void*) {};
    get_func_t    m_get          = [](void*, void*&, size_t) {};
    delete_func_t m_del          = [](void*) {};
    sample_func_t m_sample       = [](void*) {};
    stats_func_t  m_update_stats = [](void*, bool) {};
};
//
template <typename Toolset>
opaque
get_opaque();
//
template <typename Toolset>
opaque
get_opaque(scope::config _scope);
//
template <typename Toolset, typename Arg, typename... Args>
opaque
get_opaque(Arg&& arg, Args&&... args);
//
template <typename Toolset>
std::set<size_t>
get_typeids();
//
namespace factory
{
//
template <typename Toolset, typename Arg, typename... Args>
opaque
get_opaque(Arg&& arg, Args&&... args);
//
template <typename Toolset>
opaque
get_opaque();
//
template <typename Toolset>
opaque
get_opaque(scope::config _scope);
//
template <typename Toolset>
std::set<size_t>
get_typeids();
//
}  // namespace factory
}  // namespace component
}  // namespace tim
