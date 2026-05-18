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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#include "timemory/components/opaque/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
template <typename InitF, typename StartF, typename StopF, typename GetF, typename DelF,
          typename SetupF, typename PushF, typename PopF, typename SampleF,
          typename UpdateStatsF>
opaque::opaque(bool _valid, size_t _typeid, InitF&& _init, StartF&& _start, StopF&& _stop,
               GetF&& _get, DelF&& _del, SetupF&& _setup, PushF&& _push, PopF&& _pop,
               SampleF&& _sample, UpdateStatsF&& _update_stats)
: m_valid(_valid)
, m_typeid(_typeid)
, m_init(std::move(_init))
, m_setup(std::move(_setup))
, m_push(std::move(_push))
, m_start(std::move(_start))
, m_stop(std::move(_stop))
, m_pop(std::move(_pop))
, m_get(std::move(_get))
, m_del(std::move(_del))
, m_sample(std::move(_sample))
, m_update_stats(std::move(_update_stats))
{}
//
//--------------------------------------------------------------------------------------//
//
inline opaque::~opaque()
{
    if(m_data)
    {
        m_stop(m_data);
        m_del(m_data);
    }
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::init() const
{
    m_init();
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::setup(string_view_cref_t _prefix, scope::config _scope)
{
    if(m_data)
    {
        stop();
        cleanup();
    }
    m_data  = m_setup(m_data, _prefix, _scope);
    m_valid = (m_data != nullptr);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::push(string_view_cref_t _prefix, scope::config _scope)
{
    if(m_data)
        m_push(m_data, _prefix, _scope);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::sample() const
{
    if(m_data)
        m_sample(m_data);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::start() const
{
    if(m_data)
        m_start(m_data);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::stop() const
{
    if(m_data)
        m_stop(m_data);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::pop() const
{
    if(m_data)
        m_pop(m_data);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::cleanup()
{
    if(m_data && !m_copy)
        m_del(m_data);
    m_data = nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::get(void*& ptr, size_t _hash) const
{
    if(m_data)
        m_get(m_data, ptr, _hash);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::update_statistics(bool _v) const
{
    if(m_data)
        m_update_stats(m_data, _v);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
opaque::set_copy(bool val)
{
    m_copy = val;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
