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

#ifndef TIMEMORY_COMPONENTS_USER_BUNDLE_COMPONENTS_CPP_
#define TIMEMORY_COMPONENTS_USER_BUNDLE_COMPONENTS_CPP_

#include "timemory/components/user_bundle/types.hpp"

#if !defined(TIMEMORY_USER_BUNDLE_COMPONENT_HEADER_MODE)
#    define TIMEMORY_USER_BUNDLE_INLINE
#    include "timemory/components/user_bundle/components.hpp"
#    include "timemory/runtime/properties.hpp"
#else
#    define TIMEMORY_USER_BUNDLE_INLINE inline
#endif

namespace tim
{
namespace env
{
//
user_bundle_variables_t& get_user_bundle_variables(TIMEMORY_API)
{
    static user_bundle_variables_t _instance = {
        { component::global_bundle_idx,
          { []() { return settings::global_components(); } } },
        { component::ompt_bundle_idx,
          { []() { return settings::ompt_components(); },
            []() { return settings::trace_components(); },
            []() { return settings::profiler_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } },
        { component::mpip_bundle_idx,
          { []() { return settings::mpip_components(); },
            []() { return settings::trace_components(); },
            []() { return settings::profiler_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } },
        { component::ncclp_bundle_idx,
          { []() { return settings::ncclp_components(); },
            []() { return settings::mpip_components(); },
            []() { return settings::trace_components(); },
            []() { return settings::profiler_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } },
        { component::trace_bundle_idx,
          { []() { return settings::trace_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } },
        { component::profiler_bundle_idx,
          { []() { return settings::profiler_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } },
    };
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
user_bundle_variables_t& get_user_bundle_variables(project::kokkosp)
{
    static user_bundle_variables_t _instance = {
        { component::kokkosp_bundle_idx,
          { []() { return settings::kokkos_components(); },
            []() { return get_env<std::string>("TIMEMORY_KOKKOSP_COMPONENTS", ""); },
            []() { return get_env<std::string>("KOKKOS_TIMEMORY_COMPONENTS", ""); },
            []() { return settings::trace_components(); },
            []() { return settings::profiler_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } }
    };
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
std::vector<TIMEMORY_COMPONENT>
get_bundle_components(const std::vector<user_bundle_spec_t>& _priority)
{
    std::string _custom{};
    bool        _fallthrough = false;
    auto        _replace     = [&_fallthrough](const std::string& _key) {
        const std::string _ft  = "fallthrough";
        auto              _pos = _key.find(_ft);
        if(_pos != std::string::npos)
        {
            _fallthrough = true;
            return _key.substr(0, _pos) + _key.substr(_pos + _ft.length() + 1);
        }
        return _key;
    };

    for(const auto& itr : _priority)
    {
        auto _spec = itr();
        if(_spec.length() > 0)
        {
            if(_spec != "none" && _spec != "NONE")
                _custom += _replace(_spec);
            else
                _fallthrough = false;
            if(!_fallthrough)
                break;
        }
    }

    auto _debug = (settings::instance()) ? (settings::instance()->get_debug()) : false;
    TIMEMORY_CONDITIONAL_PRINT_HERE(_debug, "getting user bundle components: %s",
                                    _custom.c_str());

    return tim::enumerate_components(tim::delimit(_custom));
}
//
}  // namespace env
//
namespace component
{
namespace internal
{
TIMEMORY_USER_BUNDLE_INLINE std::string
                            user_bundle::label()
{
    return "user_bundle";
}
TIMEMORY_USER_BUNDLE_INLINE std::string
                            user_bundle::description()
{
    return "Generic bundle of components designed for runtime configuration by a "
           "user via environment variables and/or direct insertion";
}

TIMEMORY_USER_BUNDLE_INLINE
user_bundle::user_bundle(scope::config _cfg, typeid_vec_t _typeids,
                         opaque_array_t _opaque_arr, const char* _prefix)
: m_scope{ _cfg }
, m_prefix{ _prefix }
, m_typeids{ std::move(_typeids) }
, m_bundle{ std::move(_opaque_arr) }
{}

TIMEMORY_USER_BUNDLE_INLINE
user_bundle::user_bundle(const user_bundle& rhs)
: m_scope{ rhs.m_scope }
, m_prefix{ rhs.m_prefix }
, m_typeids{ rhs.m_typeids }
, m_bundle{ rhs.m_bundle }
{
    for(auto& itr : m_bundle)
        itr.set_copy(true);
}

TIMEMORY_USER_BUNDLE_INLINE user_bundle::~user_bundle()
{
    for(auto& itr : m_bundle)
        itr.cleanup();
}

TIMEMORY_USER_BUNDLE_INLINE user_bundle&
user_bundle::operator=(const user_bundle& rhs)
{
    if(this == &rhs)
        return *this;

    m_scope   = rhs.m_scope;
    m_prefix  = rhs.m_prefix;
    m_typeids = rhs.m_typeids;
    m_bundle  = rhs.m_bundle;
    for(auto& itr : m_bundle)
        itr.set_copy(true);

    return *this;
}

TIMEMORY_USER_BUNDLE_INLINE
user_bundle::user_bundle(user_bundle&& rhs) noexcept
: m_scope{ rhs.m_scope }
, m_prefix{ rhs.m_prefix }
, m_typeids{ std::move(rhs.m_typeids) }
, m_bundle{ std::move(rhs.m_bundle) }
{
    rhs.m_bundle.clear();
}

TIMEMORY_USER_BUNDLE_INLINE user_bundle&
user_bundle::operator=(user_bundle&& rhs) noexcept
{
    if(this != &rhs)
    {
        m_scope   = rhs.m_scope;
        m_prefix  = rhs.m_prefix;
        m_typeids = std::move(rhs.m_typeids);
        m_bundle  = std::move(rhs.m_bundle);
        rhs.m_bundle.clear();
    }
    return *this;
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::configure(opaque_array_t& _data, typeid_vec_t& _typeids, mutex_t& _mtx,
                       opaque&& obj, std::set<size_t>&& _inp)
{
    if(obj)
    {
        lock_t lk{ _mtx };
        size_t sum = 0;
        for(auto&& itr : _inp)
        {
            if(itr > 0 && contains(itr, _typeids))
            {
                if(settings::verbose() > 1 || settings::debug())
                    TIMEMORY_PRINT_HERE("Skipping duplicate typeid: %lu",
                                        (unsigned long) itr);
                return;
            }
            sum += itr;
            if(itr > 0)
                _typeids.emplace_back(itr);
        }
        if(sum == 0)
        {
            TIMEMORY_PRINT_HERE("No typeids. Sum: %lu", (unsigned long) sum);
            return;
        }
        _data.emplace_back(std::move(obj));
    }
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::reset(opaque_array_t& _data, typeid_vec_t& _typeids, mutex_t& _mtx)
{
    lock_t lk{ _mtx };
    _data.clear();
    _typeids.clear();
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::setup()
{
    if(!m_setup)
    {
        m_setup = true;
        for(auto& itr : m_bundle)
            itr.setup(m_prefix, m_scope);
    }
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::push()
{
    setup();
    for(auto& itr : m_bundle)
        itr.push(m_prefix, m_scope);
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::sample()
{
    setup();
    for(auto& itr : m_bundle)
        itr.sample();
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::start()
{
    setup();
    for(auto& itr : m_bundle)
        itr.start();
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::stop()
{
    for(auto& itr : m_bundle)
        itr.stop();
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::pop()
{
    for(auto& itr : m_bundle)
        itr.pop();
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::get(void*& ptr, size_t _hash) const
{
    if(ptr == nullptr)
    {
        for(const auto& itr : m_bundle)
        {
            itr.get(ptr, _hash);
            if(ptr)
                break;
        }
    }
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::set_prefix(const char* _prefix)
{
    m_prefix = _prefix;
    m_setup  = false;
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::set_scope(const scope::config& val)
{
    m_scope = val;
    m_setup = false;
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::insert(opaque&& obj, typeid_set_t&& _typeids)
{
    if(obj)
    {
        size_t sum = 0;
        for(auto&& itr : _typeids)
        {
            if(itr > 0 && contains(itr, m_typeids))
            {
                if(settings::verbose() > 1 || settings::debug())
                    TIMEMORY_PRINT_HERE("Skipping duplicate typeid: %lu",
                                        (unsigned long) itr);
                return;
            }
            sum += itr;
            if(itr > 0)
                m_typeids.emplace_back(itr);
        }
        if(sum == 0)
        {
            TIMEMORY_PRINT_HERE("No typeids. Sum: %lu", (unsigned long) sum);
            return;
        }
        m_bundle.emplace_back(std::move(obj));
    }
}

TIMEMORY_USER_BUNDLE_INLINE void
user_bundle::update_statistics(bool _v) const
{
    for(const auto& itr : m_bundle)
        itr.update_statistics(_v);
}

TIMEMORY_USER_BUNDLE_INLINE bool
user_bundle::contains(size_t _val, const typeid_vec_t& _targ)
{
    return std::any_of(_targ.begin(), _targ.end(),
                       [&_val](auto itr) { return (itr == _val); });
}

}  // namespace internal
}  // namespace component
}  // namespace tim

#endif  // TIMEMORY_COMPONENTS_USER_BUNDLE_COMPONENTS_CPP_
