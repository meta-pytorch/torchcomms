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

#ifndef TIMEMORY_ENVIRONMENT_ENV_SETTINGS_CPP_
#    define TIMEMORY_ENVIRONMENT_ENV_SETTINGS_CPP_
#endif

#include "timemory/environment/macros.hpp"

#if !defined(TIMEMORY_ENVIRONMENT_ENV_SETTINGS_HPP_)
#    include "timemory/environment/env_settings.hpp"
#endif

#include "timemory/utility/locking.hpp"
#include "timemory/utility/types.hpp"

#include <atomic>
#include <iomanip>
#include <iosfwd>
#include <sstream>
#include <string>

namespace tim
{
//
// if passed a global, insert m_env into it's array
//
TIMEMORY_ENVIRONMENT_LINKAGE(env_settings::env_settings)
(env_settings* _global, int _id)
: m_id(_id)
, m_env(new env_uomap_t)
{
    if(_global && _id != 0)
    {
        auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
        lock_flag().store(true);
        if(!lk.owns_lock())
            lk.lock();
        _global->m_env_other.emplace_back(m_env);
        lock_flag().store(false);
    }
}
//
//--------------------------------------------------------------------------------------//
//
// only the zero id deletes
//
TIMEMORY_ENVIRONMENT_LINKAGE(env_settings::~env_settings())
{
    if(m_id == 0)
    {
        delete m_env;
        for(auto& itr : m_env_other)
            delete itr;
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_ENVIRONMENT_LINKAGE(env_settings::env_map_t)
env_settings::get() const
{
    auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
    lock_flag().store(true);
    if(!lk.owns_lock())
        lk.lock();

    auto      _tmp = *m_env;
    env_map_t _ret;
    for(const auto& itr : _tmp)
        _ret[itr.first] = itr.second;

    lock_flag().store(false);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_ENVIRONMENT_LINKAGE(void)
env_settings::print(std::ostream& os, filter_func_t&& _filter) const
{
    if(m_id == 0)
        const_cast<env_settings&>(*this).collapse();

    auto _data = get();

    size_t _wl    = 35;
    size_t _wr    = 0;
    size_t _count = 0;
    for(const auto& itr : _data)
    {
        if(_filter(itr.first))
        {
            _wl = std::max<size_t>(itr.first.length(), _wl);
            _wr = std::max<size_t>(itr.second.length(), _wr);
            ++_count;
        }
    }

    if(_count == 0)
        return;

    std::stringstream filler{};
    filler.fill('-');
    filler << '#';
    {
        std::stringstream _tmp{};
        _tmp << " " << std::setw(_wl) << std::right << " "
             << "  =  " << std::setw(_wr) << std::left << " "
             << "\n";
        filler << std::setw(_tmp.str().length() + 1) << "";
    }
    filler << '#';

    std::stringstream ss;
    ss << filler.str() << '\n';
    ss << "# Environment settings:\n";
    for(const auto& itr : _data)
    {
        if(_filter(itr.first))
        {
            ss << "# " << std::setw(_wl) << std::right << itr.first << "  =  "
               << std::setw(_wr) << std::left << itr.second << '\n';
        }
    }
    ss << filler.str();
    os << ss.str() << '\n';
}
//
//--------------------------------------------------------------------------------------//
//
// NOLINTNEXTLINE
TIMEMORY_ENVIRONMENT_LINKAGE(void)
env_settings::collapse()
{
    if(m_id != 0)
        return;

    auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
    if(!lk.owns_lock())
        lk.lock();

    lock_flag().store(true);

    // insert all the other maps into this map
    for(size_t i = 0; i < m_env_other.size(); ++i)
    {
        // get the map instance
        auto* itr = m_env_other[i];
        if(itr)
        {
            // loop over entries
            for(const auto& mitr : *m_env_other[i])
            {
                auto key = mitr.first;
                if(m_env->find(key) != m_env->end())
                {
                    // if the key already exists and if the entries differ,
                    // create a new unique entry for the index
                    if(m_env->find(key)->second != mitr.second)
                    {
                        key +=
                            std::string("[@") + std::to_string(i + 1) + std::string("]");
                        (*m_env)[key] = mitr.second;
                    }
                }
                else
                {
                    // if a new entry, insert without unique tag to avoid duplication
                    // when thread count is very high
                    (*m_env)[key] = mitr.second;
                }
            }
        }
    }
    lock_flag().store(false);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_ENVIRONMENT_LINKAGE(tim::env_settings*)
env_settings::instance()
{
    static std::atomic<int>           _count{ 0 };
    static env_settings               _instance{};
    static thread_local int           _id = _count++;
    static thread_local env_settings* _local =
        (_id == 0) ? (&_instance) : new env_settings{ &_instance, _id };
    static thread_local auto _ldtor = scope::destructor{ []() {
        if(_local == &_instance || _id == 0)
            return;
        delete _local;
        _local                      = nullptr;
    } };
    return _local;
    (void) _ldtor;
}
}  // namespace tim
