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

#ifndef TIMEMORY_ENVIRONMENT_ENV_SETTINGS_HPP_
#    define TIMEMORY_ENVIRONMENT_ENV_SETTINGS_HPP_
#endif

#include "timemory/environment/macros.hpp"
#include "timemory/environment/types.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/macros.hpp"

#include <atomic>
#include <cstdlib>
#include <functional>
#include <iosfwd>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              environment
//
//--------------------------------------------------------------------------------------//
//
class env_settings
{
public:
    using env_map_t      = std::map<std::string, std::string>;
    using env_uomap_t    = std::map<std::string, std::string>;
    using env_pair_t     = std::pair<std::string, std::string>;
    using iterator       = typename env_map_t::iterator;
    using const_iterator = typename env_map_t::const_iterator;
    using filter_func_t  = std::function<bool(const std::string&)>;

public:
    static env_settings* instance();

private:
    // if passed a global, insert m_env into it's array
    env_settings(env_settings* _global = nullptr, int _id = 0);

    // only the zero id deletes
    ~env_settings();

public:
    template <typename Tp>
    void insert(const std::string& env_id, Tp val);

    env_map_t      get() const;
    iterator       get(const std::string& _entry) { return m_env->find(_entry); }
    const_iterator get(const std::string& _entry) const { return m_env->find(_entry); }
    iterator       begin() { return m_env->begin(); }
    iterator       end() { return m_env->end(); }
    const_iterator begin() const { return m_env->begin(); }
    const_iterator end() const { return m_env->end(); }

    void print(
        std::ostream&,
        filter_func_t&& _filter = [](const std::string&) { return true; }) const;
    friend std::ostream& operator<<(std::ostream& os, const env_settings& env)
    {
        env.print(os);
        return os;
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, unsigned int);

    template <typename Archive>
    static void serialize_environment(Archive& ar)
    {
        if(instance())
            instance()->serialize(ar, TIMEMORY_GET_CLASS_VERSION(tim::env_settings));
    }

private:
    static std::atomic_bool& lock_flag()
    {
        static std::atomic_bool _instance(false);
        return _instance;
    }

    void collapse();

private:
    int                       m_id  = 0;
    env_uomap_t*              m_env = new env_uomap_t{};
    std::vector<env_uomap_t*> m_env_other;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
env_settings::insert(const std::string& env_id, Tp val)
{
#if !defined(TIMEMORY_DISABLE_STORE_ENVIRONMENT)
    std::stringstream ss;
    ss << std::boolalpha << val;

    auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
    if(lock_flag().load() && !lk.owns_lock())
        lk.lock();

    if(m_env &&
       (m_env->find(env_id) == m_env->end() || m_env->find(env_id)->second != ss.str()))
        (*m_env)[env_id] = ss.str();
#else
    consume_parameters(env_id, val);
#endif
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
env_settings::serialize(Archive& ar, const unsigned int)
{
    auto _tmp = env_uomap_t{};
    if(!m_env)
        m_env = &_tmp;

    collapse();

    auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
    lock_flag().store(true);

    if(!lk.owns_lock())
        lk.lock();

    ar(cereal::make_nvp("environment", *m_env));

    if(m_env == &_tmp)
        m_env = nullptr;

    lock_flag().store(false);
}
}  // namespace tim

#if !defined(TIMEMORY_USE_ENVIRONMENT_EXTERN) &&                                         \
    !defined(TIMEMORY_ENVIRONMENT_SOURCE) &&                                             \
    !defined(TIMEMORY_ENVIRONMENT_ENV_SETTINGS_CPP_)
#    include "timemory/environment/env_settings.cpp"
#endif
