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

#include "timemory/environment/env_settings.hpp"
#include "timemory/environment/macros.hpp"
#include "timemory/environment/types.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/macros.hpp"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iosfwd>
#include <map>
#include <mutex>
#include <string>
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
template <typename Tp>
Tp
get_env(const std::string& env_id, Tp _default, bool _store)
{
    if constexpr(std::is_enum<Tp>::value)
    {
        using Up = std::underlying_type_t<Tp>;
        static_assert(!std::is_enum<Up>::value,
                      "Error! type will cause recursion. Please cast to a fundamental "
                      "type and recast the result");
        // cast to underlying type -> get_env -> cast to enum type
        return static_cast<Tp>(get_env<Up>(env_id, static_cast<Up>(_default), _store));
    }
    else
    {
        if(env_id.empty())
            return _default;

        auto* _env_settings = env_settings::instance();
        char* env_var       = std::getenv(env_id.c_str());
        if(env_var)
        {
            std::string       str_var = std::string(env_var);
            std::stringstream iss{ str_var };
            auto              var = Tp{};
            iss >> var;
            if(_env_settings && _store)
                _env_settings->insert<Tp>(env_id, var);
            return var;
        }
        // record default value
        if(_env_settings && _store)
            _env_settings->insert<Tp>(env_id, _default);

        // return default if not specified in environment
        return _default;
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
Tp
get_env_choice(const std::string& env_id, Tp _default, std::set<Tp> _choices, bool _store)
{
    assert(!_choices.empty());
    auto _choice = get_env<Tp>(env_id, _default, _store);

    // check that the choice is valid
    if(_choices.find(_choice) == _choices.end())
    {
        std::ostringstream _msg{};
        _msg << "Error! Invalid value \"" << _choice << "\" for " << env_id
             << ". Valid choices are: ";
        std::ostringstream _opts{};
        for(const auto& itr : _choices)
            _opts << ", \"" << itr << "\"";
        if(_opts.str().length() >= 2)
            _msg << _opts.str().substr(2);
        throw std::runtime_error(_msg.str());
    }

    return _choice;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
Tp
load_env(const std::string& env_id, Tp _default)
{
    if(env_id.empty())
        return _default;

    auto* _env_settings = env_settings::instance();
    if(!_env_settings)
        return _default;

    auto itr = _env_settings->get(env_id);
    if(itr != _env_settings->end())
    {
        std::stringstream iss{ itr->second };
        auto              var = Tp{};
        iss >> var;
        return var;
    }

    // return default if not specified in environment
    return _default;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
set_env(const std::string& env_var, const Tp& _val, int override)
{
    static bool _debug = []() {
        bool _env_val = get_env(TIMEMORY_SETTINGS_PREFIX "DEBUG_ENV", false);
        return get_env(TIMEMORY_SETTINGS_PREFIX "DEBUG_SETTINGS", _env_val);
    }();

    std::stringstream ss_val;
    ss_val << _val;

    if(_debug)
    {
        std::ostringstream oss;
        oss << "[" << TIMEMORY_PROJECT_NAME << "] set_env(\"" << env_var << "\", \""
            << ss_val.str() << "\", " << override << ");\n";
        std::cerr << log::color::warning() << oss.str() << log::color::end();
    }

#if defined(TIMEMORY_MACOS) || (defined(TIMEMORY_LINUX) && (_POSIX_C_SOURCE >= 200112L))
    setenv(env_var.c_str(), ss_val.str().c_str(), override);
#elif defined(TIMEMORY_WINDOWS)
    auto _curr = get_env<std::string>(env_var, "");
    if(_curr.empty() || override > 0)
        _putenv_s(env_var.c_str(), ss_val.str().c_str());
#else
    consume_parameters(env_var, _val, override, ss_val);
#endif
}
//
//--------------------------------------------------------------------------------------//
//
template <typename FuncT>
void
print_env(std::ostream& os, FuncT&& _filter)
{
    static_assert(
        std::is_same<bool, decltype(_filter(std::declval<std::string>()))>::value,
        "Error! filter must accept string and return bool");
    if(env_settings::instance())
        env_settings::instance()->print(os, std::forward<FuncT>(_filter));
}

}  // namespace tim
