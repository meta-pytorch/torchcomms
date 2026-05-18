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

#include "timemory/api.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/environment/types.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/types.hpp"

#include <cstdint>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

namespace tim
{
//
namespace argparse
{
struct argument_parser;
}
//
/// \struct tim::vsettings
/// \brief Virtual base class for storing settings
struct vsettings
{
    using parser_t         = argparse::argument_parser;
    using parser_func_t    = std::function<void(parser_t&)>;
    using display_map_t    = std::map<std::string, std::string>;
    using shared_pointer_t = std::shared_ptr<vsettings>;
    using update_type      = setting_update_type;
    using parse_callback_t =
        std::function<void(vsettings*, std::string_view, update_type)>;

    struct noparse
    {};

    // vsettings() = default;

    vsettings() = default;

    vsettings(std::string _name, std::string _env_name, std::string _descript,
              std::set<std::string> _categories, std::vector<std::string> _cmdline = {},
              int32_t _count = -1, int32_t _max_count = -1,
              std::vector<std::string> _choices = {});

    vsettings(std::string _name, std::string _env_name, std::string _descript,
              std::vector<std::string> _cmdline = {}, int32_t _count = -1,
              int32_t _max_count = -1, std::vector<std::string> _choices = {},
              std::set<std::string> _categories = {});

    vsettings(std::string _name, std::string _env_name, std::string _descript,
              parse_callback_t&& _callback, std::set<std::string> _categories,
              std::vector<std::string> _cmdline = {}, int32_t _count = -1,
              int32_t _max_count = -1, std::vector<std::string> _choices = {});

    virtual ~vsettings() = default;

    vsettings(const vsettings&) = default;
    vsettings(vsettings&&)      = default;

    vsettings& operator=(const vsettings&) = default;
    vsettings& operator=(vsettings&&) = default;

    virtual std::string as_string() const                                           = 0;
    virtual bool        reset()                                                     = 0;
    virtual bool        parse()                                                     = 0;
    virtual bool  parse(const std::string&, update_type = update_type::unspecified) = 0;
    virtual bool  is_updated()                                                      = 0;
    virtual void* add_argument(argparse::argument_parser&)                          = 0;
    virtual shared_pointer_t clone()                                                = 0;
    virtual void             clone(const shared_pointer_t& rhs);

    virtual display_map_t get_display(std::ios::fmtflags fmt = {}, int _w = -1,
                                      int _p = -1);

    const auto& get_name() const { return m_name; }
    const auto& get_env_name() const { return m_env_name; }
    const auto& get_description() const { return m_description; }
    const auto& get_command_line() const { return m_cmdline; }
    const auto& get_choices() const { return m_choices; }
    const auto& get_categories() const { return m_categories; }
    const auto& get_count() const { return m_count; }
    const auto& get_max_count() const { return m_max_count; }

    void set_hidden(bool _v) { m_hidden = _v; }
    void set_count(int32_t v) { m_count = v; }
    void set_max_count(int32_t v) { m_max_count = v; }
    void set_description(const std::string& v) { m_description = v; }
    void set_choices(const std::vector<std::string>& v) { m_choices = v; }
    void set_command_line(const std::vector<std::string>& v) { m_cmdline = v; }
    void set_categories(const std::set<std::string>& v) { m_categories = v; }
    auto set_updated(update_type _v) { return (std::swap(m_updated, _v), _v); }
    auto set_user_updated() { return set_updated(update_type::user); }
    auto set_config_updated() { return set_updated(update_type::config); }
    auto set_environ_updated() { return set_updated(update_type::env); }

    auto get_hidden() const { return m_hidden; }
    auto get_type_index() const { return m_type_index; }
    auto get_value_index() const { return m_value_index; }
    auto get_updated() const { return (m_updated != update_type::default_value); }
    auto get_updated_type() const { return m_updated; }
    auto get_user_updated() const { return (m_updated == update_type::user); }
    auto get_config_updated() const { return (m_updated == update_type::config); }
    auto get_environ_updated() const { return (m_updated == update_type::env); }

    // enabled = true/false does not affect the return value, it is provided
    // so that is can be specified to various tools, e.g. timemory-avail, whether
    // or not the setting is relevant in the context of the current build/configuration
    // or whether it should be added as a command-line option, etc.
    void set_enabled(bool _v) { m_enabled = _v; }
    auto get_enabled() const { return m_enabled; }

    virtual bool matches(const std::string&, bool&& exact = true) const;
    virtual bool matches(const std::string&, const std::string&, bool exact = true) const;
    virtual bool matches(const std::string&, const char*, bool _exact = true) const;

    template <typename Tp>
    std::pair<bool, Tp> get() const;

    template <typename Tp>
    bool get(Tp& _val) const;

    template <typename Tp, enable_if_t<std::is_fundamental<decay_t<Tp>>::value> = 0>
    bool set(Tp _val, update_type = update_type::unspecified);
    bool set(const std::string& _val, update_type _upd = update_type::unspecified)
    {
        return parse(_val, _upd);
    }

    virtual parser_func_t get_action(TIMEMORY_API) = 0;

    template <typename Tp>
    static auto cast(std::shared_ptr<vsettings>& _val)
    {
        return static_cast<tsettings<decay_t<Tp>, Tp>*>(_val.get());
    }

    template <typename Tp>
    static auto cast(const std::shared_ptr<vsettings>& _val)
    {
        return static_cast<const tsettings<decay_t<Tp>, Tp>*>(_val.get());
    }

    template <typename FuncT>
    void set_parse_callback(FuncT&& _func)
    {
        m_callback = std::forward<FuncT>(_func);
    }

protected:
    friend struct settings;

    static int get_debug()
    {
        static bool _bool_val = get_env(TIMEMORY_SETTINGS_PREFIX "DEBUG_SETTINGS", false);
        static int  _int_val  = get_env(TIMEMORY_SETTINGS_PREFIX "DEBUG_SETTINGS", 0);
        return (_bool_val) ? _int_val : 0;
    }

    template <typename Tp>
    bool report_change(Tp _old, const Tp& _new, update_type _upd);

    vsettings(std::string _name, std::string _env_name, std::string _descript,
              std::set<std::string> _categories, std::vector<std::string> _cmdline,
              int32_t _count, int32_t _max_count, std::vector<std::string> _choices,
              update_type _upd, bool _enabled, bool _hidden);

protected:
    bool                     m_enabled     = true;
    bool                     m_hidden      = false;
    update_type              m_updated     = update_type::default_value;
    std::type_index          m_type_index  = std::type_index(typeid(void));
    std::type_index          m_value_index = std::type_index(typeid(void));
    int32_t                  m_count       = -1;
    int32_t                  m_max_count   = -1;
    std::string              m_name        = {};
    std::string              m_env_name    = {};
    std::string              m_description = {};
    std::vector<std::string> m_cmdline     = {};
    std::vector<std::string> m_choices     = {};
    std::set<std::string>    m_categories  = {};
    parse_callback_t         m_callback    = {};
};
//
template <typename Tp>
std::pair<bool, Tp>
vsettings::get() const
{
    auto _ref = dynamic_cast<const tsettings<Tp, Tp&>*>(this);
    if(_ref)
    {
        return { true, _ref->get() };
    }

    auto _nref = dynamic_cast<const tsettings<Tp, Tp>*>(this);
    if(_nref)
        return { true, _nref->get() };

    return { false, Tp{} };
}
//
template <typename Tp>
bool
vsettings::get(Tp& _val) const
{
    auto&& _ret = this->get<Tp>();
    if(_ret.first)
        _val = _ret.second;
    return _ret.first;
}
//
template <typename Tp, enable_if_t<std::is_fundamental<decay_t<Tp>>::value>>
bool
vsettings::set(Tp _val, update_type _upd)
{
    auto _ref = dynamic_cast<tsettings<Tp, Tp&>*>(this);
    if(_ref)
    {
        _ref->set(_val, _upd);
        return true;
    }

    auto _nref = dynamic_cast<tsettings<Tp, Tp>*>(this);
    if(_nref)
    {
        _nref->set(_val, _upd);
        return true;
    }

    return false;
}
//
template <typename Tp>
bool
vsettings::report_change(Tp _old, const Tp& _new, update_type _upd)
{
    if(_upd != update_type::unspecified)
        m_updated = _upd;
    if(get_debug() >= 1 && _old != _new)
    {
        std::ostringstream oss;
        oss << std::boolalpha;
        oss << "[" << TIMEMORY_PROJECT_NAME << "][settings] " << m_name << " ("
            << m_env_name << ") changed: " << _old << " --> " << _new;
        switch(m_updated)
        {
            case update_type::config:
            {
                oss << " [via config]\n";
                break;
            }
            case update_type::env:
            {
                oss << " [via environ]\n";
                break;
            }
            case update_type::user:
            {
                oss << " [via user]\n";
                break;
            }
            case update_type::default_value:
            case update_type::unspecified:
            default:
            {
                oss << "\n";
                break;
            }
        };

        if(get_debug() >= 2)
        {
            timemory_print_demangled_backtrace<6, 3>(oss);
        }
        log::stream(std::cerr, log::color::warning()) << oss.str();
        std::cerr << std::flush;
    }
    return (_old != _new);
}
//
}  // namespace tim

#if defined(TIMEMORY_SETTINGS_HEADER_MODE)
#    include "timemory/settings/vsettings.cpp"
#endif
