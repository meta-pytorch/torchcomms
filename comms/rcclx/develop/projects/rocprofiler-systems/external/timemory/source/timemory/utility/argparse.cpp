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

#ifndef TIMEMORY_UTILITY_ARGPARSE_CPP_
#define TIMEMORY_UTILITY_ARGPARSE_CPP_

#include "timemory/log/color.hpp"
#include "timemory/log/logger.hpp"
#include "timemory/operations/types/file_output_message.hpp"
#include "timemory/tpls/cereal/archives.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/tpls/cereal/types.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/join.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#if !defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/argparse.hpp"
#endif

namespace tim
{
namespace argparse
{
TIMEMORY_UTILITY_INLINE
argument_vector::argument_vector(int& argc, char**& argv)
: base_type()
{
    reserve(argc);
    for(int i = 0; i < argc; ++i)
        emplace_back(argv[i]);
}

TIMEMORY_UTILITY_INLINE
argument_vector::argument_vector(int& argc, const char**& argv)
: base_type()
{
    reserve(argc);
    for(int i = 0; i < argc; ++i)
        emplace_back(argv[i]);
}

TIMEMORY_UTILITY_INLINE
argument_vector::argument_vector(int& argc, const char* const*& argv)
{
    reserve(argc);
    for(int i = 0; i < argc; ++i)
        emplace_back(argv[i]);
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_vector::cargs_t
argument_vector::get_execv(const base_type& _prepend, size_t _beg, size_t _end) const
{
    std::stringstream cmdss;
    // find the end if not specified
    _end = std::min<size_t>(size(), _end);
    // determine the number of arguments
    auto _argc = (_end - _beg) + _prepend.size();
    // create the new C argument array, add an extra entry at the end which will
    // always be a null pointer because that is how execv determines the end
    char** _argv = new char*[_argc + 1];

    // ensure all arguments are null pointers initially
    for(size_t i = 0; i < _argc + 1; ++i)
        _argv[i] = nullptr;

    // add the prepend list
    size_t _idx = 0;
    for(const auto& itr : _prepend)
        _argv[_idx++] = helpers::strdup(itr.c_str());

    // copy over the arguments stored internally from the range specified
    for(auto i = _beg; i < _end; ++i)
        _argv[_idx++] = helpers::strdup(this->at(i).c_str());

    // add check that last argument really is a nullptr
    assert(_argv[_argc] == nullptr);

    // create the command string
    for(size_t i = 0; i < _argc; ++i)
        cmdss << " " << _argv[i];
    auto cmd = cmdss.str().substr(1);

    // return a new (int argc, char** argv) and subtract 1 bc nullptr in last entry
    // does not count as argc
    return cargs_t(_argc - 1, _argv, cmd);
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_vector::cargs_t
argument_vector::get_execv(size_t _beg, size_t _end) const
{
    return get_execv(base_type{}, _beg, _end);
}

TIMEMORY_UTILITY_INLINE
argument_parser::argument&
argument_parser::argument::choice_aliases(
    const std::map<std::string, std::vector<std::string>>& _v)
{
    for(const auto& itr : _v)
    {
        TIMEMORY_REQUIRE(m_choices.find(itr.first) != m_choices.end())
            << "Error! provided choice aliases for choice which does not exist: "
            << itr.first << " for command line argument "
            << ((m_names.empty() ? "unknown" : m_names.back())) << "\n";

        for(const auto& vitr : itr.second)
        {
            m_choice_aliases[itr.first].emplace_back(vitr);
        }
    }
    return *this;
}

TIMEMORY_UTILITY_INLINE
argument_parser::argument&
argument_parser::argument::choice_alias(
    const std::string& _choice, const std::initializer_list<std::string>& _aliases)
{
    for(const auto& itr : _aliases)
    {
        TIMEMORY_REQUIRE(m_choices.find(_choice) != m_choices.end())
            << "Error! provided choice aliases for choice which does not exist: "
            << _choice << " for command line argument "
            << ((m_names.empty() ? "unknown" : m_names.back())) << "\n";

        m_choice_aliases[_choice].emplace_back(itr);
    }
    return *this;
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::arg_result
argument_parser::argument::check_choice(std::string& value) const
{
    if(!m_choices.empty())
    {
        auto citr = m_choices.find(value);

        // if value not initially found, check aliases. if value aliases a choice,
        // re-assign value to the name of the choice that it aliases
        if(citr == m_choices.end())
        {
            auto used_alias = std::string{};
            auto init_value = value;
            for(const auto& aitr : m_choice_aliases)
            {
                for(const auto& itr : aitr.second)
                {
                    if(value == itr)
                    {
                        TIMEMORY_REQUIRE(citr == m_choices.end() &&
                                         used_alias != aitr.first)
                            << "Error! " << init_value << " was already aliased to "
                            << value << " via alias for " << used_alias
                            << ". argument parser specifed the same alias for difference "
                               "choices\n";
                        value      = aitr.first;
                        used_alias = aitr.first;
                        citr       = m_choices.find(value);
                        break;  // break out of alias loop for this choice only
                    }
                }
            }
        }

        if(citr == m_choices.end())
        {
            std::stringstream ss;
            ss << "Invalid choice: '" << value << "'. Valid choices: ";
            for(const auto& itr : m_choices)
            {
                ss << "'" << itr << "' ";
                if(m_choice_aliases.find(itr) != m_choice_aliases.end())
                {
                    ss << "(aliases: ";
                    for(const auto& aitr : m_choice_aliases.at(itr))
                        ss << "'" << aitr << "' ";
                    ss << ")";
                }
            }
            return arg_result(ss.str());
        }
    }
    return arg_result{};
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
std::string
argument_parser::argument::as_string() const
{
    std::stringstream ss;
    ss << "names: ";
    for(const auto& itr : m_names)
        ss << itr << " ";
    ss << ", index: " << m_index << ", count: " << m_count
       << ", min count: " << m_min_count << ", max count: " << m_max_count
       << ", found: " << std::boolalpha << m_found << ", required: " << std::boolalpha
       << m_required << ", position: " << m_position << ", values: ";
    for(const auto& itr : m_values)
        ss << itr << " ";
    return ss.str();
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
bool
argument_parser::argument::is_separator() const
{
    int32_t _v = m_count + m_min_count + m_max_count;
    int32_t _l = m_desc.length() + m_dtype.length() + m_choices.size() + m_values.size() +
                 m_actions.size();
    if((_v + _l) == -3)
    {
        if(m_names.empty())
            return true;
        else if(m_names.size() == 1)
        {
            if(m_names.at(0).empty() ||
               (m_names.at(0).front() == '[' && m_names.at(0).back() == ']'))
                return true;
        }
    }
    return false;
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
std::ostream*
argument_parser::set_ostream(std::ostream* _v)
{
    std::swap(m_clog, _v);
    return _v;
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::argument&
argument_parser::enable_help()
{
    m_help_enabled = true;
    return add_argument()
        .names({ "-h", "-?", "--help" })
        .description("Shows this page")
        .count(0);
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::argument&
argument_parser::enable_help(const std::string _extra, const std::string _epilogue,
                             int _exit_code)
{
    m_help_enabled = true;
    return add_argument()
        .names({ "-h", "-?", "--help" })
        .description("Shows this page")
        .count(0)
        .action([_exit_code, _extra, _epilogue](argument_parser& _p) {
            _p.print_help(_extra, _epilogue);
            exit(_exit_code);
        });
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::argument&
argument_parser::enable_serialize()
{
    m_serialize_enabled = true;
    return add_argument()
        .names({ "--serialize-argparser" })
        .description("Serializes the instance to provided JSON")
        .dtype("filepath")
        .count(1);
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
bool
argument_parser::exists(const std::string& name) const
{
    std::string n =
        helpers::ltrim(name, [](int c) -> bool { return c != static_cast<int>('-'); });
    auto itr = m_name_map.find(n);
    if(itr != m_name_map.end())
        return m_arguments[static_cast<size_t>(itr->second)].m_found;
    return false;
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
int64_t
argument_parser::get_count(const std::string& name)
{
    auto itr = m_name_map.find(name);
    if(itr != m_name_map.end())
        return m_arguments[static_cast<size_t>(itr->second)].size();
    return 0;
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::argument&
argument_parser::enable_version(
    const std::string& _name, std::vector<int> _versions, const std::string& _tag,
    const std::string&                                      _rev,
    const std::vector<std::pair<std::string, std::string>>& _properties)
{
    return enable_version(
        _name, timemory::join::join(timemory::join::array_config{ "." }, _versions), _tag,
        _rev, _properties);
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::argument&
argument_parser::enable_version(
    const std::string& _name, const std::string& _version, const std::string& _tag,
    const std::string&                                      _rev,
    const std::vector<std::pair<std::string, std::string>>& _properties)
{
    auto** _cout_v = &m_clog;
    return add_argument()
        .names({ "--version" })
        .description("Prints the version and exit")
        .count(0)
        .action([_name, _version, _tag, _rev, _properties, _cout_v](argument_parser&) {
            namespace join  = ::timemory::join;
            using strpair_t = std::pair<std::string, std::string>;

            auto* _cout = *_cout_v;
            if(!_cout)
                _cout = &std::cout;

            // <NAME> <VERSION>
            (*_cout) << _name << " " << _version << std::flush;

            // assemble the list of properties
            auto _property_info  = std::vector<std::string>{};
            auto _add_properties = [&_property_info](
                                       const std::vector<strpair_t>& _data) {
                for(const auto& itr : _data)
                {
                    if(!itr.second.empty())
                        _property_info.emplace_back(
                            itr.first.empty() ? itr.second
                                              : join::join(": ", itr.first, itr.second));
                }
            };
            _add_properties({ { "rev", _rev }, { "tag", _tag } });
            _add_properties(_properties);

            if(!_property_info.empty())
            {
                // <NAME> <VERSION> (<PROPERTIES>)
                (*_cout) << join::join(join::array_config{ ", ", " (", ")" },
                                       _property_info);
            }

            // end the line and exit
            (*_cout) << std::endl;
            exit(EXIT_SUCCESS);
        });
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
void
argument_parser::print_help(const std::string _extra, const std::string _epilogue)
{
    end_group();

    std::stringstream _usage;
    if(!m_desc.empty())
        _usage << "[" << m_desc << "] ";
    _usage << "Usage: " << m_bin;

    if(!m_clog)
        m_clog = &std::clog;

    (*m_clog) << _usage.str();

    auto _get_usage_desc = [](const argument& _arg) {
        auto _usage_desc = std::stringstream{};
        _usage_desc << " (";
        if(_arg.m_count != argument::Count::ANY)
            _usage_desc << "count: " << _arg.m_count;
        else if(_arg.m_min_count != argument::Count::ANY)
            _usage_desc << "min: " << _arg.m_min_count;
        else if(_arg.m_max_count != argument::Count::ANY)
            _usage_desc << "max: " << _arg.m_max_count;
        else
            _usage_desc << "count: unlimited";
        if(!_arg.m_dtype.empty())
            _usage_desc << ", dtype: " << _arg.m_dtype;
        else if(_arg.m_count == 0 ||
                (_arg.m_count == argument::Count::ANY && _arg.m_max_count == 1))
            _usage_desc << ", dtype: bool" << _arg.m_dtype;
        _usage_desc << ")";
        return _usage_desc.str();
    };

    std::stringstream _sshort_desc;
    auto              _indent = _usage.str().length() + 2;
    size_t            _ncnt   = 0;
    for(auto& a : m_arguments)
    {
        std::string name = a.m_names.at(0);
        if(name.empty() || name.find_first_of('-') > name.find_first_not_of(" -"))
            continue;
        // select the first long option
        for(size_t n = 1; n < a.m_names.size(); ++n)
        {
            if(name.find("--") == 0)
                break;
            else if(a.m_names.at(n).find("--") == 0)
            {
                name = a.m_names.at(n);
                break;
            }
        }
        if(name.length() > 0)
        {
            if(_ncnt++ > 0)
                _sshort_desc << "\n " << std::setw(_indent) << " " << name;
            else
                _sshort_desc << " " << name;

            _sshort_desc << _get_usage_desc(a);
            a.m_desc += _get_usage_desc(a);
        }
    }

    std::string _short_desc;
    if(!_sshort_desc.str().empty())
    {
        _short_desc.append("[" + _sshort_desc.str());
        std::stringstream _tmp;
        _tmp << "\n" << std::setw(_indent) << "]";
        _short_desc.append(_tmp.str());
    }

    if(m_positional_arguments.empty())
    {
        (*m_clog) << " " << _short_desc << " " << _extra << std::endl;
    }
    else
    {
        (*m_clog) << " " << _short_desc;
        if(!_short_desc.empty())
            (*m_clog) << "\n" << std::setw(_indent - 2) << " ";
        for(auto& itr : m_positional_arguments)
        {
            (*m_clog) << " " << helpers::ltrim(itr.m_names.at(0), [](int c) -> bool {
                return c != static_cast<int>('-');
            });
        }

        int current = 0;
        for(auto& v : m_positional_map)
        {
            if(v.first != argument::Position::LastArgument)
            {
                for(; current < v.first; ++current)
                    (*m_clog) << " [" << current << "]";
                (*m_clog) << " ["
                          << helpers::ltrim(
                                 m_arguments[static_cast<size_t>(v.second)].m_names.at(0),
                                 [](int c) -> bool { return c != static_cast<int>('-'); })
                          << "]";
            }
            else
            {
                (*m_clog) << " ... ["
                          << helpers::ltrim(
                                 m_arguments[static_cast<size_t>(v.second)].m_names.at(0),
                                 [](int c) -> bool { return c != static_cast<int>('-'); })
                          << "]";
            }
        }
        (*m_clog) << " " << _extra << std::endl;
    }

    if(!m_long_desc.empty())
        (*m_clog) << m_long_desc << std::endl;

    (*m_clog) << "\nOptions:" << std::endl;
    size_t _arguments_idx = 0;
    for(auto& a : m_arguments)
    {
        auto      _idx      = ++_arguments_idx;
        argument* _next_arg = nullptr;
        if(_idx < m_arguments.size())
            _next_arg = &m_arguments.at(_idx);
        bool _next_arg_is_spacer = (_next_arg) ? _next_arg->get_name().empty() : false;

        (void) _next_arg_is_spacer;

        auto   ss     = std::stringstream{};
        auto   _name  = std::stringstream{};
        size_t _width = 0;
        {
            auto _nprefix = (m_use_color) ? a.m_color : std::string{};
            auto _nsuffix = (_nprefix.empty()) ? std::string{} : log::color::end();
            _name << _nprefix << a.m_names.at(0) << _nsuffix;
            for(size_t n = 1; n < a.m_names.size(); ++n)
                _name << ", " << _nprefix << a.m_names.at(n) << _nsuffix;
            ss << _name.str();
        }

        for(const auto& itr : a.m_names)
            _width += itr.length() + 2;

        if(_width >= 2 && a.m_names.size() > 1)
            _width -= 2;

        if(!a.m_choices.empty())
        {
            auto _choice_names = std::vector<std::string>{};
            for(const auto& itr : a.m_choices)
            {
                std::string _choice = itr;
                if(a.m_choice_aliases.find(itr) != a.m_choice_aliases.end())
                {
                    _choice += " " + timemory::join::join(
                                         timemory::join::array_config{ "|", "(", ")" },
                                         a.m_choice_aliases.at(itr));
                }
                _choice_names.emplace_back(_choice);
            }
            auto _choices = timemory::join::join(
                timemory::join::array_config{ " | ", "[ ", " ]" }, _choice_names);

            if(_width + _choices.length() <
               static_cast<size_t>(m_width + m_desc_width + 8))
                _width += _choices.length();
            else
            {
                auto _spacer = std::stringstream{};
                // 5 == four spaces + space after option name(s)
                _spacer << std::setw(_width + 5) << "";
                _choices = timemory::join::join(
                    timemory::join::array_config{ std::string{ "\n" } + _spacer.str(),
                                                  "[ ", " ]" },
                    _choice_names);
                _spacer = std::stringstream{};
                _spacer << "\n"
                        << std::setw(m_width) << ""
                        << "    ";
                _choices += _spacer.str();
            }
            ss << " " << _choices;
        }

        std::stringstream prefix;
        prefix << "    " << std::setw(m_width) << std::left << ss.str();
        (*m_clog) << std::left << prefix.str();

        if(static_cast<int64_t>(_width) >= static_cast<int64_t>(m_width))
            (*m_clog) << "\n" << std::setw(m_width + 4) << "";

        bool _autoformat = a.m_desc.find("%{NEWLINE}%") == std::string::npos &&
                           a.m_desc.find("%{INDENT}%") == std::string::npos;

        std::stringstream _newline;
        _newline << "\n" << std::setw(m_width + 5) << "";

        std::stringstream _indent_opt;
        _indent_opt << std::setw(m_width + 5) << "";

        auto _replace = [](std::string& _desc, const std::string& _key,
                           const std::string& _str) {
            const auto npos = std::string::npos;
            auto       pos  = npos;
            while((pos = _desc.find(_key)) != npos)
                _desc = _desc.replace(pos, _key.length(), _str);
        };

        if(!_autoformat)
        {
            auto desc = ((ss.str().length() > static_cast<size_t>(m_width))
                             ? std::string{ "%{NEWLINE}%" }
                             : std::string{}) +
                        a.m_desc;

            if(a.m_required)
                desc += " (Required)";

            _replace(desc, "%{INDENT}%", _indent_opt.str());
            _replace(desc, "%{NEWLINE}%", _newline.str());

            (*m_clog) << " " << std::setw(m_width) << desc;
        }
        else
        {
            std::string desc = a.m_desc;

            if(a.m_required)
                desc += " (Required)";

            _replace(desc, "%{INDENT}%", _indent_opt.str());
            _replace(desc, "%{NEWLINE}%", _newline.str());

            auto              _desc = delimit(desc, " \n");
            std::stringstream _desc_ss{};
            size_t            _w = 0;
            for(auto& itr : _desc)
            {
                if(itr.length() > m_desc_width)
                {
                    _desc_ss << itr << _newline.str();
                    _w = 0;
                }
                else if(_w + itr.length() < m_desc_width)
                {
                    _desc_ss << itr << " ";
                    _w += itr.length() + 1;
                }
                else
                {
                    _desc_ss << _newline.str() << itr << " ";
                    _w = itr.length() + 1;
                }
            }
            desc = _desc_ss.str();

            (*m_clog) << " " << std::setw(m_width) << desc;
        }

        (*m_clog) << std::endl;
    }
    (*m_clog) << _epilogue << std::endl;
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::arg_result
argument_parser::parse_known_args(int* argc, char*** argv, strvec_t& _args,
                                  const std::string& _delim, int verbose_level)
{
    // check for help flag
    auto help_check = [&](int _argc, char** _argv) {
        strset_t help_args = { "-h", "--help", "-?" };
        auto     _help_req = (exists("help") ||
                          (_argc > 1 && help_args.find(_argv[1]) != help_args.end()));
        if(_help_req && !exists("help"))
        {
            for(const auto& hitr : help_args)
            {
                auto hstr = hitr.substr(hitr.find_first_not_of('-'));
                auto itr  = m_name_map.find(hstr);
                if(itr != m_name_map.end())
                    m_arguments[static_cast<size_t>(itr->second)].m_found = true;
            }
        }
        return _help_req;
    };

    // check for a dash in th command line
    bool _pdash = false;
    for(int i = 1; i < *argc; ++i)
    {
        if(std::string_view{ (*argv)[i] } == _delim)
            _pdash = true;
    }

    // parse the known args and get the remaining argc/argv
    auto  _pargs = parse_known_args(*argc, *argv, _args, _delim, verbose_level);
    auto  _perrc = std::get<0>(_pargs);
    auto  _pargc = std::get<1>(_pargs);
    auto* _pargv = std::get<2>(_pargs);

    // check if help was requested before the dash (if dash exists)
    if(help_check((_pdash) ? 0 : _pargc, _pargv))
        return arg_result{ "help requested" };

    // assign the argc and argv
    *argc = _pargc;
    *argv = _pargv;

    return _perrc;
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::known_args
argument_parser::parse_known_args(int argc, char** argv, strvec_t& _args,
                                  const std::string& _delim, int verbose_level)
{
    int    _cmdc = argc;  // the argc after known args removed
    char** _cmdv = argv;  // the argv after known args removed
    // _cmdv and argv are same pointer unless delimiter is found

    if(argc > 0)
    {
        m_bin = std::string((const char*) argv[0]);
        _args.emplace_back(std::string((const char*) argv[0]));
    }

    for(int i = 1; i < argc; ++i)
    {
        std::string _arg = argv[i];
        if(_arg == _delim)
        {
            _cmdc        = argc - i;
            _cmdv        = new char*[_cmdc + 1];
            _cmdv[_cmdc] = nullptr;
            _cmdv[0]     = helpers::strdup(argv[0]);
            int k        = 1;
            for(int j = i + 1; j < argc; ++j, ++k)
                _cmdv[k] = helpers::strdup(argv[j]);
            break;
        }
        else
        {
            _args.emplace_back(std::string((const char*) argv[i]));
        }
    }

    auto cmd_string = [](int _ac, char** _av) {
        std::stringstream ss;
        for(int i = 0; i < _ac; ++i)
            ss << _av[i] << " ";
        return ss.str();
    };

    if(verbose_level >= 3)
    {
        namespace join = ::timemory::join;
        TIMEMORY_PRINTF(stderr, "[argparse][original] %s\n", cmd_string(argc, argv));
        TIMEMORY_PRINTF(stderr, "[argparse][cfg-args] %s\n",
                        join::join(join::array_config{ " " }, _args));

        if(_cmdc > 0)
        {
            TIMEMORY_PRINTF(stderr, "[argparse][command] %s\n", cmd_string(_cmdc, _cmdv));
        }
    }

    return known_args{ parse(_args, verbose_level), _cmdc, _cmdv };
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::arg_result
argument_parser::parse(int argc, char** argv, std::string_view _delim, int verbose_level)
{
    std::vector<std::string> _args;
    _args.reserve(argc);
    for(int i = 0; i < argc; ++i)
    {
        if(std::string_view{ argv[i] } == _delim)
            break;
        _args.emplace_back((const char*) argv[i]);
    }
    return parse(_args, verbose_level);
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::arg_result
argument_parser::parse(int argc, char** argv, int verbose_level, std::string_view _delim)
{
    return parse(argc, argv, _delim, verbose_level);
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::arg_result
argument_parser::parse(const std::vector<std::string>& _args, int verbose_level)
{
    if(verbose_level > 0)
    {
        std::stringstream _msg{};
        for(const auto& itr : _args)
            _msg << " " << itr;
        auto _cmd = _msg.str();
        if(_cmd.length() > 0)
        {
            _cmd = _cmd.substr(1);
            if(verbose_level >= 2)
            {
                if(!m_clog)
                    m_clog = &std::clog;
                log::stream(*m_clog, log::color::info())
                    << "[argparse::parse]> parsing '" << _cmd << "'...\n";
            }
        }
    }

    for(auto& a : m_arguments)
        a.m_callback(a.m_default);
    for(auto& a : m_positional_arguments)
        a.m_callback(a.m_default);

    using argmap_t = std::map<std::string, argument*>;

    argmap_t   m_arg_map = {};
    arg_result err;
    int        argc = _args.size();
    // the set of options which use a single leading dash but are longer than
    // one character, e.g. -LS ...
    std::set<std::string> long_short_opts;
    if(_args.size() > 1)
    {
        auto is_leading_dash = [](int c) -> bool { return c != static_cast<int>('-'); };
        // build name map
        for(auto& a : m_arguments)
        {
            for(auto& n : a.m_names)
            {
                auto        nleading_dash = helpers::lcount(n, is_leading_dash);
                std::string name          = helpers::ltrim(n, is_leading_dash);
                if(name.empty())
                    continue;
                if(m_name_map.find(name) != m_name_map.end())
                    return construct_error("Duplicate of argument name: " + n);
                m_name_map[name] = a.m_index;
                m_arg_map[name]  = &a;
                if(nleading_dash == 1 && name.length() > 1)
                    long_short_opts.insert(name);
            }
            if(a.m_position >= 0 || a.m_position == argument::Position::LastArgument)
                m_positional_map.at(a.m_position) = a.m_index;
        }

        m_bin = _args.at(0);

        // parse
        std::string current_arg;
        size_t      arg_len;
        for(int argv_index = 1; argv_index < argc; ++argv_index)
        {
            current_arg = _args.at(argv_index);
            arg_len     = current_arg.length();
            if(arg_len == 0)
                continue;
            if(argv_index == argc - 1 &&
               m_positional_map.find(argument::Position::LastArgument) !=
                   m_positional_map.end())
            {
                err          = end_argument();
                arg_result b = err;
                err          = add_value(current_arg, argument::Position::LastArgument);
                if(b)
                    return b;
                // return (m_error_func(*this, b), b);
                if(err)
                    return (m_error_func(*this, err), err);
                continue;
            }

            // count number of leading dashes
            auto nleading_dash = helpers::lcount(current_arg, is_leading_dash);
            // ignores the case if the arg is just a '-'
            // look for -a (short) or --arg (long) args
            bool is_arg = (nleading_dash > 0 && arg_len > 1 && arg_len != nleading_dash)
                              ? true
                              : false;

            if(is_arg && !helpers::is_numeric(current_arg))
            {
                err = end_argument();
                if(err)
                    return (m_error_func(*this, err), err);

                auto name   = current_arg.substr(nleading_dash);
                auto islong = (nleading_dash > 1 || long_short_opts.count(name) > 0);
                err         = begin_argument(name, islong, argv_index);
                if(err)
                    return (m_error_func(*this, err), err);
            }
            else if(current_arg.length() > 0)
            {
                // argument value
                err = add_value(current_arg, argv_index);
                if(err)
                    return (m_error_func(*this, err), err);
            }
        }
    }

    err = end_argument();
    if(err)
        return (m_error_func(*this, err), err);

    if(!exists("help"))
    {
        // check requirements
        for(auto& a : m_arguments)
        {
            if(a.m_required && !a.m_found)
            {
                return construct_error("Required argument not found: " + a.m_names.at(0) +
                                       a.m_required_info);
            }
            if(a.m_position >= 0 && argc >= a.m_position && !a.m_found)
            {
                return construct_error("argument " + a.m_names.at(0) +
                                       " expected in position " +
                                       std::to_string(a.m_position) + a.m_required_info);
            }
        }

        // check requirements
        for(auto& a : m_positional_arguments)
        {
            if(a.m_required && !a.m_found)
                return construct_error("Required argument not found: " + a.m_names.at(0) +
                                       a.m_required_info);
        }
    }

    // check all the counts have been satisfied
    for(auto& a : m_arguments)
    {
        if(a.m_found && a.m_default == nullptr)
        {
            auto cnt_err = check_count(a);
            if(cnt_err)
                return cnt_err;
        }
    }

    // execute the global actions
    for(auto& itr : m_actions)
    {
        if(itr.first(*this))
            itr.second(*this);
    }

    // execute the argument-specific actions
    for(auto& itr : m_arg_map)
    {
        if(exists(itr.first) || itr.second->m_default)
            itr.second->execute_actions(*this);
    }

    // check all requirements have been satisfied and conflicts have not violated
    for(auto& itr : m_arguments)
    {
        if(itr.m_found)
        {
            for(const auto& iitr : itr.m_requires)
            {
                if(iitr.empty())
                    continue;
                if(iitr.find('|') != std::string::npos)
                {
                    bool                  _found = false;
                    std::set<std::string> _opts{};
                    for(auto&& oitr : delimit(iitr, "|"))
                    {
                        _opts.emplace(std::string{ "--" } + oitr);
                        if(exists(oitr))
                            _found = true;
                    }
                    if(!_found)
                    {
                        using namespace timemory::join;
                        return construct_error(
                            itr.get_name() +
                            " requires one of the options: " + join("", _opts));
                    }
                }
                else if(!exists(iitr))
                {
                    return construct_error(itr.get_name() + " requires option --" + iitr);
                }
            }
            for(const auto& iitr : itr.m_conflicts)
            {
                if(exists(iitr))
                    return construct_error(itr.get_name() + " conflicts with option --" +
                                           iitr);
            }
        }
    }

    // return the help
    if(m_help_enabled && exists("help"))
        return arg_result("help requested");

    if(m_serialize_enabled && exists("serialize-argparser"))
    {
        auto _fname = get<std::string>("serialize-argparser");
        auto _ss    = std::stringstream{};
        {
            cereal::PrettyJSONOutputArchive _ar{ _ss };
            _ar.setNextName("timemory");
            _ar.startNode();
            _ar(cereal::make_nvp("argument_parser", *this));
            _ar.finishNode();
        }
        std::ofstream _ofs{};
        if(filepath::open(_ofs, _fname))
        {
            operation::file_output_message<argument_parser>{}(
                _fname, std::string{ "argument_parser" });
            _ofs << _ss.str() << "\n";
        }
        _ofs.close();
        std::exit(EXIT_SUCCESS);
    }

    return arg_result{};
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::arg_result
argument_parser::begin_argument(const std::string& arg, bool longarg, int position)
{
    auto it = m_positional_map.find(position);
    if(it != m_positional_map.end())
    {
        arg_result err = end_argument();
        argument&  a   = m_arguments[static_cast<size_t>(it->second)];
        a.m_values.emplace_back(arg);
        a.m_found = true;
        return err;
    }
    if(m_current != -1)
    {
        return construct_error("Current argument left open");
    }
    size_t      name_end = helpers::find_punct(arg);
    std::string arg_name = arg.substr(0, name_end);
    if(longarg)
    {
        int  equal_pos = helpers::find_equiv(arg);
        auto nmf       = m_name_map.find(arg_name);
        if(nmf == m_name_map.end())
        {
            arg_name = arg.substr(0, equal_pos);
            nmf      = m_name_map.find(arg_name);
        }
        if(nmf == m_name_map.end())
        {
            return construct_error("Unrecognized command line option '" + arg_name + "'");
        }
        m_current                                             = nmf->second;
        m_arguments[static_cast<size_t>(nmf->second)].m_found = true;
        if(equal_pos == 0 || (equal_pos < 0 && arg_name.length() < arg.length()))
        {
            // malformed argument
            return construct_error("Malformed argument: " + arg);
        }
        else if(equal_pos > 0)
        {
            std::string arg_value = arg.substr(name_end + 1);
            return add_value(arg_value, position);
        }
    }
    else
    {
        arg_result r;
        if(arg_name.length() == 1)
        {
            return begin_argument(arg, true, position);
        }
        else
        {
            for(char& c : arg_name)
            {
                r = begin_argument(std::string(1, c), true, position);
                if(r)
                {
                    return r;
                }
                r = end_argument();
                if(r)
                {
                    return r;
                }
            }
        }
    }
    return arg_result{};
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::arg_result
argument_parser::add_value(std::string& value, int location)
{
    auto unnamed = [&]() {
        auto itr = m_positional_map.find(location);
        if(itr != m_positional_map.end())
        {
            argument& a = m_arguments[static_cast<size_t>(itr->second)];
            a.m_values.emplace_back(value);
            a.m_found = true;
        }
        else
        {
            auto idx = m_positional_values.size();
            m_positional_values.emplace(idx, value);
            if(idx < m_positional_arguments.size())
            {
                auto& a   = m_positional_arguments.at(idx);
                a.m_found = true;
                auto err  = a.check_choice(value);
                if(err)
                    return err;
                a.m_values.emplace_back(value);
                a.execute_actions(*this);
            }
        }
        return arg_result{};
    };

    if(m_current >= 0)
    {
        arg_result err;
        size_t     c = static_cast<size_t>(m_current);
        consume_parameters(c);
        argument& a = m_arguments[static_cast<size_t>(m_current)];

        err = a.check_choice(value);
        if(err)
            return err;

        auto num_values = [&]() { return static_cast<int>(a.m_values.size()); };

        // check {m_count, m_max_count} > COUNT::ANY && m_values.size() >= {value}
        if((a.m_count >= 0 && num_values() >= a.m_count) ||
           (a.m_max_count >= 0 && num_values() >= a.m_max_count))
        {
            err = end_argument();
            if(err)
                return err;
            return unnamed();
        }

        a.m_values.emplace_back(value);

        // check {m_count, m_max_count} > COUNT::ANY && m_values.size() >= {value}
        if((a.m_count >= 0 && num_values() >= a.m_count) ||
           (a.m_max_count >= 0 && num_values() >= a.m_max_count))
        {
            err = end_argument();
            if(err)
                return err;
        }
        return arg_result{};
    }

    return unnamed();
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser::arg_result
argument_parser::end_argument()
{
    if(m_current >= 0)
    {
        argument& a = m_arguments[static_cast<size_t>(m_current)];
        m_current   = -1;
        if(static_cast<int>(a.m_values.size()) < a.m_count)
            return construct_error("Too few arguments given for " + a.m_names.at(0));
        if(a.m_max_count >= 0)
        {
            if(static_cast<int>(a.m_values.size()) > a.m_max_count)
                return construct_error("Too many arguments given for " + a.m_names.at(0));
        }
        else if(a.m_count >= 0)
        {
            if(static_cast<int>(a.m_values.size()) > a.m_count)
                return construct_error("Too many arguments given for " + a.m_names.at(0));
        }
    }
    return arg_result{};
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
std::ostream&
operator<<(std::ostream& os, const argument_parser::arg_result& r)
{
    os << r.what();
    return os;
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser&
argument_parser::start_group(std::string _v, const std::string& _desc)
{
    if(!m_group.empty())
        end_group();
    m_group           = _v;
    std::string _name = "[";
    for(auto& itr : _v)
        itr = toupper(itr);
    bool _has_options = (_v.find(" OPTIONS") != std::string::npos);
    _name += _v + ((_has_options) ? std::string{ "]" } : std::string{ " OPTIONS]" });

    if(!m_arguments.back().is_separator())
        add_argument({ "" }, "");

    add_argument({ _name }, _desc).color((m_use_color) ? m_color : "");
    add_argument({ "" }, "");

    return *this;
}

// clang-format off
TIMEMORY_UTILITY_INLINE
// clang-format on
argument_parser&
argument_parser::end_group()
{
    if(!m_group.empty())
    {
        add_argument({ "" }, "");
        m_group = std::string{};
    }
    return *this;
}
}  // namespace argparse
}  // namespace tim

#endif
