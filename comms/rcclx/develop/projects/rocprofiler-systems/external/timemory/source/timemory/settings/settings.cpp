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

#ifndef TIMEMORY_SETTINGS_SETTINGS_CPP_
#define TIMEMORY_SETTINGS_SETTINGS_CPP_

#if !defined(__USE_XOPEN_EXTENDED)
#    define __USE_XOPEN_EXTENDED
#endif

#include "timemory/settings/settings.hpp"

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/defines.h"
#include "timemory/mpl/policy.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/tpls/cereal/archives.hpp"
#include "timemory/utility/argparse.hpp"
#include "timemory/utility/declaration.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/md5.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/variadic/macros.hpp"

#include <cctype>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <locale>
#include <regex>
#include <string>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
const std::shared_ptr<settings>&
settings::shared_instance()
{
    // do not take reference to ensure push/pop w/o template parameters do not change
    // the settings
    static auto _instance = shared_instance<TIMEMORY_API>();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings*
settings::instance()
{
    // do not take reference to ensure push/pop w/o template parameters do not change
    // the settings
    static const auto& _instance = shared_instance();
    return _instance.get();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::strvector_t&
settings::command_line()
{
    return instance()->get_command_line();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::strvector_t&
settings::environment()
{
    return instance()->get_environment();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::strvector_t
settings::get_global_environment()
{
#if defined(TIMEMORY_UNIX)
    strvector_t _environ;
    if(environ != nullptr)
    {
        int idx = 0;
        while(environ[idx] != nullptr)
            _environ.push_back(environ[idx++]);
    }
    return _environ;
#else
    return std::vector<std::string>();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
get_local_datetime(const char* dt_format, std::time_t* dt_curr)
{
    char mbstr[512];
    if(!dt_curr)
        dt_curr = settings::get_launch_time(TIMEMORY_API{});

    if(std::strftime(mbstr, sizeof(mbstr), dt_format, std::localtime(dt_curr)) != 0)
        return std::string{ mbstr };
    return std::string{};
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::tolower(std::string str)
{
    for(auto& itr : str)
        itr = ::tolower(itr);
    return str;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::toupper(std::string str)
{
    for(auto& itr : str)
        itr = ::toupper(itr);
    return str;
}
//
//--------------------------------------------------------------------------------------//
//
namespace
{
inline std::string&
_get_prefix_with_subdirectory(std::string& _prefix, std::string _subdir)
{
    if(_subdir.empty())
        return _prefix;
    if(_subdir.back() != '/')
        _subdir += "/";
    if(_prefix.empty())
        _prefix = _subdir;
    else
    {
        if(_prefix.back() == '-')
            _prefix = _subdir + _prefix;
        else if(_prefix.back() == '/')
            _prefix += _subdir;
        else
            _prefix += std::string{ "/" } + _subdir;
    }
    return _prefix;
}
}  // namespace
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::get_global_input_prefix(std::string _subdir)
{
    auto _dir    = input_path();
    auto _prefix = input_prefix();

    _get_prefix_with_subdirectory(_prefix, std::move(_subdir));

    return filepath::osrepr(_dir + std::string{ "/" } + _prefix);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::get_global_output_prefix(bool _make_dir, std::string _subdir)
{
    using str_t = std::string;

    auto _out_path =
        get_with_env_fallback<str_t>(TIMEMORY_SETTINGS_KEY("OUTPUT_PATH"), ".");
    auto _out_prefix =
        get_with_env_fallback<str_t>(TIMEMORY_SETTINGS_KEY("OUTPUT_PREFIX"), "");
    auto _time_format =
        get_with_env_fallback<str_t>(TIMEMORY_SETTINGS_KEY("TIME_FORMAT"), "%F_%H.%M");
    auto _time_output =
        get_with_env_fallback<bool>(TIMEMORY_SETTINGS_KEY("TIME_OUTPUT"), false);

    if(_time_output)
    {
        // get the statically stored launch time
        auto* _launch_time    = get_launch_time(TIMEMORY_API{});
        auto  _local_datetime = get_local_datetime(_time_format.c_str(), _launch_time);
        if(_out_path.find(_local_datetime) == std::string::npos)
        {
            if(_out_path.length() > 0 && _out_path[_out_path.length() - 1] != '/')
                _out_path += "/";
            _out_path += _local_datetime;
        }
    }

    _get_prefix_with_subdirectory(_out_prefix, std::move(_subdir));

    // always return zero if not making dir. if makedir failed, don't prefix with
    // directory
    auto ret = (_make_dir) ? makedir(_out_path) : 0;
    return (ret == 0) ? filepath::osrepr(_out_path + std::string{ "/" } + _out_prefix)
                      : filepath::osrepr(std::string{ "./" } + _out_prefix);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::store_command_line(int argc, char** argv)
{
    auto& _v = command_line();
    _v.clear();
    for(int i = 0; i < argc; ++i)
    {
        _v.emplace_back(std::string(argv[i]));
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE std::vector<settings::output_key>
                         settings::output_keys(const std::string& _tag)
{
    auto        _cmdline     = command_line();
    std::string _argv_string = {};    // entire argv cmd
    std::string _args_string = {};    // cmdline args
    std::string _argt_string = _tag;  // prefix + cmdline args
    std::string _tag0_string = _tag;  // only the basic prefix
    auto        _options     = std::vector<output_key>{};

    auto _replace = [](auto& _v, const strpair_t& pitr) {
        auto pos = std::string::npos;
        while((pos = _v.find(pitr.first)) != std::string::npos)
            _v.replace(pos, pitr.first.length(), pitr.second);
    };

    if(_cmdline.size() > 1 && _cmdline.at(1) == "--")
        _cmdline.erase(_cmdline.begin() + 1);

    for(auto& itr : _cmdline)
    {
        itr = argparse::helpers::trim(itr);
        _replace(itr, { "/", "_" });
        while(!itr.empty() && itr.at(0) == '.')
            itr = itr.substr(1);
        while(!itr.empty() && itr.at(0) == '_')
            itr = itr.substr(1);
    }

    if(!_cmdline.empty())
    {
        for(size_t i = 0; i < _cmdline.size(); ++i)
        {
            const auto _l = std::string{ (i == 0) ? "" : "_" };
            auto       _v = _cmdline.at(i);
            _argv_string += _l + _v;
            if(i > 0)
            {
                _argt_string += (i > 1) ? (_l + _v) : _v;
                _args_string += (i > 1) ? (_l + _v) : _v;
            }
        }
    }

    auto* _launch_time = get_launch_time(TIMEMORY_API{});
    auto  _time_format = get_with_env_fallback<std::string>(
        TIMEMORY_SETTINGS_KEY("TIME_FORMAT"), "%F_%H.%M");

    auto _dmp_size      = TIMEMORY_JOIN("", dmp::size());
    auto _dmp_rank      = TIMEMORY_JOIN("", dmp::rank());
    auto _proc_id       = TIMEMORY_JOIN("", process::get_id());
    auto _parent_id     = TIMEMORY_JOIN("", process::get_parent_id());
    auto _pgroup_id     = TIMEMORY_JOIN("", process::get_group_id());
    auto _session_id    = TIMEMORY_JOIN("", process::get_session_id());
    auto _proc_size     = TIMEMORY_JOIN("", process::get_num_siblings());
    auto _pwd_string    = get_env<std::string>("PWD", ".", false);
    auto _slurm_job_id  = get_env<std::string>("SLURM_JOB_ID", "0", false);
    auto _slurm_proc_id = get_env<std::string>("SLURM_PROCID", _dmp_rank, false);
    auto _launch_string = get_local_datetime(_time_format.c_str(), _launch_time);

    auto _uniq_id = _proc_id;
    if(get_env<int32_t>("SLURM_PROCID", -1, false) >= 0)
        _uniq_id = _slurm_proc_id;
    else if(dmp::is_initialized() || dmp::rank() > 0)
        _uniq_id = _dmp_rank;

    for(auto&& itr : std::initializer_list<output_key>{
            { "%argv%", _argv_string,
              "Entire command-line condensed into a single string" },
            { "%argt%", _argt_string,
              "Similar to `%argv%` except basename of first command line argument" },
            { "%args%", _args_string,
              "All command line arguments condensed into a single string" },
            { "%tag%", _tag0_string, "Basename of first command line argument" } })
    {
        _options.emplace_back(itr);
    }

    if(!_cmdline.empty())
    {
        for(size_t i = 0; i < _cmdline.size(); ++i)
        {
            auto _v = _cmdline.at(i);
            _options.emplace_back(TIMEMORY_JOIN("", "%arg", i, "%"), _v,
                                  TIMEMORY_JOIN("", "Argument #", i));
        }
    }

    auto _hashes = _options;

    for(auto&& itr : std::initializer_list<output_key>{
            { "%pid%", _proc_id, "Process identifier" },
            { "%ppid%", _parent_id, "Parent process identifier" },
            { "%pgid%", _pgroup_id, "Process group identifier" },
            { "%psid%", _session_id, "Process session identifier" },
            { "%psize%", _proc_size, "Number of sibling process" },
            { "%job%", _slurm_job_id, "SLURM_JOB_ID env variable" },
            { "%rank%", _slurm_proc_id, "MPI/UPC++ rank" },
            { "%size%", _dmp_size, "MPI/UPC++ size" },
            { "%nid%", _uniq_id, "%rank% if possible, otherwise %pid%" },
            { "%launch_time%", _launch_string,
              "Data and/or time of run according to time format" },
        })
    {
        _options.emplace_back(itr);
    }

    for(auto&& itr : std::initializer_list<output_key>{
            { "%m", md5::compute_md5(_argt_string), "Shorthand for %argt_hash%" },
            { "%p", _proc_id, "Shorthand for %pid%" },
            { "%j", _slurm_job_id, "Shorthand for %job%" },
            { "%r", _slurm_proc_id, "Shorthand for %rank%" },
            { "%s", _dmp_size, "Shorthand for %size" },
        })
    {
        _options.emplace_back(itr);
    }

    for(auto& itr : _hashes)
    {
        if(itr.key.empty() || itr.key.find("_hash%") != std::string::npos)
            continue;
        // if shorthand, skip
        if(itr.key.find_last_of('%') != itr.key.length() - 1)
            continue;

        itr.description = TIMEMORY_JOIN(" ", "MD5 sum of", itr.key);
        itr.value       = md5::compute_md5(itr.value);
        itr.key = std::regex_replace(itr.key, std::regex{ "%(.*)%$" }, "%$1_hash%");
        _options.emplace_back(std::move(itr));
    }

    return _options;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE std::string
                         settings::format(std::string _fpath, const std::string& _tag)
{
    auto _replace = [](auto& _v, const output_key& pitr) {
        auto pos = std::string::npos;
        while((pos = _v.find(pitr.key)) != std::string::npos)
            _v.replace(pos, pitr.key.length(), pitr.value);
    };

    _fpath = filepath::canonical(_fpath);

    if(_fpath.find('%') == std::string::npos)
        return _fpath;

    for(auto&& itr : output_keys(_tag))
        _replace(_fpath, itr);

    auto _verbose =
        get_with_env_fallback(TIMEMORY_SETTINGS_KEY("VERBOSE"), 0) +
        (get_with_env_fallback(TIMEMORY_SETTINGS_KEY("DEBUG"), false) ? 16 : 0);

    // environment and configuration variables
    try
    {
        for(const auto& _expr :
            { std::string{ "(.*)%(env|ENV)\\{([A-Z0-9_]+)\\}%(.*)" },
              std::string{ "(.*)\\$(env|ENV)\\{([A-Z0-9_]+)\\}(.*)" },
              std::string{ "(.*)%(cfg|CFG)\\{([A-Z0-9_]+)\\}%(.*)" },
              std::string{ "(.*)\\$(cfg|CFG)\\{([A-Z0-9_]+)\\}(.*)" } })
        {
            std::regex  _re{ _expr };
            std::string _cbeg   = (_expr.find("(.*)%") == 0) ? "%" : "$";
            std::string _cend   = (_expr.find("(.*)%") == 0) ? "}%" : "}";
            bool        _is_env = (_expr.find("(env|ENV)") != std::string::npos);
            _cbeg += (_is_env) ? "env{" : "cfg{";
            while(std::regex_search(_fpath, _re))
            {
                auto        _var = std::regex_replace(_fpath, _re, "$3");
                std::string _val = {};
                if(_is_env)
                {
                    _val = get_env<std::string>(_var, "");
                }
                else
                {
                    auto _settings = shared_instance();
                    if(_settings)
                    {
                        auto _cfg = _settings->find(_var);
                        if(_cfg != _settings->end())
                        {
                            _val = _cfg->second->as_string();
                            _replace(_val, output_key{ ",", "-", "" });
                        }
                    }
                }
                if(_verbose >= 1 && _val.empty())
                    TIMEMORY_PRINTF(
                        stderr,
                        "[%s][settings][%s] '%s' not found! Removing '%s%s%s' from "
                        "'%s'...\n",
                        TIMEMORY_PROJECT_NAME, __FUNCTION__, _var.c_str(), _cbeg.c_str(),
                        _var.c_str(), _cend.c_str(), _fpath.c_str());
                else if(_verbose >= 4 && !_val.empty())
                    TIMEMORY_PRINTF(
                        stderr,
                        "[%s][settings][%s] replacing '%s%s%s' in '%s' with '%s'...\n",
                        TIMEMORY_PROJECT_NAME, __FUNCTION__, _cbeg.c_str(), _var.c_str(),
                        _cend.c_str(), _fpath.c_str(), _val.c_str());
                auto _beg = std::regex_replace(_fpath, _re, "$1");
                auto _end = std::regex_replace(_fpath, _re, "$4");
                _fpath    = _beg + _val + _end;
                if(_verbose >= 3)
                    TIMEMORY_PRINTF(
                        stderr,
                        "[%s][settings][%s] replacing '%s%s%s' resulted in '%s'...\n",
                        TIMEMORY_PROJECT_NAME, __FUNCTION__, _cbeg.c_str(), _var.c_str(),
                        _cend.c_str(), _fpath.c_str());
            }
        }
    } catch(std::exception& _e)
    {
        TIMEMORY_PRINT_HERE("Warning! settings::%s throw exception :: %s", __FUNCTION__,
                            _e.what());
    }

    // remove %arg<N>% and %arg<N>_hash% where N >= argc
    try
    {
        std::regex _re{ "(.*)%(arg[0-9]+|arg[0-9]+_hash)%([-/_]*)(.*)" };
        while(std::regex_search(_fpath, _re))
            _fpath = std::regex_replace(_fpath, _re, "$1$4");
    } catch(std::exception& _e)
    {
        TIMEMORY_PRINT_HERE("Warning! settings::%s threw exception :: %s", __FUNCTION__,
                            _e.what());
    }

    return _fpath;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::format(std::string _prefix, std::string _tag, std::string _suffix,
                 std::string _ext)
{
    // add period before extension
    if(!_ext.empty() && _ext.find('.') != 0)
        _ext = std::string{ "." } + _ext;

    // if the tag contains the extension, remove it
    auto _ext_pos = _tag.length() - _ext.length();
    if(_tag.find(_ext) == _ext_pos)
        _tag = _tag.substr(0, _ext_pos);

    auto plast = static_cast<intmax_t>(_prefix.length()) - 1;
    // add dash if not empty, not ends in '/', and last char is alphanumeric
    if(!_prefix.empty() && _prefix[plast] != '/' && isalnum(_prefix[plast]) != 0)
        _prefix += "-";

    settings*   _instance   = instance();
    std::string _global_tag = {};
    if(_instance)
        _global_tag = _instance->get_tag();
    else
        _global_tag = get_fallback_tag();

    return format(_prefix + _tag + std::move(_suffix) + _ext, _global_tag);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::compose_output_filename(std::string _tag, std::string _ext,
                                  const compose_filename_config& _config)
{
    bool _is_explicit = (!_config.explicit_path.empty());
    // if there isn't an explicit prefix, get the <OUTPUT_PATH>/<OUTPUT_PREFIX>
    auto _prefix =
        (!_config.explicit_path.empty())
            ? ((_config.subdirectory.empty())
                   ? _config.explicit_path
                   : (_config.explicit_path + std::string{ "/" } + _config.subdirectory))
            : get_global_output_prefix(_config.make_dir, _config.subdirectory);

    // return on empty
    if(_prefix.empty())
        return "";

    auto only_ascii = [](char c) { return isascii(c) == 0; };

    _prefix.erase(std::remove_if(_prefix.begin(), _prefix.end(), only_ascii),
                  _prefix.end());

    // if explicit prefix is provided, then make the directory
    if(_is_explicit && _config.make_dir)
    {
        auto ret = makedir(_prefix);
        if(ret != 0)
            _prefix = filepath::osrepr(std::string{ "./" });
    }

    std::string _suffix = {};
    if(_config.use_suffix)
    {
        std::visit(
            [&_suffix](auto _val) {
                if constexpr(!std::is_same<decltype(_val), process::id_t>::value)
                {
                    _suffix = std::string{ "-" } + _val;
                }
                else
                {
                    if(_val >= 0)
                        _suffix = std::string{ "-" } + std::to_string(_val);
                }
            },
            _config.suffix);
    }

    // create the path
    std::string _fpath = format(_prefix, std::move(_tag), _suffix, std::move(_ext));

    return filepath::osrepr(_fpath);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::compose_input_filename(std::string _tag, std::string _ext,
                                 const compose_filename_config& _config)
{
    if(settings::input_path().empty())
        settings::input_path() = settings::output_path();

    if(settings::input_prefix().empty())
        settings::input_prefix() = settings::output_prefix();

    auto _prefix =
        (_config.explicit_path.length() > 0)
            ? ((_config.subdirectory.empty())
                   ? _config.explicit_path
                   : (_config.explicit_path + std::string{ "/" } + _config.subdirectory))
            : get_global_input_prefix(_config.subdirectory);

    auto only_ascii = [](char c) { return isascii(c) == 0; };

    _prefix.erase(std::remove_if(_prefix.begin(), _prefix.end(), only_ascii),
                  _prefix.end());

    std::string _suffix = {};
    if(_config.use_suffix)
    {
        std::visit(
            [&_suffix](auto _val) {
                if constexpr(!std::is_same<decltype(_val), process::id_t>::value)
                {
                    _suffix = std::string{ "-" } + _val;
                }
                else
                {
                    if(_val >= 0)
                        _suffix = std::string{ "-" } + std::to_string(_val);
                }
            },
            _config.suffix);
    }

    // create the path
    std::string _fpath = format(_prefix, std::move(_tag), _suffix, std::move(_ext));

    return filepath::osrepr(_fpath);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::parse(const std::shared_ptr<settings>& _settings)
{
    if(_settings)
        parse(_settings.get());
}
//
//--------------------------------------------------------------------------------------//
//
// function to parse the environment for settings
//
// Nearly all variables will parse env when first access but this allows provides a
// way to reparse the environment so that default settings (possibly from previous
// invocation) can be overwritten
//
TIMEMORY_SETTINGS_INLINE
void
settings::parse(settings* _settings)
{
    if(!_settings)
    {
        TIMEMORY_PRINT_HERE("%s", "nullptr to tim::settings");
        return;
    }

    bool _suppress = false;
    auto _v        = _settings->find("suppress_parsing");
    if(_v != _settings->end())
    {
        _v->second->parse();
        _suppress = _v->second->get<bool>().second;
    }
    else
    {
        _suppress = _settings->get_suppress_parsing();
    }

    if(_suppress)
    {
        static auto _once = false;
        if(!_once)
        {
            TIMEMORY_PRINT_HERE("%s", "settings parsing has been suppressed");
            _once = true;
        }
        return;
    }

    for(const auto& itr : *_settings)
    {
        itr.second->parse();
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::settings()
{
    auto _reserve_size = get_env<size_t>(TIMEMORY_SETTINGS_KEY("TOTAL_SETTINGS"), 4096);
    m_order.reserve(_reserve_size);
    m_data.reserve(_reserve_size);
    initialize();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::settings(const settings& rhs)
: m_initialized{ rhs.m_initialized }
, m_tag{ rhs.m_tag }
, m_read_configs{ rhs.m_read_configs }
, m_config_stack{ rhs.m_config_stack }
, m_order{ rhs.m_order }
, m_command_line{ rhs.m_command_line }
, m_environment{ rhs.m_environment }
, m_unknown_configs{ rhs.m_unknown_configs }
{
    m_order.reserve(rhs.m_order.capacity());
    for(const auto& itr : rhs.m_data)
        m_data.emplace(itr.first, itr.second->clone());

    for(auto& itr : m_order)
    {
        if(m_data.find(itr) == m_data.end())
        {
            auto ritr = rhs.m_data.find(itr);
            if(ritr == rhs.m_data.end())
            {
                std::stringstream _sorder;
                {
                    std::set<std::string> _v = {};
                    for(const auto& ditr : m_data)
                        _v.emplace(ditr.first);
                    for(const auto& ditr : _v)
                        _sorder << "\n    " << ditr;
                }
                {
                    _sorder << "\n  ORIGINAL:";
                    std::set<std::string> _v = {};
                    for(const auto& ditr : rhs.m_data)
                        _v.emplace(ditr.first);
                    for(const auto& ditr : _v)
                        _sorder << "\n    " << ditr;
                }
                TIMEMORY_EXCEPTION("Error! Missing ordered entry: " << itr << ". Known: "
                                                                    << _sorder.str());
            }
            else
            {
                m_data.emplace(itr, ritr->second->clone());
            }
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings&
settings::operator=(const settings& rhs)
{
    if(this == &rhs)
        return *this;

    for(const auto& itr : rhs.m_data)
        m_data[itr.first] = itr.second->clone();
    m_initialized     = rhs.m_initialized;
    m_tag             = rhs.m_tag;
    m_config_stack    = rhs.m_config_stack;
    m_order           = rhs.m_order;
    m_command_line    = rhs.m_command_line;
    m_environment     = rhs.m_environment;
    m_read_configs    = rhs.m_read_configs;
    m_unknown_configs = rhs.m_unknown_configs;
    m_order.reserve(rhs.m_order.capacity());
    for(auto& itr : m_order)
    {
        if(m_data.find(itr) == m_data.end())
        {
            auto ritr = rhs.m_data.find(itr);
            if(ritr == rhs.m_data.end())
            {
                TIMEMORY_EXCEPTION(std::string("Error! Missing ordered entry: ") + itr)
            }
            else
            {
                m_data.emplace(itr, ritr->second->clone());
            }
        }
    }
    return *this;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::get_fallback_tag()
{
    std::string _tag = {};

    if(command_line().empty())
        command_line() = read_command_line(process::get_id());

    if(command_line().empty())
    {
        _tag      = std::string{ TIMEMORY_SETTINGS_PREFIX };
        auto _pos = std::string::npos;
        while((_pos = _tag.find_last_of('_')) == _tag.length() - 1)
            _tag = _tag.substr(0, _tag.length() - 1);
        return _tag;
    }

    _tag = command_line().front();

    while(_tag.find('\\') != std::string::npos)
        _tag = _tag.substr(_tag.find_last_of('\\') + 1);

    while(_tag.find('/') != std::string::npos)
        _tag = _tag.substr(_tag.find_last_of('/') + 1);

    for(auto&& itr : { std::string{ ".py" }, std::string{ ".exe" } })
    {
        if(_tag.find(itr) != std::string::npos)
            _tag.erase(_tag.find(itr), itr.length() + 1);
    }

    return _tag;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::get_tag() const
{
    if(m_tag.empty())
    {
        if(command_line().empty() && !m_command_line.empty())
            command_line() = m_command_line;

        const_cast<settings*>(this)->m_tag = get_fallback_tag();
    }

    return m_tag;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
bool
settings::disable(string_view_cref_t _key, bool _exact)
{
    auto itr = find(_key.data(), _exact);
    if(itr != m_data.end() && itr->second)
    {
        itr->second->set_enabled(false);
        return true;
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::set<std::string>
settings::disable_category(string_view_cref_t _category)
{
    std::set<std::string> _v{};
    for(auto&& itr : m_data)
    {
        if(itr.second->matches(".*", _category.data()))
        {
            itr.second->set_enabled(false);
            _v.emplace(itr.first);
        }
    }
    return _v;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
bool
settings::enable(string_view_cref_t _key, bool _exact)
{
    auto itr = find(_key.data(), _exact);
    if(itr != m_data.end() && itr->second)
    {
        itr->second->set_enabled(true);
        return true;
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::set<std::string>
settings::enable_category(string_view_cref_t _category)
{
    std::set<std::string> _v{};
    for(auto&& itr : m_data)
    {
        if(itr.second->matches(".*", _category.data()))
        {
            itr.second->set_enabled(true);
            _v.emplace(itr.first);
        }
    }
    return _v;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_core()
{
    const auto* homedir = "%env{HOME}%";

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, config_file, TIMEMORY_SETTINGS_KEY("CONFIG_FILE"),
        "Configuration file for " TIMEMORY_PROJECT_NAME,
        TIMEMORY_JOIN(';', TIMEMORY_JOIN('/', homedir, "." TIMEMORY_PROJECT_NAME ".cfg"),
                      TIMEMORY_JOIN('/', homedir, "." TIMEMORY_PROJECT_NAME ".json")),
        TIMEMORY_ESC(strset_t{ "native", "core", "config" }),
        strvector_t({ "-C", "--" TIMEMORY_PROJECT_NAME "-config" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, suppress_config, TIMEMORY_SETTINGS_KEY("SUPPRESS_CONFIG"),
        "Disable processing of setting configuration files", false,
        TIMEMORY_ESC(strset_t{ "native", "core", "config" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-suppress-config",
                      "--" TIMEMORY_PROJECT_NAME "-no-config" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, strict_config, TIMEMORY_SETTINGS_KEY("STRICT_CONFIG"),
        "Throw errors for unknown setting names in configuration files instead of "
        "emitting a warning",
        true, TIMEMORY_ESC(strset_t{ "native", "core", "config" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-strict-config" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, suppress_parsing, TIMEMORY_SETTINGS_KEY("SUPPRESS_PARSING"),
        "Disable parsing environment", false,
        TIMEMORY_ESC(strset_t{ "native", "core", "config" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-suppress-parsing" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, enabled, TIMEMORY_SETTINGS_KEY("ENABLED"), "Activation state of timemory",
        TIMEMORY_DEFAULT_ENABLED, TIMEMORY_ESC(strset_t{ "native", "core" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-enabled" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int, verbose, TIMEMORY_SETTINGS_KEY("VERBOSE"), "Verbosity level", 0,
        TIMEMORY_ESC(strset_t{ "native", "core", "debugging" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-verbose" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, debug, TIMEMORY_SETTINGS_KEY("DEBUG"), "Enable debug output", false,
        TIMEMORY_ESC(strset_t{ "native", "core", "debugging" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-debug" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        bool, flat_profile, TIMEMORY_SETTINGS_KEY("FLAT_PROFILE"),
        "Set the label hierarchy mode to default to flat",
        scope::get_fields()[scope::flat::value],
        TIMEMORY_ESC(strset_t{ "native", "core", "data", "data_layout" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-flat-profile" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        bool, timeline_profile, TIMEMORY_SETTINGS_KEY("TIMELINE_PROFILE"),
        "Set the label hierarchy mode to default to timeline",
        scope::get_fields()[scope::timeline::value],
        TIMEMORY_ESC(strset_t{ "native", "core", "data", "data_layout" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-timeline-profile" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        uint16_t, max_depth, TIMEMORY_SETTINGS_KEY("MAX_DEPTH"),
        "Set the maximum depth of label hierarchy reporting",
        std::numeric_limits<uint16_t>::max(),
        TIMEMORY_ESC(strset_t{ "native", "core", "data" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-max-depth" }), 1);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_components()
{
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, global_components, TIMEMORY_SETTINGS_KEY("GLOBAL_COMPONENTS"),
        "A specification of components which is used by multiple variadic bundlers and "
        "user_bundles as the fall-back set of components if their specific variable is "
        "not set. E.g. user_mpip_bundle will use this if MPIP_COMPONENTS is not "
        "specified",
        "", TIMEMORY_ESC(strset_t{ "native", "component" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-global-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, ompt_components, TIMEMORY_SETTINGS_KEY("OMPT_COMPONENTS"),
        "A specification of components which will be added to structures containing "
        "the 'user_ompt_bundle'. Priority: TRACE_COMPONENTS -> PROFILER_COMPONENTS -> "
        "COMPONENTS -> GLOBAL_COMPONENTS",
        "", TIMEMORY_ESC(strset_t{ "native", "component", "ompt", "gotcha" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-ompt-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, mpip_components, TIMEMORY_SETTINGS_KEY("MPIP_COMPONENTS"),
        "A specification of components which will be added to structures containing "
        "the 'user_mpip_bundle'. Priority: TRACE_COMPONENTS -> PROFILER_COMPONENTS -> "
        "COMPONENTS -> GLOBAL_COMPONENTS",
        "", TIMEMORY_ESC(strset_t{ "native", "component", "mpip", "gotcha" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-mpip-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, ncclp_components, TIMEMORY_SETTINGS_KEY("NCCLP_COMPONENTS"),
        "A specification of components which will be added to structures containing "
        "the 'user_ncclp_bundle'. Priority: MPIP_COMPONENTS -> TRACE_COMPONENTS -> "
        "PROFILER_COMPONENTS -> COMPONENTS -> GLOBAL_COMPONENTS",
        "", TIMEMORY_ESC(strset_t{ "native", "component", "ncclp", "gotcha" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-ncclp-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, trace_components, TIMEMORY_SETTINGS_KEY("TRACE_COMPONENTS"),
        "A specification of components which will be used by the interfaces which "
        "are designed for full profiling. These components will be subjected to "
        "throttling. "
        "Priority: COMPONENTS -> GLOBAL_COMPONENTS",
        "", TIMEMORY_ESC(strset_t{ "native", "component" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-trace-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, profiler_components, TIMEMORY_SETTINGS_KEY("PROFILER_COMPONENTS"),
        "A specification of components which will be used by the interfaces which "
        "are designed for full python profiling. This specification will be overridden "
        "by a trace_components specification. Priority: COMPONENTS -> GLOBAL_COMPONENTS",
        "", TIMEMORY_ESC(strset_t{ "native", "component" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-profiler-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, kokkos_components, TIMEMORY_SETTINGS_KEY("KOKKOS_COMPONENTS"),
        "A specification of components which will be used by the interfaces which "
        "are designed for kokkos profiling. Priority: TRACE_COMPONENTS -> "
        "PROFILER_COMPONENTS -> COMPONENTS -> GLOBAL_COMPONENTS",
        "", TIMEMORY_ESC(strset_t{ "native", "component" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-kokkos-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, components, TIMEMORY_SETTINGS_KEY("COMPONENTS"),
        "A specification of components which is used by the library interface. This "
        "falls back to GLOBAL_COMPONENTS.",
        "", TIMEMORY_ESC(strset_t{ "native", "component" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, network_interface, TIMEMORY_SETTINGS_KEY("NETWORK_INTERFACE"),
        "Default network interface", std::string{},
        TIMEMORY_ESC(strset_t{ "native", "component" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-network-interface" }), -1, 1);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_io()
{
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, auto_output, TIMEMORY_SETTINGS_KEY("AUTO_OUTPUT"),
        "Generate output at application termination", true,
        TIMEMORY_ESC(strset_t{ "native", "io" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-auto-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, cout_output, TIMEMORY_SETTINGS_KEY("COUT_OUTPUT"), "Write output to stdout",
        false, TIMEMORY_ESC(strset_t{ "native", "io", "console" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-cout-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, file_output, TIMEMORY_SETTINGS_KEY("FILE_OUTPUT"), "Write output to files",
        true, TIMEMORY_ESC(strset_t{ "native", "io" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-file-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, text_output, TIMEMORY_SETTINGS_KEY("TEXT_OUTPUT"),
        "Write text output files", true, TIMEMORY_ESC(strset_t{ "native", "io", "text" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-text-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, json_output, TIMEMORY_SETTINGS_KEY("JSON_OUTPUT"),
        "Write json output files", true, TIMEMORY_ESC(strset_t{ "native", "io", "json" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-json-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, tree_output, TIMEMORY_SETTINGS_KEY("TREE_OUTPUT"),
        "Write hierarchical json output files", true,
        TIMEMORY_ESC(strset_t{ "native", "io", "json" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-tree-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, dart_output, TIMEMORY_SETTINGS_KEY("DART_OUTPUT"),
        "Write dart measurements for CDash", false,
        TIMEMORY_ESC(strset_t{ "native", "io", "dart", "cdash", "console" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-dart-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, time_output, TIMEMORY_SETTINGS_KEY("TIME_OUTPUT"),
        "Output data to subfolder w/ a timestamp (see also: TIME_FORMAT)", false,
        TIMEMORY_ESC(strset_t{ "native", "io", "filename" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-time-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, plot_output, TIMEMORY_SETTINGS_KEY("PLOT_OUTPUT"),
        "Generate plot outputs from json outputs", TIMEMORY_DEFAULT_PLOTTING,
        TIMEMORY_ESC(strset_t{ "native", "io", "plotting" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-plot-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, diff_output, TIMEMORY_SETTINGS_KEY("DIFF_OUTPUT"),
        "Generate a difference output vs. a pre-existing output (see also: "
        "INPUT_PATH and INPUT_PREFIX)",
        false, TIMEMORY_ESC(strset_t{ "native", "io" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-diff-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, flamegraph_output, TIMEMORY_SETTINGS_KEY("FLAMEGRAPH_OUTPUT"),
        "Write a json output for flamegraph visualization (use chrome://tracing)", true,
        TIMEMORY_ESC(strset_t{ "native", "io", "flamegraph", "json" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-flamegraph-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, ctest_notes, TIMEMORY_SETTINGS_KEY("CTEST_NOTES"),
        "Write a CTestNotes.txt for each text output", false,
        TIMEMORY_ESC(strset_t{ "native", "io", "ctest" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-ctest-notes" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, output_path, TIMEMORY_SETTINGS_KEY("OUTPUT_PATH"),
        "Explicitly specify the output folder for results",
        TIMEMORY_PROJECT_NAME "-%tag%-output",
        TIMEMORY_ESC(strset_t{ "native", "io", "filename" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-output-path" }),
        1);  // folder

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, output_prefix, TIMEMORY_SETTINGS_KEY("OUTPUT_PREFIX"),
        "Explicitly specify a prefix for all output files", "",
        TIMEMORY_ESC(strset_t{ "native", "io", "filename" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-output-prefix" }),
        1);  // file prefix

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, input_path, TIMEMORY_SETTINGS_KEY("INPUT_PATH"),
        "Explicitly specify the input folder for difference "
        "comparisons (see also: DIFF_OUTPUT)",
        "", TIMEMORY_ESC(strset_t{ "native", "io", "filename" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-input-path" }),
        1);  // folder

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, input_prefix, TIMEMORY_SETTINGS_KEY("INPUT_PREFIX"),
        "Explicitly specify the prefix for input files used in difference "
        "comparisons (see also: DIFF_OUTPUT)",
        "", TIMEMORY_ESC(strset_t{ "native", "io", "filename" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-input-prefix" }),
        1);  // file prefix

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, input_extensions, TIMEMORY_SETTINGS_KEY("INPUT_EXTENSIONS"),
        "File extensions used when searching for input files used in difference "
        "comparisons (see also: DIFF_OUTPUT)",
        "json,xml", TIMEMORY_ESC(strset_t{ "native", "io", "filename" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-input-extensions" }));  // extensions
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_format()
{
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, time_format, TIMEMORY_SETTINGS_KEY("TIME_FORMAT"),
        "Customize the folder generation when TIME_OUTPUT is enabled (see also: "
        "strftime)",
        "%F_%H.%M", TIMEMORY_ESC(strset_t{ "native", "io", "format", "filename" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-time-format" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int16_t, precision, TIMEMORY_SETTINGS_KEY("PRECISION"),
        "Set the global output precision for components", -1,
        TIMEMORY_ESC(strset_t{ "native", "io", "format" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-precision" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int16_t, width, TIMEMORY_SETTINGS_KEY("WIDTH"),
        "Set the global output width for components", -1,
        TIMEMORY_ESC(strset_t{ "native", "io", "format" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-width" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int32_t, max_width, TIMEMORY_SETTINGS_KEY("MAX_WIDTH"),
        "Set the maximum width for component label outputs", 120,
        TIMEMORY_ESC(strset_t{ "native", "io", "format" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-max-width" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, scientific, TIMEMORY_SETTINGS_KEY("SCIENTIFIC"),
        "Set the global numerical reporting to scientific format", false,
        TIMEMORY_ESC(strset_t{ "native", "io", "format" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-scientific" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, timing_precision, TIMEMORY_SETTINGS_KEY("TIMING_PRECISION"),
        "Set the precision for components with 'is_timing_category' type-trait", -1,
        TIMEMORY_ESC(strset_t{ "native", "io", "format" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, timing_width, TIMEMORY_SETTINGS_KEY("TIMING_WIDTH"),
        "Set the output width for components with 'is_timing_category' type-trait", -1,
        TIMEMORY_ESC(strset_t{ "native", "io", "format" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, timing_units, TIMEMORY_SETTINGS_KEY("TIMING_UNITS"),
        "Set the units for components with "
        "'uses_timing_units' type-trait",
        "", TIMEMORY_ESC(strset_t{ "native", "io", "format" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-timing-units" }), 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, timing_scientific, TIMEMORY_SETTINGS_KEY("TIMING_SCIENTIFIC"),
        "Set the numerical reporting format for components "
        "with 'is_timing_category' type-trait",
        false, TIMEMORY_ESC(strset_t{ "native", "io", "format" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, memory_precision, TIMEMORY_SETTINGS_KEY("MEMORY_PRECISION"),
        "Set the precision for components with 'is_memory_category' type-trait", -1,
        TIMEMORY_ESC(strset_t{ "native", "io", "format" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, memory_width, TIMEMORY_SETTINGS_KEY("MEMORY_WIDTH"),
        "Set the output width for components with 'is_memory_category' type-trait", -1,
        TIMEMORY_ESC(strset_t{ "native", "io", "format" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, memory_units, TIMEMORY_SETTINGS_KEY("MEMORY_UNITS"),
        "Set the units for components with "
        "'uses_memory_units' type-trait",
        "", TIMEMORY_ESC(strset_t{ "native", "io", "format" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-memory-units" }), 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, memory_scientific, TIMEMORY_SETTINGS_KEY("MEMORY_SCIENTIFIC"),
        "Set the numerical reporting format for components "
        "with 'is_memory_category' type-trait",
        false, TIMEMORY_ESC(strset_t{ "native", "io", "format" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(int64_t, separator_frequency,
                                  TIMEMORY_SETTINGS_KEY("SEPARATOR_FREQ"),
                                  "Frequency of dashed separator lines in text output", 0,
                                  TIMEMORY_ESC(strset_t{ "native", "io", "format" }));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_parallel()
{
    TIMEMORY_SETTINGS_MEMBER_IMPL(size_t, max_thread_bookmarks,
                                  TIMEMORY_SETTINGS_KEY("MAX_THREAD_BOOKMARKS"),
                                  "Maximum number of times a worker thread bookmarks the "
                                  "call-graph location w.r.t. the master thread. Higher "
                                  "values tend to increase the finalization merge time",
                                  50, TIMEMORY_ESC(strset_t{ "native", "parallelism" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, collapse_threads, TIMEMORY_SETTINGS_KEY("COLLAPSE_THREADS"),
        "Enable/disable combining thread-specific data", true,
        TIMEMORY_ESC(strset_t{ "native", "parallelism", "data_layout" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-collapse-threads" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, collapse_processes, TIMEMORY_SETTINGS_KEY("COLLAPSE_PROCESSES"),
        "Enable/disable combining process-specific data", true,
        TIMEMORY_ESC(strset_t{ "native", "parallelism", "data_layout" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-collapse-processes" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, cpu_affinity, TIMEMORY_SETTINGS_KEY("CPU_AFFINITY"),
        "Enable pinning threads to CPUs (Linux-only)", false,
        TIMEMORY_ESC(strset_t{ "native", "parallelism" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-cpu-affinity" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_IMPL(
        process::id_t, target_pid, TIMEMORY_SETTINGS_KEY("TARGET_PID"),
        "Process ID for the components which require this", process::get_target_id(),
        TIMEMORY_ESC(strset_t{ "native", "parallelism" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, mpi_init, TIMEMORY_SETTINGS_KEY("MPI_INIT"),
        "Enable/disable timemory calling MPI_Init / MPI_Init_thread during certain "
        "timemory_init(...) invocations",
        false, TIMEMORY_ESC(strset_t{ "native", "parallelism", "mpi", "dmp" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-mpi-init" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, mpi_finalize, TIMEMORY_SETTINGS_KEY("MPI_FINALIZE"),
        "Enable/disable timemory calling MPI_Finalize "
        "during timemory_finalize(...) invocations",
        false, TIMEMORY_ESC(strset_t{ "native", "parallelism", "mpi", "dmp" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-mpi-finalize" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        bool, mpi_thread, TIMEMORY_SETTINGS_KEY("MPI_THREAD"),
        "Call MPI_Init_thread instead of MPI_Init (see also: MPI_INIT)",
        mpi::use_mpi_thread(),
        TIMEMORY_ESC(strset_t{ "native", "parallelism", "mpi", "dmp" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-mpi-thread" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        std::string, mpi_thread_type, TIMEMORY_SETTINGS_KEY("MPI_THREAD_TYPE"),
        "MPI_Init_thread mode: 'single', 'serialized', 'funneled', or 'multiple' "
        "(see also: MPI_INIT and MPI_THREAD)",
        mpi::use_mpi_thread_type(),
        TIMEMORY_ESC(strset_t{ "native", "parallelism", "mpi", "dmp" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-mpi-thread-type" }), 1, 1,
        strvector_t{ "single", "serialized", "funneled", "multiple" });

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, upcxx_init, TIMEMORY_SETTINGS_KEY("UPCXX_INIT"),
        "Enable/disable timemory calling upcxx::init() "
        "during certain timemory_init(...) invocations",
        false, TIMEMORY_ESC(strset_t{ "native", "parallelism", "upcxx", "dmp" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-upcxx-init" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, upcxx_finalize, TIMEMORY_SETTINGS_KEY("UPCXX_FINALIZE"),
        "Enable/disable timemory calling upcxx::finalize() during "
        "timemory_finalize()",
        false, TIMEMORY_ESC(strset_t{ "native", "parallelism", "upcxx", "dmp" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-upcxx-finalize" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int32_t, node_count, TIMEMORY_SETTINGS_KEY("NODE_COUNT"),
        "Total number of nodes used in application. Setting this value > 1 will "
        "result in aggregating N processes into groups of N / NODE_COUNT",
        0, TIMEMORY_ESC(strset_t{ "native", "parallelism", "dmp" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-node-count" }), 1);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_tpls()
{
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, papi_threading, TIMEMORY_SETTINGS_KEY("PAPI_THREADING"),
        "Enable multithreading support when using PAPI", true,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "papi" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-papi-threading" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, papi_multiplexing, TIMEMORY_SETTINGS_KEY("PAPI_MULTIPLEXING"),
        "Enable multiplexing when using PAPI", false,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "papi" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-papi-multiplexing" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, papi_fail_on_error, TIMEMORY_SETTINGS_KEY("PAPI_FAIL_ON_ERROR"),
        "Configure PAPI errors to trigger a runtime error", false,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "papi" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-papi-fail-on-error" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, papi_quiet, TIMEMORY_SETTINGS_KEY("PAPI_QUIET"),
        "Configure suppression of reporting PAPI errors/warnings", false,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "papi" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-papi-quiet" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, papi_events, TIMEMORY_SETTINGS_KEY("PAPI_EVENTS"),
        "PAPI presets and events to collect (see also: papi_avail)", "",
        TIMEMORY_ESC(strset_t{ "native", "tpl", "papi" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-papi-events" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, papi_attach, TIMEMORY_SETTINGS_KEY("PAPI_ATTACH"),
        "Configure PAPI to attach to another process (see also: TARGET_PID)", false,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "papi" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int, papi_overflow, TIMEMORY_SETTINGS_KEY("PAPI_OVERFLOW"),
        "Value at which PAPI hw counters trigger an overflow callback", 0,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "papi" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-papi-overflow" }), 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, cuda_event_batch_size, TIMEMORY_SETTINGS_KEY("CUDA_EVENT_BATCH_SIZE"),
        "Batch size for create cudaEvent_t in cuda_event components", 5,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, nvtx_marker_device_sync, TIMEMORY_SETTINGS_KEY("NVTX_MARKER_DEVICE_SYNC"),
        "Use cudaDeviceSync when stopping NVTX marker (vs. cudaStreamSychronize)", true,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "nvtx" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int32_t, cupti_activity_level, TIMEMORY_SETTINGS_KEY("CUPTI_ACTIVITY_LEVEL"),
        "Default group of kinds tracked via CUpti Activity API", 1,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-cupti-activity-level" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, cupti_activity_kinds, TIMEMORY_SETTINGS_KEY("CUPTI_ACTIVITY_KINDS"),
        "Specific cupti activity kinds to track", "",
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-cupti-activity-kinds" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, cupti_events, TIMEMORY_SETTINGS_KEY("CUPTI_EVENTS"),
        "Hardware counter event types to collect on NVIDIA "
        "GPUs",
        "", TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-cupti-events" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, cupti_metrics, TIMEMORY_SETTINGS_KEY("CUPTI_METRICS"),
        "Hardware counter metric types to collect on "
        "NVIDIA GPUs",
        "", TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-cupti-metrics" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int, cupti_device, TIMEMORY_SETTINGS_KEY("CUPTI_DEVICE"),
        "Target device for CUPTI data collection", 0,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-cupti-device" }), 1);

    insert<int>(
        TIMEMORY_SETTINGS_KEY("CUPTI_PCSAMPLING_PERIOD"), "cupti_pcsampling_period",
        "The period for PC sampling. Must be >= 5 and <= 31", 8,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti", "cupti_pcsampling" }),
        strvector_t{ "--" TIMEMORY_PROJECT_NAME "-cupti-pcsampling-period" });

    insert<bool>(
        TIMEMORY_SETTINGS_KEY("CUPTI_PCSAMPLING_PER_LINE"), "cupti_pcsampling_per_line",
        "Report the PC samples per-line or collapse into one entry for entire "
        "function",
        false,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti", "cupti_pcsampling" }),
        strvector_t{ "--" TIMEMORY_PROJECT_NAME "-cupti-pcsampling-per-line" });

    insert<bool>(
        TIMEMORY_SETTINGS_KEY("CUPTI_PCSAMPLING_REGION_TOTALS"),
        "cupti_pcsampling_region_totals",
        "When enabled, region markers will report total samples from all child "
        "functions",
        true,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti", "cupti_pcsampling" }),
        strvector_t{ "--" TIMEMORY_PROJECT_NAME "-cupti-pcsampling-region-totals" });

    insert<bool>(
        TIMEMORY_SETTINGS_KEY("CUPTI_PCSAMPLING_SERIALIZED"),
        "cupti_pcsampling_serialized", "Serialize all the kernel functions", false,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti", "cupti_pcsampling" }),
        strvector_t{ "--" TIMEMORY_PROJECT_NAME "-cupti-pcsampling-serialize" });

    insert<size_t>(
        TIMEMORY_SETTINGS_KEY("CUPTI_PCSAMPLING_NUM_COLLECT"),
        "cupti_pcsampling_num_collect", "Number of PCs to be collected", size_t{ 100 },
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti", "cupti_pcsampling" }),
        strvector_t{ "--" TIMEMORY_PROJECT_NAME "-cupti-pcsampling-num-collect" });

    insert<std::string>(
        TIMEMORY_SETTINGS_KEY("CUPTI_PCSAMPLING_STALL_REASONS"),
        "cupti_pcsampling_stall_reasons", "The PC sampling stall reasons to count",
        std::string{},
        TIMEMORY_ESC(strset_t{ "native", "tpl", "cuda", "cupti", "cupti_pcsampling" }),
        strvector_t{ "--" TIMEMORY_PROJECT_NAME "-cupti-pcsampling-stall-reasons" });

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        std::string, craypat_categories, TIMEMORY_SETTINGS_KEY("CRAYPAT"),
        "Configure the CrayPAT categories to collect (same as PAT_RT_PERFCTR)",
        get_env<std::string>("PAT_RT_PERFCTR", "", false),
        TIMEMORY_ESC(strset_t{ "native", "tpl", "craypat" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, python_exe, TIMEMORY_SETTINGS_KEY("PYTHON_EXE"),
        "Configure the python executable to use", TIMEMORY_PYTHON_PLOTTER,
        TIMEMORY_ESC(strset_t{ "native", "tpl", "python" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-python-exe" }));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_roofline()
{
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, roofline_mode, TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE"),
        "Configure the roofline collection mode. Options: 'op' 'ai'.", "op",
        TIMEMORY_ESC(strset_t{ "native", "component", "roofline" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-roofline-mode" }), 1, 1,
        strvector_t({ "op", "ai" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, cpu_roofline_mode, TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE_CPU"),
        "Configure the roofline collection mode for CPU "
        "specifically. Options: 'op', 'ai'",
        "op", TIMEMORY_ESC(strset_t{ "native", "component", "roofline", "cpu_roofline" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-cpu-roofline-mode" }), 1, 1,
        strvector_t({ "op", "ai" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, gpu_roofline_mode, TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE_GPU"),
        "Configure the roofline collection mode for GPU specifically. Options: 'op', "
        "'ai'.",
        static_cast<tsettings<std::string>*>(
            m_data[TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE")].get())
            ->get(),
        TIMEMORY_ESC(strset_t{ "native", "component", "roofline", "gpu_roofline" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-gpu-roofline-mode" }), 1, 1,
        strvector_t({ "op", "ai" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        std::string, cpu_roofline_events, TIMEMORY_SETTINGS_KEY("ROOFLINE_EVENTS_CPU"),
        "Configure custom hw counters to add to the cpu roofline", "",
        TIMEMORY_ESC(strset_t{ "native", "component", "roofline", "cpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        std::string, gpu_roofline_events, TIMEMORY_SETTINGS_KEY("ROOFLINE_EVENTS_GPU"),
        "Configure custom hw counters to add to the gpu roofline", "",
        TIMEMORY_ESC(strset_t{ "native", "component", "roofline", "gpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, roofline_type_labels, TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS"),
        "Configure roofline labels/descriptions/output-files encode the list of data "
        "types",
        false, TIMEMORY_ESC(strset_t{ "native", "component", "roofline", "io" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, roofline_type_labels_cpu, TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS_CPU"),
        "Configure labels, etc. for the roofline components for CPU (see also: "
        "ROOFLINE_TYPE_LABELS)",
        static_cast<tsettings<bool>*>(
            m_data[TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS")].get())
            ->get(),
        TIMEMORY_ESC(
            strset_t{ "native", "component", "roofline", "cpu_roofline", "io" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, roofline_type_labels_gpu, TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS_GPU"),
        "Configure labels, etc. for the roofline components for GPU (see also: "
        "ROOFLINE_TYPE_LABELS)",
        static_cast<tsettings<bool>*>(
            m_data[TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS")].get())
            ->get(),
        TIMEMORY_ESC(
            strset_t{ "native", "component", "roofline", "gpu_roofline", "io" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, instruction_roofline, TIMEMORY_SETTINGS_KEY("INSTRUCTION_ROOFLINE"),
        "Configure the roofline to include the hw counters "
        "required for generating an instruction roofline",
        false, TIMEMORY_ESC(strset_t{ "native", "component", "roofline" }));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE void
settings::initialize_miscellaneous()
{
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, add_secondary, TIMEMORY_SETTINGS_KEY("ADD_SECONDARY"),
        "Enable/disable components adding secondary (child) entries when available. "
        "E.g. "
        "suppress individual CUDA kernels, etc. when using Cupti components",
        true, TIMEMORY_ESC(strset_t{ "native", "component", "data" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-add-secondary" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        size_t, throttle_count, TIMEMORY_SETTINGS_KEY("THROTTLE_COUNT"),
        "Minimum number of laps before checking whether a key should be throttled", 10000,
        TIMEMORY_ESC(strset_t{ "native", "component", "data", "throttle" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-throttle-count" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        size_t, throttle_value, TIMEMORY_SETTINGS_KEY("THROTTLE_VALUE"),
        "Average call time in nanoseconds when # laps > throttle_count that triggers "
        "throttling",
        10000, TIMEMORY_ESC(strset_t{ "native", "component", "data", "throttle" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-throttle-value" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, enable_signal_handler, TIMEMORY_SETTINGS_KEY("ENABLE_SIGNAL_HANDLER"),
        "Enable signals in timemory_init", false,
        TIMEMORY_ESC(strset_t{ "native", "debugging", "signals" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-enable-signal-handler" }), -1, 1)

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        bool, allow_signal_handler, TIMEMORY_SETTINGS_KEY("ALLOW_SIGNAL_HANDLER"),
        "Allow signal handling to be activated", signals::signal_settings::allow(),
        TIMEMORY_ESC(strset_t{ "native", "debugging", "signals" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-allow-signal-handler" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_IMPL(
        bool, enable_all_signals, TIMEMORY_SETTINGS_KEY("ENABLE_ALL_SIGNALS"),
        "Enable catching all signals", signals::signal_settings::enable_all(),
        TIMEMORY_ESC(strset_t{ "native", "debugging", "signals" }));

    TIMEMORY_SETTINGS_REFERENCE_IMPL(
        bool, disable_all_signals, TIMEMORY_SETTINGS_KEY("DISABLE_ALL_SIGNALS"),
        "Disable catching any signals", signals::signal_settings::disable_all(),
        TIMEMORY_ESC(strset_t{ "native", "debugging", "signals" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, destructor_report, TIMEMORY_SETTINGS_KEY("DESTRUCTOR_REPORT"),
        "Configure default setting for auto_{list,tuple,hybrid} to write to stdout "
        "during destruction of the bundle",
        false, TIMEMORY_ESC(strset_t{ "native", "debugging" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-destructor-report" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, stack_clearing, TIMEMORY_SETTINGS_KEY("STACK_CLEARING"),
        "Enable/disable stopping any markers still running during finalization", true,
        TIMEMORY_ESC(strset_t{ "native", "debugging" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-stack-clearing" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, banner, TIMEMORY_SETTINGS_KEY("BANNER"),
        "Notify about tim::manager creation and destruction",
        (get_env<bool>(TIMEMORY_SETTINGS_KEY("LIBRARY_CTOR"), false)),
        TIMEMORY_ESC(strset_t{ "native", "debugging" }));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_ert()
{
    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_num_threads, TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS"),
        "Number of threads to use when running ERT", 0,
        TIMEMORY_ESC(strset_t{ "native", "ert", "parallelism", "roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(uint64_t, ert_num_threads_cpu,
                                  TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS_CPU"),
                                  "Number of threads to use when running ERT on CPU",
                                  std::thread::hardware_concurrency(),
                                  TIMEMORY_ESC(strset_t{ "native", "ert", "parallelism",
                                                         "roofline", "cpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_num_threads_gpu, TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS_GPU"),
        "Number of threads which launch kernels when running ERT on the GPU", 1,
        TIMEMORY_ESC(
            strset_t{ "native", "ert", "parallelism", "roofline", "gpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_num_streams, TIMEMORY_SETTINGS_KEY("ERT_NUM_STREAMS"),
        "Number of streams to use when launching kernels in ERT on the GPU", 1,
        TIMEMORY_ESC(
            strset_t{ "native", "ert", "parallelism", "roofline", "gpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_grid_size, TIMEMORY_SETTINGS_KEY("ERT_GRID_SIZE"),
        "Configure the grid size (number of blocks) for ERT on GPU (0 == "
        "auto-compute)",
        0,
        TIMEMORY_ESC(
            strset_t{ "native", "ert", "parallelism", "roofline", "gpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_block_size, TIMEMORY_SETTINGS_KEY("ERT_BLOCK_SIZE"),
        "Configure the block size (number of threads per block) for ERT on GPU", 1024,
        TIMEMORY_ESC(
            strset_t{ "native", "ert", "parallelism", "roofline", "gpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_alignment, TIMEMORY_SETTINGS_KEY("ERT_ALIGNMENT"),
        "Configure the alignment (in bits) when running ERT on CPU (0 == 8 * "
        "sizeof(T))",
        0, TIMEMORY_ESC(strset_t{ "native", "ert", "roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_min_working_size, TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE"),
        "Configure the minimum working size when running ERT (0 == device specific)", 0,
        TIMEMORY_ESC(strset_t{ "native", "ert", "roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_min_working_size_cpu,
        TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE_CPU"),
        "Configure the minimum working size when running ERT on CPU", 64,
        TIMEMORY_ESC(strset_t{ "native", "ert", "roofline", "cpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_min_working_size_gpu,
        TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE_GPU"),
        "Configure the minimum working size when running ERT on GPU", 10 * 1000 * 1000,
        TIMEMORY_ESC(strset_t{ "native", "ert", "roofline", "gpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(uint64_t, ert_max_data_size,
                                  TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE"),
                                  "Configure the max data size when running ERT", 0,
                                  TIMEMORY_ESC(strset_t{ "native", "ert", "roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_max_data_size_cpu, TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE_CPU"),
        "Configure the max data size when running ERT on CPU", 0,
        TIMEMORY_ESC(strset_t{ "native", "ert", "roofline", "cpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_max_data_size_gpu, TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE_GPU"),
        "Configure the max data size when running ERT on GPU", 500 * 1000 * 1000,
        TIMEMORY_ESC(strset_t{ "native", "ert", "roofline", "gpu_roofline" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        std::string, ert_skip_ops, TIMEMORY_SETTINGS_KEY("ERT_SKIP_OPS"),
        "Skip these number of ops (i.e. ERT_FLOPS) when were set at compile time", "",
        TIMEMORY_ESC(strset_t{ "native", "ert", "roofline" }));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_dart()
{
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        std::string, dart_type, TIMEMORY_SETTINGS_KEY("DART_TYPE"),
        "Only echo this measurement type (see also: DART_OUTPUT)", "",
        TIMEMORY_ESC(strset_t{ "native", "io", "dart", "cdash" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-dart-type" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        uint64_t, dart_count, TIMEMORY_SETTINGS_KEY("DART_COUNT"),
        "Only echo this number of dart tags (see also: DART_OUTPUT)", 1,
        TIMEMORY_ESC(strset_t{ "native", "io", "dart", "cdash" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-dart-count" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, dart_label, TIMEMORY_SETTINGS_KEY("DART_LABEL"),
        "Echo the category instead of the label (see also: DART_OUTPUT)", true,
        TIMEMORY_ESC(strset_t{ "native", "io", "dart", "cdash" }),
        strvector_t({ "--" TIMEMORY_PROJECT_NAME "-dart-label" }), -1, 1);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_disabled()
{
#if !defined(TIMEMORY_USE_OMPT)
    disable_category("ompt");
#endif

#if !defined(TIMEMORY_USE_MPI)
    disable_category("mpi");
#endif

#if !defined(TIMEMORY_USE_UPCXX)
    disable_category("upcxx");
#endif

#if !defined(TIMEMORY_USE_MPI) && !defined(TIMEMORY_USE_UPCXX)
    disable_category("dmp");
#endif

#if !defined(TIMEMORY_USE_PAPI)
    disable_category("papi");
    disable_category("cpu_roofline");
#endif

#if !defined(TIMEMORY_USE_CUDA)
    disable_category("cuda");
#endif

#if !defined(TIMEMORY_USE_NVTX)
    disable_category("nvtx");
#endif

#if !defined(TIMEMORY_USE_CUPTI)
    disable_category("cupti");
    disable_category("gpu_roofline");
#endif

#if !defined(TIMEMORY_USE_CUPTI_PCSAMPLING)
    disable_category("cupti_pcsampling");
#endif

#if !defined(TIMEMORY_USE_PAPI) && !defined(TIMEMORY_USE_CUPTI)
    disable_category("roofline");
    disable_category("ert");
#endif

#if !defined(TIMEMORY_USE_CRAYPAT)
    disable_category("craypat");
#endif

#if !defined(TIMEMORY_USE_GOTCHA)
    disable_category("gotcha");
#endif
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize()
{
    // m_data.clear();
    if(m_data.empty())
        m_data.reserve(160);

    initialize_core();
    initialize_components();
    initialize_io();
    initialize_format();
    initialize_parallel();
    initialize_tpls();
    initialize_roofline();
    initialize_miscellaneous();
    initialize_ert();
    initialize_dart();
    initialize_disabled();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
bool
settings::read(std::string inp)
{
    auto _debug   = get_with_env_fallback(TIMEMORY_SETTINGS_KEY("DEBUG"), false);
    auto _verbose = get_with_env_fallback(TIMEMORY_SETTINGS_KEY("VERBOSE"), 0);
    if(_debug)
        _verbose += 16;

    auto _orig = inp;
    inp        = format(inp, get_tag());

    if(inp != _orig && _verbose >= 3)
    {
        TIMEMORY_PRINTF(stderr, "[%s][settings][%s]> '%s' was expanded to '%s'...\n",
                        TIMEMORY_PROJECT_NAME, __FUNCTION__, _orig.c_str(), inp.c_str());
    }

#if defined(TIMEMORY_UNIX)
    auto file_exists = [](const std::string& _fname) {
        struct stat _buffer;
        if(stat(_fname.c_str(), &_buffer) == 0)
            return (S_ISREG(_buffer.st_mode) != 0 || S_ISLNK(_buffer.st_mode) != 0);
        return false;
    };
#else
    auto file_exists = [](const std::string&) { return true; };
#endif

    if(file_exists(inp))
    {
        std::ifstream ifs{ inp };
        if(ifs.is_open())
        {
            return read(ifs, inp);
        }
        else
        {
            TIMEMORY_EXCEPTION("Error reading configuration file: " << inp)
        }
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
bool
settings::read(std::istream& ifs, std::string inp)
{
    auto _debug   = get_with_env_fallback(TIMEMORY_SETTINGS_KEY("DEBUG"), false);
    auto _verbose = get_with_env_fallback(TIMEMORY_SETTINGS_KEY("VERBOSE"), 0);
    if(_debug)
        _verbose += 16;

    inp = format(inp, get_tag());
    if(!inp.empty())
    {
        if(m_read_configs.find(inp) != m_read_configs.end())
        {
            if(_verbose >= 2)
            {
                TIMEMORY_PRINT_HERE("Warning! Re-reading config file: %s", inp.c_str());
            }
        }
        m_read_configs.emplace(inp);
    }

    if(!inp.empty())
    {
        for(auto& itr : m_config_stack)
        {
            if(itr == inp)
            {
                if(_verbose >= 2)
                {
                    TIMEMORY_PRINT_HERE("Config file '%s' is already being read. "
                                        "Preventing recursion",
                                        inp.c_str());
                }
                return false;
            }
        }
        m_config_stack.emplace_back(inp);
    }

    scope::destructor _dtor{ [this, inp]() {
        if(!inp.empty() && !m_config_stack.empty())
            m_config_stack.pop_back();
    } };

    std::string _inp = inp;
    if(_inp.length() > 30)
    {
        auto _inp_delim = delimit(filepath::canonical(inp), "/");
        auto _sz        = _inp_delim.size();
        if(_sz > 4)
            _inp = TIMEMORY_JOIN('/', "", _inp_delim.at(0), _inp_delim.at(1), "...",
                                 _inp_delim.at(_sz - 2), _inp_delim.at(_sz - 1));
        else if(_sz > 3)
            _inp = TIMEMORY_JOIN('/', "", _inp_delim.at(0), "...", _inp_delim.at(_sz - 2),
                                 _inp_delim.at(_sz - 1));
        else if(_sz > 2)
            _inp = TIMEMORY_JOIN('/', "", "...", _inp_delim.at(_sz - 2),
                                 _inp_delim.at(_sz - 1));
        _inp = filepath::osrepr(_inp);
    }

    if(inp.find(".json") != std::string::npos || inp == "json")
    {
        using policy_type = policy::input_archive<cereal::JSONInputArchive, TIMEMORY_API>;
        auto ia           = policy_type::get(ifs);
        try
        {
            ia->setNextName(TIMEMORY_PROJECT_NAME);
            ia->startNode();
            {
                try
                {
                    ia->setNextName("metadata");
                    ia->startNode();
                    // settings
                    (*ia)(cereal::make_nvp("settings", *this));
                    ia->finishNode();
                } catch(...)
                {
                    // settings
                    (*ia)(cereal::make_nvp("settings", *this));
                }
            }
            ia->finishNode();
        } catch(tim::cereal::Exception& e)
        {
            TIMEMORY_PRINT_HERE("Exception reading %s :: %s", _inp.c_str(), e.what());
#if defined(TIMEMORY_INTERNAL_TESTING)
            TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(true, 8);
#endif
            return false;
        }
        return true;
    }
#if defined(TIMEMORY_USE_XML)
    else if(inp.find(".xml") != std::string::npos || inp == "xml")
    {
        using policy_type = policy::input_archive<cereal::XMLInputArchive, TIMEMORY_API>;
        auto ia           = policy_type::get(ifs);
        try
        {
            ia->setNextName(TIMEMORY_PROJECT_NAME);
            ia->startNode();
            {
                try
                {
                    ia->setNextName("metadata");
                    ia->startNode();
                    // settings
                    (*ia)(cereal::make_nvp("settings", *this));
                    ia->finishNode();
                } catch(...)
                {
                    // settings
                    (*ia)(cereal::make_nvp("settings", *this));
                }
            }
            ia->finishNode();
        } catch(tim::cereal::Exception& e)
        {
            TIMEMORY_PRINT_HERE("Exception reading %s :: %s", _inp.c_str(), e.what());
#    if defined(TIMEMORY_INTERNAL_TESTING)
            TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(true, 8);
#    endif
            return false;
        }
        return true;
    }
#endif
    else
    {
        if(inp.empty())
            inp = "text";

        auto _is_comment = [](std::string _s) {
            if(_s.empty())
                return true;
            {
                auto _spos = _s.find_first_of(" \t\n\r\v\f");
                auto _cpos = _s.find_first_not_of(" \t\n\r\v\f");
                if(_spos < _cpos && _cpos != std::string::npos)
                    _s = _s.substr(_cpos);
                if(_s.empty())
                    return true;
            }
            std::locale _lc{};
            for(const auto& itr : _s)
            {
                // if graphical character is # then it a comment
                if(std::isgraph(itr, _lc))
                    return (itr == '#');
            }
            // if there were no printable characters, treat as comment
            return true;
        };

        int                                                      expected = 0;
        int                                                      valid    = 0;
        std::map<std::string, std::string>                       _variables{};
        std::function<std::pair<std::string, bool>(std::string)> _resolve_variables{};

        auto _resolve = [&_resolve_variables](std::string _v) {
            std::pair<std::string, bool> _resolved{};
            while((_resolved = _resolve_variables(_v)).second)
                _v = _resolved.first;
            return _v;
        };

        // replaces VAR in $env:VAR with the value of the environment variable VAR
        auto _resolve_env = [&](std::string _v) -> std::pair<std::string, bool> {
            const char* _env_syntax = "$env:";
            auto        _pos        = _v.find(_env_syntax);
            if(_pos == std::string::npos)
                return std::make_pair(_v, false);

            if(_verbose >= 5)
                TIMEMORY_PRINTF(
                    stderr,
                    "[%s][settings]['%s']> Resolving environment variables in '%s'\n",
                    TIMEMORY_PROJECT_NAME, _inp.c_str(), _v.c_str());

            // remove the $env: and store everything before it
            std::string _prefix = _v.substr(0, _pos);
            _v                  = _v.substr(_pos + strlen(_env_syntax));

            // resolve any remaining variables or environment variables in the
            // substring
            _v = _resolve(_v);

            auto _re  = std::regex{ "^([A-Z])([A-Z0-9_]+)" };
            auto _beg = std::sregex_iterator{ _v.begin(), _v.end(), _re };
            auto _end = std::sregex_iterator{};
            for(auto itr = _beg; itr != _end; ++itr)
            {
                auto _env_name = itr->str();
                if(_env_name.empty())
                {
                    std::stringstream _msg{};
                    _msg << "Error! evaluation of $env: syntax returned an empty string. "
                            "Environment variables must consist solely of uppercase "
                            "letters, digits, and '_' and do not begin with a digit :: "
                         << _v;
                    throw std::runtime_error(_msg.str());
                }

                return std::make_pair(_prefix +
                                          _v.replace(0, _env_name.length(),
                                                     get_env<std::string>(_env_name, "")),
                                      true);
            }
            return std::make_pair(_prefix + _v, false);
        };

        // replaces VAR in $VAR or ${VAR} with the value of the config defined
        // variable or the settings value
        auto _resolve_var = [&](std::string _v) -> std::pair<std::string, bool> {
            auto _pos = _v.find('$');
            if(_pos == std::string::npos)
                return std::make_pair(_v, false);

            if(_verbose >= 5)
                TIMEMORY_PRINTF(stderr,
                                "[%s][settings]['%s']> Expanding settings in '%s'\n",
                                TIMEMORY_PROJECT_NAME, _inp.c_str(), _v.c_str());

            // remove the $ and store everything before it
            std::string _prefix = _v.substr(0, _pos);
            _v                  = _v.substr(_pos + 1);

            // resolve any remaining variables or environment variables in the
            // substring
            _v = _resolve(_v);

            for(const auto& itr : _variables)
            {
                auto _var = itr.first.substr(1);
                if(_v.find(std::string{ "{" } + _var + "}") == 0)
                    return std::make_pair(
                        _prefix + _v.replace(0, _var.length() + 2, itr.second), true);
                else if(_v.find(_var) == 0)
                    return std::make_pair(
                        _prefix + _v.replace(0, _var.length(), itr.second), true);
            }

            auto _re  = std::regex{ "^[{|]([A-Za-z])([A-Za-z0-9_]+)[|}]" };
            auto _beg = std::sregex_iterator{ _v.begin(), _v.end(), _re };
            auto _end = std::sregex_iterator{};
            for(auto itr = _beg; itr != _end; ++itr)
            {
                auto _var_name = itr->str();
                if(_var_name.empty())
                {
                    std::stringstream _msg{};
                    _msg << "Error! evaluation of setting variable: syntax returned an "
                            "empty string. Settings must consist solely of alphanumeric "
                            "characters and '_' and do not begin with a digit :: "
                         << _v;
                    throw std::runtime_error(_msg.str());
                }

                for(const auto& iitr : *this)
                {
                    if(iitr.second->matches(_v))
                        return std::make_pair(_prefix +
                                                  _v.replace(0, _var_name.length(),
                                                             iitr.second->as_string()),
                                              true);
                }
            }
            return std::make_pair(_prefix + _v, false);
        };

        _resolve_variables = [&](std::string _v) {
            std::pair<std::string, bool> _results = { _v, false };

            if(_v.empty())
                return _results;

            // if not $ then just return
            auto _pos = _v.find('$');
            if(_pos == std::string::npos)
                return _results;

            if(_verbose >= 5)
                TIMEMORY_PRINTF(stderr,
                                "[%s][settings]['%s']> Resolving variables in '%s'\n",
                                TIMEMORY_PROJECT_NAME, _inp.c_str(), _v.c_str());

            std::pair<std::string, bool> _replace_results = {};
            while((_replace_results = _resolve_env(_v)).second)
                _v = _replace_results.first;

            while((_replace_results = _resolve_var(_v)).second)
                _v = _replace_results.first;

            if(_results.first != _v)
                return std::make_pair(_v, true);
            return _results;
        };

        while(ifs)
        {
            std::string line = {};
            std::getline(ifs, line);
            if(!ifs || line.empty())
                continue;
            if(line.empty())
                continue;
            if(_verbose >= 5)
                TIMEMORY_PRINTF(stderr, "[%s][settings]['%s']> %s\n",
                                TIMEMORY_PROJECT_NAME, _inp.c_str(), line.c_str());
            if(_is_comment(line))
                continue;
            ++expected;
            // tokenize the string
            auto _equal_pos = line.find('=');
            auto delim      = std::vector<std::string>{};
            if(_equal_pos != std::string::npos)
            {
                auto line_key  = line.substr(0, _equal_pos);
                auto line_data = line.substr(_equal_pos + 1);
                delim          = delimit(line_key, "\n\t ");
                while(delim.size() > 1)
                    delim.pop_back();
                for(const auto& ditr : delimit(line_data, "\n\t,; "))
                    delim.emplace_back(ditr);
            }
            else
            {
                delim = delimit(line, "\n\t,;= ");
            }
            if(!delim.empty())
            {
                std::string key = delim.front();
                std::string val = {};
                // combine into another string separated by commas
                for(size_t i = 1; i < delim.size(); ++i)
                {
                    if(delim.empty() || delim.at(i) == "#" || delim.at(i).at(0) == '#')
                        continue;
                    auto _v = _resolve(delim.at(i));
                    if(_v != delim.at(i))
                    {
                        std::string _nv = {};
                        for(const auto& itr : delimit(_v, "\n\t,; "))
                            _nv += "," + itr;
                        if(!_nv.empty())
                            _nv = _nv.substr(1);
                        _v = _nv;
                    }
                    val += "," + _v;
                }
                // if there was any fields, remove the leading comma
                if(val.length() > 0)
                    val = val.substr(1);
                // extremely unlikely
                if(key.empty())
                    continue;
                // handle comment
                if(key == "#" || key.at(0) == '#')
                    continue;
                // this is a variable, e.g.:
                // $MYVAR = ON      # this is a variable
                // TIMEMORY_PRINT_STATS = $MYVAR
                // TIMEMORY_PRINT_MIN   = $MYVAR
                if(key.at(0) == '$')
                {
                    const char* _env_syntax = "$env:";
                    auto        _pos        = key.find(_env_syntax);
                    if(_pos == 0)
                    {
                        auto _v = key.substr(strlen(_env_syntax));
                        set_env(_v, val, 0);
                        _variables.emplace(std::string{ "$" } + _v, val);
                    }
                    else
                    {
                        _variables.emplace(key, val);
                    }
                    continue;
                }

                auto incr = valid;
                for(const auto& itr : *this)
                {
                    if(itr.second->matches(key))
                    {
                        if(_verbose >= 2)
                            TIMEMORY_PRINTF(stderr, "[%s][settings]['%s']> %-30s :: %s\n",
                                            TIMEMORY_PROJECT_NAME, _inp.c_str(),
                                            key.c_str(), val.c_str());
                        ++valid;
                        if(itr.second->matches("config_file"))
                        {
                            auto _cfgs = delimit(val, ";,: ");
                            for(const auto& fitr : _cfgs)
                            {
                                if(format(fitr, get_tag()) != format(inp, get_tag()))
                                {
                                    try
                                    {
                                        read(fitr);
                                    } catch(std::runtime_error& _e)
                                    {
                                        if(_verbose >= 0)
                                            TIMEMORY_PRINTF(
                                                stderr,
                                                "[%s][settings]['%s']> Error reading "
                                                "'%s' :: %s\n",
                                                TIMEMORY_PROJECT_NAME, _inp.c_str(),
                                                fitr.c_str(), _e.what());
                                    }
                                }
                            }
                        }
                        else
                        {
                            itr.second->parse(val, update_type::config);
                        }
                    }
                }

                if(incr == valid)
                {
                    if(get_strict_config())
                        throw std::runtime_error(TIMEMORY_JOIN(
                            "", "Unknown setting '", key, "' (value = '", val, "')"));
                    auto _key = key;
                    for(auto& itr : _key)
                        itr = std::toupper(itr);
                    if(_key.find(TIMEMORY_SETTINGS_PREFIX) == 0)
                    {
                        if(_verbose >= 3)
                        {
                            TIMEMORY_PRINTF(
                                stderr,
                                "[settings]['%s']> Unknown setting with "
                                "recognized prefix ('%s') will be exported to "
                                "environment after timemory_init (if not found later): "
                                "'%s' (value = '%s')\n",
                                _inp.c_str(), TIMEMORY_SETTINGS_PREFIX, _key.c_str(),
                                val.c_str());
                        }
                        if(!std::any_of(m_unknown_configs.begin(),
                                        m_unknown_configs.end(),
                                        [&key, &val](auto&& itr) {
                                            return std::tie(key, val) ==
                                                   std::tie(itr.first, itr.second);
                                        }))
                            m_unknown_configs.emplace_back(key, val);
                    }
                    else
                    {
                        if(_verbose >= 2)
                        {
                            TIMEMORY_PRINTF(stderr,
                                            "[settings]['%s']> WARNING! Unknown setting "
                                            "ignored: '%s' (value = '%s')\n",
                                            _inp.c_str(), key.c_str(), val.c_str());
                        }
                    }
                }
            }
        }
        return (expected == valid);
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::init_config(bool _search_default)
{
    if(get_debug() || get_verbose() > 3)
        TIMEMORY_PRINT_HERE("%s", "");

    static const auto* const _homedir = "%env{HOME}%";
    const auto               _dcfgs   = std::set<std::string>{
        format(_homedir + std::string("/." TIMEMORY_PROJECT_NAME ".cfg"), get_tag()),
        format(_homedir + std::string("/." TIMEMORY_PROJECT_NAME ".json"), get_tag()),
        format(_homedir + std::string("/." TIMEMORY_PROJECT_NAME ".xml"), get_tag())
    };

    auto _cfg   = get_config_file();
    auto _files = delimit(_cfg, ",;:");
    for(auto citr : _files)
    {
        // resolve any keys
        citr = format(citr, get_tag());

        // a previous config file may have suppressed it
        if(get_suppress_config())
            break;

        // skip defaults
        if(!_search_default && _dcfgs.find(citr) != _dcfgs.end())
            continue;

        if(m_read_configs.find(citr) != m_read_configs.end())
            continue;

        std::ifstream ifs{ citr };
        if(ifs)
        {
            if(read(ifs, citr))
                m_read_configs.emplace(citr);
        }
        else if(_dcfgs.find(citr) == _dcfgs.end())
        {
            TIMEMORY_EXCEPTION(std::string("Error reading configuration file: ") + citr);
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::handle_exception(std::string_view _env_var, std::string_view _func,
                           std::string_view _type, std::string_view _msg)
{
    TIMEMORY_PRINTF_FATAL(stderr, "[%s] Error! %s& settings::%s() failed: %s\n",
                          _env_var.data(), _type.data(), _func.data(), _msg.data());
    timemory_print_demangled_backtrace<8>(
        std::cerr, std::string{},
        TIMEMORY_JOIN("", _type, "& settings::", _func, "() :: ", _msg));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, config_file,
                             TIMEMORY_SETTINGS_KEY("CONFIG_FILE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, suppress_parsing,
                             TIMEMORY_SETTINGS_KEY("SUPPRESS_PARSING"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, suppress_config,
                             TIMEMORY_SETTINGS_KEY("SUPPRESS_CONFIG"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, strict_config, TIMEMORY_SETTINGS_KEY("STRICT_CONFIG"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, enabled, TIMEMORY_SETTINGS_KEY("ENABLED"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, auto_output, TIMEMORY_SETTINGS_KEY("AUTO_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, cout_output, TIMEMORY_SETTINGS_KEY("COUT_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, file_output, TIMEMORY_SETTINGS_KEY("FILE_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, text_output, TIMEMORY_SETTINGS_KEY("TEXT_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, json_output, TIMEMORY_SETTINGS_KEY("JSON_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, tree_output, TIMEMORY_SETTINGS_KEY("TREE_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, dart_output, TIMEMORY_SETTINGS_KEY("DART_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, time_output, TIMEMORY_SETTINGS_KEY("TIME_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, plot_output, TIMEMORY_SETTINGS_KEY("PLOT_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, diff_output, TIMEMORY_SETTINGS_KEY("DIFF_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, flamegraph_output,
                             TIMEMORY_SETTINGS_KEY("FLAMEGRAPH_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, ctest_notes, TIMEMORY_SETTINGS_KEY("CTEST_NOTES"))
TIMEMORY_SETTINGS_MEMBER_DEF(int, verbose, TIMEMORY_SETTINGS_KEY("VERBOSE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, debug, TIMEMORY_SETTINGS_KEY("DEBUG"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, banner, TIMEMORY_SETTINGS_KEY("BANNER"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, collapse_threads,
                             TIMEMORY_SETTINGS_KEY("COLLAPSE_THREADS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, collapse_processes,
                             TIMEMORY_SETTINGS_KEY("COLLAPSE_PROCESSES"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint16_t, max_depth, TIMEMORY_SETTINGS_KEY("MAX_DEPTH"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, time_format,
                             TIMEMORY_SETTINGS_KEY("TIME_FORMAT"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, precision, TIMEMORY_SETTINGS_KEY("PRECISION"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, width, TIMEMORY_SETTINGS_KEY("WIDTH"))
TIMEMORY_SETTINGS_MEMBER_DEF(int32_t, max_width, TIMEMORY_SETTINGS_KEY("MAX_WIDTH"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, scientific, TIMEMORY_SETTINGS_KEY("SCIENTIFIC"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, timing_precision,
                             TIMEMORY_SETTINGS_KEY("TIMING_PRECISION"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, timing_width, TIMEMORY_SETTINGS_KEY("TIMING_WIDTH"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, timing_units,
                             TIMEMORY_SETTINGS_KEY("TIMING_UNITS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, timing_scientific,
                             TIMEMORY_SETTINGS_KEY("TIMING_SCIENTIFIC"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, memory_precision,
                             TIMEMORY_SETTINGS_KEY("MEMORY_PRECISION"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, memory_width, TIMEMORY_SETTINGS_KEY("MEMORY_WIDTH"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, memory_units,
                             TIMEMORY_SETTINGS_KEY("MEMORY_UNITS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, memory_scientific,
                             TIMEMORY_SETTINGS_KEY("MEMORY_SCIENTIFIC"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, output_path,
                             TIMEMORY_SETTINGS_KEY("OUTPUT_PATH"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, output_prefix,
                             TIMEMORY_SETTINGS_KEY("OUTPUT_PREFIX"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, input_path, TIMEMORY_SETTINGS_KEY("INPUT_PATH"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, input_prefix,
                             TIMEMORY_SETTINGS_KEY("INPUT_PREFIX"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, input_extensions,
                             TIMEMORY_SETTINGS_KEY("INPUT_EXTENSIONS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, dart_type, TIMEMORY_SETTINGS_KEY("DART_TYPE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, dart_count, TIMEMORY_SETTINGS_KEY("DART_COUNT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, dart_label, TIMEMORY_SETTINGS_KEY("DART_LABEL"))
TIMEMORY_SETTINGS_MEMBER_DEF(size_t, max_thread_bookmarks,
                             TIMEMORY_SETTINGS_KEY("MAX_THREAD_BOOKMARKS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, cpu_affinity, TIMEMORY_SETTINGS_KEY("CPU_AFFINITY"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, stack_clearing,
                             TIMEMORY_SETTINGS_KEY("STACK_CLEARING"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, add_secondary, TIMEMORY_SETTINGS_KEY("ADD_SECONDARY"))
TIMEMORY_SETTINGS_MEMBER_DEF(size_t, throttle_count,
                             TIMEMORY_SETTINGS_KEY("THROTTLE_COUNT"))
TIMEMORY_SETTINGS_MEMBER_DEF(size_t, throttle_value,
                             TIMEMORY_SETTINGS_KEY("THROTTLE_VALUE"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, global_components,
                             TIMEMORY_SETTINGS_KEY("GLOBAL_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, tuple_components,
                             TIMEMORY_SETTINGS_KEY("TUPLE_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, list_components,
                             TIMEMORY_SETTINGS_KEY("LIST_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, ompt_components,
                             TIMEMORY_SETTINGS_KEY("OMPT_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, mpip_components,
                             TIMEMORY_SETTINGS_KEY("MPIP_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, ncclp_components,
                             TIMEMORY_SETTINGS_KEY("NCCLP_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, trace_components,
                             TIMEMORY_SETTINGS_KEY("TRACE_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, profiler_components,
                             TIMEMORY_SETTINGS_KEY("PROFILER_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, kokkos_components,
                             TIMEMORY_SETTINGS_KEY("KOKKOS_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, components, TIMEMORY_SETTINGS_KEY("COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, network_interface,
                             TIMEMORY_SETTINGS_KEY("NETWORK_INTERFACE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, mpi_init, TIMEMORY_SETTINGS_KEY("MPI_INIT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, mpi_finalize, TIMEMORY_SETTINGS_KEY("MPI_FINALIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, mpi_thread, TIMEMORY_SETTINGS_KEY("MPI_THREAD"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, mpi_thread_type,
                             TIMEMORY_SETTINGS_KEY("MPI_THREAD_TYPE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, upcxx_init, TIMEMORY_SETTINGS_KEY("UPCXX_INIT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, upcxx_finalize,
                             TIMEMORY_SETTINGS_KEY("UPCXX_FINALIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, papi_threading,
                             TIMEMORY_SETTINGS_KEY("PAPI_THREADING"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, papi_multiplexing,
                             TIMEMORY_SETTINGS_KEY("PAPI_MULTIPLEXING"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, papi_fail_on_error,
                             TIMEMORY_SETTINGS_KEY("PAPI_FAIL_ON_ERROR"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, papi_quiet, TIMEMORY_SETTINGS_KEY("PAPI_QUIET"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, papi_events,
                             TIMEMORY_SETTINGS_KEY("PAPI_EVENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, papi_attach, TIMEMORY_SETTINGS_KEY("PAPI_ATTACH"))
TIMEMORY_SETTINGS_MEMBER_DEF(int, papi_overflow, TIMEMORY_SETTINGS_KEY("PAPI_OVERFLOW"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, cuda_event_batch_size,
                             TIMEMORY_SETTINGS_KEY("CUDA_EVENT_BATCH_SIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, nvtx_marker_device_sync,
                             TIMEMORY_SETTINGS_KEY("NVTX_MARKER_DEVICE_SYNC"))
TIMEMORY_SETTINGS_MEMBER_DEF(int32_t, cupti_activity_level,
                             TIMEMORY_SETTINGS_KEY("CUPTI_ACTIVITY_LEVEL"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, cupti_activity_kinds,
                             TIMEMORY_SETTINGS_KEY("CUPTI_ACTIVITY_KINDS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, cupti_events,
                             TIMEMORY_SETTINGS_KEY("CUPTI_EVENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, cupti_metrics,
                             TIMEMORY_SETTINGS_KEY("CUPTI_METRICS"))
TIMEMORY_SETTINGS_MEMBER_DEF(int, cupti_device, TIMEMORY_SETTINGS_KEY("CUPTI_DEVICE"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, roofline_mode,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, cpu_roofline_mode,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, gpu_roofline_mode,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, cpu_roofline_events,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_EVENTS_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, gpu_roofline_events,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_EVENTS_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, roofline_type_labels,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, roofline_type_labels_cpu,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, roofline_type_labels_gpu,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, instruction_roofline,
                             TIMEMORY_SETTINGS_KEY("INSTRUCTION_ROOFLINE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_num_threads,
                             TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_num_threads_cpu,
                             TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_num_threads_gpu,
                             TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_num_streams,
                             TIMEMORY_SETTINGS_KEY("ERT_NUM_STREAMS"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_grid_size,
                             TIMEMORY_SETTINGS_KEY("ERT_GRID_SIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_block_size,
                             TIMEMORY_SETTINGS_KEY("ERT_BLOCK_SIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_alignment,
                             TIMEMORY_SETTINGS_KEY("ERT_ALIGNMENT"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_min_working_size,
                             TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_min_working_size_cpu,
                             TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_min_working_size_gpu,
                             TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_max_data_size,
                             TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_max_data_size_cpu,
                             TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_max_data_size_gpu,
                             TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, ert_skip_ops,
                             TIMEMORY_SETTINGS_KEY("ERT_SKIP_OPS"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, craypat_categories,
                             TIMEMORY_SETTINGS_KEY("CRAYPAT"))
TIMEMORY_SETTINGS_MEMBER_DEF(int32_t, node_count, TIMEMORY_SETTINGS_KEY("NODE_COUNT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, destructor_report,
                             TIMEMORY_SETTINGS_KEY("DESTRUCTOR_REPORT"))
TIMEMORY_SETTINGS_MEMBER_DEF(std::string, python_exe, TIMEMORY_SETTINGS_KEY("PYTHON_EXE"))
// stream
TIMEMORY_SETTINGS_MEMBER_DEF(int64_t, separator_frequency,
                             TIMEMORY_SETTINGS_KEY("SEPARATOR_FREQ"))
// signals
TIMEMORY_SETTINGS_MEMBER_DEF(bool, enable_signal_handler,
                             TIMEMORY_SETTINGS_KEY("ENABLE_SIGNAL_HANDLER"))
TIMEMORY_SETTINGS_REFERENCE_DEF(bool, allow_signal_handler,
                                TIMEMORY_SETTINGS_KEY("ALLOW_SIGNAL_HANDLER"))
TIMEMORY_SETTINGS_REFERENCE_DEF(bool, enable_all_signals,
                                TIMEMORY_SETTINGS_KEY("ENABLE_ALL_SIGNALS"))
TIMEMORY_SETTINGS_REFERENCE_DEF(bool, disable_all_signals,
                                TIMEMORY_SETTINGS_KEY("DISABLE_ALL_SIGNALS"))
// miscellaneous ref
TIMEMORY_SETTINGS_REFERENCE_DEF(bool, flat_profile, TIMEMORY_SETTINGS_KEY("FLAT_PROFILE"))
TIMEMORY_SETTINGS_REFERENCE_DEF(bool, timeline_profile,
                                TIMEMORY_SETTINGS_KEY("TIMELINE_PROFILE"))
TIMEMORY_SETTINGS_REFERENCE_DEF(process::id_t, target_pid,
                                TIMEMORY_SETTINGS_KEY("TARGET_PID"))
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim

#include "timemory/tpls/cereal/archives.hpp"

TIMEMORY_SETTINGS_EXTERN_TEMPLATE(TIMEMORY_API)

#endif  // TIMEMORY_SETTINGS_SETTINGS_CPP_
