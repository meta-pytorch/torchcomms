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

#include "timemory/log/color.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/tpls/cereal/cereal/cereal.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/join.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iosfwd>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tim
{
namespace argparse
{
namespace helpers
{
//
//--------------------------------------------------------------------------------------//
//
inline bool
not_is_space(int ch)
{
    return std::isspace(ch) == 0;
}
//
//--------------------------------------------------------------------------------------//
//
inline uint64_t
lcount(const std::string& s, bool (*f)(int) = not_is_space)
{
    uint64_t c = 0;
    for(size_t i = 0; i < s.length(); ++i, ++c)
    {
        if(f(s.at(i)))
            break;
    }
    return c;
}
//
//--------------------------------------------------------------------------------------//
//
inline std::string
ltrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), f));
    return s;
}
//
//--------------------------------------------------------------------------------------//
//
inline std::string
rtrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), f).base(), s.end());
    return s;
}
//
//--------------------------------------------------------------------------------------//
//
inline std::string
trim(std::string s, bool (*f)(int) = not_is_space)
{
    ltrim(s, f);
    rtrim(s, f);
    return s;
}
//
//--------------------------------------------------------------------------------------//
//
inline char*
strdup(const char* s)
{
    auto  slen   = strlen(s);
    auto* result = new char[slen + 1];
    if(result)
    {
        memcpy(result, s, slen * sizeof(char));
        result[slen] = '\0';
        return result;
    }
    return nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename InputIt>
inline std::string
join(InputIt begin, InputIt end, const std::string& separator = " ")
{
    std::ostringstream ss;
    if(begin != end)
    {
        ss << *begin++;
    }
    while(begin != end)
    {
        ss << separator;
        ss << *begin++;
    }
    return ss.str();
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
is_numeric(const std::string& arg)
{
    auto _nidx = arg.find_first_of("0123456789");
    auto _oidx = arg.find_first_not_of("0123456789.Ee+-*/");

    // must have number somewhere
    if(_nidx == std::string::npos)
        return false;

    // if something other than number or scientific notation
    if(_oidx != std::string::npos)
        return false;

    // numbers + possible scientific notation
    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline int
find_equiv(const std::string& s)
{
    for(size_t i = 0; i < s.length(); ++i)
    {
        // if find graph symbol before equal, end search
        // i.e. don't accept --asd)f=0 arguments
        // but allow --asd_f and --asd-f arguments
        if(std::ispunct(static_cast<int>(s[i])) != 0)
        {
            if(s[i] == '=')
            {
                return static_cast<int>(i);
            }
            if(s[i] == '_' || s[i] == '-')
            {
                continue;
            }
            return -1;
        }
    }
    return -1;
}
//
//--------------------------------------------------------------------------------------//
//
inline size_t
find_punct(const std::string& s)
{
    size_t i;
    for(i = 0; i < s.length(); ++i)
    {
        if((std::ispunct(static_cast<int>(s[i])) != 0) && s[i] != '-')
        {
            break;
        }
    }
    return i;
}
//
//--------------------------------------------------------------------------------------//
//
namespace is_container_impl
{
//
template <typename T>
struct is_container : std::false_type
{};
//
template <typename... Args>
struct is_container<std::vector<Args...>> : std::true_type
{};
template <typename... Args>
struct is_container<std::set<Args...>> : std::true_type
{};
template <typename... Args>
struct is_container<std::deque<Args...>> : std::true_type
{};
template <typename... Args>
struct is_container<std::list<Args...>> : std::true_type
{};
//
template <typename T>
struct is_initializing_container : is_container<T>::type
{};
//
template <typename... Args>
struct is_initializing_container<std::initializer_list<Args...>> : std::true_type
{};
}  // namespace is_container_impl
//
//--------------------------------------------------------------------------------------//
//
// type trait to utilize the implementation type traits as well as decay the type
template <typename T>
using is_container = is_container_impl::is_container<decay_t<T>>;
//
//--------------------------------------------------------------------------------------//
//
// type trait to utilize the implementation type traits as well as decay the type
template <typename T>
using is_initializing_container =
    is_container_impl::is_initializing_container<decay_t<T>>;
//
//--------------------------------------------------------------------------------------//
//
template <typename ContainerT, typename Tp>
auto
emplace(ContainerT& _target, Tp&& _arg, int)
    -> decltype(_target.emplace_back(std::forward<Tp>(_arg)))
{
    return _target.emplace_back(std::forward<Tp>(_arg));
}

template <typename ContainerT, typename Tp>
auto
emplace(ContainerT& _target, Tp&& _arg, long)
    -> decltype(_target.emplace(std::forward<Tp>(_arg)))
{
    return _target.emplace(std::forward<Tp>(_arg));
}

template <typename ContainerT, typename Tp>
auto
emplace(ContainerT& _target, Tp&& _arg)
{
    return emplace(_target, std::forward<Tp>(_arg), 0);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace helpers

//
//--------------------------------------------------------------------------------------//
//
//                      argument vector
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::argparse::argument_vector
/// \brief This class exists to simplify creating argument arrays compatible with execv*
/// routines and MPI_Comm_spawn/MPI_Comm_spawn_multiple
///
struct argument_vector : std::vector<std::string>
{
    struct c_args : std::tuple<int, char**, std::string>
    {
        using base_type = std::tuple<int, char**, std::string>;
        template <typename... Args>
        c_args(Args&&... args)
        : base_type(std::forward<Args>(args)...)
        {}

        auto& argc() { return std::get<0>(*this); }
        auto& argv() { return std::get<1>(*this); }
        auto& args() { return std::get<2>(*this); }

        const auto& argc() const { return std::get<0>(*this); }
        const auto& argv() const { return std::get<1>(*this); }
        const auto& args() const { return std::get<2>(*this); }

        void clear()
        {
            // uses comma operator to execute delete and return nullptr
            for(int i = 0; i < argc(); ++i)
                argv()[i] = (delete[] argv()[i], nullptr);
            argv() = (delete[] argv(), nullptr);
        }
    };

    using base_type = std::vector<std::string>;
    using cargs_t   = c_args;

    template <typename... Args>
    argument_vector(Args&&... args)
    : base_type(std::forward<Args>(args)...)
    {}

    explicit argument_vector(int& argc, char**& argv);
    explicit argument_vector(int& argc, const char**& argv);
    explicit argument_vector(int& argc, const char* const*& argv);
    cargs_t get_execv(const base_type& _prepend, size_t _beg = 0,
                      size_t _end = std::numeric_limits<size_t>::max()) const;
    cargs_t get_execv(size_t _beg = 0,
                      size_t _end = std::numeric_limits<size_t>::max()) const;

    // helper function to free the memory created by get_execv, pass by reference
    // so that we can set values to nullptr and avoid multiple delete errors
    static void free_execv(cargs_t& itr) { itr.clear(); }
};
//
//--------------------------------------------------------------------------------------//
//
//                      argument parser
//
//--------------------------------------------------------------------------------------//
//
struct argument_parser
{
    struct arg_result;

    using this_type     = argument_parser;
    using result_type   = arg_result;
    using bool_func_t   = std::function<bool(this_type&)>;
    using action_func_t = std::function<void(this_type&)>;
    using action_pair_t = std::pair<bool_func_t, action_func_t>;
    using error_func_t  = std::function<void(this_type&, arg_result&)>;
    using strvec_t      = std::vector<std::string>;
    using strset_t      = std::set<std::string>;
    //
    //----------------------------------------------------------------------------------//
    //
    struct arg_result
    {
        arg_result() = default;
        arg_result(std::string err) noexcept
        : m_error(true)
        , m_what(std::move(err))
        {}

        operator bool() const { return m_error; }

        friend std::ostream& operator<<(std::ostream& os, const arg_result& dt);

        const std::string& what() const { return m_what; }

    private:
        bool        m_error = false;
        std::string m_what  = {};
    };
    //
    //----------------------------------------------------------------------------------//
    //
    struct known_args : std::tuple<arg_result, int, char**>
    {
        using base_type = std::tuple<arg_result, int, char**>;

        known_args(arg_result _err, int _argc, char** _argv)
        : base_type{ std::move(_err), _argc, _argv }
        {}

        ~known_args()                 = default;
        known_args(const known_args&) = default;
        known_args(known_args&&)      = default;

        known_args& operator=(const known_args&) = default;
        known_args& operator=(known_args&&) = default;

        const arg_result& error() const { return std::get<0>(*this); }
        int               argc() const { return std::get<1>(*this); }
        char**            argv() const { return std::get<2>(*this); }
    };
    //
    //----------------------------------------------------------------------------------//
    //
    struct argument
    {
        using callback_t             = std::function<void(void*&)>;
        using choice_alias_mapping_t = std::map<std::string, std::vector<std::string>>;

        enum Position : int
        {
            LastArgument   = -1,
            IgnoreArgument = -2
        };

        enum Count : int
        {
            ANY = -1
        };

        ~argument() { m_destroy(m_default); }

        argument& name(const std::string& name)
        {
            m_names.emplace_back(name);
            return *this;
        }

        argument& names(const std::vector<std::string>& names)
        {
            for(const auto& itr : names)
                m_names.emplace_back(itr);
            return *this;
        }

        argument& description(std::string description)
        {
            m_desc = std::move(description);
            return *this;
        }

        argument& color(std::string _color)
        {
            m_color = std::move(_color);
            return *this;
        }

        argument& dtype(std::string _dtype)
        {
            m_dtype = std::move(_dtype);
            return *this;
        }

        argument& group(const std::string& _group)
        {
            m_group = _group;
            return *this;
        }

        template <typename Tp>
        argument& required(std::initializer_list<Tp>&& _v)
        {
            container_append(m_requires, std::vector<Tp>{ _v });
            return *this;
        }

        template <typename Tp>
        argument& required(Tp&& _v, std::string_view _msg = {})
        {
            using core_type = concepts::unqualified_type_t<Tp>;
            if constexpr(std::is_same<core_type, bool>::value)
            {
                m_required = _v;
            }
            else
            {
                static_assert(helpers::is_initializing_container<core_type>::value,
                              "Error! Expected a container or initializer_list");
                container_append(m_requires, std::forward<Tp>(_v));
            }

            if(!_msg.empty())
                m_required_info += std::string{ " :: " } + std::string{ _msg };

            return *this;
        }

        argument& position(int position)
        {
            if(position != Position::LastArgument)
            {
                // position + 1 because technically argument zero is the name of the
                // executable
                m_position = position + 1;
            }
            else
            {
                m_position = position;
            }
            return *this;
        }

        argument& max_count(int count)
        {
            m_max_count = count;
            return *this;
        }

        argument& min_count(int count)
        {
            m_min_count = count;
            return *this;
        }

        argument& count(int count)
        {
            m_count = count;
            return *this;
        }

        template <typename T>
        argument& set_default(const T& val)
        {
            using type =
                std::conditional_t<concepts::is_string_type<T>::value, std::string, T>;
            m_default_tidx = std::type_index{ typeid(decay_t<type>) };
            auto _val      = std::make_shared<type>(val);
            m_callback     = [_val](void*& obj) {
                if(!obj)
                    obj = (void*) new type{};
                (*static_cast<type*>(obj)) = type{ *_val };
            };
            m_destroy = [](void*& obj) { delete static_cast<type*>(obj); };
            return *this;
        }

        template <typename T>
        argument& set_default(T& val)
        {
            using type =
                std::conditional_t<concepts::is_string_type<T>::value, std::string, T>;
            m_default_tidx = std::type_index{ typeid(decay_t<type>) };
            m_callback     = [&](void*& obj) { obj = (void*) &val; };
            if(!m_names.empty())
            {
                auto _name = m_names.front();
                // remove leading dashes
                while(_name.find('-') == 0)
                    _name = _name.substr(1);
                if(m_actions.empty())
                    m_actions.emplace_back(
                        [_name](argument_parser& _p) { _p.get<T>(_name); });
            }
            return *this;
        }

        template <typename Tp>
        argument& choices(const std::initializer_list<Tp>& _v)
        {
            container_append(m_choices, _v);
            return *this;
        }

        argument& choice_aliases(const std::map<std::string, std::vector<std::string>>&);

        argument& choice_alias(const std::string&,
                               const std::initializer_list<std::string>&);

        template <typename Tp>
        argument& choices(const Tp& _v)
        {
            static_assert(helpers::is_initializing_container<Tp>::value,
                          "Error! Expected a container or initializer_list");
            container_append(m_choices, _v);
            return *this;
        }

        template <typename Tp>
        argument& conflicts(std::initializer_list<Tp>&& _v)
        {
            container_append(m_conflicts, std::vector<Tp>{ _v });
            return *this;
        }

        template <typename Tp>
        argument& conflicts(Tp&& _v)
        {
            static_assert(helpers::is_initializing_container<Tp>::value,
                          "Error! Expected a container or initializer_list");
            container_append(m_conflicts, std::forward<Tp>(_v));
            return *this;
        }

        template <typename ActionFuncT>
        argument& action(ActionFuncT&& _func)
        {
            m_actions.emplace_back(std::forward<ActionFuncT>(_func));
            return *this;
        }

        bool found() const { return m_found; }

        template <typename T>
        std::enable_if_t<helpers::is_container<T>::value, T> get()
        {
            if(m_values.empty() && m_default &&
               m_default_tidx == std::type_index{ typeid(T) })
                return (*static_cast<T*>(m_default));
            T                      t = T{};
            typename T::value_type vt;
            for(auto& s : m_values)
            {
                std::istringstream in(s);
                in >> vt;
                t.insert(t.end(), vt);
            }
            return t;
        }

        template <typename T>
        std::enable_if_t<
            !helpers::is_container<T>::value && !std::is_same<T, bool>::value, T>
        get()
        {
            auto inp = get<std::string>();
            if(inp.empty() && m_default && m_default_tidx == std::type_index{ typeid(T) })
                return (*static_cast<T*>(m_default));
            std::istringstream iss{ inp };
            T                  t = T{};
            iss >> t >> std::ws;
            return t;
        }

        template <typename T>
        std::enable_if_t<std::is_same<T, bool>::value, T> get()
        {
            if(m_count == 0 && m_default &&
               m_default_tidx == std::type_index{ typeid(T) })
                return (*static_cast<T*>(m_default));

            if(m_count == 0)
                return found();

            auto inp = get<std::string>();
            if(inp.empty() && m_default && m_default_tidx == std::type_index{ typeid(T) })
                return (*static_cast<T*>(m_default));
            else if(inp.empty())
                return found();

            return get_bool(inp, found());
        }

        size_t size() const { return m_values.size(); }

        std::string get_name() const
        {
            std::stringstream ss;
            for(const auto& itr : m_names)
                ss << "/" << itr;
            return ss.str().substr(1);
        }

        template <typename ArchiveT>
        void serialize(ArchiveT& ar, const unsigned) const
        {
            ar(cereal::make_nvp("count", m_count));
            ar(cereal::make_nvp("min_count", m_min_count));
            ar(cereal::make_nvp("max_count", m_max_count));
            ar(cereal::make_nvp("names", m_names));
            ar(cereal::make_nvp("description", m_desc));
            ar(cereal::make_nvp("dtype", m_dtype));
            ar(cereal::make_nvp("group", m_group));
            ar(cereal::make_nvp("required", m_required));
            ar(cereal::make_nvp("choices", m_choices));
            ar(cereal::make_nvp("choice_aliases", m_choice_aliases));
            ar(cereal::make_nvp("values", m_values));
        }

    private:
        argument(const std::string& name, std::string desc, bool req = false)
        : m_desc(std::move(desc))
        , m_required(req)
        {
            m_names.emplace_back(name);
        }

        argument() = default;

        arg_result check_choice(std::string& value) const;

        void execute_actions(argument_parser& p)
        {
            for(auto& itr : m_actions)
                itr(p);
        }

        std::string as_string() const;

        friend std::ostream& operator<<(std::ostream& os, const argument& arg)
        {
            return (os << arg.as_string());
        }

        template <typename TargetT, typename Tp>
        static void container_append(TargetT& _target, Tp&& _val)
        {
            static_assert(helpers::is_initializing_container<Tp>::value,
                          "Error! Expected a container or initializer_list");
            for(auto&& itr : std::forward<Tp>(_val))
            {
                std::stringstream ss;
                ss << itr;
                helpers::emplace(_target, ss.str());
            }
        }

        bool is_separator() const;

        friend struct argument_parser;
        int                        m_position       = Position::IgnoreArgument;
        int                        m_count          = Count::ANY;
        int                        m_min_count      = Count::ANY;
        int                        m_max_count      = Count::ANY;
        std::string                m_color          = {};
        std::string                m_desc           = {};
        std::string                m_dtype          = {};
        std::string                m_group          = {};
        std::string                m_required_info  = {};
        bool                       m_found          = false;
        bool                       m_required       = false;
        int                        m_index          = -1;
        std::type_index            m_default_tidx   = std::type_index{ typeid(void) };
        void*                      m_default        = nullptr;
        callback_t                 m_callback       = [](void*&) {};
        callback_t                 m_destroy        = [](void*&) {};
        std::set<std::string>      m_choices        = {};
        std::set<std::string>      m_conflicts      = {};
        std::set<std::string>      m_requires       = {};
        std::vector<std::string>   m_names          = {};
        std::vector<std::string>   m_values         = {};
        std::vector<action_func_t> m_actions        = {};
        choice_alias_mapping_t     m_choice_aliases = {};
    };
    //
    //----------------------------------------------------------------------------------//
    //
    argument_parser(std::string _desc, std::string _long_desc = {})
    : m_desc{ std::move(_desc) }
    , m_long_desc{ std::move(_long_desc) }
    {}
    //
    //----------------------------------------------------------------------------------//
    //
    argument& add_argument()
    {
        m_arguments.emplace_back(argument{});
        m_arguments.back().m_index = static_cast<int>(m_arguments.size()) - 1;
        if(!m_group.empty())
            m_arguments.back().group(m_group);
        return m_arguments.back();
    }
    //
    //----------------------------------------------------------------------------------//
    //
    argument& add_argument(const std::initializer_list<std::string>& _names,
                           const std::string& desc, bool req = false)
    {
        return add_argument().names(_names).description(desc).required(req);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    argument& add_argument(const std::vector<std::string>& _names,
                           const std::string& desc, bool req = false)
    {
        return add_argument().names(_names).description(desc).required(req);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    argument& add_positional_argument(const std::string& _name)
    {
        m_positional_arguments.emplace_back(argument{});
        auto& _entry = m_positional_arguments.back();
        _entry.name(_name);
        _entry.count(1);
        _entry.m_index = m_positional_arguments.size();
        return _entry;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    argument_parser& start_group(std::string _v, const std::string& _desc = {});
    //
    argument_parser& end_group();
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename Tp>
    arg_result get(size_t _idx, Tp& _value)
    {
        if(m_positional_values.find(_idx) == m_positional_values.end())
            return arg_result{ "Positional value not found at index " +
                               std::to_string(_idx) };
        if(_idx >= m_positional_arguments.size())
            return arg_result{ "No positional argument was specified for index " +
                               std::to_string(_idx) };
        _value = m_positional_arguments.at(_idx).get<Tp>();
        return arg_result{};
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename Tp>
    arg_result get(const std::string& _name, Tp& _value)
    {
        // loop over parsed positional args
        for(size_t i = 0; i < m_positional_values.size(); ++i)
        {
            if(i >= m_positional_arguments.size())
                break;

            // loop over added positional args
            auto& itr = m_positional_arguments.at(i);
            for(auto& nitr : itr.m_names)
            {
                if(nitr == _name)
                    return get(i, _value);
            }
        }

        // not found, check if required
        for(auto& itr : m_positional_arguments)
        {
            for(auto& nitr : itr.m_names)
            {
                if(nitr == _name)
                {
                    if(itr.m_default &&
                       itr.m_default_tidx == std::type_index{ typeid(decay_t<Tp>) })
                        _value = (*static_cast<Tp*>(itr.m_default));
                    else if(itr.m_required)
                        return arg_result{
                            _name + " not parsed from the command line (required)"
                        };
                    return arg_result{};
                }
            }
        }

        return arg_result{ _name + " is not a named positional argument" };
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename BoolFuncT, typename ActionFuncT>
    this_type& add_action(BoolFuncT&& _b, ActionFuncT& _act)
    {
        m_actions.emplace_back(
            { std::forward<BoolFuncT>(_b), std::forward<ActionFuncT>(_act) });
        return *this;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ActionFuncT>
    this_type& add_action(const std::string& _name, ActionFuncT& _act)
    {
        auto _b = [=](this_type& p) { return p.exists(_name); };
        m_actions.emplace_back({ _b, std::forward<ActionFuncT>(_act) });
        return *this;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void print_help(const std::string _extra = {}, const std::string _epilogue = {});
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn arg_result parse_known_args(int argc, char** argv, const std::string& delim,
    ///                                 int verb)
    /// \param[in,out] argc Number of arguments (i.e. # of command-line args)
    /// \param[in,out] argv Array of strings (i.e. command-line)
    /// \param[in] delim Delimiter which separates this argparser's opts from user's
    /// arguments
    /// \param[in] verb verbosity
    ///
    /// \brief Basic variant of \ref parse_known_args which does not replace argc/argv
    /// and does not provide an array of strings that it processed
    ///
    known_args parse_known_args(int argc, char** argv, const std::string& _delim = "--",
                                int verbose_level = 0)
    {
        strvec_t _args{};
        return parse_known_args(argc, argv, _args, _delim, verbose_level);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result parse_known_args(int* argc, char*** argv, const std::string& _delim = "--",
                                int verbose_level = 0)
    {
        strvec_t args{};
        return parse_known_args(argc, argv, args, _delim, verbose_level);
    }

    /// \fn arg_result parse_known_args(int* argc, char*** argv, const std::string& delim,
    ///                                 int verb)
    /// \param[in,out] argc Pointer to number of arguments (i.e. # of command-line args)
    /// \param[in,out] argv Pointer to array of strings (i.e. command-line)
    /// \param[in] delim Delimiter which separates this argparser's opts from user's
    /// arguments
    /// \param[in] verb verbosity
    ///
    /// \brief This variant calls \ref parse_known_args and replaces argc and argv with
    /// the argv[0] + anything after delimiter (if the delimiter is provided). If the
    /// delimiter does not exist, argc and argv are unchanged.
    ///
    arg_result parse_known_args(int* argc, char*** argv, strvec_t& _args,
                                const std::string& _delim = "--", int verbose_level = 0);

    /// \fn arg_result parse_known_args(int argc, char** argv, strvec_t& args, const
    ///                                 std::string& delim, int verb)
    /// \param[in,out] argc Number of arguments (i.e. # of command-line args)
    /// \param[in,out] argv Array of strings (i.e. command-line)
    /// \param[in,out] args Array of strings processed by this parser
    /// \param[in] delim Delimiter which separates this argparser's opts from user's
    /// arguments
    /// \param[in] verb verbosity
    ///
    /// \brief Parses all options until argv[argc-1] or delimiter is found.
    /// Returns a tuple containing an argument error object (operator bool will return
    /// true if there was an error) and the new argc and argv after the known arguments
    /// have been processed. This is slightly different from the Python
    /// argparse.ArgumentParser.parse_known_args: if the delimiter is not found, it will
    /// not remove the arguments that it recognizes.
    /// To distinguish this parsers options from user arguments, use the syntax:
    ///
    ///    ./<CMD> <PARSER_OPTIONS> -- <APP> <USER_ARGS>
    ///
    /// And std::get<1>(...) on the return value will be the new argc.
    /// and std::get<2>(...) on the return value will be the new argv.
    /// Other valid usages:
    ///
    ///    ./<CMD> --help           (will report this parser's help message)
    ///    ./<CMD> -- <APP> --help  (will report the applications help message)
    ///    ./<CMD> <USER_ARGS>
    ///    ./<CMD> <PARSER_OPTIONS>
    ///    ./<CMD> <PARSER_OPTIONS> <APP> <USER_ARGS> (intermixed)
    ///
    /// will not remove any of the known options.
    /// In other words, this will remove all arguments after <CMD> until the first "--" if
    /// reached and everything after the "--" will be placed in argv[1:]
    ///
    known_args parse_known_args(int argc, char** argv, strvec_t& _args,
                                const std::string& _delim = "--", int verbose_level = 0);

    arg_result parse(int argc, char** argv, std::string_view _delim = "--",
                     int verbose_level = 0);
    arg_result parse(int argc, char** argv, int verbose_level,
                     std::string_view _delim = "--");
    template <typename... Args>
    arg_result parse_args(Args&&... args)
    {
        return parse(std::forward<Args>(args)...);
    }

    /// \fn arg_result parse(const std::vector<std::string>& args, int verb)
    /// \param[in] args Array of strings (i.e. command-line arguments)
    /// \param[in] verb Verbosity
    ///
    /// \brief This is the primary function for parsing the command line arguments.
    /// This is where the map of the options is built and the loop over the
    /// arguments is performed.
    arg_result parse(const std::vector<std::string>& _args, int verbose_level = 0);

    /// \fn argument& enable_help()
    /// \brief Add a help command
    argument& enable_help();

    /// \fn argument& enable_help()
    /// \brief Add a help command
    argument& enable_help(const std::string _extra, const std::string _epilogue = {},
                          int _exit_code = EXIT_SUCCESS);

    /// \fn argument& enable_version(
    ///         const std::string& name,
    ///         const std::string& version,
    ///         const std::string& tag,
    ///         const std::string& revision,
    ///         const std::vector<std::pair<std::string, std::string>>& properties)
    /// \param name [in] Name of the exe/project (e.g. timemory)
    /// \param version [in] Version of the project (e.g. 1.0.0)
    /// \param tag [in] git tag (e.g. output from `git describe --tags`)
    /// \param revision [in] git revision (e.g. output from `git rev-parse HEAD`)
    /// \param properties [in] list of pairs of descriptor + value; produces
    ///     "<descriptor>: <value>" (i.e. adds colon + space) if value is not an empty
    ///     string, e.g. { { "compiler", "GCC 9.2" }, { "linker", "BFD" }, { "backends",
    ///     "" } } would produce "compiler: GCC 9.2, linker: BFD"
    ///
    /// \brief Add a command printing the version + optional properties
    argument& enable_version(
        const std::string& _name, const std::string& _version,
        const std::string& _tag = {}, const std::string& _rev = {},
        const std::vector<std::pair<std::string, std::string>>& = {});

    /// \brief Add a command printing the version (from a vector of version numbers) +
    /// optional properties
    argument& enable_version(
        const std::string& _name, std::vector<int> _versions,
        const std::string& _tag = {}, const std::string& _rev = {},
        const std::vector<std::pair<std::string, std::string>>& = {});

    /// \fn argument& enable_serialize()
    /// \brief Add a serializer-argparser command
    argument& enable_serialize();

    /// \fn bool exists(const std::string& name) const
    /// \brief Returns whether or not an option was found in the arguments. Only
    /// useful after a call to \ref parse or \ref parse_known_args.
    ///
    /// \code{.cpp}
    ///
    /// int main(int argc, char** argv)
    /// {
    ///     argument_parser p{ argv[0] };
    ///     p.add_argument()
    ///         .names({ "-h", "--help"})
    ///         .description("Help message")
    ///         .count(0);
    ///
    ///     auto ec = p.parse(argc, argv);
    ///     if(ec)
    ///     {
    ///         std::cerr << "Error: " << ec << std::endl;
    ///         exit(EXIT_FAILURE);
    ///     }
    ///
    ///     if(p.exists("help"))
    ///     {
    ///         p.print_help();
    ///         exit(EXIT_FAILURE);
    ///     }
    ///
    ///     // ...
    /// }
    /// \endcode
    bool exists(const std::string& name) const;

    template <typename T>
    T get(const std::string& name);

    static int64_t get_count(argument& a) { return a.m_values.size(); }
    int64_t        get_count(const std::string& name);
    int64_t        get_positional_count() const { return m_positional_values.size(); }
    int            get_help_width() const { return m_width; }
    int            get_description_width() const { return m_desc_width; }
    bool           get_use_color() const { return m_use_color; }
    auto           get_color() const { return m_color; }

    void          set_help_width(int _v) { m_width = _v; }
    void          set_description_width(int _v) { m_desc_width = _v; }
    void          set_use_color(bool _v) { m_use_color = _v; }
    void          set_color(const char* _v) { m_color = _v; }
    std::ostream* set_ostream(std::ostream*);

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned) const;

    template <typename ErrorFuncT>
    void on_error(ErrorFuncT&&);

private:
    template <typename... Args>
    arg_result construct_error(Args&&... args)
    {
        auto _err = arg_result{ timemory::join::join("", std::forward<Args>(args)...) };
        return (m_error_func(*this, _err), _err);
    }

    template <typename FuncT = std::function<bool(int, int)>>
    arg_result check_count(argument& a, const std::string& _do_str = "111",
                           const FuncT& _func = std::not_equal_to<int>{});

    template <typename FuncT = std::function<bool(int, int)>>
    arg_result check_count(const std::string& name, const std::string& _do = "111",
                           const FuncT& _func = std::not_equal_to<int>{});

    template <typename ErrorFuncT>
    auto on_error_sfinae(ErrorFuncT&& _func, int)
        -> decltype(_func(std::declval<this_type&>(), std::declval<result_type>()),
                    void())
    {
        m_error_func = std::forward<ErrorFuncT>(_func);
    }

    template <typename ErrorFuncT>
    auto on_error_sfinae(ErrorFuncT&& _func, long)
        -> decltype(_func(std::declval<result_type>()), void())
    {
        auto _wrap_func = [=](this_type&, result_type ret) { _func(ret); };
        m_error_func    = _wrap_func;
    }

    arg_result begin_argument(const std::string& arg, bool longarg, int position);
    arg_result add_value(std::string& value, int location);
    arg_result end_argument();

private:
    bool                       m_help_enabled      = false;
    bool                       m_use_color         = true;
    bool                       m_serialize_enabled = false;
    int                        m_current           = -1;
    int                        m_width             = 30;
    size_t                     m_desc_width        = 90;
    std::ostream*              m_clog              = &std::cout;
    std::string                m_desc              = {};
    std::string                m_long_desc         = {};
    std::string                m_bin               = {};
    std::string                m_group             = {};
    std::string                m_color             = log::color::info();
    error_func_t               m_error_func     = [](this_type&, const result_type&) {};
    std::vector<argument>      m_arguments      = {};
    std::map<int, int>         m_positional_map = {};
    std::map<std::string, int> m_name_map       = {};
    std::vector<action_pair_t> m_actions        = {};
    std::vector<argument>      m_positional_arguments = {};
    std::map<int, std::string> m_positional_values    = {};
};
//
//--------------------------------------------------------------------------------------//
//
/// \fn T get(const std::string& name)
/// \tparam T Data type to convert the argument into
/// \param[in] name An identifier of the option
///
/// \brief Get the value(s) associated with an argument. If option, it should
/// be used in conjunction with \ref exists(name). Only useful after a call to \ref
/// parse or \ref parse_known_args.
///
/// \code{.cpp}
///
/// int main(int argc, char** argv)
/// {
///     argument_parser p{ argv[0] };
///     p.add_argument()
///         .names({ "-n", "--iterations"})
///         .description("Number of iterations")
///         .count(1);
///
///     // ... etc.
///
///     auto nitr = p.get<size_t>("iteration");
///
///     // ... etc.
/// }
/// \endcode
template <typename Tp>
Tp
argument_parser::get(const std::string& name)
{
    if(name.empty())
    {
        timemory_print_demangled_backtrace<8>(std::cerr, "",
                                              "no argument name requested");
        throw std::runtime_error("argparser::get requested with no name");
    }

    auto itr = m_name_map.find(name);
    if(itr != m_name_map.end())
        return m_arguments[static_cast<size_t>(itr->second)].get<Tp>();

    construct_error("No argument option found with name: \"", name,
                    "\" [type: ", demangle<Tp>(), "] (ignoring leading dashes)");
    return Tp{};
}
//
//--------------------------------------------------------------------------------------//
//
template <>
inline std::string
argument_parser::argument::get<std::string>()
{
    using T = std::string;
    if(m_values.empty() && m_default != nullptr &&
       m_default_tidx == std::type_index{ typeid(T) })
        return (*static_cast<T*>(m_default));
    return helpers::join(m_values.begin(), m_values.end());
}
//
//--------------------------------------------------------------------------------------//
//
template <>
inline std::vector<std::string>
argument_parser::argument::get<std::vector<std::string>>()
{
    using T = std::vector<std::string>;
    if(m_values.empty() && m_default != nullptr &&
       m_default_tidx == std::type_index{ typeid(T) })
        return (*static_cast<T*>(m_default));
    return m_values;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename ArchiveT>
void
argument_parser::serialize(ArchiveT& ar, const unsigned) const
{
    ar(cereal::make_nvp("help_enabled", m_help_enabled));
    ar(cereal::make_nvp("width", m_width));
    ar(cereal::make_nvp("description_width", m_desc_width));
    ar(cereal::make_nvp("description", m_desc));
    ar(cereal::make_nvp("long_description", m_long_desc));
    ar(cereal::make_nvp("bin", m_bin));
    ar(cereal::make_nvp("positional_map", m_positional_map));
    ar(cereal::make_nvp("name_map", m_name_map));
    if constexpr(concepts::is_output_archive<ArchiveT>::value)
    {
        auto _arguments = std::vector<argument>{};
        _arguments.reserve(m_arguments.size());
        for(const auto& itr : m_arguments)
            if(!itr.is_separator())
                _arguments.emplace_back(itr);
        ar(cereal::make_nvp("arguments", _arguments));
    }
    else
    {
        ar(cereal::make_nvp("arguments", m_arguments));
    }
    ar(cereal::make_nvp("positional_arguments", m_positional_arguments));
    ar(cereal::make_nvp("positional_values", m_positional_values));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename ErrorFuncT>
void
argument_parser::on_error(ErrorFuncT&& _func)
{
    on_error_sfinae(std::forward<ErrorFuncT>(_func), 0);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename FuncT>
argument_parser::arg_result
argument_parser::check_count(argument& a, const std::string& _do_str, const FuncT& _func)
{
    int _sz  = static_cast<int>(a.m_values.size());
    int _cnt = a.m_count;
    int _max = a.m_max_count;
    int _min = a.m_min_count;

    std::bitset<3> _do{ _do_str };
    std::bitset<3> _checks;
    _checks.reset();  // set all to false
    // if <val> > ANY AND <val does not satisfies condition> -> true
    _checks.set(0, _do.test(0) && _cnt > argument::Count::ANY && _func(_sz, _cnt));
    _checks.set(1, _do.test(1) && _max > argument::Count::ANY && _sz > _max);
    _checks.set(2, _do.test(2) && _min > argument::Count::ANY && _sz < _min);
    // if no checks failed, return non-error
    if(_checks.none())
        return arg_result{};
    // otherwise, compose an error message
    std::stringstream msg;
    msg << "Argument: " << a.get_name() << " failed to satisfy its argument count "
        << "requirements. Number of arguments: " << _sz << ".";
    if(_checks.test(0))
    {
        msg << "\n[" << a.get_name() << "]> Requires exactly " << _cnt << " values.";
    }
    else
    {
        if(_checks.test(1))
        {
            msg << "\n[" << a.get_name() << "]> Requires less than " << _max + 1
                << " values.";
        }
        if(_checks.test(2))
        {
            msg << "\n[" << a.get_name() << "]> Requires more than " << _min - 1
                << " values.";
        }
    }
    return arg_result(msg.str());
}
//
//----------------------------------------------------------------------------------//
//
template <typename FuncT>
argument_parser::arg_result
argument_parser::check_count(const std::string& name, const std::string& _do,
                             const FuncT& _func)
{
    auto itr = m_name_map.find(name);
    if(itr != m_name_map.end())
        return check_count(m_arguments[static_cast<size_t>(itr->second)], _do, _func);
    return arg_result{};
}

}  // namespace argparse
}  // namespace tim

#if defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/argparse.cpp"
#endif
