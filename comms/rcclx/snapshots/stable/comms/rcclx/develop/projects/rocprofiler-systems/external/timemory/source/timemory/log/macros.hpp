//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.
//

#pragma once

#include "timemory/log/color.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/process/process.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iosfwd>
#include <string>
#include <utility>

//======================================================================================//
//
//      DEBUG
//
//======================================================================================//

// stringify some macro -- uses TIMEMORY_STRINGIZE2 which does the actual
//   "stringify-ing" after the macro has been substituted by it's result
#if !defined(TIMEMORY_STRINGIZE)
#    define TIMEMORY_STRINGIZE(X) TIMEMORY_STRINGIZE2(X)
#endif

// actual stringifying
#if !defined(TIMEMORY_STRINGIZE2)
#    define TIMEMORY_STRINGIZE2(X) #    X
#endif

#if !defined(TIMEMORY_TRUNCATED_FILE_STRING)
#    define TIMEMORY_TRUNCATED_FILE_STRING(FILE)                                         \
        []() {                                                                           \
            std::string TIMEMORY_VAR_NAME_COMBINE(_f, __LINE__){ FILE };                 \
            auto        TIMEMORY_VAR_NAME_COMBINE(_pos, __LINE__) =                      \
                TIMEMORY_VAR_NAME_COMBINE(_f, __LINE__).find("/timemory/");              \
            if(TIMEMORY_VAR_NAME_COMBINE(_pos, __LINE__) != std::string::npos)           \
            {                                                                            \
                return TIMEMORY_VAR_NAME_COMBINE(_f, __LINE__)                           \
                    .substr(TIMEMORY_VAR_NAME_COMBINE(_pos, __LINE__) + 1);              \
            }                                                                            \
            return TIMEMORY_VAR_NAME_COMBINE(_f, __LINE__);                              \
        }()
#endif

#if !defined(TIMEMORY_FILE_LINE_FUNC_STRING)
#    define TIMEMORY_FILE_LINE_FUNC_STRING                                               \
        std::string                                                                      \
        {                                                                                \
            std::string{ "[" } + TIMEMORY_TRUNCATED_FILE_STRING(__FILE__) + ":" +        \
                std::to_string(__LINE__) + "@'" + __FUNCTION__ + "']"                    \
        }
#endif

#if !defined(TIMEMORY_PID_TID_STRING)
#    define TIMEMORY_PID_TID_STRING                                                      \
        std::string                                                                      \
        {                                                                                \
            std::string{ "[pid=" } + std::to_string(::tim::process::get_id()) +          \
                std::string{ "][tid=" } + std::to_string(::tim::threading::get_id()) +   \
                "]"                                                                      \
        }
#endif

TIMEMORY_INLINE auto
timemory_proxy_value(std::string& arg, int)
{
    return arg.c_str();
}

template <typename Arg>
TIMEMORY_INLINE auto
timemory_proxy_value(Arg arg, int) -> decltype(arg.proxy_value())
{
    using type = std::decay_t<decltype(arg.proxy_value())>;
    return static_cast<type>(arg.proxy_value());
}

template <typename Arg>
TIMEMORY_INLINE auto
timemory_proxy_value(Arg arg, long)
{
    if constexpr(std::is_same<Arg, std::string_view>::value)
    {
        auto _len = arg.length();
        return (strnlen(arg.data(), _len + 1) == _len) ? arg.data()
                                                       : "unterminated-string-view";
    }
    else
    {
        return static_cast<std::decay_t<Arg>>(arg);
    }
}

template <typename... Args>
TIMEMORY_NOINLINE void
timemory_print_here(const char* _pid_tid, const char* _file, int _line, const char* _func,
                    Args... args)
{
#if defined(TIMEMORY_PROJECT_NAME)
    fprintf(stderr, "%s[%s]%s[%s:%i][%s]> ", ::tim::log::color::info(),
            TIMEMORY_PROJECT_NAME, _pid_tid, _file, _line, _func);
#else
    fprintf(stderr, "%s%s[%s:%i@'%s']> ", ::tim::log::color::info(), _pid_tid, _file,
            _line, _func);
#endif
    fprintf(stderr, timemory_proxy_value(args, 0)...);
    fprintf(stderr, "...\n%s", ::tim::log::color::end());
    fflush(stderr);
}

inline void
timemory_printf(const char* _color, FILE* _file, const char* _fmt)
{
    if(!_fmt)
        return;
    if(_file == stdout || _file == stderr)
    {
#if defined(TIMEMORY_PROJECT_NAME)
        fprintf(_file, "%s[%s][%i]", _color, TIMEMORY_PROJECT_NAME,
                ::tim::process::get_id());
        if(strnlen(_fmt, 2) > 0 && _fmt[0] != '[')
            fprintf(_file, " ");
#else
        fprintf(_file, "%s[%i]", _color, ::tim::process::get_id());
#endif
    }
    fprintf(_file, "%s", _fmt);
    if(_file == stdout || _file == stderr)
        fprintf(_file, "%s", ::tim::log::color::end());
}

template <typename Arg, typename... Args>
inline void
timemory_printf(const char* _color, FILE* _file, const char* _fmt, Arg arg, Args... args)
{
    if(!_fmt)
        return;
    if(_file == stdout || _file == stderr)
    {
#if defined(TIMEMORY_PROJECT_NAME)
        if(std::string_view{ _fmt }.find("[" TIMEMORY_PROJECT_NAME "]") != 0)
        {
            fprintf(_file, "%s[%s][%i]", _color, TIMEMORY_PROJECT_NAME,
                    ::tim::process::get_id());
            if(strnlen(_fmt, 2) > 0 && _fmt[0] != '[')
                fprintf(_file, " ");
        }
        else
            fprintf(_file, "%s", _color);
#else
        fprintf(_file, "%s[%i]", _color, ::tim::process::get_id());
#endif
    }
    fprintf(_file, _fmt, timemory_proxy_value(arg, 0), timemory_proxy_value(args, 0)...);
    if(_file == stdout || _file == stderr)
        fprintf(_file, "%s", ::tim::log::color::end());
}

template <typename... Args>
inline void
timemory_printf(FILE* _file, const char* _fmt, Args... args)
{
    return timemory_printf(::tim::log::color::info(), _file, _fmt, args...);
}

#if !defined(TIMEMORY_PRINTF)
#    define TIMEMORY_PRINTF(...) timemory_printf(::tim::log::color::info(), __VA_ARGS__)
#endif

#if !defined(TIMEMORY_PRINTF_NOCOLOR)
#    define TIMEMORY_PRINTF_NOCOLOR(...) timemory_printf("", __VA_ARGS__)
#endif

#if !defined(TIMEMORY_PRINTF_INFO)
#    define TIMEMORY_PRINTF_INFO(...)                                                    \
        timemory_printf(::tim::log::color::info(), __VA_ARGS__)
#endif

#if !defined(TIMEMORY_PRINTF_WARNING)
#    define TIMEMORY_PRINTF_WARNING(...)                                                 \
        timemory_printf(::tim::log::color::warning(), __VA_ARGS__)
#endif

#if !defined(TIMEMORY_PRINTF_FATAL)
#    define TIMEMORY_PRINTF_FATAL(...)                                                   \
        timemory_printf(::tim::log::color::fatal(), __VA_ARGS__)
#endif

#if !defined(TIMEMORY_PRINTF_SOURCE)
#    define TIMEMORY_PRINTF_SOURCE(...)                                                  \
        timemory_printf(::tim::log::color::source(), __VA_ARGS__)
#endif

#if !defined(TIMEMORY_PRINT_HERE)
#    define TIMEMORY_PRINT_HERE(...)                                                     \
        timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                             \
                            TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(), __LINE__,  \
                            __FUNCTION__, __VA_ARGS__)
#endif

#if !defined(TIMEMORY_DEBUG_PRINT_HERE)
#    if defined(DEBUG)
#        define TIMEMORY_DEBUG_PRINT_HERE(...)                                           \
            if(::tim::settings::debug())                                                 \
            {                                                                            \
                timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                     \
                                    TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),    \
                                    __LINE__, __FUNCTION__, __VA_ARGS__);                \
            }
#    else
#        define TIMEMORY_DEBUG_PRINT_HERE(...)
#    endif
#endif

#if !defined(TIMEMORY_VERBOSE_PRINT_HERE)
#    define TIMEMORY_VERBOSE_PRINT_HERE(VERBOSE_LEVEL, ...)                              \
        if(::tim::settings::verbose() >= VERBOSE_LEVEL)                                  \
        {                                                                                \
            timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                         \
                                TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),        \
                                __LINE__, __FUNCTION__, __VA_ARGS__);                    \
        }
#endif

#if !defined(TIMEMORY_CONDITIONAL_PRINT_HERE)
#    define TIMEMORY_CONDITIONAL_PRINT_HERE(CONDITION, ...)                              \
        if(CONDITION)                                                                    \
        {                                                                                \
            timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                         \
                                TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),        \
                                __LINE__, __FUNCTION__, __VA_ARGS__);                    \
        }
#endif

#if !defined(TIMEMORY_CONDITIONAL_BACKTRACE)
#    define TIMEMORY_CONDITIONAL_BACKTRACE(CONDITION, DEPTH)                             \
        if(CONDITION)                                                                    \
        {                                                                                \
            timemory_print_backtrace<DEPTH>(std::cerr, TIMEMORY_PID_TID_STRING,          \
                                            TIMEMORY_FILE_LINE_FUNC_STRING);             \
        }
#endif

#if !defined(TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE)
#    define TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(CONDITION, DEPTH)                   \
        if(CONDITION)                                                                    \
        {                                                                                \
            timemory_print_demangled_backtrace<DEPTH>(                                   \
                std::cerr, TIMEMORY_PID_TID_STRING, TIMEMORY_FILE_LINE_FUNC_STRING);     \
        }
#endif

#if !defined(TIMEMORY_PRETTY_PRINT_HERE)
#    if defined(TIMEMORY_GNU_COMPILER) || defined(TIMEMORY_CLANG_COMPILER)
#        define TIMEMORY_PRETTY_PRINT_HERE(...)                                          \
            timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                         \
                                TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),        \
                                __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)
#    else
#        define TIMEMORY_PRETTY_PRINT_HERE(...)                                          \
            timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                         \
                                TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),        \
                                __LINE__, __FUNCTION__, __VA_ARGS__)
#    endif
#endif

#if !defined(TIMEMORY_CONDITIONAL_BACKTRACE)
#    define TIMEMORY_CONDITIONAL_BACKTRACE(CONDITION, DEPTH)                             \
        if(CONDITION)                                                                    \
        {                                                                                \
            timemory_print_backtrace<DEPTH>(std::cerr, TIMEMORY_PID_TID_STRING);         \
        }
#endif

#if !defined(TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE)
#    define TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(CONDITION, DEPTH)                   \
        if(CONDITION)                                                                    \
        {                                                                                \
            timemory_print_demangled_backtrace<DEPTH>(std::cerr,                         \
                                                      TIMEMORY_PID_TID_STRING);          \
        }
#endif

#if !defined(TIMEMORY_LOG)
#    define TIMEMORY_LOG(COLOR, EXIT_CODE)                                               \
        (::tim::log::logger(EXIT_CODE)                                                   \
         << ::tim::log::color::end() << ::tim::log::color::source() << "["               \
         << TIMEMORY_PROJECT_NAME << "][" << __FILE__ << ":" << __LINE__ << "]["         \
         << getpid() << "] " << ::tim::log::color::end() << COLOR)
#endif

#if defined(NDEBUG)
#    if !defined(TIMEMORY_INFO)
#        define TIMEMORY_INFO (::tim::log::base())
#    endif
#    if !defined(TIMEMORY_ASSERT)
#        define TIMEMORY_ASSERT(COND) (::tim::log::base())
#    endif
#else
#    if !defined(TIMEMORY_INFO)
#        define TIMEMORY_INFO TIMEMORY_LOG(::tim::log::color::info(), false)
#    endif
#    if !defined(TIMEMORY_ASSERT)
#        define TIMEMORY_ASSERT(COND) (COND) ? ::tim::log::base() : TIMEMORY_FATAL
#    endif
#endif

#if !defined(TIMEMORY_WARNING)
#    define TIMEMORY_WARNING TIMEMORY_LOG(::tim::log::color::warning(), false)
#endif

#if !defined(TIMEMORY_FATAL)
#    define TIMEMORY_FATAL TIMEMORY_LOG(::tim::log::color::fatal(), true)
#endif

#if !defined(TIMEMORY_PREFER)
#    define TIMEMORY_PREFER(COND) (COND) ? ::tim::log::base() : TIMEMORY_WARNING
#endif

#if !defined(TIMEMORY_REQUIRE)
#    define TIMEMORY_REQUIRE(COND) (COND) ? ::tim::log::base() : TIMEMORY_FATAL
#endif
