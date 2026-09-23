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

#include "timemory/defines.h"
#include "timemory/log/logger.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/unwind.hpp"

#if defined(TIMEMORY_USE_LIBUNWIND)
#    include <libunwind.h>
#endif

#if defined(TIMEMORY_UNIX)
#    include <execinfo.h>
#endif

// C library
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace tim
{
inline namespace backtrace
{
//
static constexpr size_t bt_max_length = 1024;
//
#if defined(TIMEMORY_UNIX)
//
template <size_t Depth, int64_t Offset = 1>
TIMEMORY_NOINLINE inline auto
get_native_backtrace()
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    // destination
    std::array<char[bt_max_length], Depth> btrace{};
    for(auto& itr : btrace)
        memset(itr, 0, bt_max_length * sizeof(char));

    // plus one for this stack-frame
    std::array<void*, Depth + Offset> buffer{};
    buffer.fill(nullptr);

    // size of returned buffer
    auto sz = ::backtrace(buffer.data(), Depth + Offset);
    // size of relevant data
    auto n = sz - Offset;

    // skip ahead (Offset + 1) stack frames
    char** bsym = ::backtrace_symbols(buffer.data() + Offset, n);

    // report errors
    if(bsym == nullptr)
    {
        ::perror("backtrace_symbols");
    }
    else
    {
        for(decltype(n) i = 0; i < n; ++i)
            ::snprintf(btrace[i], bt_max_length, "%s", bsym[i]);
        for(auto& itr : btrace)
            if(strnlen(itr, bt_max_length + 1) >= bt_max_length + 1)
                itr[bt_max_length - 1] = '\0';
        free(bsym);
    }

    return btrace;
}
//
#    if defined(TIMEMORY_USE_LIBUNWIND)
//
template <size_t Depth, int64_t Offset = 1, bool WSignalFrame = false>
TIMEMORY_NOINLINE inline auto
get_unw_stack(unw_frame_regnum_t _reg = UNW_REG_IP)
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    // destination
    auto _stack = unwind::stack<Depth>{ _reg };

    // Initialize cursor to current frame for local unwinding.
    unw_getcontext(&_stack.context);
    if(unw_init_local(&_stack.cursor, &_stack.context) < 0)
        return _stack;

    int     unw_ret = 0;
    int64_t tot_idx = 0;
    while((unw_ret = unw_step(&_stack.cursor)) != 0)
    {
        if(unw_ret < 0)
        {
            switch(unw_ret)
            {
                // instrumentation may cause one of these so continue
                case UNW_ENOINFO:
                case UNW_EBADVERSION:
                case UNW_EINVALIDIP:
                case UNW_EBADFRAME:
                {
                    goto unwind_continue;
                    break;
                }
                // if error not specified or should stop, break from loop
                case UNW_EUNSPEC:
                case UNW_ESTOPUNWIND:
                {
                    goto unwind_break;
                    break;
                }
                // if not one of cases above, break from the loop
                default:
                {
                    goto unwind_break;
                    break;
                }
            }
        unwind_break:
        {
            break;
        }
        unwind_continue:
        {
            continue;
        }
        }

        auto _idx = tot_idx++;

        // skip all frames less than the offset
        if(_idx < Offset)
            continue;
        // skip the signal frames
        IF_CONSTEXPR(!WSignalFrame)
        {
            if(unw_is_signal_frame(&_stack.cursor) > 0)
                continue;
        }
        // break when _idx - Offset will be >= max size
        if(_idx >= static_cast<int64_t>(Depth + Offset))
            break;
        _idx -= Offset;             // index in stack
        auto _addr = unw_word_t{};  // instruction pointer
        if(unw_get_reg(&_stack.cursor, _reg, &_addr) < 0)
            --tot_idx;
        else
        {
            if(_reg == UNW_REG_IP && _addr == 0)
                break;
            _stack.at(_idx) = { _addr, &_stack.cursor };
        }
    }
    return _stack;
}
//
template <size_t Depth, int64_t Offset = 0>
TIMEMORY_NOINLINE inline auto
get_unw_signal_frame_stack(unw_frame_regnum_t _reg = UNW_REG_IP)
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    // destination
    auto _stack = unwind::stack<Depth>{ _reg };

    // Initialize cursor to current frame for local unwinding.
    unw_getcontext(&_stack.context);
    if(unw_init_local(&_stack.cursor, &_stack.context) < 0)
        return _stack;

    int     unw_ret  = 0;
    int64_t tot_idx  = 0;
    bool    found_sf = false;
    while((unw_ret = unw_step(&_stack.cursor)) != 0)
    {
        if(unw_ret < 0)
            continue;

        if(unw_is_signal_frame(&_stack.cursor) > 0)
        {
            found_sf = true;
            continue;
        }

        if(!found_sf)
            continue;

        auto _idx = tot_idx++;

        // skip all frames less than the offset
        if(_idx < Offset)
            continue;
        // break when _idx - Offset will be >= max size
        if(_idx >= static_cast<int64_t>(Depth + Offset))
            break;
        _idx -= Offset;             // index in stack
        auto _addr = unw_word_t{};  // instruction pointer
        if(unw_get_reg(&_stack.cursor, _reg, &_addr) < 0)
            --tot_idx;
        else
        {
            if(_reg == UNW_REG_IP && _addr == 0)
                break;
            _stack.at(_idx) = { _addr, &_stack.cursor };
        }
    }
    return _stack;
}
//
//
template <size_t Depth, int64_t Offset = 0>
TIMEMORY_NOINLINE inline auto
get_unw_stack_raw(unw_frame_regnum_t _reg = UNW_REG_IP)
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    // destination
    auto   _stack = std::array<uintptr_t, Depth>{};
    size_t _n     = 0;
    _stack.fill(0);

    // Initialize cursor to current frame for local unwinding.
    auto _cursor  = unw_cursor_t{};
    auto _context = unw_context_t{};

    // Initialize cursor to current frame for local unwinding.
    unw_getcontext(&_context);
    if(unw_init_local(&_cursor, &_context) < 0)
        return std::make_pair(_stack, _n);

    int     unw_ret = 0;
    int64_t tot_idx = 0;
    while((unw_ret = unw_step(&_cursor)) != 0)
    {
        if(unw_ret < 0)
            continue;

        auto _idx = tot_idx++;

        // skip all frames less than the offset
        if(_idx < Offset)
            continue;
        // break when _idx - Offset will be >= max size
        if(_idx >= static_cast<int64_t>(Depth + Offset))
            break;
        auto _addr = unw_word_t{};  // instruction pointer
        if(unw_get_reg(&_cursor, _reg, &_addr) < 0)
            --tot_idx;
        else
        {
            if(_reg == UNW_REG_IP && _addr == 0)
                break;
            _stack.at(_n++) = _addr;
        }
    }

    return std::make_pair(_stack, _n);
}
//
//
template <size_t Depth, int64_t Offset = 0>
TIMEMORY_NOINLINE inline auto
get_unw_signal_frame_stack_raw(unw_frame_regnum_t _reg = UNW_REG_IP)
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    // destination
    auto   _stack = std::array<uintptr_t, Depth>{};
    size_t _n     = 0;
    _stack.fill(0);

    // Initialize cursor to current frame for local unwinding.
    auto _cursor  = unw_cursor_t{};
    auto _context = unw_context_t{};

    // Initialize cursor to current frame for local unwinding.
    unw_getcontext(&_context);
    if(unw_init_local(&_cursor, &_context) < 0)
        return std::make_pair(_stack, _n);

    int     unw_ret  = 0;
    int64_t tot_idx  = 0;
    bool    found_sf = false;
    while((unw_ret = unw_step(&_cursor)) != 0)
    {
        if(unw_ret < 0)
            continue;

        if(unw_is_signal_frame(&_cursor) > 0)
        {
            found_sf = true;
            continue;
        }

        if(!found_sf)
            continue;

        auto _idx = tot_idx++;

        // skip all frames less than the offset
        if(_idx < Offset)
            continue;
        // break when _idx - Offset will be >= max size
        if(_idx >= static_cast<int64_t>(Depth + Offset))
            break;
        auto _addr = unw_word_t{};  // instruction pointer
        if(unw_get_reg(&_cursor, _reg, &_addr) < 0)
            --tot_idx;
        else
        {
            if(_reg == UNW_REG_IP && _addr == 0)
                break;
            _stack.at(_n++) = _addr;
        }
    }

    return std::make_pair(_stack, _n);
}
//
#    else
//
template <size_t Depth, int64_t Offset = 1, bool WSignalFrame = false, typename Tp = int>
TIMEMORY_NOINLINE inline auto get_unw_stack(Tp = {})
{
    unwind::stack<Depth> _stack = {};
    throw std::runtime_error("[timemory]> libunwind not available");
    return _stack;
}
#    endif
//
template <size_t Depth, int64_t Offset = 1, bool WFuncOffset = true,
          bool WSignalFrame = true>
TIMEMORY_NOINLINE inline auto
get_unw_backtrace()
{
#    if defined(TIMEMORY_USE_LIBUNWIND)
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    // raw backtrace
    auto _raw = get_unw_stack<Depth, Offset, WSignalFrame>(UNW_REG_IP);

    // destination
    std::array<char[bt_max_length], Depth> btrace{};
    for(auto& itr : btrace)
        memset(itr, 0, bt_max_length * sizeof(char));

    for(size_t i = 0; i < _raw.size(); ++i)
    {
        auto _context = _raw.context;
        if(!_raw.at(i))
            continue;
        unw_word_t       _off     = {};  // offset
        constexpr size_t NameSize = (WFuncOffset) ? (bt_max_length - 16) : bt_max_length;
        char             _name[NameSize];
        memset(_name, 0, NameSize * sizeof(char));
        if(_raw.at(i)->get_name(_context, _name, sizeof(_name), &_off) == 0)
        {
            IF_CONSTEXPR(WFuncOffset)
            {
                if(_off != 0)
                    snprintf(btrace[i], sizeof(btrace[i]), "%s +0x%lx", _name,
                             (long) _off);
                else
                    snprintf(btrace[i], sizeof(btrace[i]), "%s", _name);
            }
            else { snprintf(btrace[i], sizeof(btrace[i]), "%s", _name); }
        }
    }
#    else
    std::array<char[bt_max_length], Depth> btrace{};
    throw std::runtime_error("[timemory]> libunwind not available");
#    endif
    return btrace;
}
//
template <size_t Depth, int64_t Offset = 1, typename Func>
TIMEMORY_NOINLINE inline auto
get_native_backtrace(Func&& func)
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    using type = std::result_of_t<Func(const char[bt_max_length])>;
    // destination
    std::array<type, Depth> btrace{};

    auto&& _data = ::tim::get_native_backtrace<Depth, Offset + 1>();
    auto   _n    = _data.size();
    for(decltype(_n) i = 0; i < _n; ++i)
        btrace[i] = func(_data[i]);
    return btrace;
}
//
template <size_t Depth, int64_t Offset = 1, bool WFuncOffset = true, typename Func>
TIMEMORY_NOINLINE inline auto
get_unw_backtrace(Func&& func)
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    using type = std::result_of_t<Func(const char[bt_max_length])>;
    // destination
    std::array<type, Depth> btrace{};

    auto&& _data = ::tim::get_unw_backtrace<Depth, Offset + 1, WFuncOffset>();
    auto   _n    = _data.size();
    for(decltype(_n) i = 0; i < _n; ++i)
        btrace[i] = func(_data[i]);
    return btrace;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, int64_t Offset = 1>
TIMEMORY_NOINLINE inline auto
get_demangled_native_backtrace()
{
    auto demangle_bt = [](const char cstr[bt_max_length]) {
        auto _len = strnlen(cstr, bt_max_length);
        return (_len > 0 && _len < bt_max_length) ? demangle_backtrace(cstr)
                                                  : std::string{};
    };
    return get_native_backtrace<Depth, Offset + 1>(demangle_bt);
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, int64_t Offset = 1, bool WFuncOffset = true>
TIMEMORY_NOINLINE inline auto
get_demangled_unw_backtrace()
{
    auto demangle_bt = [](const char cstr[bt_max_length]) {
        auto _len = strnlen(cstr, bt_max_length);
        return (_len > 0 && _len < bt_max_length) ? demangle_backtrace(cstr)
                                                  : std::string{};
    };
    return get_unw_backtrace<Depth, Offset + 1, WFuncOffset>(demangle_bt);
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, int64_t Offset = 1>
TIMEMORY_NOINLINE inline std::ostream&
print_native_backtrace(std::ostream& os = std::cerr, std::string _prefix = {},
                       const std::string& _info = {}, const std::string& _indent = "    ",
                       bool _use_lock = true)
{
    auto_lock_t _lk{ type_mutex<std::ostream>(), std::defer_lock };
    if(_use_lock && !_lk.owns_lock())
        _lk.lock();
    os << log::warning;
    if(_indent.length() > 2)
        os << _indent.substr(0, _indent.length() / 2);
    os << "[" << TIMEMORY_PROJECT_NAME << "] Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << " [tid=" << std::this_thread::get_id() << "]:\n" << std::flush;
    auto bt = ::tim::get_native_backtrace<Depth, Offset + 1>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(strnlen(itr, bt_max_length) > 0 && strnlen(itr, bt_max_length) < bt_max_length)
            log::stream(os, log::color::source()) << _indent << _prefix << itr << "\n";
    }
    os << log::flush;
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, int64_t Offset = 1>
TIMEMORY_NOINLINE inline std::ostream&
print_demangled_native_backtrace(std::ostream& os = std::cerr, std::string _prefix = {},
                                 const std::string& _info     = {},
                                 const std::string& _indent   = "    ",
                                 bool               _use_lock = true)
{
    auto_lock_t _lk{ type_mutex<std::ostream>(), std::defer_lock };
    if(_use_lock && !_lk.owns_lock())
        _lk.lock();
    os << log::warning;
    if(_indent.length() > 2)
        os << _indent.substr(0, _indent.length() / 2);
    os << "[" << TIMEMORY_PROJECT_NAME << "] Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << " [tid=" << std::this_thread::get_id() << "]:\n" << std::flush;
    auto bt = ::tim::get_demangled_native_backtrace<Depth, Offset + 1>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(itr.length() > 0)
            log::stream(os, log::color::source()) << _indent << _prefix << itr << "\n";
    }
    os << log::flush;
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, int64_t Offset = 1, bool WFuncOffset = true>
TIMEMORY_NOINLINE inline std::ostream&
print_unw_backtrace(std::ostream& os = std::cerr, std::string _prefix = {},
                    const std::string& _info = {}, const std::string& _indent = "    ",
                    bool _use_lock = true)
{
    auto_lock_t _lk{ type_mutex<std::ostream>(), std::defer_lock };
    if(_use_lock && !_lk.owns_lock())
        _lk.lock();
    os << log::warning;
    if(_indent.length() > 2)
        os << _indent.substr(0, _indent.length() / 2);
    os << "[" << TIMEMORY_PROJECT_NAME << "] Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << " [tid=" << std::this_thread::get_id() << "]:\n" << std::flush;
    auto bt = ::tim::get_unw_backtrace<Depth, Offset + 1, WFuncOffset>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(strnlen(itr, bt_max_length) > 0 && strnlen(itr, bt_max_length) < bt_max_length)
            log::stream(os, log::color::source()) << _indent << _prefix << itr << "\n";
    }
    os << log::flush;
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, int64_t Offset = 1>
TIMEMORY_NOINLINE inline std::ostream&
print_demangled_unw_backtrace(std::ostream& os = std::cerr, std::string _prefix = {},
                              const std::string& _info   = {},
                              const std::string& _indent = "    ", bool _use_lock = true)
{
    auto_lock_t _lk{ type_mutex<std::ostream>(), std::defer_lock };
    if(_use_lock && !_lk.owns_lock())
        _lk.lock();
    os << log::warning;
    if(_indent.length() > 2)
        os << _indent.substr(0, _indent.length() / 2);
    os << "[" << TIMEMORY_PROJECT_NAME << "] Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << " [tid=" << std::this_thread::get_id() << "]:\n" << std::flush;
    auto bt = ::tim::get_demangled_unw_backtrace<Depth, Offset + 1>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(itr.length() > 0)
            log::stream(os, log::color::source()) << _indent << _prefix << itr << "\n";
    }
    os << log::flush;
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_DISABLE_BACKTRACE_MACROS)
#        if defined(TIMEMORY_SOURCE)
#            define TIMEMORY_DISABLE_BACKTRACE_MACROS 0
#        else
#            define TIMEMORY_DISABLE_BACKTRACE_MACROS 1
#        endif
#    endif
//
// using macros here is not ideal but saves another frame being created
#    if TIMEMORY_DISABLE_BACKTRACE_MACROS == 0
#        if defined(TIMEMORY_USE_LIBUNWIND)
#            if !defined(get_backtrace)
#                define get_backtrace get_unw_backtrace
#            endif
#            if !defined(get_demangled_backtrace)
#                define get_demangled_backtrace get_demangled_unw_backtrace
#            endif
#            if !defined(print_backtrace)
#                define print_backtrace print_unw_backtrace
#            endif
#            if !defined(print_demangled_backtrace)
#                define print_demangled_backtrace print_demangled_unw_backtrace
#            endif
#        else
#            if !defined(get_backtrace)
#                define get_backtrace get_native_backtrace
#            endif
#            if !defined(get_demangled_backtrace)
#                define get_demangled_backtrace get_demangled_native_backtrace
#            endif
#            if !defined(print_backtrace)
#                define print_backtrace print_native_backtrace
#            endif
#            if !defined(print_demangled_backtrace)
#                define print_demangled_backtrace print_demangled_native_backtrace
#            endif
#        endif
#    endif
//
#else
//
// define these dummy functions since they are used in operation::decode
//
template <size_t Depth, int64_t Offset = 2>
static inline std::ostream&
print_native_backtrace(std::ostream& os = std::cerr, std::string = {}, std::string = {},
                       std::string = {}, bool = false)
{
    log::stream(os, log::color::warning())
        << "[timemory]> Backtrace not supported on this platform\n";
    return os;
}
//
template <size_t Depth, int64_t Offset = 3>
static inline std::ostream&
print_demangled_native_backtrace(std::ostream& os = std::cerr, std::string = {},
                                 std::string = {}, std::string = {}, bool = false)
{
    log::stream(os, log::color::warning())
        << "[timemory]> Backtrace not supported on this platform\n";
    return os;
}
//
template <size_t Depth, int64_t Offset = 2>
static inline std::ostream&
print_unw_backtrace(std::ostream& os = std::cerr, std::string = {}, std::string = {},
                    std::string = {}, bool = false)
{
    log::stream(os, log::color::warning())
        << "[timemory]> libunwind backtrace not supported on this platform\n";
    return os;
}
//
template <size_t Depth, int64_t Offset = 3>
static inline std::ostream&
print_demangled_unw_backtrace(std::ostream& os = std::cerr, std::string = {},
                              std::string = {}, std::string = {}, bool = false)
{
    log::stream(os, log::color::warning())
        << "[timemory]> libunwind backtrace not supported on this platform\n";
    return os;
}
//
#endif
//
}  // namespace backtrace
}  // namespace tim

// using macros here is not ideal but saves another frame being created
#if defined(TIMEMORY_USE_LIBUNWIND)
#    if !defined(timemory_get_backtrace)
#        define timemory_get_backtrace ::tim::backtrace::get_unw_backtrace
#    endif
#    if !defined(timemory_get_demangled_backtrace)
#        define timemory_get_demangled_backtrace                                         \
            ::tim::backtrace::get_demangled_unw_backtrace
#    endif
#    if !defined(timemory_print_backtrace)
#        define timemory_print_backtrace ::tim::backtrace::print_unw_backtrace
#    endif
#    if !defined(timemory_print_demangled_backtrace)
#        define timemory_print_demangled_backtrace                                       \
            ::tim::backtrace::print_demangled_unw_backtrace
#    endif
#else
#    if !defined(timemory_get_backtrace)
#        define timemory_get_backtrace ::tim::backtrace::get_native_backtrace
#    endif
#    if !defined(timemory_get_demangled_backtrace)
#        define timemory_get_demangled_backtrace                                         \
            ::tim::backtrace::get_demangled_native_backtrace
#    endif
#    if !defined(timemory_print_backtrace)
#        define timemory_print_backtrace ::tim::backtrace::print_native_backtrace
#    endif
#    if !defined(timemory_print_demangled_backtrace)
#        define timemory_print_demangled_backtrace                                       \
            ::tim::backtrace::print_demangled_native_backtrace
#    endif
#endif
