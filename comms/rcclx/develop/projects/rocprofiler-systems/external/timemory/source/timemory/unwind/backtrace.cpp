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

#if !defined(TIMEMORY_UNWIND_BACKTRACE_HPP_)
#    include "timemory/unwind/backtrace.hpp"
#endif

#include "timemory/defines.h"
#include "timemory/log/logger.hpp"
#include "timemory/log/macros.hpp"
#include "timemory/process/process.hpp"
#include "timemory/process/threading.hpp"
#include "timemory/unwind/bfd.hpp"
#include "timemory/unwind/macros.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/procfs/maps.hpp"
#include "timemory/variadic/macros.hpp"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <ostream>

namespace tim
{
namespace unwind
{
TIMEMORY_UNWIND_INLINE
file_map_t&
library_file_maps()
{
    static file_map_t _v = {};
    return _v;
}

TIMEMORY_UNWIND_INLINE
void
update_file_maps()
{
#if defined(TIMEMORY_USE_BFD)
    static auto _mutex = std::mutex{};
    auto        _lk    = std::scoped_lock<std::mutex>{ _mutex };

    auto& _maps = procfs::get_maps(process::get_id(), true);
    for(const auto& itr : _maps)
    {
        auto& _files = library_file_maps();
        auto  _loc   = itr.pathname;
        if(!_loc.empty() && filepath::exists(_loc))
        {
            auto _add_file = [&_files](const auto& _val) {
                if(_files.find(_val) == _files.end())
                {
                    std::stringstream _msg{};
                    _msg << "Reading '" << _val << "'...";
                    unwind::bfd_message(2, _msg.str());
                    _files.emplace(_val, std::make_shared<unwind::bfd_file>(_val));
                    auto _val_real = filepath::realpath(_val, nullptr, false);
                    if(_val != _val_real)
                        _files.emplace(_val_real, _files.at(_val));
                }
            };

            _add_file(_loc);
        }
    }
#endif
}

template <size_t OffsetV, size_t DepthV>
TIMEMORY_UNWIND_INLINE void
detailed_backtrace(std::ostream& os, detailed_backtrace_config cfg)
{
    static_assert(OffsetV >= 0 && OffsetV < 4,
                  "Error! detailed_backtrace only supports offset >= 0 and < 4");

    constexpr size_t buffer_size = bt_max_length;

    const auto* _src_color =
        (&os == &std::cerr || cfg.force_color) ? log::color::source() : "";
    const auto* _fatal_color =
        (&os == &std::cerr || cfg.force_color) ? log::color::fatal() : "";
    auto message = log::stream(os, _fatal_color);

    (void) _src_color;
    (void) _fatal_color;

    char prefix[buffer_size];
    memset(prefix, '\0', buffer_size * sizeof(char));
    snprintf(prefix, buffer_size - 1, "[PID=%i][TID=%i]", (int) process::get_id(),
             (int) threading::get_id());

    auto _replace = [](std::string _v, const std::string& _old, const std::string& _new) {
        // start at 1 to avoid replacing when it starts with string, e.g. do not replace:
        //      std::__cxx11::basic_string<...>::~basic_string
        auto _pos = size_t{ 1 };
        while((_pos = _v.find(_old, _pos)) != std::string::npos)
            _v = _v.replace(_pos, _old.length(), _new);
        return _v;
    };

    bool _do_patch =
        std::getenv(TIMEMORY_SETTINGS_PREFIX "BACKTRACE_DISABLE_PATCH") == nullptr;
    auto _patch_demangled = [_replace, _do_patch](std::string _v) {
        _v =
            (_do_patch)
                ? _replace(
                      _replace(_replace(_replace(std::move(_v), demangle<std::string>(),
                                                 "std::string"),
                                        demangle<std::string_view>(), "std::string_view"),
                               " > >", ">>"),
                      "> >", ">>")
                : _v;
        return _v;
    };

    if(cfg.native_mangled)
    {
        message << "\nBacktrace:\n";
        message.flush();

        size_t ntot = 0;
        auto   bt   = timemory_get_backtrace<DepthV, OffsetV + 1>();
        for(const auto& itr : bt)
        {
            auto _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            ++ntot;
        }

        for(size_t i = 0; i < bt.size(); ++i)
        {
            auto* itr  = bt.at(i);
            auto  _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            message << prefix << "[" << i << '/' << ntot << "] " << itr << "\n";
            os << std::flush;
        }
    }

    if(cfg.native_demangled)
    {
        message << "\nBacktrace (demangled):\n";
        message.flush();

        size_t ntot = 0;
        auto   bt   = get_native_backtrace<DepthV, OffsetV + 1>();
        for(const auto& itr : bt)
        {
            auto _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            ++ntot;
        }

        for(size_t i = 0; i < bt.size(); ++i)
        {
            auto* itr  = bt.at(i);
            auto  _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            message << prefix << "[" << i << '/' << ntot << "] "
                    << _patch_demangled(demangle_native_backtrace(itr)) << "\n";
            os << std::flush;
        }
    }

    if(cfg.proc_pid_maps)
    {
        auto _maps_file = TIMEMORY_JOIN("/", "/proc", process::get_id(), "maps");
        auto _ifs       = std::ifstream{ _maps_file };
        if(_ifs)
        {
            message << "\n" << _maps_file << ":\n";
            while(_ifs)
            {
                std::string _line{};
                getline(_ifs, _line);
                if(!_line.empty())
                    message << "    " << _line << "\n";
            }
        }
    }

#if defined(TIMEMORY_USE_LIBUNWIND)
    if(cfg.unwind_demangled)
    {
        message << "\nBacktrace (demangled):\n";
        message.flush();

        size_t ntot = 0;
        auto   bt   = get_unw_backtrace<DepthV, OffsetV + 1>();
        for(const auto& itr : bt)
        {
            auto _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            ++ntot;
        }

        for(size_t i = 0; i < bt.size(); ++i)
        {
            auto* itr  = bt.at(i);
            auto  _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            auto _v = std::string{};
            _v.resize(_len);
            for(size_t j = 0; j < _len; ++j)
                _v.at(j) = itr[j];
            message << prefix << "[" << i << '/' << ntot << "] "
                    << _patch_demangled(demangle_backtrace(_v)) << "\n";
            os << std::flush;
        }
    }
#    if defined(TIMEMORY_USE_BFD)
    if(cfg.unwind_lineinfo)
    {
        message << "\nBacktrace (lineinfo):\n";
        message.flush();

        unwind::set_bfd_verbose(8);
        struct bt_line_info
        {
            bool        first    = false;
            int64_t     index    = 0;
            int64_t     line     = 0;
            std::string name     = {};
            std::string location = {};
        };

        size_t ntot   = 0;
        auto   _bt    = get_unw_stack<DepthV, OffsetV>();
        auto   _lines = std::vector<bt_line_info>{};
        auto&  _files = library_file_maps();
        for(const auto& itr : _bt)
        {
            if(!itr)
                continue;
            unwind::processed_entry _entry{};
            _entry.address = itr->address();
            _entry.name = itr->template get_name<1024, false>(_bt.context, &_entry.offset,
                                                              &_entry.error);

            unwind::processed_entry::construct(_entry, &_files, true);
            int64_t _idx = ntot++;
            if(_entry.lineinfo.found && !_entry.lineinfo.lines.empty())
            {
                bool _first = true;
                // make sure the function at the top is the one that
                // is shown in the other backtraces
                auto _line_info = _entry.lineinfo.lines;
                std::reverse(_line_info.begin(), _line_info.end());
                for(const auto& litr : _line_info)
                {
                    _lines.emplace_back(bt_line_info{
                        _first, _idx, int64_t{ litr.line },
                        _patch_demangled(demangle(litr.name)), demangle(litr.location) });
                    _first = false;
                }
            }
            else
            {
                _lines.emplace_back(bt_line_info{ true, _idx, int64_t{ 0 },
                                                  _patch_demangled(demangle(_entry.name)),
                                                  demangle(_entry.location) });
            }
        }

        for(const auto& itr : _lines)
        {
            auto _get_loc = [&]() -> std::string {
                auto&& _loc = (itr.location.empty()) ? std::string{ "??" } : itr.location;
                return (itr.line == 0) ? TIMEMORY_JOIN(":", _loc, "?")
                                       : TIMEMORY_JOIN(":", _loc, itr.line);
            };

            if(itr.first)
                message << prefix << "[" << itr.index << '/' << ntot << "]\n";

            message << "    " << _src_color << "[" << _get_loc() << "]" << _fatal_color
                    << " " << itr.name << "\n";
        }
    }
#    endif
#endif
}

template <size_t OffsetV, size_t DepthV>
void
detailed_backtrace(std::ostream& os, bool&& force_color, detailed_backtrace_config cfg)
{
    cfg.force_color = force_color;
    detailed_backtrace<OffsetV, DepthV>(os, cfg);
}

#if defined(TIMEMORY_UNWIND_SOURCE) && TIMEMORY_UNWIND_SOURCE > 0
#    define TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE_IMPL(OFFSET, DEPTH)           \
        template void detailed_backtrace<OFFSET, DEPTH>(std::ostream&,                   \
                                                        detailed_backtrace_config);      \
        template void detailed_backtrace<OFFSET, DEPTH>(std::ostream&, bool&&,           \
                                                        detailed_backtrace_config);

#    define TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE(DEPTH)                        \
        TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE_IMPL(0, DEPTH)                    \
        TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE_IMPL(1, DEPTH)                    \
        TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE_IMPL(2, DEPTH)                    \
        TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE_IMPL(3, DEPTH)

TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE(8)
TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE(16)
TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE(32)
TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE(64)

#    undef TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE_IMPL
#    undef TIMEMORY_UNWIND_DETAILED_BACKTRACE_INSTANTIATE
#endif

}  // namespace unwind
}  // namespace tim
