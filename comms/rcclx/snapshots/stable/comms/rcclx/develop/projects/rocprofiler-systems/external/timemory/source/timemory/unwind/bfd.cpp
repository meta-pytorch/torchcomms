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

#include "timemory/unwind/bfd.hpp"

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/log/macros.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/settings/settings.hpp"

#include <cstdarg>
#include <fcntl.h>
#include <mutex>

#if defined(TIMEMORY_USE_BFD)
#    if !defined(PACKAGE)
#        define PACKAGE TIMEMORY_PROJECT_NAME
#    endif
#    include <bfd.h>
#    include <elf-bfd.h>
#endif

#ifndef TIMEMORY_BFD_ERROR_MESSAGE_MAX
#    define TIMEMORY_BFD_ERROR_MESSAGE_MAX 4096
#endif

namespace tim
{
namespace unwind
{
namespace
{
auto&
get_bfd_verbose()
{
    static int _v = 1;
    return _v;
}
}  // namespace

#if defined(TIMEMORY_USE_BFD)

namespace
{
void
timemory_bfd_error_handler(const char* _format, va_list _arglist)
{
    const auto& _instance = settings::shared_instance();
    if(_instance && _instance->get_verbose() >= get_bfd_verbose())
    {
        char _buffer[TIMEMORY_BFD_ERROR_MESSAGE_MAX];
        vsnprintf(_buffer, TIMEMORY_BFD_ERROR_MESSAGE_MAX, _format, _arglist);
        TIMEMORY_PRINTF_WARNING(stderr, "[%li] BFD error: %s\n", threading::get_id(),
                                _buffer);
    }
}

void
initialize_bfd()
{
    static std::once_flag _once{};
    std::call_once(_once, []() {
        bfd_init();
        bfd_set_error_handler(timemory_bfd_error_handler);
    });
}
}  // namespace

int
bfd_error(const char* string)
{
    const auto& _instance = settings::shared_instance();
    auto        _verbose  = get_bfd_verbose();
    auto        _err      = bfd_get_error();
    if(_err == bfd_error_file_not_recognized && _verbose == 1)
        _verbose = 2;
    if(_err > 0 && _instance && _instance->get_verbose() >= _verbose)
    {
        const char* errmsg = bfd_errmsg(_err);
        if(string)
            TIMEMORY_PRINTF_WARNING(stderr, "[%li] BFD error %i: %s: %s\n",
                                    threading::get_id(), (int) _err, string, errmsg);
        else
            TIMEMORY_PRINTF_WARNING(stderr, "[%li] BFD error %i: %s\n",
                                    threading::get_id(), (int) _err, errmsg);
    }
    return -1;
}

int
bfd_message(int _lvl, std::string_view _msg)
{
    if(!_msg.empty())
    {
        const auto& _instance = settings::shared_instance();
        if(_instance && _instance->get_verbose() >= get_bfd_verbose() &&
           _instance->get_verbose() >= _lvl)
        {
            TIMEMORY_PRINTF_INFO(stderr, "[%i][%li] BFD info: %s\n", process::get_id(),
                                 threading::get_id(), std::string{ _msg }.c_str());
        }
    }
    return 0;
}

bfd_file::bfd_file(std::string _inp)
: name{ std::move(_inp) }
, data{ open(name, &fd) }
{
    if(data != nullptr && fd < 0)
        throw std::runtime_error("fd not set");
    read_symtab();
}

bfd_file::~bfd_file()
{
    delete[] reinterpret_cast<asymbol**>(syms);
    if(data)
        bfd_close(static_cast<bfd*>(data));
    if(fd > 0)
        ::close(fd);
}

void*
bfd_file::open(const std::string& _v, int* _fd)
{
    initialize_bfd();

    bfd* _data = nullptr;
    if(_fd)
    {
        auto _fd_v = ::open(_v.c_str(), O_RDONLY);
        if(_fd_v < 0)
        {
            auto _err = errno;
            errno     = 0;
            if(get_bfd_verbose() >= 1)
            {
                TIMEMORY_PRINTF_INFO(
                    stderr, "[%i][%li][bfd_file] Error opening '%s': %s\n",
                    process::get_id(), threading::get_id(), _v.c_str(), strerror(_err));
            }
            return nullptr;
        }
        else
        {
            *_fd  = _fd_v;
            _data = bfd_fdopenr(_v.c_str(), nullptr, _fd_v);
        }
    }
    else
    {
        _data = bfd_openr(_v.c_str(), nullptr);
    }

    if(_data)
    {
        _data->flags |= BFD_DECOMPRESS;
        if(!bfd_check_format(_data, bfd_object) && !bfd_check_format(_data, bfd_archive))
        {
            auto _err = bfd_get_error();
            bfd_message(3, bfd_errmsg(_err));
            return nullptr;
        }
    }
    else
    {
        auto _err = bfd_get_error();
        bfd_message(0, bfd_errmsg(_err));
    }

    return _data;
}

int
bfd_file::read_symtab()
{
    if(!data)
        return bfd_error(name.c_str());

    bfd* _data = static_cast<bfd*>(data);
    if((bfd_get_file_flags(_data) & HAS_SYMS) == 0)
        return bfd_error(bfd_get_filename(_data));

    auto _nbytes_sym = bfd_get_symtab_upper_bound(_data);
    auto _nbytes_dyn = bfd_get_dynamic_symtab_upper_bound(_data);
    if(_nbytes_sym < 0L && _nbytes_dyn <= 0)
        return bfd_error(bfd_get_filename(_data));

    using nbytes_type = decltype(_nbytes_sym);
    auto _nbytes =
        std::max<nbytes_type>(_nbytes_sym, 0) + std::max<nbytes_type>(_nbytes_dyn, 0);

    // if(nbytes <= 0)
    //    return bfd_error(bfd_get_filename(_data));

    auto*   _syms     = new asymbol*[_nbytes];
    int64_t _num_syms = 0;
    int64_t _num_dyns = 0;

    if(_nbytes_sym > 0)
        _num_syms = bfd_canonicalize_symtab(_data, _syms);
    if(_nbytes_dyn > 0)
        _num_dyns = bfd_canonicalize_dynamic_symtab(_data, _syms + _num_syms);
    nsyms = _num_syms + _num_dyns;

    if(nsyms < 0)
    {
        delete[] _syms;
        return bfd_error(bfd_get_filename(_data));
    }

    syms = reinterpret_cast<void**>(_syms);

    return 0;
}

std::vector<bfd_file::symbol>
bfd_file::get_symbols(bool _include_undefined) const
{
    auto _syms   = std::vector<bfd_file::symbol>{};
    auto _flavor = bfd_get_flavour(static_cast<bfd*>(data));
    _syms.reserve(nsyms);
    for(int64_t i = 0; i < nsyms; ++i)
    {
        auto* _sym = reinterpret_cast<asymbol**>(syms)[i];
        if(_sym)
        {
            auto        _binding    = bfd_binding::Unknown;
            auto        _visibility = bfd_visibility::Unknown;
            const auto* _name       = bfd_asymbol_name(_sym);
            auto*       _section    = bfd_asymbol_section(_sym);
            auto        _vmaddr     = bfd_asymbol_value(_sym);
            bfd_vma     _sym_sz     = 0;
            if(_vmaddr <= 0 && !_include_undefined)
                continue;
            if((_sym->flags & (BSF_SECTION_SYM | BSF_SYNTHETIC)) == 0 &&
               _flavor == bfd_target_elf_flavour)
            {
                auto* _elf_sym = reinterpret_cast<elf_symbol_type*>(_sym);
                _sym_sz        = _elf_sym->internal_elf_sym.st_size;
                _binding       = static_cast<bfd_binding>(
                    ELF64_ST_BIND(_elf_sym->internal_elf_sym.st_info));
                _visibility = static_cast<bfd_visibility>(
                    ELF64_ST_VISIBILITY(_elf_sym->internal_elf_sym.st_other));
            }
            else if((_sym->flags & (BSF_SECTION_SYM | BSF_SYNTHETIC)) == 0 &&
                    bfd_is_com_section(_section))
            {
                _sym_sz = _sym->value;
            }

            _syms.emplace_back(symbol{ _binding, _visibility, _vmaddr, _sym_sz,
                                       static_cast<void*>(_section),
                                       std::string_view{ _name } });
        }
    }
    std::sort(_syms.begin(), _syms.end(), [](auto _lhs, auto _rhs) {
        return std::tie(_lhs.address, _lhs.symsize) <
               std::tie(_rhs.address, _rhs.symsize);
    });
    _syms.shrink_to_fit();
    return _syms;
}

#else

int
bfd_error(const char*)
{
    return -1;
}

#endif

void
set_bfd_verbose(int _v)
{
    get_bfd_verbose() = _v;
}
}  // namespace unwind
}  // namespace tim
