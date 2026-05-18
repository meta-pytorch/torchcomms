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
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/tpls/cereal/cereal/cereal.hpp"
#include "timemory/unwind/types.hpp"
#include "timemory/utility/macros.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace tim
{
namespace unwind
{
struct entry
{
    using error_handler_func_t = std::function<void(int, std::string&)>;

    TIMEMORY_DEFAULT_OBJECT(entry)

    entry(unw_word_t _addr, unw_cursor_t*);
    explicit entry(unw_word_t _addr);

#if TIMEMORY_LIBUNWIND_HAS_PROC_NAME_BY_IP == 1
    unw_word_t register_addr = {};  // instruction/stack pointer
#else
    unw_word_t   register_addr = {};  // instruction/stack pointer
    unw_cursor_t cursor        = {};  // copy of cursor is required
#endif

    unw_word_t address() const { return register_addr; }

    int get_name(unw_context_t _context, char* _buffer, size_t _size,
                 unw_word_t* _off) const;

    template <size_t BufferSize = 1024>
    std::string& get_name(unw_context_t _context, std::string&,
                          unw_word_t* _off = nullptr, int* _err = nullptr) const;

    template <size_t BufferSize = 1024, bool Shrink = true>
    std::string get_name(unw_context_t _context, unw_word_t* _off = nullptr,
                         int* _err = nullptr) const;

    template <typename FuncT>
    static auto set_error_handler(FuncT&& _func)
    {
        auto _old       = error_handler();
        error_handler() = std::forward<FuncT>(_func);
        return _old;
    }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned)
    {
        ar(cereal::make_nvp("address", register_addr));
    }

    bool operator==(entry _rhs) const { return (register_addr == _rhs.register_addr); }
    bool operator<(entry _rhs) const { return (register_addr < _rhs.register_addr); }
    bool operator>(entry _rhs) const { return (register_addr > _rhs.register_addr); }
    bool operator!=(entry _rhs) const { return !(*this == _rhs); }
    bool operator<=(entry _rhs) const { return (*this < _rhs) || (*this == _rhs); }
    bool operator>=(entry _rhs) const { return (*this > _rhs) || (*this == _rhs); }

private:
    static std::function<void(int, std::string&)>& error_handler()
    {
        static error_handler_func_t _v = [](int _err, std::string& _dest) {
            // provide the error info if failed but if the procedure name is too long to
            // fit in the buffer provided and a truncated version of the name has been
            // returned, keep the truncated version of the name
            if(_err != UNW_ENOMEM)
                _dest.assign(unw_strerror(_err));
        };
        return _v;
    };
};

inline entry::entry(unw_word_t _addr)
: register_addr{ _addr }
{}

#if TIMEMORY_LIBUNWIND_HAS_PROC_NAME_BY_IP == 1
inline entry::entry(unw_word_t _addr, unw_cursor_t*)
: register_addr{ _addr }
{}
#else
inline entry::entry(unw_word_t _addr, unw_cursor_t* _cursor)
: register_addr{ _addr }
, cursor{ *_cursor }
{}
#endif

inline int
entry::get_name(unw_context_t _context, char* _buffer, size_t _size,
                unw_word_t* _off) const
{
#if TIMEMORY_LIBUNWIND_HAS_PROC_NAME_BY_IP == 1
    return unw_get_proc_name_by_ip(unw_local_addr_space, register_addr, _buffer, _size,
                                   _off, &_context);
#else
    (void) _context;
    auto* _cursor = &const_cast<unw_cursor_t&>(cursor);
    return unw_get_proc_name(_cursor, _buffer, _size, _off);
#endif
}

template <size_t DefaultBufferSize>
inline std::string&
entry::get_name(unw_context_t _context, std::string& _name, unw_word_t* _off,
                int* _err_ret) const
{
    // use this as a reference capacity for a default string since std::string
    // in most (if not all) C++ STL libraries have a default capacity > 0
    // (small string optimization)
    const auto _default_string_capacity = std::string{}.capacity();

    unw_word_t _off_v = {};  // offset

    // make sure there is enough space in the string
    if(_name.capacity() <= _default_string_capacity)
        _name.reserve(DefaultBufferSize);

    // unw_get_proc_name will fill until it hits \0 so move it to the end of
    // the strings capacity. Since we only resize to the capacity, there
    // is no reallocation
    if(_name.length() < _name.capacity())
        _name.resize(_name.capacity(), ' ');

    // get the procedure name and offset
    auto _err = get_name(_context, _name.data(), _name.capacity(), &_off_v);

    // provide the error
    if(_err_ret)
        *_err_ret = _err;

    // call the error handler
    if(_err != 0)
        error_handler()(_err, _name);

    // provide the offset if requested
    if(_off)
        *_off = _off_v;

    // there are going to be multiple \0 in the string because of the resize
    // and unw_get_proc_name. this should just update the string metadata
    // and not reallocate
    auto _pos = _name.find('\0');
    if(_pos < _name.length())
        _name = _name.assign(_name.data(), _name.data() + _pos);

    return _name;
}

template <size_t BufferSize, bool Shrink>
inline std::string
entry::get_name(unw_context_t _context, unw_word_t* _off, int* _err_ptr) const
{
    if(_err_ptr)
    {
        std::string _name = {};
        _name.reserve(BufferSize);
        get_name<BufferSize>(_context, _name, _off, _err_ptr);
        if(Shrink)
            _name.shrink_to_fit();
        return _name;
    }
    else
    {
        size_t      _buffer_size = BufferSize;
        int         _err         = 0;
        std::string _name        = {};
        do
        {
            _name.reserve(_buffer_size);
            get_name<BufferSize>(_context, _name, _off, &_err);
            if(_err == ENOMEM)
            {
                _buffer_size *= 2;
            }
        } while(_err == ENOMEM);
        if(Shrink)
            _name.shrink_to_fit();
        return _name;
    }
}
}  // namespace unwind
}  // namespace tim
