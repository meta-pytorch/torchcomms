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

#ifndef TIMEMORY_DATA_RING_BUFFER_CPP_
#define TIMEMORY_DATA_RING_BUFFER_CPP_

#include "timemory/data/macros.hpp"

#if !defined(TIMEMORY_DATA_RING_BUFFER_HPP_)
#    include "timemory/data/ring_buffer.hpp"
#endif

#include "timemory/log/macros.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/units.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(TIMEMORY_LINUX)
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

namespace tim
{
namespace base
{
TIMEMORY_DATA_INLINE
ring_buffer::ring_buffer(size_t _size, bool _use_mmap)
{
    set_use_mmap(_use_mmap);
    init(_size);
}

TIMEMORY_DATA_INLINE
ring_buffer::~ring_buffer() { destroy(); }

TIMEMORY_DATA_INLINE
ring_buffer::ring_buffer(const ring_buffer& rhs)
: m_use_mmap{ rhs.m_use_mmap }
, m_use_mmap_explicit{ rhs.m_use_mmap_explicit }
{
    init(rhs.m_size);
}

TIMEMORY_DATA_INLINE
ring_buffer::ring_buffer(ring_buffer&& rhs) noexcept
: m_init{ rhs.m_init }
, m_use_mmap{ rhs.m_use_mmap }
, m_use_mmap_explicit{ rhs.m_use_mmap_explicit }
, m_ptr{ rhs.m_ptr }
, m_size{ rhs.m_size }
, m_read_count{ rhs.m_read_count }
, m_write_count{ rhs.m_write_count }
{
    rhs.reset();
}

TIMEMORY_DATA_INLINE
ring_buffer&
ring_buffer::operator=(const ring_buffer& rhs)
{
    if(this == &rhs)
        return *this;
    destroy();
    m_use_mmap          = rhs.m_use_mmap;
    m_use_mmap_explicit = rhs.m_use_mmap_explicit;
    init(rhs.m_size);
    return *this;
}

TIMEMORY_DATA_INLINE
ring_buffer&
ring_buffer::operator=(ring_buffer&& rhs) noexcept
{
    if(this == &rhs)
        return *this;
    destroy();
    m_init              = rhs.m_init;
    m_use_mmap          = rhs.m_use_mmap;
    m_use_mmap_explicit = rhs.m_use_mmap_explicit;
    m_ptr               = rhs.m_ptr;
    m_size              = rhs.m_size;
    m_read_count        = rhs.m_read_count;
    m_write_count       = rhs.m_write_count;
    rhs.reset();
    return *this;
}

TIMEMORY_DATA_INLINE
void
ring_buffer::init(size_t _size)
{
    if(m_init)
        throw std::runtime_error(
            "tim::base::ring_buffer::init(size_t) :: already initialized");

    m_init = true;

    // Round up to multiple of page size.
    _size += units::get_page_size() - ((_size % units::get_page_size() > 0)
                                           ? (_size % units::get_page_size())
                                           : units::get_page_size());

    if((_size % units::get_page_size()) > 0)
    {
        std::ostringstream _oss{};
        _oss << "Error! size is not a multiple of page size: " << _size << " % "
             << units::get_page_size() << " = " << (_size % units::get_page_size());
        throw std::runtime_error(_oss.str());
    }

    m_size        = _size;
    m_read_count  = 0;
    m_write_count = 0;

    if(!m_use_mmap_explicit)
        m_use_mmap = get_env(TIMEMORY_SETTINGS_KEY("USE_MMAP"), m_use_mmap);

#if defined(TIMEMORY_LINUX)
    if(!m_use_mmap)
    {
        m_ptr = malloc(m_size * sizeof(char));
        return;
    }

    // Map twice the buffer size.
    if((m_ptr = mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE,
                     -1, 0)) == MAP_FAILED)
    {
        destroy();
        auto _err = errno;
        TIMEMORY_PRINTF_FATAL(stderr, "Error using mmap: %s\n", strerror(_err));
        throw std::runtime_error(strerror(_err));
    }
#else
    m_use_mmap = false;
    m_ptr      = malloc(m_size * sizeof(char));
#endif
}

TIMEMORY_DATA_INLINE
void
ring_buffer::destroy()
{
    if(m_ptr && m_init)
    {
#if defined(TIMEMORY_LINUX)
        if(!m_use_mmap)
        {
            ::free(m_ptr);
        }
        else
        {
            // Unmap the mapped virtual memmory.
            auto ret = munmap(m_ptr, m_size);
            if(ret != 0)
                perror("munmap");
        }
#else
        ::free(m_ptr);
#endif
    }
    m_init        = false;
    m_size        = 0;
    m_read_count  = 0;
    m_write_count = 0;
    m_ptr         = nullptr;
}

TIMEMORY_DATA_INLINE
void
ring_buffer::set_use_mmap(bool _v)
{
    if(!m_init)
    {
        m_use_mmap          = _v;
        m_use_mmap_explicit = true;
    }
    else
    {
        throw std::runtime_error("tim::base::ring_buffer::set_use_mmap(bool) cannot be "
                                 "called after initialization");
    }
}

TIMEMORY_DATA_INLINE
std::string
ring_buffer::as_string() const
{
    std::ostringstream ss{};
    ss << std::boolalpha << "is_initialized: " << is_initialized()
       << ", capacity: " << capacity() << ", count: " << count() << ", free: " << free()
       << ", is_empty: " << is_empty() << ", is_full: " << is_full()
       << ", pointer: " << m_ptr << ", read count: " << m_read_count
       << ", write count: " << m_write_count;
    return ss.str();
}
//
TIMEMORY_DATA_INLINE
void*
ring_buffer::request(size_t _length)
{
    if(m_ptr == nullptr)
        return nullptr;

    // Make sure we don't put in more than there's room for, by writing no
    // more than there is free.
    if(_length > free())
        throw std::runtime_error("heap-buffer-overflow :: ring buffer is full. read data "
                                 "to avoid data corruption");

    // if write count is at the tail of buffer, bump to the end of buffer
    auto _modulo = m_size - (m_write_count % m_size);
    if(_modulo < _length)
        m_write_count += _modulo;

    // pointer in buffer
    void* _out = write_ptr();

    // Update write count
    m_write_count += _length;

    return _out;
}
//
TIMEMORY_DATA_INLINE
void*
ring_buffer::retrieve(size_t _length)
{
    if(m_ptr == nullptr)
        return nullptr;

    // Make sure we don't put in more than there's room for, by writing no
    // more than there is free.
    if(_length > count())
        throw std::runtime_error("ring buffer is empty");

    // if read count is at the tail of buffer, bump to the end of buffer
    auto _modulo = m_size - (m_read_count % m_size);
    if(_modulo < _length)
        m_read_count += _modulo;

    // pointer in buffer
    void* _out = read_ptr();

    // Update write count
    m_read_count += _length;

    return _out;
}
//
TIMEMORY_DATA_INLINE
size_t
ring_buffer::rewind(size_t n) const
{
    if(n > m_read_count)
        n = m_read_count;
    m_read_count -= n;
    return n;
}
//
TIMEMORY_DATA_INLINE
void
ring_buffer::reset()
{
    m_init        = false;
    m_ptr         = nullptr;
    m_size        = 0;
    m_read_count  = 0;
    m_write_count = 0;
}
//
TIMEMORY_DATA_INLINE
void
ring_buffer::save(std::fstream& _fs)
{
    _fs.write(reinterpret_cast<char*>(&m_use_mmap), sizeof(m_use_mmap));
    _fs.write(reinterpret_cast<char*>(&m_use_mmap_explicit), sizeof(m_use_mmap_explicit));
    _fs.write(reinterpret_cast<char*>(&m_size), sizeof(m_size));
    _fs.write(reinterpret_cast<char*>(&m_read_count), sizeof(m_read_count));
    _fs.write(reinterpret_cast<char*>(&m_write_count), sizeof(m_write_count));
    _fs.write(reinterpret_cast<char*>(m_ptr), m_size * sizeof(char));
}
//
TIMEMORY_DATA_INLINE
void
ring_buffer::load(std::fstream& _fs)
{
    destroy();

    _fs.read(reinterpret_cast<char*>(&m_use_mmap), sizeof(m_use_mmap));
    _fs.read(reinterpret_cast<char*>(&m_use_mmap_explicit), sizeof(m_use_mmap_explicit));
    _fs.read(reinterpret_cast<char*>(&m_size), sizeof(m_size));

    init(m_size);
    if(!m_ptr)
        m_ptr = malloc(m_size);

    _fs.read(reinterpret_cast<char*>(&m_read_count), sizeof(m_read_count));
    _fs.read(reinterpret_cast<char*>(&m_write_count), sizeof(m_write_count));
    _fs.read(reinterpret_cast<char*>(m_ptr), m_size * sizeof(char));
}
}  // namespace base
}  // namespace tim

#endif
