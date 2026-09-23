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

#ifndef TIMEMORY_HASH_STATIC_STRING_CPP_
#define TIMEMORY_HASH_STATIC_STRING_CPP_

#include "timemory/hash/macros.hpp"

#if !defined(TIMEMORY_HASH_HEADER_MODE) ||                                               \
    (defined(TIMEMORY_HASH_HEADER_MODE) && TIMEMORY_HASH_HEADER_MODE == 0)
#    include "timemory/hash/static_string.hpp"
#endif

namespace tim
{
inline namespace hash
{
TIMEMORY_HASH_INLINE
static_string::static_string(const char* _str)
: m_string{ _str }
{
    if(get_private_registry())
        get_private_registry()->emplace(_str);
}

TIMEMORY_HASH_INLINE
bool
static_string::is_registered(const char* _str)
{
    if(!get_private_registry())
        return false;
    return get_private_registry()->find(_str) != get_private_registry()->end();
}

TIMEMORY_HASH_INLINE
bool
static_string::is_registered(std::size_t _hash)
{
    return is_registered(reinterpret_cast<const char*>(_hash));
}

TIMEMORY_HASH_INLINE
static_string::string_registry_t
static_string::get_registry()
{
    if(!get_private_registry())
        return string_registry_t{};
    return *get_private_registry();
}

TIMEMORY_HASH_INLINE
static_string::string_registry_t*&
static_string::get_private_registry()
{
    // this is an intentional data "leak"
    static thread_local auto* _instance = new string_registry_t{};
    return _instance;
}
}  // namespace hash
}  // namespace tim

namespace std
{
TIMEMORY_HASH_INLINE
std::size_t
hash<tim::static_string>::operator()(const tim::static_string& _static_str)
{
    return _static_str.hash();
}
}  // namespace std

#endif
