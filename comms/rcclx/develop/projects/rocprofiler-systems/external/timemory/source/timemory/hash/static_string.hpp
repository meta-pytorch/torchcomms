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

#include "timemory/hash/macros.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_set>

namespace tim
{
inline namespace hash
{
struct static_string
{
    using string_registry_t = std::unordered_set<const char*>;

    static_string(const char* _str);
    ~static_string()                        = default;
    static_string(const static_string&)     = default;
    static_string(static_string&&) noexcept = default;
    static_string& operator=(const static_string&) = default;
    static_string& operator=(static_string&&) noexcept = default;

    const char* c_str() const { return m_string; }
    std::size_t hash() const { return reinterpret_cast<std::size_t>(m_string); }
    std::size_t operator()() const { return hash(); }

    operator const char*() const { return m_string; }
    operator std::size_t() const { return hash(); }

    static bool              is_registered(const char*);
    static bool              is_registered(std::size_t);
    static string_registry_t get_registry();

private:
    static string_registry_t*& get_private_registry();

    const char* m_string = nullptr;
};
}  // namespace hash
}  // namespace tim

namespace std
{
template <>
struct hash<tim::static_string>
{
    std::size_t operator()(const tim::static_string&);
};
}  // namespace std

// include the definitions inline if header-only mode
#if defined(TIMEMORY_HASH_HEADER_MODE) && TIMEMORY_HASH_HEADER_MODE > 0
#    include "timemory/hash/static_string.cpp"
#endif
