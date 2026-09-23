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

#ifndef TIMEMORY_UTILITY_PROCFS_MAPS_HPP_
#    define TIMEMORY_UTILITY_PROCFS_MAPS_HPP_
#endif

#include "timemory/backends/process.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/tpls/cereal/cereal/cereal.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/transient_function.hpp"
#include "timemory/variadic/macros.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace tim
{
namespace utility
{
template <typename Tp>
std::string
as_hex(Tp _v, size_t _width = 16)
{
    uintptr_t _vp = 0;
    if constexpr(std::is_pointer<Tp>::value)
        _vp = reinterpret_cast<uintptr_t>(_v);
    else
        _vp = _v;

    std::stringstream _ss;
    _ss.fill('0');
    _ss << "0x" << std::hex << std::setw(_width) << _vp;
    return _ss.str();
}
}  // namespace utility

namespace procfs
{
struct maps
{
    TIMEMORY_DEFAULT_OBJECT(maps)

    maps(const std::vector<std::string>& _line);

    size_t              load_address = 0;
    size_t              last_address = 0;
    std::array<char, 4> permissions  = {};
    size_t              offset       = 0;
    std::string         device       = {};
    size_t              inode        = {};
    std::string         pathname     = {};

    static std::vector<maps> iterate_program_headers();

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned);

    size_t length() const;
    bool   is_empty() const;
    bool   is_file_mapping() const;
    bool   has_read_perms() const { return permissions.at(0) == 'r'; }
    bool   has_write_perms() const { return permissions.at(1) == 'w'; }
    bool   has_exec_perms() const { return permissions.at(2) == 'x'; }
    bool   is_private() const { return permissions.at(3) == 'p'; }
    bool   is_contiguous_with(const maps&) const;
    bool   contains(size_t) const;

    explicit operator bool() const { return !is_empty(); }
    maps&    operator+=(const maps& rhs);

    bool operator<(const maps& _rhs) const;
    bool operator==(const maps& _rhs) const;

    std::string as_string() const;

    friend std::ostream& operator<<(std::ostream& _os, const maps& _v)
    {
        return (_os << _v.as_string());
    }

private:
    friend std::vector<maps> read_maps(pid_t _pid);
};

template <typename ArchiveT>
inline void
maps::serialize(ArchiveT& ar, const unsigned)
{
    auto _perms = std::string(permissions.size(), '-');
    for(size_t i = 0; i < permissions.size(); ++i)
        _perms.at(i) = permissions.at(i);

    ar(cereal::make_nvp("load_address", TIMEMORY_JOIN("", std::hex, load_address)),
       cereal::make_nvp("last_address", TIMEMORY_JOIN("", std::hex, last_address)),
       cereal::make_nvp("permissions", _perms),
       cereal::make_nvp("offset", TIMEMORY_JOIN("", std::hex, offset)),
       cereal::make_nvp("device", device), cereal::make_nvp("inode", inode),
       cereal::make_nvp("pathname", pathname));
}

std::vector<maps>
read_maps(pid_t _pid = process::get_target_id());

std::vector<maps>&
get_maps(pid_t _pid, bool _update = false);

std::vector<maps>
get_maps(pid_t _pid, utility::transient_function<bool(const maps&)>&& _filter,
         bool _update = false);

std::vector<maps>
get_contiguous_maps(pid_t _pid, utility::transient_function<bool(const maps&)>&& _filter,
                    bool _update = false);

std::vector<maps>
get_contiguous_maps(pid_t _pid, bool _update = false);

maps
find_map(uint64_t _addr, pid_t _pid = process::get_target_id());
}  // namespace procfs
}  // namespace tim

#if defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/procfs/maps.cpp"
#endif
