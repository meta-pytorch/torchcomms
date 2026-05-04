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

#include "timemory/defines.h"

#ifndef TIMEMORY_UTILITY_PROCFS_MAPS_HPP_
#    include "timemory/utility/procfs/maps.hpp"
#    define TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
#else
#    define TIMEMORY_UTILITY_PROCFS_MAPS_INLINE inline
#endif

#ifndef TIMEMORY_UTILITY_PROCFS_MAPS_CPP_
#    define TIMEMORY_UTILITY_PROCFS_MAPS_CPP_

#    include "timemory/environment/types.hpp"
#    include "timemory/macros/os.hpp"
#    include "timemory/process/process.hpp"
#    include "timemory/utility/utility.hpp"

#    if defined(TIMEMORY_LINUX)
#        include <elf.h>
#        include <link.h>
#    endif

#    include <string>
#    include <vector>

namespace tim
{
namespace procfs
{
#    if defined(TIMEMORY_LINUX)
namespace
{
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
int
dl_iterate_callback(dl_phdr_info* _info, size_t, void* _pdata)
{
    auto* _data = static_cast<std::vector<maps>*>(_pdata);

    static auto _exe_name = []() {
        auto _cmd = read_command_line(process::get_id());
        auto _cmd_env =
            tim::get_env<std::string>(TIMEMORY_SETTINGS_PREFIX "COMMAND_LINE", "");
        if(!_cmd_env.empty())
            _cmd = tim::delimit(_cmd_env, " ");
        return (_cmd.empty()) ? std::string{} : _cmd.front();
    }();

    auto _name = std::string{};
    if(_info->dlpi_name)
        _name = _info->dlpi_name;
    else
        _name = _exe_name;

    if(!filepath::exists(_name))
        return 0;

    auto _base_addr = _info->dlpi_addr;

    for(int j = 0; j < _info->dlpi_phnum; j++)
    {
        auto _hdr = _info->dlpi_phdr[j];
        if(_hdr.p_type == PT_LOAD)
        {
            auto _v         = maps{};
            _v.pathname     = _name;
            _v.offset       = _hdr.p_offset;
            _v.load_address = _base_addr + _hdr.p_vaddr;
            _v.last_address =
                _v.load_address + std::max<size_t>(_hdr.p_filesz, _hdr.p_memsz);
            _v.permissions.fill('-');
            if((_hdr.p_flags & PF_R) == PF_R)
                _v.permissions.at(0) = 'r';
            if((_hdr.p_flags & PF_W) == PF_W)
                _v.permissions.at(1) = 'w';
            if((_hdr.p_flags & PF_X) == PF_X)
                _v.permissions.at(2) = 'x';
            _data->emplace_back(_v);
        }
    }

    return 0;
}
}  // namespace

TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
std::vector<maps>
maps::iterate_program_headers()
{
    auto _data = std::vector<maps>{};
    dl_iterate_phdr(dl_iterate_callback, static_cast<void*>(&_data));

    // filter out duplicates
    std::sort(_data.begin(), _data.end());
    _data.erase(std::unique(_data.begin(), _data.end()), _data.end());

    return _data;
}

#    else

TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
std::vector<maps>
maps::iterate_program_headers()
{
    return std::vector<maps>{};
}

#    endif

TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
maps::maps(const std::vector<std::string>& _delim)
{
    auto _addr_str = _delim.front();
    auto _addr     = delimit(_delim.front(), "-");
    load_address   = std::stoull(_addr.front(), nullptr, 16);
    last_address   = std::stoull(_addr.back(), nullptr, 16);
    auto _perm     = _delim.at(1);
    for(size_t i = 0; i < permissions.size(); ++i)
        permissions.at(i) = _perm.at(i);
    offset = std::stoull(_delim.at(2), nullptr, 16);
    device = _delim.at(3);
    inode  = std::stoull(_delim.at(4));
    if(_delim.size() > 5)
        pathname = _delim.at(5);
}

TIMEMORY_UTILITY_PROCFS_MAPS_INLINE size_t
maps::length() const
{
    return (last_address >= load_address) ? (last_address - load_address) : 0;
}

TIMEMORY_UTILITY_PROCFS_MAPS_INLINE bool
maps::is_empty() const
{
    // sum will be zero if default initialized
    return ((load_address + last_address + offset + inode + device.length() +
             pathname.length()) == 0);
}

TIMEMORY_UTILITY_PROCFS_MAPS_INLINE bool
maps::is_file_mapping() const
{
    return !pathname.empty() && filepath::exists(pathname);
}

TIMEMORY_UTILITY_PROCFS_MAPS_INLINE bool
maps::contains(size_t _addr) const
{
    return _addr >= load_address && _addr < last_address;
}

TIMEMORY_UTILITY_PROCFS_MAPS_INLINE bool
maps::is_contiguous_with(const maps& _v) const
{
    return ((last_address == _v.load_address || load_address == _v.last_address) &&
            pathname == _v.pathname);
}

// clang-format off
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
maps&
maps::operator+=(const maps& _v)
// clang-format on
{
    if(is_contiguous_with(_v))
    {
        load_address = std::min<size_t>(load_address, _v.load_address);
        last_address = std::max<size_t>(last_address, _v.last_address);
        offset       = std::min<size_t>(offset, _v.offset);
        for(size_t i = 0; i < permissions.size(); ++i)
        {
            // downgrade permissions if not equal
            if(permissions.at(i) != _v.permissions.at(i))
                permissions.at(i) = (i == 4) ? 'p' : '-';
        }
        if(device != _v.device)
            device += std::string{ ";" } + _v.device;
        if(inode != _v.inode)
            inode = 0;
    }
    return *this;
}

// clang-format off
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
bool
maps::operator<(const maps& _rhs) const
// clang-format on
{
    return std::tie(load_address, last_address, offset, pathname) <
           std::tie(_rhs.load_address, _rhs.last_address, _rhs.offset, _rhs.pathname);
}

// clang-format off
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
bool
maps::operator==(const maps& _rhs) const
// clang-format on
{
    return std::tie(load_address, last_address, offset, pathname) ==
           std::tie(_rhs.load_address, _rhs.last_address, _rhs.offset, _rhs.pathname);
}

// clang-format off
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
std::string
maps::as_string() const
// clang-format on
{
    auto _ss = std::stringstream{};
    _ss << utility::as_hex(load_address) << "-" << utility::as_hex(last_address) << " ";
    for(auto itr : permissions)
        _ss << itr;
    _ss << " " << utility::as_hex(offset) << " " << device << " " << inode;
    if(!pathname.empty())
        _ss << " " << pathname;
    return _ss.str();
}

// clang-format off
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
std::vector<maps>
read_maps(pid_t _pid)
// clang-format on
{
    auto _data = (_pid == process::get_id()) ? maps::iterate_program_headers()
                                             : std::vector<maps>{};
    auto          _fname = TIMEMORY_JOIN('/', "/proc", _pid, "maps");
    std::ifstream ifs{ _fname };
    if(!ifs)
    {
        fprintf(stderr, "Failure opening %s\n", _fname.c_str());
    }
    else
    {
        while(ifs)
        {
            std::string _line = {};
            if(std::getline(ifs, _line) && !_line.empty())
            {
                auto _delim = delimit(_line, " \t\n\r");
                if(_delim.size() >= 4)
                    _data.emplace_back(_delim);
                else
                {
                    fprintf(stderr, "Discarding '%s'...\n", _line.c_str());
                }
            }
        }
    }
    return _data;
}

// clang-format off
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
std::vector<maps>&
get_maps(pid_t _pid, bool _update)
// clang-format on
{
    static auto                  _data  = std::unordered_map<pid_t, std::vector<maps>>{};
    static auto                  _mutex = std::mutex{};
    std::scoped_lock<std::mutex> _lk{ _mutex };

    auto itr = _data.find(_pid);
    if(itr == _data.end())
        _data.emplace(_pid, read_maps(_pid));
    else if(_update)
        _data.at(_pid) = read_maps(_pid);

    return _data.at(_pid);
}

// clang-format off
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
std::vector<maps>
get_maps(pid_t _pid, utility::transient_function<bool(const maps&)>&& _filter,
         bool _update)
// clang-format on
{
    auto _orig = get_maps(_pid, _update);
    auto _data = std::vector<maps>{};
    _data.reserve(_orig.size());
    for(auto&& itr : _orig)
        if(_filter(itr))
            _data.emplace_back(itr);
    return _data;
}

// clang-format off
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
std::vector<maps>
get_contiguous_maps(pid_t _pid, utility::transient_function<bool(const maps&)>&& _filter,
                    bool _update)
// clang-format on
{
    auto&& _find_contiguous = [](const auto& _v, auto _beg, auto _end) {
        for(auto iitr = _beg; iitr != _end; ++iitr)
            if(_v.pathname == iitr->pathname && _v.is_contiguous_with(*iitr))
                return iitr;
        return _end;
    };

    auto _data = std::vector<procfs::maps>{};
    auto _orig = get_maps(_pid, std::move(_filter), _update);
    for(const auto& itr : _orig)
    {
        auto citr = _find_contiguous(itr, _data.begin(), _data.end());
        if(citr == _data.end())
            _data.emplace_back(itr);
        else
            *citr += itr;
    }

    return _data;
}

// clang-format off
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
std::vector<maps>
get_contiguous_maps(pid_t _pid, bool _update)
// clang-format on
{
    return get_contiguous_maps(
        _pid, [](const maps&) { return true; }, _update);
}

// clang-format off
TIMEMORY_UTILITY_PROCFS_MAPS_INLINE
maps
find_map(uint64_t _addr, pid_t _pid)
// clang-format on
{
    auto _search_map = [_addr, _pid]() {
        // make sure it is copied
        auto _v = get_maps(_pid);
        for(const auto& itr : _v)
        {
            if(_addr >= itr.load_address && _addr <= itr.last_address)
                return itr;
        }
        return maps{};
    };

    auto _v = _search_map();

    // if not found, update
    if(_v.is_empty())
        get_maps(_pid, true);

    return _search_map();
}
}  // namespace procfs
}  // namespace tim

#endif
