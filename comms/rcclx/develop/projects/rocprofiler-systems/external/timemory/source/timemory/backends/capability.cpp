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

#include "timemory/backends/capability.hpp"

#include "timemory/utility/join.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(TIMEMORY_LINUX)
#    include <linux/capability.h>
#endif

namespace timemory
{
namespace linux
{
namespace capability
{
#define TIMEMORY_INFO_ENTRY_(VAL)                                                        \
    cap_info { #VAL, VAL }

namespace
{
std::initializer_list<cap_info> known_capabilities = {
#if defined(CAP_CHOWN)
    TIMEMORY_INFO_ENTRY_(CAP_CHOWN),
#endif

#if defined(CAP_DAC_OVERRIDE)
    TIMEMORY_INFO_ENTRY_(CAP_DAC_OVERRIDE),
#endif

#if defined(CAP_DAC_READ_SEARCH)
    TIMEMORY_INFO_ENTRY_(CAP_DAC_READ_SEARCH),
#endif

#if defined(CAP_FOWNER)
    TIMEMORY_INFO_ENTRY_(CAP_FOWNER),
#endif

#if defined(CAP_FSETID)
    TIMEMORY_INFO_ENTRY_(CAP_FSETID),
#endif

#if defined(CAP_KILL)
    TIMEMORY_INFO_ENTRY_(CAP_KILL),
#endif

#if defined(CAP_SETGID)
    TIMEMORY_INFO_ENTRY_(CAP_SETGID),
#endif

#if defined(CAP_SETUID)
    TIMEMORY_INFO_ENTRY_(CAP_SETUID),
#endif

#if defined(CAP_SETPCAP)
    TIMEMORY_INFO_ENTRY_(CAP_SETPCAP),
#endif

#if defined(CAP_LINUX_IMMUTABLE)
    TIMEMORY_INFO_ENTRY_(CAP_LINUX_IMMUTABLE),
#endif

#if defined(CAP_NET_BIND_SERVICE)
    TIMEMORY_INFO_ENTRY_(CAP_NET_BIND_SERVICE),
#endif

#if defined(CAP_NET_BROADCAST)
    TIMEMORY_INFO_ENTRY_(CAP_NET_BROADCAST),
#endif

#if defined(CAP_NET_ADMIN)
    TIMEMORY_INFO_ENTRY_(CAP_NET_ADMIN),
#endif

#if defined(CAP_NET_RAW)
    TIMEMORY_INFO_ENTRY_(CAP_NET_RAW),
#endif

#if defined(CAP_IPC_LOCK)
    TIMEMORY_INFO_ENTRY_(CAP_IPC_LOCK),
#endif

#if defined(CAP_IPC_OWNER)
    TIMEMORY_INFO_ENTRY_(CAP_IPC_OWNER),
#endif

#if defined(CAP_SYS_MODULE)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_MODULE),
#endif

#if defined(CAP_SYS_RAWIO)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_RAWIO),
#endif

#if defined(CAP_SYS_CHROOT)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_CHROOT),
#endif

#if defined(CAP_SYS_PTRACE)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_PTRACE),
#endif

#if defined(CAP_SYS_PACCT)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_PACCT),
#endif

#if defined(CAP_SYS_ADMIN)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_ADMIN),
#endif

#if defined(CAP_SYS_BOOT)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_BOOT),
#endif

#if defined(CAP_SYS_NICE)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_NICE),
#endif

#if defined(CAP_SYS_RESOURCE)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_RESOURCE),
#endif

#if defined(CAP_SYS_TIME)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_TIME),
#endif

#if defined(CAP_SYS_TTY_CONFIG)
    TIMEMORY_INFO_ENTRY_(CAP_SYS_TTY_CONFIG),
#endif

#if defined(CAP_MKNOD)
    TIMEMORY_INFO_ENTRY_(CAP_MKNOD),
#endif

#if defined(CAP_LEASE)
    TIMEMORY_INFO_ENTRY_(CAP_LEASE),
#endif

#if defined(CAP_AUDIT_WRITE)
    TIMEMORY_INFO_ENTRY_(CAP_AUDIT_WRITE),
#endif

#if defined(CAP_AUDIT_CONTROL)
    TIMEMORY_INFO_ENTRY_(CAP_AUDIT_CONTROL),
#endif

#if defined(CAP_SETFCAP)
    TIMEMORY_INFO_ENTRY_(CAP_SETFCAP),
#endif

#if defined(CAP_MAC_OVERRIDE)
    TIMEMORY_INFO_ENTRY_(CAP_MAC_OVERRIDE),
#endif

#if defined(CAP_MAC_ADMIN)
    TIMEMORY_INFO_ENTRY_(CAP_MAC_ADMIN),
#endif

#if defined(CAP_SYSLOG)
    TIMEMORY_INFO_ENTRY_(CAP_SYSLOG),
#endif

#if defined(CAP_WAKE_ALARM)
    TIMEMORY_INFO_ENTRY_(CAP_WAKE_ALARM),
#endif

#if defined(CAP_BLOCK_SUSPEND)
    TIMEMORY_INFO_ENTRY_(CAP_BLOCK_SUSPEND),
#endif

#if defined(CAP_AUDIT_READ)
    TIMEMORY_INFO_ENTRY_(CAP_AUDIT_READ),
#endif

#if defined(CAP_PERFMON)
    TIMEMORY_INFO_ENTRY_(CAP_PERFMON),
#endif

#if defined(CAP_BPF)
    TIMEMORY_INFO_ENTRY_(CAP_BPF),
#endif

#if defined(CAP_CHECKPOINT_RESTORE)
    TIMEMORY_INFO_ENTRY_(CAP_CHECKPOINT_RESTORE),
#endif

#if defined(CAP_LAST_CAP)
    TIMEMORY_INFO_ENTRY_(CAP_LAST_CAP),
#endif
};

#undef TIMEMORY_INFO_ENTRY_

template <typename Tp = unsigned>
Tp
cap_max_bits()
{
    auto _value = Tp{ 0 };
    for(const auto& itr : known_capabilities)
        _value = std::max<Tp>(_value, itr.value + 1);
    return _value;
}
}  // namespace

cap_status
cap_read(pid_t _pid)
{
    auto ifs = std::ifstream{ join::join('/', "/proc", _pid, "status") };
    if(!ifs)
        return cap_status{};

    auto _lines = std::vector<std::string>{};

    while(ifs && ifs.good())
    {
        auto _line = std::string{};
        std::getline(ifs, _line);
        if(ifs && ifs.good() && !_line.empty())
            _lines.emplace_back(std::move(_line));
    }

    auto _data = cap_status{};
    for(const auto& itr : _lines)
    {
        auto iss    = std::istringstream{ itr };
        auto _key   = std::string{};
        auto _value = std::string{};
        iss >> _key;

        if(_key.find("Cap") == 0)
            iss >> _value;

        if(!_value.empty())
        {
            auto _key_matches = [&_key](std::string_view _cap_id_str) {
                return (_key.find(_cap_id_str) == 0);
            };
            if(_key_matches("CapInh"))
                _data.inherited = std::stoull(_value, nullptr, 16);
            else if(_key_matches("CapPrm"))
                _data.permitted = std::stoull(_value, nullptr, 16);
            else if(_key_matches("CapEff"))
                _data.effective = std::stoull(_value, nullptr, 16);
            else if(_key_matches("CapBnd"))
                _data.bounding = std::stoull(_value, nullptr, 16);
            else if(_key_matches("CapAmb"))
                _data.ambient = std::stoull(_value, nullptr, 16);
        }
    }

    return _data;
}

std::string
cap_name(cap_value_t _v)
{
    auto to_lower = [](std::string&& _s) {
        for(auto& citr : _s)
            citr = tolower(citr);
        return std::move(_s);
    };

    for(const auto& itr : known_capabilities)
        if(itr.value == _v)
            return to_lower(std::string{ itr.name });

    return std::string{};
}

std::vector<cap_value_t>
cap_decode(unsigned long long value)
{
    auto _cap_max_bits_v = cap_max_bits();
    auto _data           = std::vector<cap_value_t>{};
    _data.reserve(known_capabilities.size());
    for(unsigned cap = 0; (cap < 64) && ((value >> cap) != 0U); ++cap)
    {
        auto _mask = value & (1ULL << cap);
        if(_mask != 0U)
        {
            if(cap < _cap_max_bits_v)
                _data.emplace_back(cap);
        }
    }

    return _data;
}

std::vector<cap_value_t>
cap_decode(const char* arg)
{
    return cap_decode(std::strtoull(arg, nullptr, 16));
}
}  // namespace capability
}  // namespace linux
}  // namespace timemory
