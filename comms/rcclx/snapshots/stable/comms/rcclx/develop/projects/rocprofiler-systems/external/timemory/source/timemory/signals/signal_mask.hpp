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

#include "timemory/backends/signals.hpp"
#include "timemory/defines.h"
#include "timemory/macros/os.hpp"

#include <csignal>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>

#if defined(TIMEMORY_UNIX)
#    include <pthread.h>
#endif

namespace tim
{
namespace signals
{
enum class sigmask_scope : short
{
    thread  = 0,
    process = 1
};

#if defined(TIMEMORY_UNIX)
inline sigset_t
block_signals(const std::set<int>& _signals, sigmask_scope _scope)
{
    sigset_t _old;
    sigset_t _new;

    sigemptyset(&_new);
    for(auto itr : _signals)
        sigaddset(&_new, itr);

    auto _err = (_scope == sigmask_scope::thread)
                    ? pthread_sigmask(SIG_BLOCK, &_new, &_old)
                    : sigprocmask(SIG_BLOCK, &_new, &_old);

    if(_err != 0)
    {
        std::string _msg =
            (_scope == sigmask_scope::thread) ? "pthread_sigmask" : "sigprocmask";
        perror(_msg.c_str());
        throw std::runtime_error(_msg);
    }

    return _old;
}

inline sigset_t
unblock_signals(const std::set<int>& _signals, sigmask_scope _scope)
{
    sigset_t _old;
    sigset_t _new;

    sigemptyset(&_new);
    for(auto itr : _signals)
        sigaddset(&_new, itr);

    auto _err = (_scope == sigmask_scope::thread)
                    ? pthread_sigmask(SIG_UNBLOCK, &_new, &_old)
                    : sigprocmask(SIG_UNBLOCK, &_new, &_old);

    if(_err != 0)
    {
        std::string _msg =
            (_scope == sigmask_scope::thread) ? "pthread_sigmask" : "sigprocmask";
        perror(_msg.c_str());
        throw std::runtime_error(_msg);
    }

    return _old;
}
#endif
}  // namespace signals
}  // namespace tim
