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

#include "timemory/macros/os.hpp"

#include <csignal>

#if defined(TIMEMORY_LINUX)
#    include <features.h>
#elif defined(TIMEMORY_MACOS)
#    include <signal.h>
#endif

//======================================================================================//
//
//      WINDOWS SIGNALS (dummy)
//
//======================================================================================//

#if defined(TIMEMORY_WINDOWS) || defined(_WIN32) || defined(_WIN64)
//   dummy definition of SIGHUP
#    ifndef SIGHUP
#        define SIGHUP 1
#    endif
//   dummy definition of SIGINT
#    ifndef SIGINT
#        define SIGINT 2
#    endif
//   dummy definition of SIGQUIT
#    ifndef SIGQUIT
#        define SIGQUIT 3
#    endif
//   dummy definition of SIGILL
#    ifndef SIGILL
#        define SIGILL 4
#    endif
//   dummy definition of SIGTRAP
#    ifndef SIGTRAP
#        define SIGTRAP 5
#    endif
//   dummy definition of SIGABRT
#    ifndef SIGABRT
#        define SIGABRT 6
#    endif
//   dummy definition of SIGEMT
#    ifndef SIGEMT
#        define SIGEMT 7
#    endif
//   dummy definition of SIGFPE
#    ifndef SIGFPE
#        define SIGFPE 8
#    endif
//   dummy definition of SIGKILL
#    ifndef SIGKILL
#        define SIGKILL 9
#    endif
//   dummy definition of SIGBUS
#    ifndef SIGBUS
#        define SIGBUS 10
#    endif
//   dummy definition of SIGSEGV
#    ifndef SIGSEGV
#        define SIGSEGV 11
#    endif
//   dummy definition of SIGSYS
#    ifndef SIGSYS
#        define SIGSYS 12
#    endif
//   dummy definition of SIGPIPE
#    ifndef SIGPIPE
#        define SIGPIPE 13
#    endif
//   dummy definition of SIGALRM
#    ifndef SIGALRM
#        define SIGALRM 14
#    endif
//   dummy definition of SIGTERM
#    ifndef SIGTERM
#        define SIGTERM 15
#    endif
//   dummy definition of SIGURG
#    ifndef SIGURG
#        define SIGURG 16
#    endif
//   dummy definition of SIGSTOP
#    ifndef SIGSTOP
#        define SIGSTOP 17
#    endif
//   dummy definition of SIGTSTP
#    ifndef SIGTSTP
#        define SIGTSTP 18
#    endif
//   dummy definition of SIGCONT
#    ifndef SIGCONT
#        define SIGCONT 19
#    endif
//   dummy definition of SIGCHLD
#    ifndef SIGCHLD
#        define SIGCHLD 20
#    endif
//   dummy definition of SIGTTIN
#    ifndef SIGTTIN
#        define SIGTTIN 21
#    endif
//   dummy definition of SIGTTOU
#    ifndef SIGTTOU
#        define SIGTTOU 22
#    endif
//   dummy definition of SIGIO
#    ifndef SIGIO
#        define SIGIO 23
#    endif
//   dummy definition of SIGXCPU
#    ifndef SIGXCPU
#        define SIGXCPU 24
#    endif
//   dummy definition of SIGXFSZ
#    ifndef SIGXFSZ
#        define SIGXFSZ 25
#    endif
//   dummy definition of SIGVTALRM
#    ifndef SIGVTALRM
#        define SIGVTALRM 26
#    endif
//   dummy definition of SIGPROF
#    ifndef SIGPROF
#        define SIGPROF 27
#    endif
//   dummy definition of SIGWINCH
#    ifndef SIGWINCH
#        define SIGWINCH 28
#    endif
//   dummy definition of SIGINFO
#    ifndef SIGINFO
#        define SIGINFO 29
#    endif
//   dummy definition of SIGUSR1
#    ifndef SIGUSR1
#        define SIGUSR1 30
#    endif
//   dummy definition of SIGUSR2
#    ifndef SIGUSR2
#        define SIGUSR2 31
#    endif
#endif  // defined(TIMEMORY_WINDOWS)

// compatible compiler
#if defined(__GNUC__) || defined(__clang__) || defined(_INTEL_COMPILER)
#    if !defined(TIMEMORY_SIGNAL_COMPAT_COMPILER)
#        define TIMEMORY_SIGNAL_COMPAT_COMPILER
#    endif
#endif

#if defined(_XOPEN_SOURCE) && defined(_POSIX_C_SOURCE) &&                                \
    (_XOPEN_SOURCE >= 700 || _POSIX_C_SOURCE >= 200809L)
#    define PSIGINFO_AVAILABLE
#endif

// compatible operating system
#if defined(__linux__) || defined(__MACH__)
#    if !defined(TIMEMORY_SIGNAL_COMPAT_OS)
#        define TIMEMORY_SIGNAL_COMPAT_OS
#    endif
#endif

#if defined(TIMEMORY_SIGNAL_COMPAT_COMPILER) && defined(TIMEMORY_SIGNAL_COMPAT_OS) &&    \
    !(defined(TIMEMORY_USE_GPERFTOOLS) || defined(TIMEMORY_USE_GPERFTOOLS_PROFILER) ||   \
      defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC))
#    if !defined(TIMEMORY_SIGNAL_AVAILABLE)
#        define TIMEMORY_SIGNAL_AVAILABLE
#    endif
#endif

//======================================================================================//
//  these are not in the original POSIX.1-1990 standard so we are defining
//  them in case the OS hasn't
//  POSIX-1.2001
#ifndef SIGTRAP
#    define SIGTRAP 5
#endif
//  not specified in POSIX.1-2001, but nevertheless appears on most other
//  UNIX systems, where its default action is typically to terminate the
//  process with a core dump.
#ifndef SIGEMT
#    define SIGEMT 7
#endif
//  POSIX-1.2001
#ifndef SIGURG
#    define SIGURG 16
#endif
//  POSIX-1.2001
#ifndef SIGXCPU
#    define SIGXCPU 24
#endif
//  POSIX-1.2001
#ifndef SIGXFSZ
#    define SIGXFSZ 25
#endif
//  POSIX-1.2001
#ifndef SIGVTALRM
#    define SIGVTALRM 26
#endif
//  POSIX-1.2001
#ifndef SIGPROF
#    define SIGPROF 27
#endif
//  POSIX-1.2001
#ifndef SIGINFO
#    define SIGINFO 29
#endif
