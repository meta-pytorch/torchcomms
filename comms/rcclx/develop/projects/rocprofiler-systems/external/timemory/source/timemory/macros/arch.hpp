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

//======================================================================================//
//
//      ARCHITECTURE
//
//======================================================================================//

#if !defined(TIMEMORY_ARCH_X86_64)
#    if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) ||                \
        defined(__amd64) || defined(_M_X64)
#        define TIMEMORY_ARCH_X86_64 1
#    else
#        define TIMEMORY_ARCH_X86_64 0
#    endif
#endif

#if !defined(TIMEMORY_ARCH_X86_32)
#    if defined(i386) || defined(__i386__) || defined(__i486__) || defined(__i586__) ||  \
        defined(__i686__) || defined(__i386) || defined(_M_IX86) || defined(_X86_) ||    \
        defined(__THW_INTEL__) || defined(__I86__) || defined(__INTEL__)
#        define TIMEMORY_ARCH_X86_32 1
#    else
#        define TIMEMORY_ARCH_X86_32 0
#    endif
#endif

#if !defined(TIMEMORY_ARCH_X86)
#    if TIMEMORY_ARCH_X86_64 > 0 || TIMEMORY_ARCH_X86_32 > 0
#        define TIMEMORY_ARCH_X86 1
#    else
#        define TIMEMORY_ARCH_X86 0
#    endif
#endif

#if !defined(TIMEMORY_ARCH_IA64)
#    if defined(__ia64__) || defined(_IA64) || defined(__IA64__) || defined(__ia64) ||   \
        defined(_M_IA64) || defined(__itanium__)
#        define TIMEMORY_ARCH_IA64 1
#    else
#        define TIMEMORY_ARCH_IA64 0
#    endif
#endif

#if !defined(TIMEMORY_ARCH_ARM)
#    if defined(__ARM_ARCH) || defined(__TARGET_ARCH_ARM) ||                             \
        defined(__TARGET_ARCH_THUMB) || defined(_M_ARM) || defined(__arm__) ||           \
        defined(__arm64) || defined(__thumb__) || defined(_M_ARM64) ||                   \
        defined(__aarch64__) || defined(__AARCH64EL__) || defined(__ARM_ARCH_7__) ||     \
        defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) ||                          \
        defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_6K__) ||                          \
        defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6KZ__) ||                         \
        defined(__ARM_ARCH_6T2__) || defined(__ARM_ARCH_5TE__) ||                        \
        defined(__ARM_ARCH_5TEJ__) || defined(__ARM_ARCH_4T__) ||                        \
        defined(__ARM_ARCH_4__)
#        define TIMEMORY_ARCH_ARM 1
#    else
#        define TIMEMORY_ARCH_ARM 0
#    endif
#endif

#if !defined(TIMEMORY_ARCH_PPC)
#    if defined(__powerpc) || defined(__powerpc__) || defined(__POWERPC__) ||            \
        defined(__ppc__) || defined(_M_PPC) || defined(_ARCH_PPC) ||                     \
        defined(__PPCGECKO__) || defined(__PPCBROADWAY__) || defined(_XENON)
#        define TIMEMORY_ARCH_PPC 1
#    else
#        define TIMEMORY_ARCH_PPC 0
#    endif
#endif

//--------------------------------------------------------------------------------------//
