/*
 * Copyright (c) 2003-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef __PFMLIB_OS_H__
#define __PFMLIB_OS_H__

#ifdef __linux__
#ifdef __ia64__
#include <perfmon/pfmlib_os_ia64.h>
#endif

#ifdef __x86_64__
#include <perfmon/pfmlib_os_x86_64.h>
#endif

#ifdef __i386__
#include <perfmon/pfmlib_os_i386.h>
#endif

#if defined(__mips__)
#include <perfmon/pfmlib_os_mips64.h>
#endif

#ifdef __powerpc__
#include <perfmon/pfmlib_os_powerpc.h>
#endif

#ifdef __sparc__
#include <perfmon/pfmlib_os_sparc.h>
#endif

#ifdef __cell__
#include <perfmon/pfmlib_os_powerpc.h>
#endif

#ifdef __crayx2
#include <perfmon/pfmlib_os_crayx2.h>
#endif
#endif /* __linux__ */
#endif /* __PFMLIB_OS_H__ */
