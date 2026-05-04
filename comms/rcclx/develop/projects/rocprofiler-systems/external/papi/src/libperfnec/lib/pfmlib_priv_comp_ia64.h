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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux/ia64.
 */
#ifndef __PFMLIB_PRIV_COMP_IA64_H__
#define __PFMLIB_PRIV_COMP_IA64_H__

#include <perfmon/pfmlib_comp.h>

#ifdef LIBPFM_USING_INTEL_ECC_COMPILER

#define ia64_get_cpuid(regnum)	__getIndReg(_IA64_REG_INDR_CPUID, (regnum))
#define ia64_getf(d)		__getf_exp(d)

#elif defined(__GNUC__)

static inline unsigned long
ia64_get_cpuid (unsigned long regnum)
{
	unsigned long r;

	asm ("mov %0=cpuid[%r1]" : "=r"(r) : "rO"(regnum));
	return r;
}

static inline unsigned long
ia64_getf(double d)
{
	unsigned long exp;

	__asm__ ("getf.exp %0=%1" : "=r"(exp) : "f"(d));
	return exp;
}
#else /* !GNUC nor INTEL_ECC */
#error "need to define a set of compiler-specific macros"
#endif

#endif /* __PFMLIB_PRIV_COMP_IA64_H__ */
