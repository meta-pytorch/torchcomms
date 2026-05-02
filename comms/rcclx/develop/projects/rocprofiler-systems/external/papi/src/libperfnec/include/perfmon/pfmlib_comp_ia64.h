/*
 * IA-64 compiler specific macros
 *
 * Copyright (c) 2002-2006 Hewlett-Packard Development Company, L.P.
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
#ifndef __PFMLIB_COMP_IA64_H__
#define __PFMLIB_COMP_IA64_H__

#ifndef __PFMLIB_COMP_H__
#error "you should never include this file directly, use pfmlib_comp.h"
#endif

#ifndef __ia64__
#error "you should not be including this file"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * this header file contains all the macros, inline assembly, instrinsics needed
 * by the library and which are compiler-specific
 */

#if defined(__ECC) && defined(__INTEL_COMPILER)
#define LIBPFM_USING_INTEL_ECC_COMPILER	1
/* if you do not have this file, your compiler is too old */
#include <ia64intrin.h>
#endif


#ifdef LIBPFM_USING_INTEL_ECC_COMPILER

#define ia64_sum(void)		__sum(1<<2)
#define ia64_rum(void)		__rum(1<<2)
#define ia64_get_pmd(regnum)	__getIndReg(_IA64_REG_INDR_PMD, (regnum))
#define pfmlib_popcnt(v)	_m64_popcnt(v)

#elif defined(__GNUC__)

static inline void
ia64_sum(void)
{
	__asm__ __volatile__("sum psr.up;;" ::: "memory" );
}

static inline void
ia64_rum(void)
{
	__asm__ __volatile__("rum psr.up;;" ::: "memory" );
}
static inline unsigned long
ia64_get_pmd(int regnum)
{
	unsigned long value;
	__asm__ __volatile__ ("mov %0=pmd[%1]" : "=r"(value) : "r"(regnum));
	return value;
}

static inline unsigned long
pfmlib_popcnt(unsigned long v)
{
	unsigned long ret;
	__asm__ __volatile__ ("popcnt %0=%1" : "=r"(ret) : "r"(v));
	return ret;
}
#else /* !GNUC nor INTEL_ECC */
#error "need to define a set of compiler-specific macros"
#endif

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_COMP_IA64_H__ */
