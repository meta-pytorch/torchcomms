/*
 * I386 P6/Pentium M compiler specific macros
 *
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
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
#ifndef __PFMLIB_COMP_I386_P6_H__
#define __PFMLIB_COMP_I386_P6_H__

#ifndef __PFMLIB_COMP_H__
#error "you should never include this file directly, use pfmlib_comp.h"
#endif

#ifndef __i386__
#error "you should not be including this file"
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline unsigned long
pfmlib_popcnt(unsigned long v)
{
	unsigned long sum = 0;

	for(; v ; v >>=1) {
		if (v & 0x1) sum++;
	}
	return sum;
}

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_COMP_IA64_H__ */
