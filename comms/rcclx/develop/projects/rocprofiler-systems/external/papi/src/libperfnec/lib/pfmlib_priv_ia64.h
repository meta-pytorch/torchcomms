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
#ifndef __PFMLIB_PRIV_IA64_H__
#define __PFMLIB_PRIV_IA64_H__

typedef struct {
	unsigned long db_mask:56;
	unsigned long db_plm:4;
	unsigned long db_ig:2;
	unsigned long db_w:1;
	unsigned long db_rx:1;
} br_mask_reg_t;

typedef union {
	unsigned long  val;
	br_mask_reg_t  db;
} dbreg_t;

static inline int
pfm_ia64_get_cpu_family(void)
{
	return (int)((ia64_get_cpuid(3) >> 24) & 0xff);
}

static inline int
pfm_ia64_get_cpu_model(void)
{
	return (int)((ia64_get_cpuid(3) >> 16) & 0xff);
}

/*
 * find last bit set
 */
static inline int
pfm_ia64_fls (unsigned long x)
{
	double d = x;
	long exp;

	exp = ia64_getf(d);
	return exp - 0xffff;

}
#endif /* __PFMLIB_PRIV_IA64_H__ */
