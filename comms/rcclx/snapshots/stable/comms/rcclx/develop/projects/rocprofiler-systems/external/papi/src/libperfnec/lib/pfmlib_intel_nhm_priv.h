/*
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
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
 * applications on Linux.
 */
#ifndef __PFMLIB_NHM_PRIV_H__
#define __PFMLIB_NHM_PRIV_H__

#define PFMLIB_NHM_MAX_UMASK 32

typedef struct {
	char			*pme_uname; /* unit mask name */
	char			*pme_udesc; /* event/umask description */
	unsigned int		pme_cntmsk; /* counter mask */
	unsigned int		pme_ucode;  /* unit mask code */
	unsigned int		pme_uflags; /* unit mask flags */
	unsigned int		pme_umodel; /* CPU model for this umask */
} pme_nhm_umask_t;

typedef struct {
	char			*pme_name;	/* event name */
	char			*pme_desc;	/* event description */
	unsigned int		pme_code; 	/* event code */
	unsigned int		pme_cntmsk;	/* counter mask */
	unsigned int		pme_numasks;	/* number of umasks */
	unsigned int		pme_flags;	/* flags */
	pme_nhm_umask_t	pme_umasks[PFMLIB_NHM_MAX_UMASK]; /* umask desc */
} pme_nhm_entry_t;

/*
 * pme_flags value (event and unit mask)
 */

/* event or unit-mask level constraints */
#define PFMLIB_NHM_UMASK_NCOMBO		0x001 /* unit mask cannot be combined (default: combination ok) */
#define PFMLIB_NHM_FIXED0		0x002 /* event supported by FIXED_CTR0, can work on generic counters */
#define PFMLIB_NHM_FIXED1		0x004 /* event supported by FIXED_CTR1, can work on generic counters */
#define PFMLIB_NHM_FIXED2_ONLY		0x008 /* only works in FIXED_CTR2 */
#define PFMLIB_NHM_OFFCORE_RSP0		0x010 /* requires OFFCORE_RSP0 register */
#define PFMLIB_NHM_PMC01		0x020 /* works only on IA32_PMC0 or IA32_PMC1  */
#define PFMLIB_NHM_PEBS			0x040 /* support PEBS (precise event) */
#define PFMLIB_NHM_UNC			0x080 /* uncore event */
#define PFMLIB_NHM_UNC_FIXED		0x100 /* uncore fixed event */
#define PFMLIB_NHM_OFFCORE_RSP1		0x200 /* requires OFFCORE_RSP1 register */
#define PFMLIB_NHM_PMC0			0x400 /* works only on IA32_PMC0 */
#define PFMLIB_NHM_EX			0x800 /*  has Nehalem-EX specific unit masks */

#endif /* __PFMLIB_NHM_PRIV_H__ */
