/*
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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
#ifndef __PFMLIB_CORE_PRIV_H__
#define __PFMLIB_CORE_PRIV_H__

#define PFMLIB_CORE_MAX_UMASK 32

typedef struct {
	char			*pme_uname; /* unit mask name */
	char			*pme_udesc; /* event/umask description */
	unsigned int		pme_ucode;  /* unit mask code */
	unsigned int		pme_flags;  /* unit mask flags */
} pme_core_umask_t;

typedef struct {
	char			*pme_name;	/* event name */
	char			*pme_desc;	/* event description */
	unsigned int		pme_code; 	/* event code */
	unsigned int		pme_numasks;	/* number of umasks */
	unsigned int		pme_flags;	/* flags */
	pme_core_umask_t	pme_umasks[PFMLIB_CORE_MAX_UMASK]; /* umask desc */
} pme_core_entry_t;

/*
 * pme_flags value (event and unit mask)
 */

/* event or unit-mask level constraints */
#define PFMLIB_CORE_FIXED0		0x02 /* event supported by FIXED_CTR0, can work on generic counters */
#define PFMLIB_CORE_FIXED1		0x04 /* event supported by FIXED_CTR1, can work on generic counters */
#define PFMLIB_CORE_FIXED2_ONLY		0x08 /* works only on FIXED_CTR2 */

/* event-level constraints */
#define PFMLIB_CORE_UMASK_NCOMBO	0x01 /* unit mask cannot be combined (default: combination ok) */

#define PFMLIB_CORE_CSPEC		0x40 /* requires a core specification */
#define PFMLIB_CORE_PEBS		0x20 /* support PEBS (precise event) */
#define PFMLIB_CORE_PMC0		0x10 /* works only on IA32_PMC0  */
#define PFMLIB_CORE_PMC1		0x80 /* works only on IA32_PMC1 */
#define PFMLIB_CORE_MESI		0x100 /* requires MESI */

#endif /* __PFMLIB_CORE_PRIV_H__ */
