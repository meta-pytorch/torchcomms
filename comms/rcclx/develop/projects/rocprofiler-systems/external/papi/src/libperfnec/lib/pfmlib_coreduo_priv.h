/*
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Based on:
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
 */
#ifndef __PFMLIB_COREDUO_PRIV_H__
#define __PFMLIB_COREDUO_PRIV_H__

#define PFMLIB_COREDUO_MAX_UMASK 16

typedef struct {
	char			*pme_uname; /* unit mask name */
	char			*pme_udesc; /* event/umask description */
	unsigned int		pme_ucode;  /* unit mask code */
	unsigned int		pme_flags;  /* unit mask flags */
} pme_coreduo_umask_t;

typedef struct {
	char			*pme_name;	/* event name */
	char			*pme_desc;	/* event description */
	unsigned int		pme_code; 	/* event code */
	unsigned int		pme_numasks;	/* number of umasks */
	unsigned int		pme_flags;	/* flags */
	pme_coreduo_umask_t	pme_umasks[PFMLIB_COREDUO_MAX_UMASK]; /* umask desc */
} pme_coreduo_entry_t;

/*
 * pme_flags value (event and unit mask)
 */

/* event-level constraints */
#define PFMLIB_COREDUO_CSPEC		0x02 /* requires a core specification */
#define PFMLIB_COREDUO_PMC0		0x04 /* works only on IA32_PMC0  */
#define PFMLIB_COREDUO_PMC1		0x08 /* works only on IA32_PMC1 */
#define PFMLIB_COREDUO_MESI		0x10 /* requires MESI */

#endif /* __PFMLIB_COREDUO_PRIV_H__ */
