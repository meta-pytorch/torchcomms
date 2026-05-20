/*
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Based on:
 * Copyright (c) 2006-2007 Hewlett-Packard Development Company, L.P.
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
#ifndef __PFMLIB_INTEL_ATOM_PRIV_H__
#define __PFMLIB_INTEL_ATOM_PRIV_H__

#define PFMLIB_INTEL_ATOM_MAX_UMASK 16

typedef struct {
	char			*pme_uname; /* unit mask name */
	char			*pme_udesc; /* event/umask description */
	unsigned int		pme_ucode;  /* unit mask code */
	unsigned int		pme_flags;  /* unit mask flags */
} pme_intel_atom_umask_t;

typedef struct {
	char			*pme_name;	/* event name */
	char			*pme_desc;	/* event description */
	unsigned int		pme_code; 	/* event code */
	unsigned int		pme_numasks;	/* number of umasks */
	unsigned int		pme_flags;	/* flags */
	unsigned int		pme_fixed;	/* fixed counter index, < FIXED_CTR0 if unsupported */
	pme_intel_atom_umask_t	pme_umasks[PFMLIB_INTEL_ATOM_MAX_UMASK]; /* umask desc */
} pme_intel_atom_entry_t;

/*
 * pme_flags value
 */

/*
 * pme_flags value (event and unit mask)
 */
#define PFMLIB_INTEL_ATOM_UMASK_NCOMBO	0x01 /* unit mask cannot be combined (default exclusive) */
#define PFMLIB_INTEL_ATOM_FIXED0	0x02 /* event supported by FIXED_CTR0, can work on generic counters */
#define PFMLIB_INTEL_ATOM_FIXED1	0x04 /* event supported by FIXED_CTR1, can work on generic counters */
#define PFMLIB_INTEL_ATOM_FIXED2_ONLY	0x08 /* works only on FIXED_CTR2 */
#define PFMLIB_INTEL_ATOM_PEBS		0x10 /* support PEBS (precise event) */
#define PFMLIB_INTEL_ATOM_PMC0		0x20 /* works only on IA32_PMC0  */
#define PFMLIB_INTEL_ATOM_PMC1		0x40 /* works only on IA32_PMC1 */

typedef struct {
	unsigned int version:8;
	unsigned int num_cnt:8;
	unsigned int cnt_width:8;
	unsigned int ebx_length:8;
} pmu_eax_t;

typedef struct {
	unsigned int num_cnt:6;
	unsigned int cnt_width:6;
	unsigned int reserved:20;
} pmu_edx_t;

typedef struct {
	unsigned int no_core_cycle:1;
	unsigned int no_inst_retired:1;
	unsigned int no_ref_cycle:1;
	unsigned int no_llc_ref:1;
	unsigned int no_llc_miss:1;
	unsigned int no_br_retired:1;
	unsigned int no_br_mispred_retired:1;
	unsigned int reserved:25;
} pmu_ebx_t;

#endif /* __PFMLIB_INTEL_ATOM_PRIV_H__ */
