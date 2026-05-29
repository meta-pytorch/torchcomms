/*
 * Intel Atom : architectural perfmon v3 + PEBS 
 *
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Based on pfmlib_intel_atom.h with
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
 */
#ifndef __PFMLIB_INTEL_ATOM_H__
#define __PFMLIB_INTEL_ATOM_H__

#include <perfmon/pfmlib.h>
/*
 * privilege level mask usage for architected PMU:
 *
 * PFM_PLM0 = OS (kernel, hypervisor, ..)
 * PFM_PLM1 = unused (ignored)
 * PFM_PLM2 = unused (ignored)
 * PFM_PLM3 = USR (user level)
 */

#ifdef __cplusplus
extern "C" {
#endif

#define PMU_INTEL_ATOM_NUM_COUNTERS	5 /* 2 generic + 3 fixed */

typedef union {
	unsigned long long val;			/* complete register value */
	struct {
		unsigned long sel_event_select:8;	/* event mask */
		unsigned long sel_unit_mask:8;		/* unit mask */
		unsigned long sel_usr:1;		/* user level */
		unsigned long sel_os:1;			/* system level */
		unsigned long sel_edge:1;		/* edge detec */
		unsigned long sel_pc:1;			/* pin control */
		unsigned long sel_int:1;		/* enable APIC intr */
		unsigned long sel_any:1;		/* any thread */
		unsigned long sel_en:1;			/* enable */
		unsigned long sel_inv:1;		/* invert counter mask */
		unsigned long sel_cnt_mask:8;		/* counter mask */
		unsigned long sel_res2:32;
	} perfevtsel;
} pfm_intel_atom_sel_reg_t;

typedef struct {
	unsigned long		cnt_mask;	/* threshold (cnt_mask)  */
	unsigned int		flags;		/* counter specific flags */
} pfmlib_intel_atom_counter_t;

#define PFM_INTEL_ATOM_SEL_INV		0x1	/* inverse */
#define PFM_INTEL_ATOM_SEL_EDGE		0x2	/* edge detect */
#define PFM_INTEL_ATOM_SEL_ANYTHR	0x4	/* measure on any of 2 threads */

/*
 * model-specific parameters for the library
 */
typedef struct {
	pfmlib_intel_atom_counter_t	pfp_intel_atom_counters[PMU_INTEL_ATOM_NUM_COUNTERS];
	unsigned int			pfp_intel_atom_pebs_used; /* set to 1 to use PEBS */
	uint64_t			reserved[4];	    /* for future use */
} pfmlib_intel_atom_input_param_t;

#ifdef __cplusplus /* extern C */
}
#endif

/*
 * Atom-specific interface
 */
extern int pfm_intel_atom_has_pebs(pfmlib_event_t *e);

#endif /* __PFMLIB_INTEL_ATOM_H__ */
