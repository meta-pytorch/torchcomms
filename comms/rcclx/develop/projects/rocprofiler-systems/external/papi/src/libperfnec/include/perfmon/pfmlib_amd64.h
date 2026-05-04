/*
 * AMD64 PMU specific types and definitions (64 and 32 bit modes)
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

#ifndef __PFMLIB_AMD64_H__
#define __PFMLIB_AMD64_H__

#include <stdint.h>

/*
 * privilege level mask usage for AMD64:
 *
 * PFM_PLM0 = OS (kernel, hypervisor, ..)
 * PFM_PLM1 = invalid parameters
 * PFM_PLM2 = invalid parameters
 * PFM_PLM3 = USR (user level)
 */

#ifdef __cplusplus
extern "C" {
#endif

#define PMU_AMD64_MAX_COUNTERS	6	/* total numbers of performance counters */

/*
 * AMD64 MSR definitions
 */

typedef union {
	uint64_t val;				/* complete register value */
	struct {
		uint64_t sel_event_mask:8;	/* event mask */
		uint64_t sel_unit_mask:8;	/* unit mask */
		uint64_t sel_usr:1;		/* user level */
		uint64_t sel_os:1;		/* system level */
		uint64_t sel_edge:1;		/* edge detec */
		uint64_t sel_pc:1;		/* pin control */
		uint64_t sel_int:1;		/* enable APIC intr */
		uint64_t sel_res1:1;		/* reserved */
		uint64_t sel_en:1;		/* enable */
		uint64_t sel_inv:1;		/* invert counter mask */
		uint64_t sel_cnt_mask:8;	/* counter mask */
		uint64_t sel_event_mask2:4;	/* from 10h: event mask [11:8] */
		uint64_t sel_res2:4;		/* reserved */
		uint64_t sel_guest:1;		/* from 10h: guest only counter */
		uint64_t sel_host:1;		/* from 10h: host only counter */
		uint64_t sel_res3:22;		/* reserved */
	} perfsel;
} pfm_amd64_sel_reg_t; /* MSR 0xc001000-0xc001003 */

typedef union {
	uint64_t val;				/* complete register value */
	struct {
		uint64_t ctr_count:48;		/* 48-bit hardware counter  */
		uint64_t ctr_res1:16;		/* reserved */
	} perfctr;
} pfm_amd64_ctr_reg_t; /* MSR 0xc001004-0xc001007 */

typedef union {
	uint64_t val;				/* complete register value */
	struct {
		uint64_t ibsfetchmaxcnt:16;
		uint64_t ibsfetchcnt:16;
		uint64_t ibsfetchlat:16;
		uint64_t ibsfetchen:1;
		uint64_t ibsfetchval:1;
		uint64_t ibsfetchcomp:1;
		uint64_t ibsicmiss:1;
		uint64_t ibsphyaddrvalid:1;
		uint64_t ibsl1tlbpgsz:2;
		uint64_t ibsl1tlbmiss:1;
		uint64_t ibsl2tlbmiss:1;
		uint64_t ibsranden:1;
		uint64_t reserved:6;
	} reg;
} ibsfetchctl_t; /* MSR 0xc0011030 */

typedef union {
	uint64_t val;				/* complete register value */
	struct {
		uint64_t ibsopmaxcnt:16;
		uint64_t reserved1:1;
		uint64_t ibsopen:1;
		uint64_t ibsopval:1;
		uint64_t ibsopcntl:1;
		uint64_t reserved2:44;
	} reg;
} ibsopctl_t; /* MSR 0xc0011033 */

typedef union {
	uint64_t val;				/* complete register value */
	struct {
		uint64_t ibscomptoretctr:16;
		uint64_t ibstagtoretctr:16;
		uint64_t ibsopbrnresync:1;
		uint64_t ibsopmispreturn:1;
		uint64_t ibsopreturn:1;
		uint64_t ibsopbrntaken:1;
		uint64_t ibsopbrnmisp:1;
		uint64_t ibsopbrnret:1;
		uint64_t reserved:26;
	} reg;
} ibsopdata_t; /* MSR 0xc0011035 */

typedef union {
	uint64_t val;				/* complete register value */
	struct {
		uint64_t nbibsreqsrc:3;
		uint64_t reserved1:1;
		uint64_t nbibsreqdstproc:1;
		uint64_t nbibsreqcachehitst:1;
		uint64_t reserved2:58;
	} reg;
} ibsopdata2_t; /* MSR 0xc0011036 */

typedef union {
	uint64_t val;				/* complete register value */
	struct {
		uint64_t ibsldop:1;
		uint64_t ibsstop:1;
		uint64_t ibsdcl1tlbmiss:1;
		uint64_t ibsdcl2tlbmiss:1;
		uint64_t ibsdcl1tlbhit2m:1;
		uint64_t ibsdcl1tlbhit1g:1;
		uint64_t ibsdcl2tlbhit2m:1;
		uint64_t ibsdcmiss:1;
		uint64_t ibsdcmissacc:1;
		uint64_t ibsdcldbnkcon:1;
		uint64_t ibsdcstbnkcon:1;
		uint64_t ibsdcsttoldfwd:1;
		uint64_t ibsdcsttoldcan:1;
		uint64_t ibsdcucmemacc:1;
		uint64_t ibsdcwcmemacc:1;
		uint64_t ibsdclockedop:1;
		uint64_t ibsdcmabhit:1;
		uint64_t ibsdclinaddrvalid:1;
		uint64_t ibsdcphyaddrvalid:1;
		uint64_t reserved1:13;
		uint64_t ibsdcmisslat:16;
		uint64_t reserved2:16;
	} reg;
} ibsopdata3_t; /* MSR 0xc0011037 */

/*
 * AMD64 specific input parameters for the library
 */

typedef struct {
	uint32_t	cnt_mask;	/* threshold ([4-255] are reserved) */
	uint32_t	flags;		/* counter specific flag */
} pfmlib_amd64_counter_t;

#define PFM_AMD64_SEL_INV	0x1	/* inverse */
#define PFM_AMD64_SEL_EDGE	0x2	/* edge detect */
#define PFM_AMD64_SEL_GUEST	0x4	/* guest only */
#define PFM_AMD64_SEL_HOST	0x8	/* host only */

/*
 * IBS input parameters
 *
 * Maxcnt specifies the maximum count value of the periodic counter,
 * 20 bits, bits 3:0 are always set to zero.
 */
typedef struct {
	unsigned int maxcnt;
	unsigned int options;
} ibs_param_t;

/*
 * values for ibs_param_t.options
 */
#define IBS_OPTIONS_RANDEN 1	/* enable randomization (IBS fetch only) */
#define IBS_OPTIONS_UOPS   1	/* count dispatched uops (IBS op only) */

typedef struct {
	pfmlib_amd64_counter_t pfp_amd64_counters[PMU_AMD64_MAX_COUNTERS]; /* extended counter features */
	uint32_t	flags;		/* use flags */
	uint32_t	reserved1;	/* for future use */
       	ibs_param_t	ibsfetch;	/* IBS fetch control */
	ibs_param_t	ibsop;		/* IBS execution control */
	uint64_t	reserved2;	/* for future use */
} pfmlib_amd64_input_param_t;

/* A bit mask, meaning multiple usage types may be defined */
#define PFMLIB_AMD64_USE_IBSFETCH	1
#define PFMLIB_AMD64_USE_IBSOP		2

/*
 * AMD64 specific output parameters for the library
 *
 * The values ibsfetch_base and ibsop_base pass back the index of the
 * ibsopctl and ibsfetchctl register in pfp_pmds[].
 */

typedef struct {
	uint32_t	ibsfetch_base;	/* Perfmon2 base register index */
	uint32_t	ibsop_base;	/* Perfmon2 base register index */
	uint64_t	reserved[7];	/* for future use */
} pfmlib_amd64_output_param_t;

/* Perfmon2 registers relative to base register */
#define PMD_IBSFETCHCTL		0
#define PMD_IBSFETCHLINAD	1
#define PMD_IBSFETCHPHYSAD	2
#define PMD_IBSOPCTL		0
#define PMD_IBSOPRIP		1
#define PMD_IBSOPDATA		2
#define PMD_IBSOPDATA2		3
#define PMD_IBSOPDATA3		4
#define PMD_IBSDCLINAD		5
#define PMD_IBSDCPHYSAD		6

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_AMD64_H__ */
