/*
 * Copyright (c) 2001-2007 Hewlett-Packard Development Company, L.P.
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
#ifndef __PFMLIB_H__
#define __PFMLIB_H__

#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <inttypes.h>

#include <perfmon/pfmlib_os.h>
#include <perfmon/pfmlib_comp.h>

#define PFMLIB_VERSION		(3 << 16 | 10)
#define PFMLIB_MAJ_VERSION(v)	((v)>>16)
#define PFMLIB_MIN_VERSION(v)	((v) & 0xffff)

/*
 * Maximum number of PMCs/PMDs supported by the library (especially bitmasks)
 */
#define PFMLIB_MAX_PMCS		512 /* maximum number of PMCS supported by the library */
#define PFMLIB_MAX_PMDS		512 /* maximum number of PMDS supported by the library */

/*
 * privilege level mask (mask can be combined)
 * The interpretation of the level is specific to each
 * architecture. Checkout the architecture specific header
 * file for more details.
 */
#define PFM_PLM0	0x1	/* priv level 0 */
#define PFM_PLM1	0x2	/* priv level 1 */
#define PFM_PLM2	0x4	/* priv level 2 */
#define PFM_PLM3	0x8	/* priv level 3 */

/*
 * type used to describe a set of bits in the mask (container type)
 */
typedef unsigned long pfmlib_regmask_bits_t;

/*
 * how many elements do we need to represent all the PMCs and PMDs (rounded up)
 */
#if PFMLIB_MAX_PMCS > PFMLIB_MAX_PMDS
#define PFMLIB_REG_MAX	PFMLIB_MAX_PMCS
#else
#define PFMLIB_REG_MAX	PFMLIB_MAX_PMDS
#endif

#ifndef SWIG
#define __PFMLIB_REG_BV_BITS (sizeof(pfmlib_regmask_bits_t)<<3)
#define PFMLIB_BVSIZE(x) (((x)+(__PFMLIB_REG_BV_BITS)-1) / __PFMLIB_REG_BV_BITS)
#define PFMLIB_REG_BV PFMLIB_BVSIZE(PFMLIB_REG_MAX)
#endif

typedef struct {
	pfmlib_regmask_bits_t bits[PFMLIB_REG_BV];
} pfmlib_regmask_t;


#define PFMLIB_MAX_MASKS_PER_EVENT 48 /* maximum number of unit masks per event */

/*
 * event definition for pfmlib_input_param_t
 */
typedef struct {
	unsigned int	event;		/* event descriptor */
	unsigned int	plm;		/* event privilege level mask */
	unsigned long	flags;		/* per-event flag */
	unsigned int	unit_masks[PFMLIB_MAX_MASKS_PER_EVENT]; /* unit-mask identifiers */
	unsigned int	num_masks;	/* number of masks specified in 'unit_masks' */
	unsigned long	reserved[2];	/* for future use */
} pfmlib_event_t;

/*
 * generic register definition
 */
typedef struct {
	unsigned long long	reg_value;	/* register value */
	unsigned long long	reg_addr;	/* hardware register addr or index */
	unsigned int 		reg_num;	/* logical register index (perfmon2) */
	unsigned int		reg_reserved1;	/* for future use */
	unsigned long		reg_alt_addr;	/* alternate hw register addr of index */
} pfmlib_reg_t;

/*
 * library generic input parameters for pfm_dispatch_event()
 */
typedef struct {
	unsigned int	 pfp_event_count;	 	/* how many events specified (input) */
	unsigned int	 pfp_dfl_plm;		 	/* default priv level : used when event.plm==0 */
	unsigned int     pfp_flags;		 	/* set of flags for all events used when event.flags==0*/
	unsigned int	 reserved1;			/* for future use */
	pfmlib_event_t	 pfp_events[PFMLIB_MAX_PMCS];	/* event descriptions */
	pfmlib_regmask_t pfp_unavail_pmcs;		/* bitmask of unavailable PMC registers */
	unsigned long	 reserved[6];			/* for future use */
} pfmlib_input_param_t;

/*
 * pfp_flags possible values (apply to all events)
 */
#define PFMLIB_PFP_SYSTEMWIDE	0x1 /* indicate monitors will be used in a system-wide session */

/*
 * library generic output parameters for pfm_dispatch_event()
 */
typedef struct {
	unsigned int	 pfp_pmc_count;		 	/* number of entries in pfp_pmcs */
	unsigned int	 pfp_pmd_count;			/* number of entries in pfp_pmds */
	pfmlib_reg_t	 pfp_pmcs[PFMLIB_MAX_PMCS];	/* PMC registers number and values */
	pfmlib_reg_t	 pfp_pmds[PFMLIB_MAX_PMDS];	/* PMD registers numbers */
	unsigned long	 reserved[7];			/* for future use */
} pfmlib_output_param_t;

/*
 * library configuration options
 */
typedef struct {
	unsigned int	pfm_debug:1;	/* set in debug  mode */
	unsigned int	pfm_verbose:1;	/* set in verbose mode */
	unsigned int	pfm_reserved:30;/* for future use */
} pfmlib_options_t;

/*
 * special data type for libpfm error value used to help
 * with Python support and in particular for SWIG. By using
 * a specific type we can detect library calls and trap errors
 * in one SWIG statement as opposed to having to keep track of
 * each call individually. Programs can use 'int' safely for
 * the return value.
 */
typedef int pfm_err_t;			/* error if !PFMLIB_SUCCESS */

extern pfm_err_t pfm_set_options(pfmlib_options_t *opt);
extern pfm_err_t pfm_initialize(void);

extern pfm_err_t pfm_list_supported_pmus(int (*pf)(const char *fmt,...));
extern pfm_err_t pfm_get_pmu_name(char *name, int maxlen);
extern pfm_err_t pfm_get_pmu_type(int *type);
extern pfm_err_t pfm_get_pmu_name_bytype(int type, char *name, size_t maxlen);
extern pfm_err_t pfm_is_pmu_supported(int type);
extern pfm_err_t pfm_force_pmu(int type);

/*
 * pfm_find_event_byname() is obsolete, use pfm_find_event
 */
extern pfm_err_t pfm_find_event(const char *str, unsigned int *idx);
extern pfm_err_t pfm_find_event_byname(const char *name, unsigned int *idx);
extern pfm_err_t pfm_find_event_bycode(int code, unsigned int *idx);
extern pfm_err_t pfm_find_event_bycode_next(int code, unsigned int start,
                                            unsigned int *next);
extern pfm_err_t pfm_find_event_mask(unsigned int event_idx, const char *str,
                                     unsigned int *mask_idx);
extern pfm_err_t pfm_find_full_event(const char *str, pfmlib_event_t *e);

extern pfm_err_t pfm_get_max_event_name_len(size_t *len);

extern pfm_err_t pfm_get_num_events(unsigned int *count);
extern pfm_err_t pfm_get_num_event_masks(unsigned int event_idx,
                                         unsigned int *count);
extern pfm_err_t pfm_get_event_name(unsigned int idx, char *name,
                                    size_t maxlen);
extern pfm_err_t pfm_get_full_event_name(pfmlib_event_t *e, char *name,
                                         size_t maxlen);
extern pfm_err_t pfm_get_event_code(unsigned int idx, int *code);
extern pfm_err_t pfm_get_event_mask_code(unsigned int idx,
                                         unsigned int mask_idx,
                                         unsigned int *code);
extern pfm_err_t pfm_get_event_counters(unsigned int idx,
                                        pfmlib_regmask_t *counters);
extern pfm_err_t pfm_get_event_description(unsigned int idx, char **str);
extern pfm_err_t pfm_get_event_code_counter(unsigned int idx, unsigned int cnt,
                                            int *code);
extern pfm_err_t pfm_get_event_mask_name(unsigned int event_idx,
                                         unsigned int mask_idx,
                                         char *name, size_t maxlen);
extern pfm_err_t pfm_get_event_mask_description(unsigned int event_idx,
                                                unsigned int mask_idx,
                                                char **desc);

extern pfm_err_t pfm_dispatch_events(pfmlib_input_param_t *p,
                                     void *model_in,
                                     pfmlib_output_param_t *q,
                                     void *model_out);

extern pfm_err_t pfm_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs);
extern pfm_err_t pfm_get_impl_pmds(pfmlib_regmask_t *impl_pmds);
extern pfm_err_t pfm_get_impl_counters(pfmlib_regmask_t *impl_counters);
extern pfm_err_t pfm_get_num_pmds(unsigned int *num);
extern pfm_err_t pfm_get_num_pmcs(unsigned int *num);
extern pfm_err_t pfm_get_num_counters(unsigned int *num);

extern pfm_err_t pfm_get_hw_counter_width(unsigned int *width);
extern pfm_err_t pfm_get_version(unsigned int *version);
extern char *pfm_strerror(int code);
extern pfm_err_t pfm_get_cycle_event(pfmlib_event_t *e);
extern pfm_err_t pfm_get_inst_retired_event(pfmlib_event_t *e);

/*
 * Supported PMU family
 */
#define PFMLIB_NO_PMU	 	 	-1	/* PMU unused (forced) */
#define PFMLIB_UNKNOWN_PMU	 	0	/* type not yet known (dynamic) */
#define PFMLIB_GEN_IA64_PMU	 	1	/* Intel IA-64 architected PMU */
#define PFMLIB_ITANIUM_PMU	 	2	/* Intel Itanium   */
#define PFMLIB_ITANIUM2_PMU 	 	3	/* Intel Itanium 2 */
#define PFMLIB_MONTECITO_PMU 	 	4	/* Intel Dual-Core Itanium 2 9000 */
#define PFMLIB_AMD64_PMU		16	/* AMD AMD64 (K7, K8, Families 10h, 15h) */
#define PFMLIB_GEN_IA32_PMU		63	/* Intel architectural PMU for X86 */
#define PFMLIB_I386_P6_PMU		32	/* Intel PIII (P6 core) */
#define PFMLIB_PENTIUM4_PMU		33	/* Intel Pentium4/Xeon/EM64T */
#define PFMLIB_COREDUO_PMU		34	/* Intel Core Duo/Core Solo */
#define PFMLIB_I386_PM_PMU		35	/* Intel Pentium M */
#define PFMLIB_CORE_PMU			36	/* obsolete, use PFMLIB_INTEL_CORE_PMU */
#define PFMLIB_INTEL_CORE_PMU		36	/* Intel Core */
#define PFMLIB_INTEL_PPRO_PMU		37	/* Intel Pentium Pro */
#define PFMLIB_INTEL_PII_PMU		38	/* Intel Pentium II */
#define PFMLIB_INTEL_ATOM_PMU		39	/* Intel Atom */
#define PFMLIB_INTEL_NHM_PMU            40      /* Intel Nehalem */
#define PFMLIB_INTEL_WSM_PMU            41      /* Intel Westmere */

#define PFMLIB_MIPS_20KC_PMU		64	/* MIPS 20KC */
#define PFMLIB_MIPS_24K_PMU		65	/* MIPS 24K */
#define PFMLIB_MIPS_25KF_PMU		66	/* MIPS 25KF */
#define PFMLIB_MIPS_34K_PMU		67	/* MIPS 34K */
#define PFMLIB_MIPS_5KC_PMU		68	/* MIPS 5KC */
#define PFMLIB_MIPS_74K_PMU		69	/* MIPS 74K */
#define PFMLIB_MIPS_R10000_PMU		70	/* MIPS R10000 */
#define PFMLIB_MIPS_R12000_PMU		71	/* MIPS R12000 */
#define PFMLIB_MIPS_RM7000_PMU		72	/* MIPS RM7000 */
#define PFMLIB_MIPS_RM9000_PMU		73	/* MIPS RM9000 */
#define PFMLIB_MIPS_SB1_PMU		74	/* MIPS SB1/SB1A */
#define PFMLIB_MIPS_VR5432_PMU		75	/* MIPS VR5432 */
#define PFMLIB_MIPS_VR5500_PMU		76	/* MIPS VR5500 */
#define PFMLIB_MIPS_ICE9A_PMU		77	/* SiCortex ICE9A */
#define PFMLIB_MIPS_ICE9B_PMU		78	/* SiCortex ICE9B */

#define PFMLIB_POWERPC_PMU		90	/* POWERPC */

#define PFMLIB_CRAYX2_PMU		96	/* Cray X2 */

#define	PFMLIB_CELL_PMU			100	/* CELL */

#define PFMLIB_PPC970_PMU               110	/* IBM PowerPC 970(FX,GX) */
#define PFMLIB_PPC970MP_PMU             111	/* IBM PowerPC 970MP */     
#define PFMLIB_POWER3_PMU               112	/* IBM POWER3 */
#define PFMLIB_POWER4_PMU               113	/* IBM POWER4 */
#define PFMLIB_POWER5_PMU               114	/* IBM POWER5 */
#define PFMLIB_POWER5p_PMU              115	/* IBM POWER5+ */
#define PFMLIB_POWER6_PMU               116	/* IBM POWER6 */
#define PFMLIB_POWER7_PMU               117	/* IBM POWER7 */

#define PFMLIB_SPARC_ULTRA12_PMU	130	/* UltraSPARC I, II, IIi, and IIe */
#define PFMLIB_SPARC_ULTRA3_PMU		131	/* UltraSPARC III */
#define PFMLIB_SPARC_ULTRA3I_PMU	132	/* UltraSPARC IIIi and IIIi+ */
#define PFMLIB_SPARC_ULTRA3PLUS_PMU	133	/* UltraSPARC III+ and IV */
#define PFMLIB_SPARC_ULTRA4PLUS_PMU	134	/* UltraSPARC IV+ */
#define PFMLIB_SPARC_NIAGARA1_PMU	135	/* Niagara-1 */
#define PFMLIB_SPARC_NIAGARA2_PMU	136	/* Niagara-2 */

/*
 * pfmlib error codes
 */
#define PFMLIB_SUCCESS		  0	/* success */
#define PFMLIB_ERR_NOTSUPP	 -1	/* function not supported */
#define PFMLIB_ERR_INVAL	 -2	/* invalid parameters */
#define PFMLIB_ERR_NOINIT	 -3	/* library was not initialized */
#define PFMLIB_ERR_NOTFOUND	 -4	/* event not found */
#define PFMLIB_ERR_NOASSIGN	 -5	/* cannot assign events to counters */
#define PFMLIB_ERR_FULL	 	 -6	/* buffer is full or too small */
#define PFMLIB_ERR_EVTMANY	 -7	/* event used more than once */
#define PFMLIB_ERR_MAGIC	 -8	/* invalid library magic number */
#define PFMLIB_ERR_FEATCOMB	 -9	/* invalid combination of features */
#define PFMLIB_ERR_EVTSET	-10	/* incompatible event sets */
#define PFMLIB_ERR_EVTINCOMP	-11	/* incompatible event combination */
#define PFMLIB_ERR_TOOMANY	-12	/* too many events or unit masks */

#define PFMLIB_ERR_IRRTOOBIG	-13	/* code range too big */
#define PFMLIB_ERR_IRREMPTY	-14	/* empty code range */
#define PFMLIB_ERR_IRRINVAL	-15	/* invalid code range */
#define PFMLIB_ERR_IRRTOOMANY	-16	/* too many code ranges */
#define PFMLIB_ERR_DRRINVAL	-17	/* invalid data range */
#define PFMLIB_ERR_DRRTOOMANY	-18	/* too many data ranges */
#define PFMLIB_ERR_BADHOST	-19	/* not supported by host CPU */
#define PFMLIB_ERR_IRRALIGN	-20	/* bad alignment for code range */
#define PFMLIB_ERR_IRRFLAGS	-21	/* code range missing flags */
#define PFMLIB_ERR_UMASK	-22	/* invalid or missing unit mask */
#define PFMLIB_ERR_NOMEM	-23	/* out of memory */

#define __PFMLIB_REGMASK_EL(g)		((g)/__PFMLIB_REG_BV_BITS)
#define __PFMLIB_REGMASK_MASK(g)	(((pfmlib_regmask_bits_t)1) << ((g) % __PFMLIB_REG_BV_BITS))

static inline int
pfm_regmask_isset(pfmlib_regmask_t *h, unsigned int b)
{
	if (b >= PFMLIB_REG_MAX)
		return 0;
	return (h->bits[__PFMLIB_REGMASK_EL(b)] & __PFMLIB_REGMASK_MASK(b)) != 0;
}

static inline int
pfm_regmask_set(pfmlib_regmask_t *h, unsigned int b)
{
	if (b >= PFMLIB_REG_MAX)
		return PFMLIB_ERR_INVAL;

	h->bits[__PFMLIB_REGMASK_EL(b)] |=  __PFMLIB_REGMASK_MASK(b);

	return PFMLIB_SUCCESS;
}

static inline int
pfm_regmask_clr(pfmlib_regmask_t *h, unsigned int b)
{
	if (h == NULL || b >= PFMLIB_REG_MAX)
		return PFMLIB_ERR_INVAL;

	h->bits[__PFMLIB_REGMASK_EL(b)] &= ~ __PFMLIB_REGMASK_MASK(b);

	return PFMLIB_SUCCESS;
}

static inline int
pfm_regmask_weight(pfmlib_regmask_t *h, unsigned int *w)
{
	unsigned int pos;
	unsigned int weight = 0;

	if (h == NULL || w == NULL)
		return PFMLIB_ERR_INVAL;

	for (pos = 0; pos < PFMLIB_REG_BV; pos++) {
		weight += (unsigned int)pfmlib_popcnt(h->bits[pos]);
	}
	*w = weight;
	return PFMLIB_SUCCESS;
}

static inline int
pfm_regmask_eq(pfmlib_regmask_t *h1, pfmlib_regmask_t *h2)
{
	unsigned int pos;

	if (h1 == NULL || h2 == NULL)
		return 0;

	for (pos = 0; pos < PFMLIB_REG_BV; pos++) {
		if (h1->bits[pos] != h2->bits[pos]) return 0;
	}
	return 1;
}

static inline int
pfm_regmask_and(pfmlib_regmask_t *dst, pfmlib_regmask_t *h1, pfmlib_regmask_t *h2)
{
	unsigned int pos;
	if (dst == NULL || h1 == NULL || h2 == NULL)
		return PFMLIB_ERR_INVAL;

	for (pos = 0; pos < PFMLIB_REG_BV; pos++) {
		dst->bits[pos] = h1->bits[pos] & h2->bits[pos];
	}
	return PFMLIB_SUCCESS;
}

static inline int
pfm_regmask_andnot(pfmlib_regmask_t *dst, pfmlib_regmask_t *h1, pfmlib_regmask_t *h2)
{
	unsigned int pos;
	if (dst == NULL || h1 == NULL || h2 == NULL)
		return PFMLIB_ERR_INVAL;

	for (pos = 0; pos < PFMLIB_REG_BV; pos++) {
		dst->bits[pos] = h1->bits[pos] & ~h2->bits[pos];
	}
	return PFMLIB_SUCCESS;
}

static inline int
pfm_regmask_or(pfmlib_regmask_t *dst, pfmlib_regmask_t *h1, pfmlib_regmask_t *h2)
{
	unsigned int pos;
	if (dst == NULL || h1 == NULL || h2 == NULL)
		return PFMLIB_ERR_INVAL;

	for (pos = 0; pos < PFMLIB_REG_BV; pos++) {
		dst->bits[pos] = h1->bits[pos] | h2->bits[pos];
	}
	return PFMLIB_SUCCESS;
}

static inline int
pfm_regmask_copy(pfmlib_regmask_t *dst, pfmlib_regmask_t *src)
{
	unsigned int pos;
	if (dst == NULL || src == NULL)
		return PFMLIB_ERR_INVAL;

	for (pos = 0; pos < PFMLIB_REG_BV; pos++) {
		dst->bits[pos] = src->bits[pos];
	}
	return PFMLIB_SUCCESS;
}
static inline int
pfm_regmask_not(pfmlib_regmask_t *dst)
{
	unsigned int pos;
	if (dst == NULL)
		return PFMLIB_ERR_INVAL;

	for (pos = 0; pos < PFMLIB_REG_BV; pos++) {
		dst->bits[pos] = ~dst->bits[pos];
	}
	return PFMLIB_SUCCESS;
}

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_H__ */
