/*
 * pfmlib_amd64.c : support for the AMD64 architected PMU
 * 		    (for both 64 and 32 bit modes)
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
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* public headers */
#include <perfmon/pfmlib_amd64.h>

/* private headers */
#include "pfmlib_priv.h"		/* library private */
#include "pfmlib_amd64_priv.h"	/* architecture private */
#include "amd64_events.h"		/* PMU private */

/* let's define some handy shortcuts! */
#define sel_event_mask	perfsel.sel_event_mask
#define sel_unit_mask	perfsel.sel_unit_mask
#define sel_usr		perfsel.sel_usr
#define sel_os		perfsel.sel_os
#define sel_edge	perfsel.sel_edge
#define sel_pc		perfsel.sel_pc
#define sel_int		perfsel.sel_int
#define sel_en		perfsel.sel_en
#define sel_inv		perfsel.sel_inv
#define sel_cnt_mask	perfsel.sel_cnt_mask
#define sel_event_mask2 perfsel.sel_event_mask2
#define sel_guest	perfsel.sel_guest
#define sel_host	perfsel.sel_host

#define CHECK_AMD_ARCH(reg) \
	((reg).sel_event_mask2 || (reg).sel_guest || (reg).sel_host)

#define PFMLIB_AMD64_HAS_COMBO(_e) \
	((pfm_amd64_get_event_entry(_e)->pme_flags & PFMLIB_AMD64_UMASK_COMBO) != 0)


#define PFMLIB_AMD64_ALL_FLAGS \
	(PFM_AMD64_SEL_INV|PFM_AMD64_SEL_EDGE|PFM_AMD64_SEL_GUEST|PFM_AMD64_SEL_HOST)
/*
 * Description of the PMC register mappings use by
 * this module:
 * pfp_pmcs[].reg_num:
 * 	0 -> PMC0 -> PERFEVTSEL0 -> MSR @ 0xc0010000
 * 	1 -> PMC1 -> PERFEVTSEL1 -> MSR @ 0xc0010001
 * 	...
 * pfp_pmds[].reg_num:
 * 	0 -> PMD0 -> PERCTR0 -> MSR @ 0xc0010004
 * 	1 -> PMD1 -> PERCTR1 -> MSR @ 0xc0010005
 * 	...
 */
#define AMD64_SEL_BASE		0xc0010000
#define AMD64_CTR_BASE		0xc0010004
#define AMD64_SEL_BASE_F15H	0xc0010200
#define AMD64_CTR_BASE_F15H	0xc0010201

static struct {
	amd64_rev_t	revision;
	char		*name;
	unsigned int	cpu_clks;
	unsigned int	ret_inst;
	int		family;
	int		model;
	int		stepping;
	pme_amd64_entry_t *events;
} amd64_pmu;

pme_amd64_entry_t unsupported_event = {
	.pme_name = "<unsupported>",
	.pme_desc = "This event is not supported be this cpu revision.",
	.pme_code = ~0,
	.pme_flags = PFMLIB_AMD64_NOT_SUPP,
};

pfm_pmu_support_t amd64_support;

#define amd64_revision    amd64_pmu.revision
#define amd64_event_count amd64_support.pme_count
#define amd64_cpu_clks    amd64_pmu.cpu_clks
#define amd64_ret_inst    amd64_pmu.ret_inst
#define amd64_events      amd64_pmu.events
#define amd64_family	  amd64_pmu.family
#define amd64_model	  amd64_pmu.model
#define amd64_stepping	  amd64_pmu.stepping

/* AMD architectural pmu features starts with family 10h */
#define IS_AMD_ARCH()	(amd64_pmu.family >= 0x10)

static amd64_rev_t
amd64_get_revision(int family, int model, int stepping)
{
	switch (family) {
	case 6:
		return AMD64_K7;
	case 0x0f:
		switch (model >> 4) {
		case 0:
			if (model == 5 && stepping < 2)
				return AMD64_K8_REV_B;
			if (model == 4 && stepping == 0)
				return AMD64_K8_REV_B;
			return AMD64_K8_REV_C;
		case 1:
			return AMD64_K8_REV_D;
		case 2:
		case 3:
			return AMD64_K8_REV_E;
		case 4:
		case 5:
		case 0xc:
			return AMD64_K8_REV_F;
		case 6:
		case 7:
		case 8:
			return AMD64_K8_REV_G;
		}
		return AMD64_K8_REV_B;
	case 0x10:
		switch (model) {
		case 4:
		case 5:
		case 6:
			return AMD64_FAM10H_REV_C;
		case 8:
		case 9:
			return AMD64_FAM10H_REV_D;
		case 10:
			return AMD64_FAM10H_REV_E;
		}
		return AMD64_FAM10H_REV_B;
	case 0x15:
		return AMD64_FAM15H_REV_B;
	}

	return AMD64_CPU_UN;
}

/*
 * .byte 0x53 == push ebx. it's universal for 32 and 64 bit
 * .byte 0x5b == pop ebx.
 * Some gcc's (4.1.2 on Core2) object to pairing push/pop and ebx in 64 bit mode.
 * Using the opcode directly avoids this problem.
 */
static inline void cpuid(unsigned int op, unsigned int *a, unsigned int *b,
                  unsigned int *c, unsigned int *d)
{
  __asm__ __volatile__ (".byte 0x53\n\tcpuid\n\tmovl %%ebx, %%esi\n\t.byte 0x5b"
       : "=a" (*a),
	     "=S" (*b),
		 "=c" (*c),
		 "=d" (*d)
       : "a" (op));
}

static void
pfm_amd64_setup(amd64_rev_t revision)
{
	amd64_pmu.revision = revision;
	amd64_pmu.name = (char *)amd64_cpu_strs[revision];
	amd64_support.pmu_name	= amd64_pmu.name;

	/* K8 (default) */
	amd64_pmu.events	= amd64_k8_table.events;
	amd64_support.pme_count	= amd64_k8_table.num;
	amd64_pmu.cpu_clks	= amd64_k8_table.cpu_clks;
	amd64_pmu.ret_inst	= amd64_k8_table.ret_inst;
	amd64_support.pmu_type	= PFMLIB_AMD64_PMU;
	amd64_support.num_cnt	= PMU_AMD64_NUM_COUNTERS;
	amd64_support.pmc_count	= PMU_AMD64_NUM_COUNTERS;
	amd64_support.pmd_count	= PMU_AMD64_NUM_COUNTERS;

	switch (amd64_pmu.family) {
	case 6:
		/* K7 */
		amd64_pmu.events	= amd64_k7_table.events;
		amd64_support.pme_count	= amd64_k7_table.num;
		amd64_pmu.cpu_clks	= amd64_k7_table.cpu_clks;
		amd64_pmu.ret_inst	= amd64_k7_table.ret_inst;
		return;
	case 0x10:
		/* Family 10h */
		amd64_pmu.events	= amd64_fam10h_table.events;
		amd64_support.pme_count	= amd64_fam10h_table.num;
		amd64_pmu.cpu_clks	= amd64_fam10h_table.cpu_clks;
		amd64_pmu.ret_inst	= amd64_fam10h_table.ret_inst;
		amd64_support.pmc_count	= PMU_AMD64_NUM_PERFSEL;
		amd64_support.pmd_count	= PMU_AMD64_NUM_PERFCTR;
		return;
	case 0x15:
		/* Family 15h */
		amd64_pmu.events	= amd64_fam15h_table.events;
		amd64_support.pme_count	= amd64_fam15h_table.num;
		amd64_pmu.cpu_clks	= amd64_fam15h_table.cpu_clks;
		amd64_pmu.ret_inst	= amd64_fam15h_table.ret_inst;
		amd64_support.num_cnt   = PMU_AMD64_NUM_COUNTERS_F15H;
		amd64_support.pmc_count	= PMU_AMD64_NUM_PERFSEL;
		amd64_support.pmd_count	= PMU_AMD64_NUM_PERFCTR;
		return;
	}
}

static int
pfm_amd64_detect(void)
{
	unsigned int a, b, c, d;
	char buffer[128];

	cpuid(0, &a, &b, &c, &d);
	strncpy(&buffer[0], (char *)(&b), 4);
	strncpy(&buffer[4], (char *)(&d), 4);
	strncpy(&buffer[8], (char *)(&c), 4);
	buffer[12] = '\0';

	if (strcmp(buffer, "AuthenticAMD"))
		return PFMLIB_ERR_NOTSUPP;

	cpuid(1, &a, &b, &c, &d);
	amd64_family = (a >> 8) & 0x0000000f;  // bits 11 - 8
	amd64_model  = (a >> 4) & 0x0000000f;  // Bits  7 - 4
	if (amd64_family == 0xf) {
		amd64_family += (a >> 20) & 0x000000ff; // Extended family
		amd64_model  |= (a >> 12) & 0x000000f0; // Extended model
	}
	amd64_stepping = a & 0x0000000f;  // bits  3 - 0

	amd64_revision = amd64_get_revision(amd64_family, amd64_model, amd64_stepping);

	if (amd64_revision == AMD64_CPU_UN)
		return PFMLIB_ERR_NOTSUPP;

	return PFMLIB_SUCCESS;
}

static void
pfm_amd64_force(void)
{
	char *str;
	int pmu_type;

	/* parses LIBPFM_FORCE_PMU=16,<family>,<model>,<stepping> */
	str = getenv("LIBPFM_FORCE_PMU");
	if (!str)
		goto failed;
	pmu_type = strtol(str, &str, 10);
	if (pmu_type != PFMLIB_AMD64_PMU)
		goto failed;
	if (!*str || *str++ != ',')
		goto failed;
	amd64_family = strtol(str, &str, 10);
	if (!*str || *str++ != ',')
		goto failed;
	amd64_model = strtol(str, &str, 10);
	if (!*str || *str++ != ',')
		goto failed;
	amd64_stepping = strtol(str, &str, 10);
	if (!*str)
		goto done;
failed:
	DPRINT("force failed at: %s\n", str ? str : "<NULL>");
	/* force AMD64 =  force to Barcelona */
	amd64_family = 16;
	amd64_model  = 2;
	amd64_stepping = 2;
done:
	amd64_revision = amd64_get_revision(amd64_family, amd64_model,
					    amd64_stepping);
}

static int
pfm_amd64_init(void)
{
	if (forced_pmu != PFMLIB_NO_PMU)
		pfm_amd64_force();

	__pfm_vbprintf("AMD family=%d model=0x%x stepping=0x%x rev=%s, %s\n",
		       amd64_family,
		       amd64_model,
		       amd64_stepping,
		       amd64_rev_strs[amd64_revision],
		       amd64_cpu_strs[amd64_revision]);

	pfm_amd64_setup(amd64_revision);

	return PFMLIB_SUCCESS;
}

static int
is_valid_rev(unsigned int flags, int revision)
{
	if (revision < from_revision(flags))
		return 0;

	if (revision > till_revision(flags))
		return 0;

	/* no restrictions or matches restrictions */
	return 1;
}

static inline pme_amd64_entry_t
*pfm_amd64_get_event_entry(unsigned int index)
{
	/*
	 * Since there are no NULL pointer checks for the return
	 * value, &unsupported_event is returned instead. Function
	 * is_valid_index() may be used to validate the index.
	 */
	pme_amd64_entry_t *event;
	if (index >= amd64_event_count)
		return &unsupported_event;
	event = &amd64_events[index];
	if (!is_valid_rev(event->pme_flags, amd64_revision))
		return &unsupported_event;
	return event;
}

static inline int
is_valid_index(unsigned int index)
{
	return (pfm_amd64_get_event_entry(index) != &unsupported_event);
}

/*
 * Automatically dispatch events to corresponding counters following constraints.
 */
static int
pfm_amd64_dispatch_counters(pfmlib_input_param_t *inp, pfmlib_amd64_input_param_t *mod_in, pfmlib_output_param_t *outp)
{
	pfmlib_amd64_input_param_t *param = mod_in;
	pfmlib_amd64_counter_t *cntrs;
	pfm_amd64_sel_reg_t reg;
	pfmlib_event_t *e;
	pfmlib_reg_t *pc, *pd;
	pfmlib_regmask_t *r_pmcs;
	unsigned long plm;
	unsigned int i, j, k, cnt, umask;
	unsigned int assign[PMU_AMD64_MAX_COUNTERS];

	e      = inp->pfp_events;
	pc     = outp->pfp_pmcs;
	pd     = outp->pfp_pmds;
	cnt    = inp->pfp_event_count;
	r_pmcs = &inp->pfp_unavail_pmcs;
	cntrs  = param ? param->pfp_amd64_counters : NULL;

	/* priviledge level 1 and 2 are not supported */
	if (inp->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		DPRINT("invalid plm=%x\n", inp->pfp_dfl_plm);
		return PFMLIB_ERR_INVAL;
	}

	if (PFMLIB_DEBUG()) {
		for (j=0; j < cnt; j++) {
			DPRINT("ev[%d]=%s\n", j, pfm_amd64_get_event_entry(e[j].event)->pme_name);
		}
	}

	if (cnt > amd64_support.num_cnt) return PFMLIB_ERR_TOOMANY;

	for(i=0, j=0; j < cnt; j++, i++) {
		/*
		 * AMD64 only supports two priv levels for perf counters
	 	 */
		if (e[j].plm & (PFM_PLM1|PFM_PLM2)) {
			DPRINT("event=%d invalid plm=%d\n", e[j].event, e[j].plm);
			return PFMLIB_ERR_INVAL;
		}
		/*
		 * check illegal unit masks combination
		 */
		if (e[j].num_masks > 1 && PFMLIB_AMD64_HAS_COMBO(e[j].event) == 0) {
			DPRINT("event does not supports unit mask combination\n");
			return PFMLIB_ERR_FEATCOMB;
		}

		/*
		 * check revision restrictions at the event level
		 * (check at the umask level later)
		 */
		if (!is_valid_rev(pfm_amd64_get_event_entry(e[i].event)->pme_flags,
				  amd64_revision)) {
			DPRINT("CPU does not have correct revision level\n");
			return PFMLIB_ERR_BADHOST;
		}
		if (cntrs && (cntrs[j].flags & ~PFMLIB_AMD64_ALL_FLAGS)) {
			DPRINT("invalid AMD64 flags\n");
			return PFMLIB_ERR_INVAL;
		}

		if (cntrs && (cntrs[j].cnt_mask >= PMU_AMD64_CNT_MASK_MAX)) {
			DPRINT("event=%d invalid cnt_mask=%d: must be < %u\n",
				e[j].event,
				cntrs[j].cnt_mask,
				PMU_AMD64_CNT_MASK_MAX);
			return PFMLIB_ERR_INVAL;
		}

		/*
		 * exclude unavailable registers from assignment
		 */
		while(i < amd64_support.num_cnt && pfm_regmask_isset(r_pmcs, i))
			i++;

		if (i == amd64_support.num_cnt)
			return PFMLIB_ERR_NOASSIGN;

		assign[j] = i;
	}

	for (j=0; j < cnt ; j++ ) {
		reg.val = 0; /* assume reserved bits are zerooed */

		/* if plm is 0, then assume not specified per-event and use default */
		plm = e[j].plm ? e[j].plm : inp->pfp_dfl_plm;

		if (!is_valid_rev(pfm_amd64_get_event_entry(e[j].event)->pme_flags,
					amd64_revision))
			return PFMLIB_ERR_BADHOST;

		reg.sel_event_mask  = pfm_amd64_get_event_entry(e[j].event)->pme_code;
		reg.sel_event_mask2 = pfm_amd64_get_event_entry(e[j].event)->pme_code >> 8;

		umask = 0;
		for(k=0; k < e[j].num_masks; k++) {
			/* check unit mask revision restrictions */
			if (!is_valid_rev(pfm_amd64_get_event_entry(e[j].event)->pme_umasks[e[j].unit_masks[k]].pme_uflags,
						amd64_revision))
				return PFMLIB_ERR_BADHOST;

			umask |= pfm_amd64_get_event_entry(e[j].event)->pme_umasks[e[j].unit_masks[k]].pme_ucode;
		}
		if (e[j].event == PME_AMD64_IBSOP) {
			ibsopctl_t ibsopctl;

			ibsopctl.val = 0;
			ibsopctl.reg.ibsopen = 1;

			if (umask == 2 && amd64_revision < from_revision(PFMLIB_AMD64_FAM10H_REV_C)) {             
                                DPRINT("IBSOP:UOPS available on Rev C and later processors\n");                    
                                return PFMLIB_ERR_BADHOST;
                        }

			/*
			 * 1: cycles
			 * 2: uops
			 */
			ibsopctl.reg.ibsopcntl = umask == 0x1 ? 0 : 1;

			pc[j].reg_value = ibsopctl.val;
			pc[j].reg_num   = PMU_AMD64_IBSOPCTL_PMC;
			pc[j].reg_addr  = 0xc0011033;

			__pfm_vbprintf("[IBSOPCTL(pmc%u)=0x%llx en=%d uops=%d maxcnt=0x%x]\n",
					PMU_AMD64_IBSOPCTL_PMD,
					ibsopctl.val,
					ibsopctl.reg.ibsopen,
					ibsopctl.reg.ibsopcntl,
					ibsopctl.reg.ibsopmaxcnt);

			pd[j].reg_num   = PMU_AMD64_IBSOPCTL_PMD;
			pd[j].reg_addr  = 0xc0011033;
			__pfm_vbprintf("[IBSOPCTL(pmd%u)]\n", PMU_AMD64_IBSOPCTL_PMD);

		} else if (e[j].event == PME_AMD64_IBSFETCH) {
			ibsfetchctl_t ibsfetchctl;

			ibsfetchctl.val = 0;
			ibsfetchctl.reg.ibsfetchen = 1;
			ibsfetchctl.reg.ibsranden = umask == 0x1 ? 1 : 0;

			pc[j].reg_value = ibsfetchctl.val;
			pc[j].reg_num   = PMU_AMD64_IBSFETCHCTL_PMC;
			pc[j].reg_addr  = 0xc0011031;

			pd[j].reg_num   = PMU_AMD64_IBSFETCHCTL_PMD;
			pd[j].reg_addr  = 0xc0011031;

			__pfm_vbprintf("[IBSFETCHCTL(pmc%u)=0x%llx en=%d maxcnt=0x%x rand=%u]\n",
					PMU_AMD64_IBSFETCHCTL_PMD,
					ibsfetchctl.val,
					ibsfetchctl.reg.ibsfetchen,
					ibsfetchctl.reg.ibsfetchmaxcnt,
					ibsfetchctl.reg.ibsranden);

			__pfm_vbprintf("[IBSOPFETCH(pmd%u)]\n", PMU_AMD64_IBSFETCHCTL_PMD);

		} else {
			reg.sel_unit_mask  = umask;
			reg.sel_usr        = plm & PFM_PLM3 ? 1 : 0;
			reg.sel_os         = plm & PFM_PLM0 ? 1 : 0;
			reg.sel_en         = 1; /* force enable bit to 1 */
			reg.sel_int        = 1; /* force APIC int to 1 */
			if (cntrs) {
				reg.sel_cnt_mask = cntrs[j].cnt_mask;
				reg.sel_edge     = cntrs[j].flags & PFM_AMD64_SEL_EDGE ? 1 : 0;
				reg.sel_inv      = cntrs[j].flags & PFM_AMD64_SEL_INV ? 1 : 0;
				reg.sel_guest    = cntrs[j].flags & PFM_AMD64_SEL_GUEST ? 1 : 0;
				reg.sel_host     = cntrs[j].flags & PFM_AMD64_SEL_HOST ? 1 : 0;
			}
			pc[j].reg_num   = assign[j];

			if ((CHECK_AMD_ARCH(reg)) && !IS_AMD_ARCH())
				return PFMLIB_ERR_BADHOST;

			if (amd64_support.num_cnt == PMU_AMD64_NUM_COUNTERS_F15H) {
				pc[j].reg_addr = AMD64_SEL_BASE_F15H + (assign[j] << 1);
				pd[j].reg_addr = AMD64_CTR_BASE_F15H + (assign[j] << 1);
			} else {
				pc[j].reg_addr = AMD64_SEL_BASE + assign[j];
				pd[j].reg_addr = AMD64_CTR_BASE + assign[j];
			}

			pc[j].reg_value = reg.val;
			pc[j].reg_alt_addr = pc[j].reg_addr;

			pd[j].reg_num  = assign[j];
			pd[j].reg_alt_addr = assign[j];		/* index to use with RDPMC */

			__pfm_vbprintf("[PERFSEL%u(pmc%u)=0x%llx emask=0x%x umask=0x%x os=%d usr=%d inv=%d en=%d int=%d edge=%d cnt_mask=%d] %s\n",
					assign[j],
					assign[j],
					reg.val,
					reg.sel_event_mask,
					reg.sel_unit_mask,
					reg.sel_os,
					reg.sel_usr,
					reg.sel_inv,
					reg.sel_en,
					reg.sel_int,
					reg.sel_edge,
					reg.sel_cnt_mask,
					pfm_amd64_get_event_entry(e[j].event)->pme_name);

			__pfm_vbprintf("[PERFCTR%u(pmd%u)]\n", pd[j].reg_num, pd[j].reg_num);
		}

	}
	/* number of evtsel/ctr registers programmed */
	outp->pfp_pmc_count = cnt;
	outp->pfp_pmd_count = cnt;

	return PFMLIB_SUCCESS;
}

static int pfm_amd64_dispatch_ibs(pfmlib_input_param_t *inp,
				  pfmlib_amd64_input_param_t *inp_mod,
				  pfmlib_output_param_t *outp,
				  pfmlib_amd64_output_param_t *outp_mod)
{
	unsigned int pmc_base, pmd_base;
	ibsfetchctl_t ibsfetchctl;
	ibsopctl_t ibsopctl;

	if (!inp_mod || !outp || !outp_mod)
		return PFMLIB_ERR_INVAL;

	if (!IS_AMD_ARCH())
		return PFMLIB_ERR_BADHOST;

	/* IBS fetch profiling */
	if (inp_mod->flags & PFMLIB_AMD64_USE_IBSFETCH) {

		/* check availability of a PMC and PMD */
		if (outp->pfp_pmc_count >= PFMLIB_MAX_PMCS)
			return PFMLIB_ERR_NOASSIGN;

		if (outp->pfp_pmd_count >= PFMLIB_MAX_PMDS)
			return PFMLIB_ERR_NOASSIGN;

		pmc_base = outp->pfp_pmc_count;
		pmd_base = outp->pfp_pmd_count;

		outp->pfp_pmcs[pmc_base].reg_num = PMU_AMD64_IBSFETCHCTL_PMC;

		ibsfetchctl.val = 0;
		ibsfetchctl.reg.ibsfetchen = 1;
		ibsfetchctl.reg.ibsfetchmaxcnt = inp_mod->ibsfetch.maxcnt >> 4;

		if (inp_mod->ibsfetch.options & IBS_OPTIONS_RANDEN)
			ibsfetchctl.reg.ibsranden = 1;

		outp->pfp_pmcs[pmc_base].reg_value = ibsfetchctl.val;
		outp->pfp_pmds[pmd_base].reg_num = PMU_AMD64_IBSFETCHCTL_PMD;
		outp_mod->ibsfetch_base = pmd_base;

		++outp->pfp_pmc_count;
		++outp->pfp_pmd_count;
	}

	/* IBS execution profiling */
	if (inp_mod->flags & PFMLIB_AMD64_USE_IBSOP) {

		/* check availability of a PMC and PMD */
		if (outp->pfp_pmc_count >= PFMLIB_MAX_PMCS)
			return PFMLIB_ERR_NOASSIGN;

		if (outp->pfp_pmd_count >= PFMLIB_MAX_PMDS)
			return PFMLIB_ERR_NOASSIGN;

		pmc_base = outp->pfp_pmc_count;
		pmd_base = outp->pfp_pmd_count;

		outp->pfp_pmcs[pmc_base].reg_num = PMU_AMD64_IBSOPCTL_PMC;

		ibsopctl.val = 0;
		ibsopctl.reg.ibsopen = 1;
		ibsopctl.reg.ibsopmaxcnt = inp_mod->ibsop.maxcnt >> 4;

		if (inp_mod->ibsop.options & IBS_OPTIONS_UOPS) {
			if (amd64_revision < from_revision(PFMLIB_AMD64_FAM10H_REV_C)) {
                                DPRINT("IBSOP:UOPS available on Rev C and later processors\n");                    
                                return PFMLIB_ERR_BADHOST;
                        }
			ibsopctl.reg.ibsopcntl = 1;
		}
		outp->pfp_pmcs[pmc_base].reg_value = ibsopctl.val;
		outp->pfp_pmds[pmd_base].reg_num = PMU_AMD64_IBSOPCTL_PMD;

		outp_mod->ibsop_base = pmd_base;
		++outp->pfp_pmc_count;
		++outp->pfp_pmd_count;
	}

	return PFMLIB_SUCCESS;
}

static int
pfm_amd64_dispatch_events(
	pfmlib_input_param_t *inp, void *_inp_mod,
	pfmlib_output_param_t *outp, void *outp_mod)
{
	pfmlib_amd64_input_param_t *inp_mod = _inp_mod;
	int ret = PFMLIB_ERR_INVAL;

	if (!outp)
		return PFMLIB_ERR_INVAL;

	/*
	 * At least one of the dispatch function calls must return
	 * PFMLIB_SUCCESS
	 */

	if (inp && inp->pfp_event_count) {
		ret = pfm_amd64_dispatch_counters(inp, inp_mod, outp);
		if (ret != PFMLIB_SUCCESS)
			return ret;
	}

	if (inp_mod && inp_mod->flags & (PFMLIB_AMD64_USE_IBSOP | PFMLIB_AMD64_USE_IBSFETCH))
		ret = pfm_amd64_dispatch_ibs(inp, inp_mod, outp, outp_mod);

	return ret;
}

static int
pfm_amd64_get_event_code(unsigned int i, unsigned int cnt, int *code)
{
	if (cnt != PFMLIB_CNT_FIRST && cnt >= amd64_support.num_cnt)
		return PFMLIB_ERR_INVAL;

	*code = pfm_amd64_get_event_entry(i)->pme_code;

	return PFMLIB_SUCCESS;
}

/*
 * This function is accessible directly to the user
 */
int
pfm_amd64_get_event_umask(unsigned int i, unsigned long *umask)
{
	if (i >= amd64_event_count || umask == NULL) return PFMLIB_ERR_INVAL;
	*umask = 0; //evt_umask(i);
	return PFMLIB_SUCCESS;
}
	
static void
pfm_amd64_get_event_counters(unsigned int j, pfmlib_regmask_t *counters)
{
	unsigned int i;

	memset(counters, 0, sizeof(*counters));

	for(i=0; i < amd64_support.num_cnt; i++)
		pfm_regmask_set(counters, i);
}

static void
pfm_amd64_get_impl_perfsel(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i = 0;

	/* all pmcs are contiguous */
	for(i=0; i < amd64_support.pmc_count; i++)
		pfm_regmask_set(impl_pmcs, i);
}

static void
pfm_amd64_get_impl_perfctr(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i = 0;

	/* all pmds are contiguous */
	for(i=0; i < amd64_support.pmd_count; i++)
		pfm_regmask_set(impl_pmds, i);
}

static void
pfm_amd64_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	unsigned int i = 0;

	/* counting pmds are contiguous */
	for(i=0; i < amd64_support.num_cnt; i++)
		pfm_regmask_set(impl_counters, i);
}

static void
pfm_amd64_get_hw_counter_width(unsigned int *width)
{
	*width = PMU_AMD64_COUNTER_WIDTH;
}

static char *
pfm_amd64_get_event_name(unsigned int i)
{
	if (!is_valid_index(i))
		return NULL;
	return pfm_amd64_get_event_entry(i)->pme_name;
}

static int
pfm_amd64_get_event_desc(unsigned int ev, char **str)
{
	char *s;
	s = pfm_amd64_get_event_entry(ev)->pme_desc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static char *
pfm_amd64_get_event_mask_name(unsigned int ev, unsigned int midx)
{
	pme_amd64_umask_t *umask;
	umask = &pfm_amd64_get_event_entry(ev)->pme_umasks[midx];
	if (!is_valid_rev(umask->pme_uflags, amd64_revision))
		return NULL;
	return umask->pme_uname;
}

static int
pfm_amd64_get_event_mask_desc(unsigned int ev, unsigned int midx, char **str)
{
	char *s;

	s = pfm_amd64_get_event_entry(ev)->pme_umasks[midx].pme_udesc;
	if (s) {
		*str = strdup(s);
	} else {
		*str = NULL;
	}
	return PFMLIB_SUCCESS;
}

static unsigned int
pfm_amd64_get_num_event_masks(unsigned int ev)
{
	return pfm_amd64_get_event_entry(ev)->pme_numasks;
}

static int
pfm_amd64_get_event_mask_code(unsigned int ev, unsigned int midx, unsigned int *code)
{
	*code = pfm_amd64_get_event_entry(ev)->pme_umasks[midx].pme_ucode;
	return PFMLIB_SUCCESS;
}

static int
pfm_amd64_get_cycle_event(pfmlib_event_t *e)
{
	e->event = amd64_cpu_clks;
	return PFMLIB_SUCCESS;

}

static int
pfm_amd64_get_inst_retired(pfmlib_event_t *e)
{
	e->event = amd64_ret_inst;
	return PFMLIB_SUCCESS;
}

pfm_pmu_support_t amd64_support = {
	.pmu_name		= "AMD64",
	.pmu_type		= PFMLIB_AMD64_PMU,
	.pme_count		= 0,
	.pmc_count		= PMU_AMD64_NUM_COUNTERS,
	.pmd_count		= PMU_AMD64_NUM_COUNTERS,
	.num_cnt		= PMU_AMD64_NUM_COUNTERS,
	.get_event_code		= pfm_amd64_get_event_code,
	.get_event_name		= pfm_amd64_get_event_name,
	.get_event_counters	= pfm_amd64_get_event_counters,
	.dispatch_events	= pfm_amd64_dispatch_events,
	.pmu_detect		= pfm_amd64_detect,
	.pmu_init		= pfm_amd64_init,
	.get_impl_pmcs		= pfm_amd64_get_impl_perfsel,
	.get_impl_pmds		= pfm_amd64_get_impl_perfctr,
	.get_impl_counters	= pfm_amd64_get_impl_counters,
	.get_hw_counter_width	= pfm_amd64_get_hw_counter_width,
	.get_event_desc         = pfm_amd64_get_event_desc,
	.get_num_event_masks	= pfm_amd64_get_num_event_masks,
	.get_event_mask_name	= pfm_amd64_get_event_mask_name,
	.get_event_mask_code	= pfm_amd64_get_event_mask_code,
	.get_event_mask_desc	= pfm_amd64_get_event_mask_desc,
	.get_cycle_event	= pfm_amd64_get_cycle_event,
	.get_inst_retired_event = pfm_amd64_get_inst_retired
};
