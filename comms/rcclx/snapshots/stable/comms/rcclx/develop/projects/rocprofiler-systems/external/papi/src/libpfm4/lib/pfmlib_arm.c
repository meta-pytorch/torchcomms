/*
 * pfmlib_arm.c : 	support for ARM chips
 * 
 * Copyright (c) 2010 University of Tennessee
 * Contributed by Vince Weaver <vweaver1@utk.edu>
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
 */

#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_arm_priv.h"

const pfmlib_attr_desc_t arm_mods[]={
	PFM_ATTR_B("k", "monitor at kernel level"),
	PFM_ATTR_B("u", "monitor at user level"),
	PFM_ATTR_B("hv", "monitor in hypervisor"),
	PFM_ATTR_NULL /* end-marker to avoid exporting number of entries */
};

pfm_arm_config_t pfm_arm_cfg = {
	.init_cpuinfo_done = 0,
};

#define MAX_ARM_CPUIDS	8

static arm_cpuid_t arm_cpuids[MAX_ARM_CPUIDS];
static int num_arm_cpuids;

static int pfmlib_find_arm_cpuid(arm_cpuid_t *attr, arm_cpuid_t *match_attr)
{
	int i;

	if (attr == NULL)
		return PFM_ERR_NOTFOUND;

	for (i=0; i < num_arm_cpuids; i++) {
#if 0
/*
 * disabled due to issues with expected arch vs. reported
 * arch by the Linux kernel cpuinfo
 */
		if (arm_cpuids[i].arch != attr->arch)
			continue;
#endif
		if (arm_cpuids[i].impl != attr->impl)
			continue;
		if (arm_cpuids[i].part != attr->part)
			continue;
		if (match_attr)
			*match_attr = arm_cpuids[i];
		return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

#ifdef CONFIG_PFMLIB_OS_LINUX
/*
 * Function populates the arm_cpuidsp[] table with each unique
 * core identifications found on the host. In the case of hybrids
 * that number is greater than 1
 */
static int
pfmlib_init_cpuids(void)
{
	arm_cpuid_t attr = {0, };
	FILE *fp = NULL;
	int ret = -1;
	size_t buf_len = 0;
	char *p, *value = NULL;
	char *buffer = NULL;
	int nattrs = 0;

	if (pfm_arm_cfg.init_cpuinfo_done == 1)
		return PFM_SUCCESS;

	fp = fopen(pfm_cfg.proc_cpuinfo, "r");
	if (fp == NULL) {
		DPRINT("pfmlib_init_cpuids: cannot open %s\n", pfm_cfg.proc_cpuinfo);
		return PFM_ERR_NOTFOUND;
	}

	while(pfmlib_getl(&buffer, &buf_len, fp) != -1){
		if (nattrs == ARM_NUM_ATTR_FIELDS) {
			if (pfmlib_find_arm_cpuid(&attr, NULL) != PFM_SUCCESS) {
				/* must add */
				if (num_arm_cpuids == MAX_ARM_CPUIDS) {
					DPRINT("pfmlib_init_cpuids: too many cpuids num_arm_cpuids=%d\n", num_arm_cpuids);
					ret = PFM_ERR_TOOMANY;
					goto error;
				}
				arm_cpuids[num_arm_cpuids++] = attr;
				__pfm_vbprintf("Detected ARM CPU impl=0x%x arch=%d part=0x%x\n", attr.impl, attr.arch, attr.part);
			}
			nattrs = 0;
		}

		/* skip  blank lines */
		if (*buffer == '\n' || *buffer == '\r')
			continue;

		p = strchr(buffer, ':');
		if (p == NULL)
			continue;

		/*
		 * p+2: +1 = space, +2= firt character
		 * strlen()-1 gets rid of \n
		 */
		*p = '\0';
		value = p+2;

		value[strlen(value)-1] = '\0';

		if (!strncmp("CPU implementer", buffer, 15)) {
			attr.impl = strtoul(value, NULL, 0);
			nattrs++;
			continue;
		}
		if (!strncmp("CPU architecture", buffer, 16)) {
			attr.arch = strtoul(value, NULL, 0);
			nattrs++;
			continue;
		}
		if (!strncmp("CPU part", buffer, 8)) {
			attr.part = strtoul(value, NULL, 0);
			nattrs++;
			continue;
		}
	}
	ret = PFM_SUCCESS;
	DPRINT("num_arm_cpuids=%d\n", num_arm_cpuids);
error:
	for (nattrs = 0; nattrs < num_arm_cpuids; nattrs++) {
		DPRINT("cpuids[%d] = impl=0x%x arch=%d part=0x%x\n", nattrs, arm_cpuids[nattrs].impl, arm_cpuids[nattrs].arch, arm_cpuids[nattrs].part);
	}
	pfm_arm_cfg.init_cpuinfo_done = 1;

	free(buffer);
	fclose(fp);

	return ret;
}
#else
static int
pfmlib_init_cpuids(void)
{
	return -1;
}
#endif

static int
arm_num_mods(void *this, int idx)
{
	const arm_entry_t *pe = this_pe(this);
	unsigned int mask;

	mask = pe[idx].modmsk;
	return pfmlib_popcnt(mask);
}

static inline int
arm_attr2mod(void *this, int pidx, int attr_idx)
{
	const arm_entry_t *pe = this_pe(this);
	size_t x;
	int n;

	n = attr_idx;

	pfmlib_for_each_bit(x, pe[pidx].modmsk) {
		if (n == 0)
			break;
		n--;
	}
	return x;
}

static void
pfm_arm_display_reg(void *this, pfmlib_event_desc_t *e, pfm_arm_reg_t reg)
{
	__pfm_vbprintf("[0x%x] %s\n", reg.val, e->fstr);
}

int
pfm_arm_detect(arm_cpuid_t *attr, arm_cpuid_t *match_attr)
{
	int ret;

	ret = pfmlib_init_cpuids();
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	return pfmlib_find_arm_cpuid(attr, match_attr);
}

int
pfm_arm_get_encoding(void *this, pfmlib_event_desc_t *e)
{

	const arm_entry_t *pe = this_pe(this);
	pfmlib_event_attr_info_t *a;
	pfm_arm_reg_t reg;
	unsigned int plm = 0;
	int i, idx, has_plm = 0;

	reg.val = pe[e->event].code;
  

	for (i = 0; i < e->nattrs; i++) {
		a = attr(e, i);

		if (a->ctrl != PFM_ATTR_CTRL_PMU)
			continue;

		if (a->type > PFM_ATTR_UMASK) {
			uint64_t ival = e->attrs[i].ival;

			switch(a->idx) {
				case ARM_ATTR_U: /* USR */
					if (ival)
						plm |= PFM_PLM3;
					has_plm = 1;
					break;
				case ARM_ATTR_K: /* OS */
					if (ival)
						plm |= PFM_PLM0;
					has_plm = 1;
					break;
				case ARM_ATTR_HV: /* HYPERVISOR */
					if (ival)
						plm |= PFM_PLMH;
					has_plm = 1;
					break;
				default:
					return PFM_ERR_ATTR;
			}
		}
	}

	if (arm_has_plm(this, e)) {
		if (!has_plm)
			plm = e->dfl_plm;
		reg.evtsel.excl_pl1 = !(plm & PFM_PLM0);
		reg.evtsel.excl_usr = !(plm & PFM_PLM3);
		reg.evtsel.excl_hyp = !(plm & PFM_PLMH);
	}

        evt_strcat(e->fstr, "%s", pe[e->event].name);

	e->codes[0] = reg.val;
	e->count    = 1;

	for (i = 0; i < e->npattrs; i++) {
		if (e->pattrs[i].ctrl != PFM_ATTR_CTRL_PMU)
			continue;

		if (e->pattrs[i].type == PFM_ATTR_UMASK)
			continue;

		idx = e->pattrs[i].idx;
		switch(idx) {
		case ARM_ATTR_K:
			evt_strcat(e->fstr, ":%s=%lu", arm_mods[idx].name, !reg.evtsel.excl_pl1);
			break;
		case ARM_ATTR_U:
			evt_strcat(e->fstr, ":%s=%lu", arm_mods[idx].name, !reg.evtsel.excl_usr);
			break;
		case ARM_ATTR_HV:
			evt_strcat(e->fstr, ":%s=%lu", arm_mods[idx].name, !reg.evtsel.excl_hyp);
			break;
		}
	}

        pfm_arm_display_reg(this, e, reg);
   
	return PFM_SUCCESS;
}

int
pfm_arm_get_event_first(void *this)
{
	return 0;
}

int
pfm_arm_get_event_next(void *this, int idx)
{
	pfmlib_pmu_t *p = this;

	if (idx >= (p->pme_count-1))
		return -1;

	return idx+1;
}

int
pfm_arm_event_is_valid(void *this, int pidx)
{
	pfmlib_pmu_t *p = this;
	return pidx >= 0 && pidx < p->pme_count;
}

int
pfm_arm_validate_table(void *this, FILE *fp)
{

	pfmlib_pmu_t *pmu = this;
	const arm_entry_t *pe = this_pe(this);
	int i, j, error = 0;

	for(i=0; i < pmu->pme_count; i++) {
		if (!pe[i].name) {
			fprintf(fp, "pmu: %s event%d: :: no name (prev event was %s)\n", pmu->name, i,
			i > 1 ? pe[i-1].name : "??");
			error++;
		}
		if (!pe[i].desc) {
			fprintf(fp, "pmu: %s event%d: %s :: no description\n", pmu->name, i, pe[i].name);
			error++;
		}
		for(j = i+1; j < pmu->pme_count; j++) {
			if (pe[i].code == pe[j].code && !(pe[j].equiv || pe[i].equiv))  {
				fprintf(fp, "pmu: %s events %s and %s have the same code 0x%x\n", pmu->name, pe[i].name, pe[j].name, pe[i].code);
				error++;
				}
		}
	}
	return error ? PFM_ERR_INVAL : PFM_SUCCESS;
}

int
pfm_arm_get_event_attr_info(void *this, int pidx, int attr_idx, pfmlib_event_attr_info_t *info)
{
	int idx;

	idx = arm_attr2mod(this, pidx, attr_idx);
	info->name = arm_mods[idx].name;
	info->desc = arm_mods[idx].desc;
	info->type = arm_mods[idx].type;
	info->code = idx;

	info->is_dfl = 0;
	info->equiv  = NULL;
	info->ctrl   = PFM_ATTR_CTRL_PMU;
	info->idx    = idx; /* namespace specific index */

	info->dfl_val64  = 0;
	info->is_precise = 0;
	info->support_hw_smpl = 0;

	return PFM_SUCCESS;
}

unsigned int
pfm_arm_get_event_nattrs(void *this, int pidx)
{
	return arm_num_mods(this, pidx);
}

int
pfm_arm_get_event_info(void *this, int idx, pfm_event_info_t *info)
{
	pfmlib_pmu_t *pmu = this;
	const arm_entry_t *pe = this_pe(this);

	info->name  = pe[idx].name;
	info->desc  = pe[idx].desc;
	info->code  = pe[idx].code;
	info->equiv = pe[idx].equiv;
	info->idx   = idx; /* private index */
	info->pmu   = pmu->pmu;
	info->is_precise = 0;
	info->support_hw_smpl = 0;

	/* no attributes defined for ARM yet */
	info->nattrs  = 0;

	return PFM_SUCCESS;
}
