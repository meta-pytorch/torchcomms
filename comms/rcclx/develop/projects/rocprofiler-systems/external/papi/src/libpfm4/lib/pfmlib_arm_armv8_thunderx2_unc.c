/*
 * pfmlib_arm_armv8_thunderx2_unc.c : support for Marvell ThunderX2 uncore PMUs
 *
 * Copyright (c) 2024 Google Inc. All rights reserved
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
 */
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>

/* private headers */
#include "pfmlib_priv.h"			/* library private */
#include "pfmlib_arm_priv.h"

#include "pfmlib_arm_armv8_unc_priv.h"
#include "events/arm_marvell_tx2_unc_events.h" 	/* Marvell ThunderX2 PMU tables */

static int
pfm_arm_detect_thunderx2(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x42) && /* Broadcom */
		(pfm_arm_cfg.part == 0x516)) { /* Thunder2x */
			return PFM_SUCCESS;
	}
	if ((pfm_arm_cfg.implementer == 0x43) && /* Cavium */
		(pfm_arm_cfg.part == 0xaf)) { /* Thunder2x */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

static int
pfm_tx2_unc_get_event_encoding(void *this, pfmlib_event_desc_t *e)
{
	//from pe field in for the uncore, get the array with all the event defs
	const arm_entry_t *event_list = this_pe(this);
	tx2_unc_data_t reg;

	//get code for the event from the table
	reg.val = event_list[e->event].code;

	//pass the data back to the caller
	e->codes[0] = reg.val;
	e->count = 1;

	evt_strcat(e->fstr, "%s", event_list[e->event].name);

	return PFM_SUCCESS;
}

// For uncore, each socket has a separate perf name, otherwise they are the same, use macro

#define DEFINE_TX2_DMC(n) \
pfmlib_pmu_t arm_thunderx2_dmc##n##_support={ \
	.desc			= "Marvell ThunderX2 Node"#n" DMC", \
	.name			= "tx2_dmc"#n, \
	.perf_name		= "uncore_dmc_"#n, \
	.pmu			= PFM_PMU_ARM_THUNDERX2_DMC##n, \
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_thunderx2_unc_dmc_pe), \
	.type			= PFM_PMU_TYPE_UNCORE, \
	.pe			= arm_thunderx2_unc_dmc_pe, \
	.pmu_detect		= pfm_arm_detect_thunderx2, \
	.max_encoding		= 1, \
	.num_cntrs		= 4, \
	.get_event_encoding[PFM_OS_NONE] = pfm_tx2_unc_get_event_encoding, \
	 PFMLIB_ENCODE_PERF(pfm_tx2_unc_get_perf_encoding),		\
	.get_event_first	= pfm_arm_get_event_first, \
	.get_event_next		= pfm_arm_get_event_next,  \
	.event_is_valid		= pfm_arm_event_is_valid,  \
	.validate_table		= pfm_arm_validate_table,  \
	.get_event_info		= pfm_arm_get_event_info,  \
	.get_event_attr_info	= pfm_arm_get_event_attr_info,	\
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),\
	.get_event_nattrs	= pfm_arm_get_event_nattrs, \
};

DEFINE_TX2_DMC(0);
DEFINE_TX2_DMC(1);

#define DEFINE_TX2_LLC(n) \
pfmlib_pmu_t arm_thunderx2_llc##n##_support={ \
	.desc			= "Marvell ThunderX2 node "#n" LLC", \
	.name			= "tx2_llc"#n, \
	.perf_name		= "uncore_l3c_"#n, \
	.pmu			= PFM_PMU_ARM_THUNDERX2_LLC##n, \
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_thunderx2_unc_llc_pe), \
	.type			= PFM_PMU_TYPE_UNCORE, \
	.pe			= arm_thunderx2_unc_llc_pe, \
	.pmu_detect		= pfm_arm_detect_thunderx2, \
	.max_encoding		= 1, \
	.num_cntrs		= 4, \
	.get_event_encoding[PFM_OS_NONE] = pfm_tx2_unc_get_event_encoding, \
	 PFMLIB_ENCODE_PERF(pfm_tx2_unc_get_perf_encoding),		\
	.get_event_first	= pfm_arm_get_event_first, \
	.get_event_next		= pfm_arm_get_event_next,  \
	.event_is_valid		= pfm_arm_event_is_valid,  \
	.validate_table		= pfm_arm_validate_table,  \
	.get_event_info		= pfm_arm_get_event_info,  \
	.get_event_attr_info	= pfm_arm_get_event_attr_info,	\
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),\
	.get_event_nattrs	= pfm_arm_get_event_nattrs, \
};

DEFINE_TX2_LLC(0);
DEFINE_TX2_LLC(1);

#define DEFINE_TX2_CCPI(n) \
pfmlib_pmu_t arm_thunderx2_ccpi##n##_support={ \
	.desc			= "Marvell ThunderX2 node "#n" Cross-Socket Interconnect", \
	.name			= "tx2_ccpi"#n, \
	.perf_name		= "uncore_ccpi2_"#n, \
	.pmu			= PFM_PMU_ARM_THUNDERX2_CCPI##n, \
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_thunderx2_unc_ccpi_pe), \
	.type			= PFM_PMU_TYPE_UNCORE, \
	.pe			= arm_thunderx2_unc_ccpi_pe, \
	.pmu_detect		= pfm_arm_detect_thunderx2, \
	.max_encoding		= 1, \
	.num_cntrs		= 4, \
	.get_event_encoding[PFM_OS_NONE] = pfm_tx2_unc_get_event_encoding, \
	 PFMLIB_ENCODE_PERF(pfm_tx2_unc_get_perf_encoding),		\
	.get_event_first	= pfm_arm_get_event_first, \
	.get_event_next		= pfm_arm_get_event_next,  \
	.event_is_valid		= pfm_arm_event_is_valid,  \
	.validate_table		= pfm_arm_validate_table,  \
	.get_event_info		= pfm_arm_get_event_info,  \
	.get_event_attr_info	= pfm_arm_get_event_attr_info,	\
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),\
	.get_event_nattrs	= pfm_arm_get_event_nattrs, \
};

DEFINE_TX2_CCPI(0);
DEFINE_TX2_CCPI(1);
