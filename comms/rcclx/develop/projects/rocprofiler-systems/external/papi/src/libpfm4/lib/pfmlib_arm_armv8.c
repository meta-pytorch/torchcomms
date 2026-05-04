/*
 * pfmlib_arm_armv8.c : support for ARMv8 processors
 *
 * Copyright (c) 2014 Google Inc. All rights reserved
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
 * Contributed by John Linford <jlinford@nvidia.com>
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

#include "events/arm_cortex_a57_events.h"    /* A57 event tables */
#include "events/arm_cortex_a53_events.h"    /* A53 event tables */
#include "events/arm_cortex_a55_events.h"    /* A53 event tables */
#include "events/arm_cortex_a76_events.h"    /* A76 event tables */
#include "events/arm_xgene_events.h"         /* Applied Micro X-Gene tables */
#include "events/arm_cavium_tx2_events.h"    	/* Marvell ThunderX2 tables */
#include "events/arm_fujitsu_a64fx_events.h"	/* Fujitsu A64FX PMU tables */
#include "events/arm_neoverse_n1_events.h"	/* ARM Neoverse N1 table */
#include "events/arm_neoverse_v1_events.h"	/* Arm Neoverse V1 table */
#include "events/arm_hisilicon_kunpeng_events.h" /* HiSilicon Kunpeng PMU tables */

static int
pfm_arm_detect_n1(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x41) && /* ARM */
		(pfm_arm_cfg.part == 0xd0c)) { /* Neoverse N1 */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

static int
pfm_arm_detect_v1(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x41) && /* ARM */
		(pfm_arm_cfg.part == 0xd40)) { /* Neoverse V1 */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

static int
pfm_arm_detect_cortex_a57(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x41) && /* ARM */
		(pfm_arm_cfg.part == 0xd07)) { /* Cortex A57 */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

static int
pfm_arm_detect_cortex_a72(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x41) && /* ARM */
		(pfm_arm_cfg.part == 0xd08)) { /* Cortex A57 */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

static int
pfm_arm_detect_cortex_a53(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x41) && /* ARM */
		(pfm_arm_cfg.part == 0xd03)) { /* Cortex A53 */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

static int
pfm_arm_detect_cortex_a55(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x41) && /* ARM */
		(pfm_arm_cfg.part == 0xd05)) { /* Cortex A55 */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

static int
pfm_arm_detect_cortex_a76(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x41) && /* ARM */
		(pfm_arm_cfg.part == 0xd0b)) { /* Cortex A76 */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

static int
pfm_arm_detect_xgene(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x50) && /* Applied Micro */
		(pfm_arm_cfg.part == 0x000)) { /* Applied Micro X-Gene */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

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
pfm_arm_detect_a64fx(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x46) && /* Fujitsu */
		(pfm_arm_cfg.part == 0x001)) { /* a64fx */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

static int
pfm_arm_detect_hisilicon_kunpeng(void *this)
{
	int ret;

	ret = pfm_arm_detect(this);
	if (ret != PFM_SUCCESS)
		return PFM_ERR_NOTSUPP;

	if ((pfm_arm_cfg.implementer == 0x48) && /* Hisilicon */
	    (pfm_arm_cfg.part == 0xd01)) { /* Kunpeng */
			return PFM_SUCCESS;
	}
	return PFM_ERR_NOTSUPP;
}

/* ARM Cortex A57 support */
pfmlib_pmu_t arm_cortex_a57_support={
	.desc			= "ARM Cortex A57",
	.name			= "arm_ac57",
	.pmu			= PFM_PMU_ARM_CORTEX_A57,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_cortex_a57_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV8_PLM,
	.pe			= arm_cortex_a57_pe,

	.pmu_detect		= pfm_arm_detect_cortex_a57,
	.max_encoding		= 1,
	.num_cntrs		= 6,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first	= pfm_arm_get_event_first,
	.get_event_next		= pfm_arm_get_event_next,
	.event_is_valid		= pfm_arm_event_is_valid,
	.validate_table		= pfm_arm_validate_table,
	.get_event_info		= pfm_arm_get_event_info,
	.get_event_attr_info	= pfm_arm_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs	= pfm_arm_get_event_nattrs,
};

/* ARM Cortex A72 support */
pfmlib_pmu_t arm_cortex_a72_support={
	.desc			= "ARM Cortex A72",
	.name			= "arm_ac72",
	.pmu			= PFM_PMU_ARM_CORTEX_A72,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_cortex_a57_pe), /* shared with a57 */
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV8_PLM,
	.pe			= arm_cortex_a57_pe, /* shared with a57 */

	.pmu_detect		= pfm_arm_detect_cortex_a72,
	.max_encoding		= 1,
	.num_cntrs		= 6,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first	= pfm_arm_get_event_first,
	.get_event_next		= pfm_arm_get_event_next,
	.event_is_valid		= pfm_arm_event_is_valid,
	.validate_table		= pfm_arm_validate_table,
	.get_event_info		= pfm_arm_get_event_info,
	.get_event_attr_info	= pfm_arm_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs	= pfm_arm_get_event_nattrs,
};

/* ARM Cortex A53 support */
pfmlib_pmu_t arm_cortex_a53_support={
	.desc			= "ARM Cortex A53",
	.name			= "arm_ac53",
	.pmu			= PFM_PMU_ARM_CORTEX_A53,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_cortex_a53_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV8_PLM,
	.pe			= arm_cortex_a53_pe,

	.pmu_detect		= pfm_arm_detect_cortex_a53,
	.max_encoding		= 1,
	.num_cntrs		= 6,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first	= pfm_arm_get_event_first,
	.get_event_next		= pfm_arm_get_event_next,
	.event_is_valid		= pfm_arm_event_is_valid,
	.validate_table		= pfm_arm_validate_table,
	.get_event_info		= pfm_arm_get_event_info,
	.get_event_attr_info	= pfm_arm_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs	= pfm_arm_get_event_nattrs,
};

/* ARM Cortex A55 support */
pfmlib_pmu_t arm_cortex_a55_support={
	.desc			= "ARM Cortex A55",
	.name			= "arm_ac55",
	.perf_name		= "armv8_cortex_a55",
	.pmu			= PFM_PMU_ARM_CORTEX_A55,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_cortex_a55_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV8_PLM,
	.pe			= arm_cortex_a55_pe,

	.pmu_detect		= pfm_arm_detect_cortex_a55,
	.max_encoding		= 1,
	.num_cntrs		= 6,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first	= pfm_arm_get_event_first,
	.get_event_next		= pfm_arm_get_event_next,
	.event_is_valid		= pfm_arm_event_is_valid,
	.validate_table		= pfm_arm_validate_table,
	.get_event_info		= pfm_arm_get_event_info,
	.get_event_attr_info	= pfm_arm_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs	= pfm_arm_get_event_nattrs,
};

/* ARM Cortex A76 support */
pfmlib_pmu_t arm_cortex_a76_support={
	.desc			= "ARM Cortex A76",
	.name			= "arm_ac76",
	.perf_name		= "armv8_cortex_a76",
	.pmu			= PFM_PMU_ARM_CORTEX_A76,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_cortex_a76_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV8_PLM,
	.pe			= arm_cortex_a76_pe,

	.pmu_detect		= pfm_arm_detect_cortex_a76,
	.max_encoding		= 1,
	.num_cntrs		= 6,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first	= pfm_arm_get_event_first,
	.get_event_next		= pfm_arm_get_event_next,
	.event_is_valid		= pfm_arm_event_is_valid,
	.validate_table		= pfm_arm_validate_table,
	.get_event_info		= pfm_arm_get_event_info,
	.get_event_attr_info	= pfm_arm_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs	= pfm_arm_get_event_nattrs,
};

/* Applied Micro X-Gene support */
pfmlib_pmu_t arm_xgene_support={
	.desc			= "Applied Micro X-Gene",
	.name			= "arm_xgene",
	.pmu			= PFM_PMU_ARM_XGENE,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_xgene_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV8_PLM,
	.pe			= arm_xgene_pe,

	.pmu_detect		= pfm_arm_detect_xgene,
	.max_encoding		= 1,
	.num_cntrs		= 4,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first	= pfm_arm_get_event_first,
	.get_event_next		= pfm_arm_get_event_next,
	.event_is_valid		= pfm_arm_event_is_valid,
	.validate_table		= pfm_arm_validate_table,
	.get_event_info		= pfm_arm_get_event_info,
	.get_event_attr_info	= pfm_arm_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs	= pfm_arm_get_event_nattrs,
};

/* Marvell ThunderX2 support */
pfmlib_pmu_t arm_thunderx2_support={
	.desc			= "Cavium ThunderX2",
	.name			= "arm_thunderx2",
	.pmu			= PFM_PMU_ARM_THUNDERX2,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_thunderx2_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV8_PLM,
	.pe			= arm_thunderx2_pe,

	.pmu_detect		= pfm_arm_detect_thunderx2,
	.max_encoding		= 1,
	.num_cntrs		= 6,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first	= pfm_arm_get_event_first,
	.get_event_next		= pfm_arm_get_event_next,
	.event_is_valid		= pfm_arm_event_is_valid,
	.validate_table		= pfm_arm_validate_table,
	.get_event_info		= pfm_arm_get_event_info,
	.get_event_attr_info	= pfm_arm_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs	= pfm_arm_get_event_nattrs,
};

/* Fujitsu A64FX support */
pfmlib_pmu_t arm_fujitsu_a64fx_support={
	.desc			= "Fujitsu A64FX",
	.name			= "arm_a64fx",
	.pmu			= PFM_PMU_ARM_A64FX,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_a64fx_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV8_PLM,
	.pe			= arm_a64fx_pe,

	.pmu_detect		= pfm_arm_detect_a64fx,
	.max_encoding		= 1,
	.num_cntrs		= 8,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first	= pfm_arm_get_event_first,
	.get_event_next		= pfm_arm_get_event_next,
	.event_is_valid		= pfm_arm_event_is_valid,
	.validate_table		= pfm_arm_validate_table,
	.get_event_info		= pfm_arm_get_event_info,
	.get_event_attr_info	= pfm_arm_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs	= pfm_arm_get_event_nattrs,
};

/* HiSilicon Kunpeng support */
pfmlib_pmu_t arm_hisilicon_kunpeng_support={
	.desc           = "Hisilicon Kunpeng",
	.name           = "arm_kunpeng",
	.pmu            = PFM_PMU_ARM_KUNPENG,
	.pme_count      = LIBPFM_ARRAY_SIZE(arm_kunpeng_pe),
	.type           = PFM_PMU_TYPE_CORE,
	.supported_plm  = ARMV8_PLM,
	.pe             = arm_kunpeng_pe,
	.pmu_detect     = pfm_arm_detect_hisilicon_kunpeng,
	.max_encoding   = 1,
	.num_cntrs      = 12,
	.num_fixed_cntrs      = 1,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first        = pfm_arm_get_event_first,
	.get_event_next         = pfm_arm_get_event_next,
	.event_is_valid         = pfm_arm_event_is_valid,
	.validate_table         = pfm_arm_validate_table,
	.get_event_info         = pfm_arm_get_event_info,
	.get_event_attr_info    = pfm_arm_get_event_attr_info,
	PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs       = pfm_arm_get_event_nattrs,
};

pfmlib_pmu_t arm_n1_support={
	.desc			= "ARM Neoverse N1",
	.name			= "arm_n1",
	.pmu			= PFM_PMU_ARM_N1,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_n1_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV8_PLM,
	.pe			= arm_n1_pe,

	.pmu_detect		= pfm_arm_detect_n1,
	.max_encoding		= 1,
	.num_cntrs		= 6,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first	= pfm_arm_get_event_first,
	.get_event_next		= pfm_arm_get_event_next,
	.event_is_valid		= pfm_arm_event_is_valid,
	.validate_table		= pfm_arm_validate_table,
	.get_event_info		= pfm_arm_get_event_info,
	.get_event_attr_info	= pfm_arm_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs	= pfm_arm_get_event_nattrs,
};

pfmlib_pmu_t arm_v1_support={
	.desc			= "Arm Neoverse V1",
	.name			= "arm_v1",
	.pmu			= PFM_PMU_ARM_V1,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_v1_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm  = ARMV8_PLM,
	.pe             = arm_v1_pe,

	.pmu_detect		= pfm_arm_detect_v1,
	.max_encoding		= 1,
	.num_cntrs		= 6,

	.get_event_encoding[PFM_OS_NONE] = pfm_arm_get_encoding,
	 PFMLIB_ENCODE_PERF(pfm_arm_get_perf_encoding),
	.get_event_first	= pfm_arm_get_event_first,
	.get_event_next		= pfm_arm_get_event_next,
	.event_is_valid		= pfm_arm_event_is_valid,
	.validate_table		= pfm_arm_validate_table,
	.get_event_info		= pfm_arm_get_event_info,
	.get_event_attr_info	= pfm_arm_get_event_attr_info,
	 PFMLIB_VALID_PERF_PATTRS(pfm_arm_perf_validate_pattrs),
	.get_event_nattrs	= pfm_arm_get_event_nattrs,
};
