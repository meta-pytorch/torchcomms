/*
 * pfmlib_arm_armv9.c : support for ARMv9 processors
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

#include "events/arm_neoverse_n2_events.h"	/* Arm Neoverse N2 table */
#include "events/arm_neoverse_n3_events.h"	/* Arm Neoverse N3 table */
#include "events/arm_neoverse_v2_events.h"	/* Arm Neoverse V2 table */
#include "events/arm_neoverse_v3_events.h"	/* Arm Neoverse V3 table */
#include "events/arm_fujitsu_monaka_events.h"	/* Fujitsu FUJITSU-MONAKA PMU tables */
#include "events/arm_cortex_x4_events.h"	/* Arm Cortex X4 table */

static int
pfm_arm_detect_n2(void *this)
{
	/* ARM Neoverse N2 */
	arm_cpuid_t attr = { .impl = 0x41, .arch = 9, .part = 0xd49 };

	return pfm_arm_detect(&attr, NULL);
}

static int
pfm_arm_detect_n3(void *this)
{
	/* ARM Neoverse N3 */
	arm_cpuid_t attr = { .impl = 0x41, .arch = 9, .part = 0xd8e };

	return pfm_arm_detect(&attr, NULL);
}

static int
pfm_arm_detect_v2(void *this)
{
	/* ARM Neoverse V2 */
	arm_cpuid_t attr = { .impl = 0x41, .arch = 9, .part = 0xd4f };

	return pfm_arm_detect(&attr, NULL);
}

static int
pfm_arm_detect_v3(void *this)
{
	/* ARM Neoverse V3 */
	arm_cpuid_t attr = { .impl = 0x41, .arch = 9, .part = 0xd84 };

	return pfm_arm_detect(&attr, NULL);
}

static int
pfm_arm_detect_monaka(void *this)
{
	/* Fujitsu Monaka */
	arm_cpuid_t attr = { .impl = 0x46, .arch = 9, .part = 0x3 };

	return pfm_arm_detect(&attr, NULL);
}

static int
pfm_arm_detect_cortex_x4(void *this)
{
	/* ARM Cortex X4 */
	arm_cpuid_t attr = { .impl = 0x41, .arch = 9, .part = 0xd82 };

	return pfm_arm_detect(&attr, NULL);
}

pfmlib_pmu_t arm_n2_support={
	.desc			= "Arm Neoverse N2",
	.name			= "arm_n2",
	.pmu			= PFM_PMU_ARM_N2,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_n2_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV9_PLM,
	.pe			= arm_n2_pe,

	.pmu_detect		= pfm_arm_detect_n2,
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

pfmlib_pmu_t arm_n3_support={
	.desc			= "Arm Neoverse N3",
	.name			= "arm_n3",
	.pmu			= PFM_PMU_ARM_N3,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_n3_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm          = ARMV9_PLM,
	.pe			= arm_n3_pe,

	.pmu_detect		= pfm_arm_detect_n3,
	.max_encoding		= 1,
	.num_cntrs		= 6, /* or 20 */

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

pfmlib_pmu_t arm_v2_support={
	.desc			= "Arm Neoverse V2",
	.name			= "arm_v2",
	.pmu			= PFM_PMU_ARM_V2,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_v2_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm  = ARMV9_PLM,
	.pe				= arm_v2_pe,

	.pmu_detect		= pfm_arm_detect_v2,
	.max_encoding	= 1,
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

pfmlib_pmu_t arm_v3_support={
	.desc			= "Arm Neoverse V3",
	.name			= "arm_v3",
	.pmu			= PFM_PMU_ARM_V3,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_neoverse_v3_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm  = ARMV9_PLM,
	.pe				= arm_neoverse_v3_pe,

	.pmu_detect		= pfm_arm_detect_v3,
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

/* Fujitsu FUJITSU-MONAKA support */
pfmlib_pmu_t arm_fujitsu_monaka_support={
	.desc			= "Fujitsu FUJITSU-MONAKA",
	.name			= "arm_monaka",
	.pmu			= PFM_PMU_ARM_MONAKA,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_monaka_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm  = ARMV9_PLM,
	.pe				= arm_monaka_pe,

	.pmu_detect		= pfm_arm_detect_monaka,
	.max_encoding	= 1,
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

pfmlib_pmu_t arm_cortex_x4_support={
	.desc			= "ARM Cortex X4",
	.name			= "arm_x4",
	.perf_name              = "armv8_pmuv3_0,armv8_pmuv3",
	.pmu			= PFM_PMU_ARM_CORTEX_X4,
	.pme_count		= LIBPFM_ARRAY_SIZE(arm_cortex_x4_pe),
	.type			= PFM_PMU_TYPE_CORE,
	.supported_plm  = ARMV9_PLM,
	.pe				= arm_cortex_x4_pe,

	.pmu_detect		= pfm_arm_detect_cortex_x4,
	.max_encoding		= 1,
	.num_cntrs		= 6, /* or 31 */

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
